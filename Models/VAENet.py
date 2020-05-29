import tensorflow as tf
from Utils.general import append_timestamp_to_file
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VAENet(tf.keras.Model):
    def __init__(self, hyper: dict):
        super(VAENet, self).__init__()
        self.cont_latent_n = hyper['latent_norm_n']
        self.cont_var_num = hyper['num_of_norm_var']
        self.cont_param_num = hyper['num_of_norm_param']
        self.disc_latent_in = hyper['n_required']
        self.disc_latent_out = hyper['latent_discrete_n']
        self.disc_var_num = hyper['num_of_discrete_var']
        self.disc_param_num = hyper['num_of_discrete_param']
        self.architecture_type = hyper['architecture']
        self.dataset_name = hyper['dataset_name']
        self.model_type = hyper['model_type']
        self.image_shape = hyper['image_shape']

        self.log_px_z_params_num = 1 if self.dataset_name == 'mnist' else 2
        self.latent_dim_in = (self.cont_param_num * self.cont_latent_n * self.cont_var_num +
                              self.disc_param_num * self.disc_latent_in * self.disc_var_num)
        self.latent_dim_out = (self.cont_var_num * self.cont_latent_n +
                               self.disc_var_num * self.disc_latent_out)
        self.split_sizes_list = [self.cont_latent_n *
                                 self.cont_var_num for _ in range(self.cont_param_num)]
        self.split_sizes_list += [self.disc_latent_in *
                                  self.disc_var_num for _ in range(self.disc_param_num)]
        self.num_var = (self.cont_var_num, self.disc_var_num)

        self.inference_net = tf.keras.Sequential
        self.generative_net = tf.keras.Sequential
        self.planar_flow = tf.keras.Sequential

    def construct_architecture(self):
        if self.architecture_type == 'dense':
            self.generate_dense_inference_net()
            if self.model_type.find('Planar') > 0:
                self.generate_planar_flow()
            self.generate_dense_generative_net()
        elif self.architecture_type == 'dense_nonlinear':
            self.generate_dense_nonlinear_inference_net()
            if self.model_type.find('Planar') > 0:
                self.generate_planar_flow()
            self.generate_dense_nonlinear_generative_net()
        elif self.architecture_type == 'dense_relax':
            self.generate_relax_inference_net()
            self.generate_relax_generative_net()
        elif self.architecture_type == 'conv':
            self.generate_convolutional_inference_net()
            self.generate_convolutional_generative_net()
        elif self.architecture_type == 'conv_jointvae':
            if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
                self.generate_convolutional_inference_net_jointvae_celeb_a()
                if self.model_type.find('Planar') > 0:
                    self.generate_planar_flow()
                self.generate_convolutional_generative_net_jointvae_celeb_a()
            else:
                self.generate_convolutional_inference_net_jointvae()
                if self.model_type.find('Planar') > 0:
                    self.generate_planar_flow()
                self.generate_convolutional_generative_net_jointvae()

    # -------------------------------------------------------------------------------------------
    def generate_dense_inference_net(self, activation='linear'):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.image_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=self.latent_dim_in, name='encoder_1'),
        ])

    def generate_dense_generative_net(self, activation='linear'):
        activation_type = self.determine_activation_from_case()
        image_flat = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        image_flat *= self.log_px_z_params_num
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_out,)),
            tf.keras.layers.Dense(units=image_flat, activation=activation_type, name='decoder_1'),
            tf.keras.layers.Reshape(target_shape=(self.image_shape[0], self.image_shape[1],
                                                  self.image_shape[2] * self.log_px_z_params_num))
        ])

    # --------------------------------------------------------------------------------------------
    def generate_relax_inference_net(self):
        input_layer = tf.keras.layers.Input(shape=self.image_shape)
        flat_layer = tf.keras.layers.Flatten()(input_layer)
        layer1 = tf.keras.layers.Dense(units=self.latent_dim_in, name='encoder_1',
                                       activation='relu')(2. * flat_layer - 1.)
        layer2 = tf.keras.layers.Dense(units=self.latent_dim_in, name='encoder_2',
                                       activation='relu')(layer1)
        layer3 = tf.keras.layers.Dense(units=self.latent_dim_in,
                                       name='encoder_out')(layer2)
        self.inference_net = tf.keras.Model(inputs=[input_layer], outputs=[layer3])

    def generate_relax_generative_net(self):
        image_flat = self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        output_layer = tf.keras.layers.Input(shape=(self.latent_dim_out,))
        layer1 = tf.keras.layers.Dense(units=200, activation='relu',
                                       name='decoder_1')(2. * output_layer - 1.)
        layer2 = tf.keras.layers.Dense(units=200, activation='relu',
                                       name='decoder_2')(layer1)
        layer3 = tf.keras.layers.Dense(units=image_flat,
                                       name='decoder_out')(layer2)
        reshaped_layer = tf.keras.layers.Reshape(target_shape=(self.image_shape[0], self.image_shape[1],
                                                               self.image_shape[2]
                                                               * self.log_px_z_params_num))(layer3)
        self.generative_net = tf.keras.Model(inputs=[output_layer], outputs=[reshaped_layer])

    # --------------------------------------------------------------------------------------------
    def generate_dense_nonlinear_inference_net(self, activation='relu'):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.image_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=512, activation=activation, name='encoder_1'),
            tf.keras.layers.Dense(units=256, activation=activation, name='encoder_2'),
            tf.keras.layers.Dense(units=self.latent_dim_in, name='encoder_3'),
        ])

    def generate_dense_nonlinear_generative_net(self, activation='relu'):
        activation_type = self.determine_activation_from_case()
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_dim_out,)),
            tf.keras.layers.Dense(units=256, activation=activation, name='decoder_1'),
            tf.keras.layers.Dense(units=512, activation=activation, name='decoder_2'),
            tf.keras.layers.Dense(units=self.image_shape[0] * self.image_shape[1] *
                                  self.image_shape[2] *
                                  self.log_px_z_params_num, activation=activation_type,
                                  name='decoder_3'),
            tf.keras.layers.Reshape(target_shape=(self.image_shape[0], self.image_shape[1],
                                                  self.image_shape[2] * self.log_px_z_params_num))
        ])

    def determine_activation_from_case(self):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            activation_type = 'elu'
        elif self.dataset_name == 'mnist':
            activation_type = 'linear'
        elif self.dataset_name == 'omniglot':
            activation_type = 'linear'
        else:
            raise RuntimeError
        return activation_type

    def generate_planar_flow(self):
        self.planar_flow = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.disc_latent_in, 1, self.disc_var_num)),
            PlanarFlowLayer(units=self.disc_latent_in, var_num=self.disc_var_num),
            PlanarFlowLayer(units=self.disc_latent_in, var_num=self.disc_var_num)])

    # --------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net_jointvae(self):
        input_layer = tf.keras.layers.Input(shape=self.image_shape)
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(input_layer)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        if self.image_shape[0] == 64:
            conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                          activation='relu', padding='same')(conv)
        flat_layer = tf.keras.layers.Flatten()(conv)
        dense_layer = tf.keras.layers.Dense(units=256, activation='relu')(flat_layer)
        if len(self.split_sizes_list) == 1:
            concat_layer = tf.keras.layers.Dense(units=self.split_sizes_list[0])(dense_layer)
        else:
            params_layer = [tf.keras.layers.Dense(units=u)(dense_layer)
                            for u in self.split_sizes_list]
            concat_layer = tf.keras.layers.Concatenate()(params_layer)
        self.inference_net = tf.keras.Model(inputs=[input_layer], outputs=[concat_layer])

    def generate_convolutional_generative_net_jointvae(self):
        output_layer = tf.keras.layers.Input(shape=(self.latent_dim_out,))
        layer = tf.keras.layers.Dense(units=256, activation='relu')(output_layer)
        layer = tf.keras.layers.Dense(units=4 * 4 * 64, activation='relu')(layer)
        layer = tf.keras.layers.Reshape(target_shape=(4, 4, 64))(layer)
        if self.image_shape[0] == 64:
            layer = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                    activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=self.image_shape[2], kernel_size=(4, 4),
                                                strides=(2, 2), padding='same')(layer)
        self.generative_net = tf.keras.Model(inputs=[output_layer], outputs=[layer])

    # --------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net_jointvae_celeb_a(self):
        input_layer = tf.keras.layers.Input(shape=self.image_shape)
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(input_layer)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                      activation='relu', padding='same')(conv)
        if self.image_shape[0] == 64:
            conv = tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                          activation='relu', padding='same')(conv)
        flat_layer = tf.keras.layers.Flatten()(conv)
        dense_layer = tf.keras.layers.Dense(units=256, activation='relu')(flat_layer)
        if len(self.split_sizes_list) == 1:
            concat_layer = tf.keras.layers.Dense(units=self.split_sizes_list[0])(dense_layer)
        else:
            params_layer = [tf.keras.layers.Dense(units=u)(dense_layer)
                            for u in self.split_sizes_list]
            concat_layer = tf.keras.layers.Concatenate()(params_layer)
        self.inference_net = tf.keras.Model(inputs=[input_layer], outputs=[concat_layer])

    def generate_convolutional_generative_net_jointvae_celeb_a(self):
        output_layer = tf.keras.layers.Input(shape=(self.latent_dim_out,))
        layer = tf.keras.layers.Dense(units=256, activation='relu')(output_layer)
        layer = tf.keras.layers.Dense(units=4 * 4 * 64, activation='relu')(layer)
        layer = tf.keras.layers.Reshape(target_shape=(4, 4, 64))(layer)
        if self.image_shape[0] == 64:
            layer = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                                    activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2),
                                                activation='relu', padding='same')(layer)
        layer = tf.keras.layers.Conv2DTranspose(filters=self.image_shape[2] *
                                                self.log_px_z_params_num,
                                                kernel_size=(4, 4), strides=(2, 2), padding='same',
                                                activation='elu')(layer)
        self.generative_net = tf.keras.Model(inputs=[output_layer], outputs=[layer])

    # --------------------------------------------------------------------------------------------
    def generate_convolutional_inference_net(self):
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=self.image_shape),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2),
                                   activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2),
                                   activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.latent_dim_in),
        ])

    def generate_convolutional_generative_net(self):
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.latent_out_n,)),
            tf.keras.layers.Dense(units=7 * 7 * 32, activation='relu'),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])

    # --------------------------------------------------------------------------------------------
    def encode(self, x):
        params = self.split_and_reshape_network_parameters(x=x)
        return params

    def split_and_reshape_network_parameters(self, x):
        params = tf.split(self.inference_net(x), num_or_size_splits=self.split_sizes_list, axis=1)
        reshaped_params = []
        for idx, param in enumerate(params):
            batch_size = param.shape[0]
            if self.disc_var_num > 1:
                param = tf.reshape(param,
                                   shape=(batch_size, self.disc_latent_in, 1, self.disc_var_num))
            else:
                param = tf.reshape(param,
                                   shape=(batch_size, self.split_sizes_list[idx], 1, self.disc_var_num))
            reshaped_params.append(param)
        return reshaped_params

    def decode(self, z):
        logits = tf.split(self.generative_net(
            z), num_or_size_splits=self.log_px_z_params_num, axis=3)
        return logits
    # ---------------------------------------------------------------------------------------------


class PlanarFlowLayer(tf.keras.layers.Layer):

    def __init__(self, units, var_num, initializer='random_normal', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.var_num = var_num
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(shape=(1, self.units, 1, self.var_num),
                                 initializer='random_normal',
                                 trainable=True, name='w')
        self.b = self.add_weight(shape=(1, 1, 1, self.var_num), initializer='zeros',
                                 trainable=True, name='b')
        self.u = self.add_weight(shape=(1, self.units, 1, self.var_num),
                                 initializer=self.initializer,
                                 trainable=True, name='u')
        super().build(input_shape)

    def call(self, inputs):
        u_tilde = self.get_u_tilde()

        batch_n, n_required, sample_size, var_num = inputs.shape
        if batch_n is None:
            batch_n = 1

        w_broad = tf.broadcast_to(self.w, shape=(batch_n, n_required, sample_size, var_num))
        u_tilde_broad = tf.broadcast_to(u_tilde, shape=(batch_n, n_required, sample_size, var_num))

        prod_wTinputs = tf.math.reduce_sum(inputs * w_broad, axis=1, keepdims=True)
        tanh = tf.math.tanh(prod_wTinputs + self.b)
        output = inputs + u_tilde_broad * tanh
        return output

    def get_u_tilde(self):
        prod_wTu = tf.math.reduce_sum(self.w * self.u, axis=1)
        alpha = -1 + tf.math.softplus(prod_wTu) - prod_wTu
        u_tilde = self.u + alpha * self.w / tf.linalg.norm(self.w)
        return u_tilde

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'units': self.units, 'var_num': self.var_num}


class RelaxCovNet(tf.keras.models.Model):

    def __init__(self, cov_net_shape, **kwargs):
        super().__init__(**kwargs)
        self.cov_net_shape = cov_net_shape
        # self.net = tf.keras.Sequential([
        #     tf.keras.layers.InputLayer(input_shape=self.cov_net_shape),
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.Dense(units=200, activation='relu'),
        #     tf.keras.layers.Dense(units=200, activation='relu'),
        #     tf.keras.layers.Dense(units=200, activation='relu'),
        #     tf.keras.layers.Dense(units=1),
        # ])

        # self.input_layer = tf.keras.layers.Input(shape=self.cov_net_shape)
        # self.flat_layer = tf.keras.layers.Flatten()(input_layer)
        # self.layer1 = tf.keras.layers.Dense(units=50, activation='relu')(2. * flat_layer - 1.)
        # self.layer2 = tf.keras.layers.Dense(units=1)(layer1)
        # self.net = tf.keras.Model(inputs=[input_layer], outputs=[layer2])

        self.input_layer = tf.keras.layers.Input(shape=self.cov_net_shape)
        self.flat_layer = tf.keras.layers.Flatten()
        self.layer1 = tf.keras.layers.Dense(units=50, activation='relu')
        self.layer2 = tf.keras.layers.Dense(units=1)

        def call(self, inputs):
            breakpoint()
            o1 = self.input_layer(inputs)
            o2 = self.flat_layer(o1)
            o3 = self.layer1(2. * o2 - 1.)
            o4 = self.layer2(o3)
            return o4


def create_nested_planar_flow(nested_layers, latent_n, var_num, initializer='random_normal'):
    sequence = [tf.keras.layers.InputLayer(input_shape=(latent_n, 1, var_num))]
    for l in range(nested_layers):
        sequence.append(PlanarFlowLayer(units=latent_n, var_num=var_num, initializer=initializer))
    planar_flow = tf.keras.Sequential(sequence)
    return planar_flow


def offload_weights_planar_flow(weights):
    num_of_planar_flow_params_per_layer = 3
    nested_layers = int(len(weights) / num_of_planar_flow_params_per_layer)
    input_shape = weights[0].shape
    batch_n, latent_n, sample_size, var_num = input_shape
    pf = create_nested_planar_flow(nested_layers, latent_n, var_num)
    for idx, w in enumerate(weights):
        pf.weights[idx].assign(w)
    return pf


def generate_random_planar_flow_weights(nested_layers, latent_n, var_num):
    weights = []
    for _ in range(nested_layers):
        outputs = generate_random_layer_weights(latent_n, var_num)
        for element in outputs:
            weights.append(element)
    return weights


def generate_random_layer_weights(latent_n, var_num):
    w = tf.Variable(tf.random.normal(mean=0., stddev=1., shape=(1, latent_n, 1, var_num)),
                    name='w')
    b = tf.Variable(tf.random.normal(mean=0., stddev=1., shape=(1, 1, 1, var_num)),
                    name='b')
    u = tf.Variable(tf.random.normal(mean=0., stddev=1., shape=(1, latent_n, 1, var_num)),
                    name='u')
    return w, b, u


def determine_path_to_save_results(model_type, dataset_name):
    results_path = './Log/' + dataset_name + '_' + \
        model_type + append_timestamp_to_file('', termination='')
    return results_path


def construct_networks(hyper):
    model = VAENet(hyper=hyper)
    model.construct_architecture()
    return model
