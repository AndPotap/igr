import tensorflow as tf
from tensorflow.keras.layers import Flatten, InputLayer, Dense, Reshape
from Utils.Distributions import IGR_I, IGR_SB_Finite, IGR_Planar
from Models.VAENet import PlanarFlowLayer


class SOP(tf.keras.Model):
    def __init__(self, hyper: dict):
        super(SOP, self).__init__()
        self.half_image_w_h = hyper['width_height']
        self.half_image_size = hyper['width_height'][0] * hyper['width_height'][1]
        self.units_per_layer = hyper['units_per_layer']
        self.temp = hyper['temp']
        self.model_type = hyper['model_type']
        self.var_num = 1 if self.model_type in ['GS', 'IGR_Iso'] else 2
        self.split_sizes_list = [self.units_per_layer for _ in range(self.var_num)]

        self.input_layer = InputLayer(input_shape=self.half_image_w_h)
        self.flat_layer = Flatten()
        # self.layer1 = tf.keras.layers.Dense(units=self.units_per_layer * self.var_num)
        self.h1_dense = Dense(units=self.units_per_layer)
        self.h2_dense = Dense(units=self.units_per_layer * self.var_num)
        self.out_dense = Dense(units=self.half_image_size)
        self.reshape_out = Reshape(self.half_image_w_h)
        if self.model_type == 'IGR_Planar':
            self.planar_flow = generate_planar_flow(disc_latent_in=1,
                                                    disc_var_num=self.units_per_layer)
        else:
            self.planar_flow = None

    @tf.function()
    def call(self, x_upper, sample_size=1, use_one_hot=False):
        out = self.h1_dense(self.flat_layer(self.input_layer(x_upper)))
        # params_1 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        # z_1 = self.sample_bernoulli(params_1, use_one_hot)

        # out = self.layer2(z_1)
        out = self.h2_dense(out)
        params_2 = tf.split(out, num_or_size_splits=self.split_sizes_list, axis=1)
        z_2 = self.sample_binary(params_2, use_one_hot, sample_size)

        logits = self.get_samples_of_logits(z_2)
        return logits

    @tf.function()
    def get_samples_of_logits(self, z_2):
        batch_n, _, sample_size = z_2.shape
        width, height, rgb = self.half_image_w_h
        logits = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                element_shape=(batch_n, width, height, rgb))
        for i in range(sample_size):
            value = self.reshape_out(self.out_dense(z_2[:, :, i]))
            logits = logits.write(index=i, value=value)
        logits = tf.transpose(logits.stack(), perm=[1, 2, 3, 4, 0])
        return logits

    def sample_binary(self, params, use_one_hot, sample_size):
        if self.model_type == 'GS':
            psi = sample_gs_binary(params=params, temp=self.temp, sample_size=sample_size)
        elif self.model_type == 'IGR_Iso':
            psi = sample_igr_binary_iso(params=params, temp=self.temp, sample_size=sample_size)
        elif self.model_type in ['IGR_I', 'IGR_SB', 'IGR_Planar']:
            psi = sample_igr_binary(model_type=self.model_type, params=params, temp=self.temp,
                                    sample_size=sample_size,
                                    planar_flow=self.planar_flow)
        else:
            raise RuntimeError
        psi = tf.math.round(psi) if use_one_hot else psi
        return psi


@tf.function()
def sample_gs_binary(params, temp, sample_size):
    # TODO: add the latex formulas
    log_alpha = params[0]
    unif = tf.random.uniform(shape=log_alpha.shape + (sample_size,))
    logistic_sample = tf.math.log(unif) - tf.math.log(1. - unif)
    log_alpha = tf.reshape(log_alpha, shape=log_alpha.shape + (1,))
    lam = (log_alpha + logistic_sample) / temp
    # making the output be in {-1, 1} as in Maddison et. al 2017
    psi = 2. * tf.math.sigmoid(lam) - 1.
    return psi


@tf.function()
def sample_igr_binary_iso(params, temp, sample_size):
    mu = params[0]
    eps = tf.random.normal(shape=(mu.shape + (sample_size,)))
    # unif = tf.random.uniform(shape=mu.shape + (sample_size,))
    # eps = tf.math.log(unif) - tf.math.log(1. - unif)
    mu_broad = tf.reshape(mu, shape=mu.shape + (1,))
    lam = mu_broad + eps
    psi = 2. * tf.math.sigmoid(lam / temp) - 1.
    return psi


@tf.function()
def sample_igr_binary(model_type, params, temp, sample_size, planar_flow):
    mu, xi = params
    mu_broad = tf.reshape(mu, shape=mu.shape + (1,))
    xi_broad = tf.reshape(xi, shape=xi.shape + (1,))
    eps = tf.random.normal(shape=(mu.shape + (sample_size,)))
    lam = mu_broad + tf.math.exp(xi_broad) * eps
    lam = mu_broad + eps
    psi = 2. * tf.math.sigmoid(lam / temp) - 1.
    return psi


# @tf.function()
# def sample_igr_binary(model_type, params, temp, sample_size, planar_flow):
#     dist = get_igr_dist(model_type, params, temp, planar_flow, sample_size)
#     dist.generate_sample()
#     lam = tf.transpose(dist.lam[:, 0, :, :], perm=[0, 2, 1])
#     psi = 2. * tf.math.sigmoid(lam / temp) - 1.
#     return psi
#

def get_igr_dist(model_type, params, temp, planar_flow, sample_size):
    mu, xi = params
    batch_n, num_of_vars = mu.shape
    mu_broad = tf.reshape(mu, shape=(batch_n, 1, 1, num_of_vars))
    xi_broad = tf.reshape(xi, shape=(batch_n, 1, 1, num_of_vars))
    if model_type == 'IGR_I':
        dist = IGR_I(mu=mu_broad, xi=xi_broad, temp=temp, sample_size=sample_size)
    elif model_type == 'IGR_Planar':
        dist = IGR_Planar(mu=mu_broad, xi=xi_broad, temp=temp, sample_size=sample_size,
                          planar_flow=planar_flow)
    elif model_type == 'IGR_SB':
        dist = IGR_SB_Finite(mu=mu_broad, xi=xi_broad, temp=temp, sample_size=sample_size)
    else:
        raise ValueError
    return dist


def generate_planar_flow(disc_latent_in, disc_var_num):
    planar_flow = tf.keras.Sequential([InputLayer(input_shape=(disc_latent_in, 1, disc_var_num)),
                                       PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num),
                                       PlanarFlowLayer(units=disc_latent_in, var_num=disc_var_num)])
    return planar_flow


def revert_samples_to_last_dim(a, sample_size):
    batch_n, width, height, rgb = a.shape
    new_shape = (int(batch_n / sample_size), width, height, rgb, sample_size)
    a = tf.reshape(a, shape=new_shape)
    return a


def brodcast_samples_to_batch(x_upper, sample_size):
    batch_n, width, height, rgb = x_upper.shape
    x_upper_broad = brodcast_to_sample_size(x_upper, sample_size=sample_size)
    x_upper_broad = tf.reshape(x_upper_broad, shape=(batch_n * sample_size, width, height, rgb))
    return x_upper_broad


def brodcast_to_sample_size(a, sample_size):
    original_shape = a.shape
    newshape = original_shape + (1,)
    broad_shape = original_shape + (sample_size,)

    a = tf.reshape(a, shape=newshape)
    a = tf.broadcast_to(a, shape=broad_shape)
    return a
