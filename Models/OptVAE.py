import pickle
import tensorflow as tf
from os import environ as os_env
from Utils.Distributions import IGR_I, IGR_Planar, IGR_SB, IGR_SB_Finite
from Utils.Distributions import GS, compute_log_exp_gs_dist
from Models.VAENet import RelaxCovNet
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'


class OptVAE:

    def __init__(self, nets, optimizer, hyper):
        self.nets = nets
        self.optimizer = optimizer
        self.batch_size = hyper['batch_n']
        self.n_required = hyper['n_required']
        self.sample_size = hyper['sample_size']
        self.sample_size_training = hyper['sample_size']
        self.sample_size_testing = hyper['sample_size_testing']
        self.num_of_vars = hyper['num_of_discrete_var']
        self.dataset_name = hyper['dataset_name']
        self.model_type = hyper['model_type']
        self.test_with_one_hot = hyper['test_with_one_hot']
        self.sample_from_cont_kl = hyper['sample_from_cont_kl']
        self.sample_from_disc_kl = hyper['sample_from_disc_kl']
        self.temp = tf.constant(value=hyper['temp'], dtype=tf.float32)
        self.stick_the_landing = hyper['stick_the_landing']

        self.run_jv = hyper['run_jv']
        self.gamma = hyper['gamma']
        self.discrete_c = tf.constant(0.)
        self.continuous_c = tf.constant(0.)

    def perform_fwd_pass(self, x, test_with_one_hot=False):
        self.set_hyper_for_testing(test_with_one_hot)
        params = self.nets.encode(x)
        z = self.reparameterize(params_broad=params)
        x_logit = self.decode_w_or_wo_one_hot(z, test_with_one_hot)
        return z, x_logit, params

    def set_hyper_for_testing(self, test_with_one_hot):
        if test_with_one_hot:
            self.sample_size = self.sample_size_testing
        else:
            self.sample_size = self.sample_size_training

    def decode_w_or_wo_one_hot(self, z, test_with_one_hot):
        if test_with_one_hot:
            batch_n, categories_n, sample_size, var_num = z[-1].shape
            zz = []
            for idx in range(len(z)):
                one_hot = tf.transpose(tf.one_hot(tf.argmax(z[idx], axis=1), depth=categories_n),
                                       perm=[0, 3, 1, 2])
                zz.append(one_hot)
            x_logit = self.decode(z=zz)
        else:
            x_logit = self.decode(z=z)
        return x_logit

    def reparameterize(self, params_broad):
        mean, log_var = params_broad
        z = sample_normal(mean=mean, log_var=log_var)
        return z

    def decode(self, z):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            x_logit = self.decode_gaussian(z=z)
        else:
            x_logit = self.decode_bernoulli(z=z)
        return x_logit

    def decode_bernoulli(self, z):
        batch_n, sample_size = z[0].shape[0], z[0].shape[2]
        z = reshape_and_stack_z(z=z)
        x_logit = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                 element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(sample_size):
            x_logit = x_logit.write(index=i, value=self.nets.decode(z[:, :, i])[0])
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0])
        return x_logit

    def decode_gaussian(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, sample_size = z.shape[0], z.shape[2]
        mu = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        xi = tf.TensorArray(dtype=tf.float32, size=sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(sample_size):
            z_mu, z_xi = self.nets.decode(z[:, :, i])
            mu = mu.write(index=i, value=z_mu)
            xi = xi.write(index=i, value=z_xi)
        mu = tf.transpose(mu.stack(), perm=[1, 2, 3, 4, 0])
        xi = tf.transpose(xi.stack(), perm=[1, 2, 3, 4, 0])
        x_logit = [mu, xi]
        return x_logit

    def compute_kl_elements(self, z, params_broad,
                            sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        mean, log_var = params_broad
        if sample_from_cont_kl:
            kl_norm = sample_kl_norm(z_norm=z, mean=mean, log_var=log_var)
        else:
            kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        kl_dis = tf.constant(0.) if sample_from_disc_kl else tf.constant(0.)
        return kl_norm, kl_dis

    # @tf.function() -- messes up all the computations. I'm not sure why this goes so wrong
    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            log_px_z = compute_log_gaussian_pdf(x=x, x_logit=x_logit)
        else:
            log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit)
        kl_norm, kl_dis = self.compute_kl_elements(z=z, params_broad=params_broad,
                                                   sample_from_cont_kl=sample_from_cont_kl,
                                                   sample_from_disc_kl=sample_from_disc_kl,
                                                   test_with_one_hot=test_with_one_hot)
        kl = kl_norm + kl_dis
        loss = compute_loss(log_px_z=log_px_z, kl_norm=kl_norm, kl_dis=kl_dis,
                            run_jv=self.run_jv, gamma=self.gamma,
                            discrete_c=self.discrete_c, continuous_c=self.continuous_c)
        output = (loss, tf.reduce_mean(log_px_z), tf.reduce_mean(kl),
                  tf.reduce_mean(kl_norm), tf.reduce_mean(kl_dis))
        return output

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl, sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                         test_with_one_hot=self.test_with_one_hot)
        output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                   sample_from_cont_kl=sample_from_cont_kl,
                                   sample_from_disc_kl=sample_from_disc_kl,
                                   test_with_one_hot=self.test_with_one_hot)
        loss, recon, kl, kl_norm, kl_dis = output
        return loss, recon, kl, kl_norm, kl_dis

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x, test_with_one_hot=False)
            output = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                       sample_from_cont_kl=self.sample_from_cont_kl,
                                       sample_from_disc_kl=self.sample_from_disc_kl,
                                       test_with_one_hot=False)
            loss, recon, kl, kl_n, kl_d = output
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        return gradients, loss, recon, kl, kl_n, kl_d

    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.nets.trainable_variables))


class OptExpGS(OptVAE):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.dist = GS(log_pi=tf.constant(1., dtype=tf.float32, shape=(1, 1, 1, 1)),
                       temp=self.temp)
        self.log_psi = tf.constant(1., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.use_continuous = True

    def reparameterize(self, params_broad):
        mean, log_var, logits = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.dist = GS(log_pi=logits, sample_size=self.sample_size, temp=self.temp)
        self.dist.generate_sample()
        self.log_psi = self.dist.log_psi
        z_discrete = self.dits.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def compute_kl_elements(self, z, params_broad,
                            sample_from_cont_kl, sample_from_disc_kl,
                            test_with_one_hot):
        if self.use_continuous:
            mean, log_var, log_alpha = params_broad
            z_norm, _ = z
            if sample_from_cont_kl:
                z_norm, _ = z
                kl_norm = sample_kl_norm(z_norm=z, mean=mean, log_var=log_var)
            else:
                kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        else:
            log_alpha = params_broad[0]
            if self.stick_the_landing:
                log_alpha = tf.stop_gradient(log_alpha)
            kl_norm = 0.
        kl_dis = self.compute_discrete_kl(log_alpha, sample_from_disc_kl)
        return kl_norm, kl_dis

    def compute_discrete_kl(self, log_alpha, sample_from_disc_kl):
        if sample_from_disc_kl:
            kl_dis = sample_kl_exp_gs(log_psi=self.log_psi, log_pi=log_alpha,
                                      temp=self.temp)
        else:
            kl_dis = calculate_categorical_closed_kl(log_alpha=log_alpha, normalize=True)
        return kl_dis


class OptExpGSDis(OptExpGS):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.use_continuous = False

    def reparameterize(self, params_broad):
        self.dist = GS(log_pi=params_broad[0], sample_size=self.sample_size, temp=self.temp)
        self.dist.generate_sample()
        self.n_required = self.dist.psi.shape[1]
        self.log_psi = self.dist.log_psi
        z_discrete = [self.dist.psi]
        return z_discrete


class OptRELAXGSDis(OptExpGSDis):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizer=optimizers[0], hyper=hyper)
        self.optimizer_encoder = self.optimizer
        self.optimizer_decoder = optimizers[1]
        self.optimizer_var = optimizers[2]
        cov_net_shape = (self.n_required, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape)
        self.log_temp = tf.Variable(self.temp, name='temp', trainable=True)

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl=None, sample_from_disc_kl=None,
                     test_with_one_hot=False):
        log_alpha = params_broad[0]
        log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit)
        log_p = compute_log_categorical_pmf(z, tf.zeros_like(log_alpha))
        log_qz_x = compute_log_categorical_pmf(z, log_alpha)
        # kl = log_p - log_qz_x
        kl = tf.reduce_mean(log_p - log_qz_x, axis=0)
        # kl = tf.reduce_mean(calculate_categorical_closed_kl(log_alpha=log_alpha, normalize=True))
        loss = -tf.math.reduce_mean(log_px_z) - kl
        return loss

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl, sample_from_disc_kl):
        one_hot, x_logit, params_broad = self.perform_fwd_pass(x=x, test_with_one_hot=True)
        loss = self.compute_loss(x, x_logit, one_hot, params_broad)
        recon = tf.constant(0.)
        kl, kl_norm, kl_dis = tf.constant(0.), tf.constant(0.), tf.constant(0.)
        return loss, recon, kl, kl_norm, kl_dis

    @tf.function()
    def compute_gradients(self, x):
        decoder_vars = [v for v in self.nets.trainable_variables if 'decoder' in v.name]
        encoder_vars = [v for v in self.nets.trainable_variables if 'encoder' in v.name]
        con_net_vars = self.relax_cov.net.trainable_variables
        with tf.GradientTape(persistent=True) as tape_cov:
            with tf.GradientTape(persistent=True) as tape:
                log_alpha = self.nets.encode(x)[0]
                tape.watch(log_alpha)
                output = self.get_relax_variables_from_params(log_alpha)
                one_hot, x_logit = output
                output = self.compute_relax_ingredients(x=x, x_logit=x_logit,
                                                        log_alpha=log_alpha,
                                                        one_hot=one_hot[0])
                log_qz_x_grad_theta = compute_log_categorical_pmf_grad(one_hot, log_alpha)
                c_phi, c_phi_tilde, log_qz_x = output
                loss = self.compute_loss(x=x, x_logit=x_logit, params_broad=[log_alpha], z=one_hot)

            c_phi_z_grad_theta = tape.gradient(target=c_phi, sources=log_alpha)
            c_phi_z_tilde_grad_theta = tape.gradient(target=c_phi_tilde, sources=log_alpha)
            log_qz_x_grad_theta = tape.gradient(target=log_qz_x, sources=log_alpha)
            breakpoint()

            c_phi_z_grad = tape.gradient(target=c_phi, sources=encoder_vars)
            c_phi_z_tilde_grad = tape.gradient(target=c_phi_tilde, sources=encoder_vars)
            # log_qz_x_grad = tape.gradient(target=log_qz_x, sources=encoder_vars)
            log_qz_x_grad = tape.gradient(target=log_qz_x_grad_theta, sources=encoder_vars)

            decoder_grads = tape.gradient(target=loss, sources=decoder_vars)
            encoder_grads = []
            diff = loss - c_phi_tilde
            for idx in range(len(encoder_vars)):
                relax_grad = self.compute_relax_grad(diff, log_qz_x_grad[idx],
                                                     c_phi_z_grad[idx], c_phi_z_tilde_grad[idx])
                encoder_grads.append(relax_grad)

            relax_grad_theta = self.compute_relax_grad(diff, log_qz_x_grad_theta,
                                                       c_phi_z_grad_theta, c_phi_z_tilde_grad_theta)
            variance = self.compute_relax_grad_variance(relax_grad_theta)
        # cov_net_grad = tape_cov.gradient(target=variance, sources=con_net_vars)
        cov_net_grad_net = tape_cov.gradient(target=variance, sources=con_net_vars)
        cov_net_grad_temp = tape_cov.gradient(target=variance, sources=self.log_temp)
        cov_net_grad = cov_net_grad_net + [cov_net_grad_temp]

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        output = (gradients, loss, tf.constant(0.), tf.constant(0.), tf.constant(0.),
                  tf.constant(0.))
        return output

    @staticmethod
    def compute_relax_grad(diff, log_qz_x_grad, c_phi_z_grad, c_phi_z_tilde_grad):
        relax_grad = diff * log_qz_x_grad
        relax_grad += c_phi_z_grad
        relax_grad -= c_phi_z_tilde_grad
        # TODO: verify this step
        relax_grad += log_qz_x_grad
        return relax_grad

    def get_relax_variables_from_params(self, log_alpha):
        z = self.reparameterize(params_broad=[log_alpha])[0]
        one_hot = tf.transpose(tf.one_hot(tf.argmax(z, axis=1), depth=self.n_required),
                               perm=[0, 3, 1, 2])
        x_logit = self.decode([one_hot])
        return one_hot, x_logit

    def compute_relax_ingredients(self, x, x_logit, log_alpha, one_hot):
        u = tf.random.uniform(shape=log_alpha.shape)
        offset = 1.e-20
        z_un = log_alpha - tf.math.log(-tf.math.log(u + offset) + offset)
        # TODO: verify what to do with the sampling
        # z_tilde_un = sample_z_tilde(one_hot, log_alpha)
        z_tilde_un = log_alpha - tf.math.log(-tf.math.log(u + offset) + offset)

        c_phi = self.compute_c_phi(z_un, x, x_logit, log_alpha)
        c_phi_tilde = self.compute_c_phi(z_tilde_un, x, x_logit, log_alpha)

        log_qz_x = compute_log_categorical_pmf(one_hot[0], log_alpha)
        return (c_phi, c_phi_tilde, log_qz_x)

    def compute_c_phi(self, z_un, x, x_logit, log_alpha):
        r = tf.math.reduce_mean(self.relax_cov.net(z_un), axis=0)
        temp = tf.math.exp(self.log_temp)
        z = tf.math.softmax(z_un / temp, axis=1)
        c_phi = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=[log_alpha]) + r
        return c_phi

    @staticmethod
    def compute_relax_grad_variance(relax_grad):
        # TODO: check on paper how to implement the dimensionality reduction for variance
        variance = tf.math.square(relax_grad)
        variance = tf.math.reduce_sum(variance, axis=(1, 2, 3))
        variance = tf.math.reduce_mean(variance, axis=0)
        return variance

    def apply_gradients(self, gradients):
        encoder_grads, decoder_grads, cov_net_grad = gradients
        encoder_vars = [v for v in self.nets.trainable_variables if 'encoder' in v.name]
        decoder_vars = [v for v in self.nets.trainable_variables if 'decoder' in v.name]
        # con_net_vars = self.relax_cov.net.trainable_variables
        con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp]
        self.optimizer_encoder.apply_gradients(zip(encoder_grads, encoder_vars))
        self.optimizer_decoder.apply_gradients(zip(decoder_grads, decoder_vars))
        self.optimizer_var.apply_gradients(zip(cov_net_grad, con_net_vars))


class OptIGR(OptVAE):
    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.mu_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.xi_0 = tf.constant(value=0., dtype=tf.float32, shape=(1, 1, 1, 1))
        self.dist = IGR_I(mu=self.mu_0, xi=self.xi_0, temp=self.temp)
        self.use_continuous = True
        self.prior_file = hyper['prior_file']

    def reparameterize(self, params_broad):
        mean, log_var, mu, xi = params_broad
        z_norm = sample_normal(mean=mean, log_var=log_var)
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        z_discrete = self.dist.psi
        self.n_required = z_discrete.shape[1]
        z = [z_norm, z_discrete]
        return z

    def select_distribution(self, mu, xi):
        self.dist = IGR_I(mu=mu, xi=xi, temp=self.temp, sample_size=self.sample_size)

    def compute_kl_elements(self, z, params_broad,
                            sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        if self.use_continuous:
            mean, log_var, mu_disc, xi_disc = params_broad
            if sample_from_cont_kl:
                z_norm, _ = z
                kl_norm = sample_kl_norm(z_norm=z_norm, mean=mean, log_var=log_var)
            else:
                kl_norm = calculate_simple_closed_gauss_kl(mean=mean, log_var=log_var)
        else:
            mu_disc, xi_disc = params_broad
            if self.stick_the_landing:
                mu_disc = tf.stop_gradient(mu_disc)
                xi_disc = tf.stop_gradient(xi_disc)
            kl_norm = 0.
        if test_with_one_hot and not sample_from_disc_kl:
            batch_n, categories_n, sample_size, var_num = z[-1].shape
            one_hot = tf.transpose(tf.one_hot(tf.argmax(z[-1], axis=1), depth=categories_n),
                                   perm=[0, 3, 1, 2])
            p_discrete = tf.reduce_mean(one_hot, axis=2, keepdims=True)
            kl_dis = calculate_categorical_closed_kl(log_alpha=p_discrete, normalize=False)
        else:
            kl_dis = self.compute_discrete_kl(mu_disc, xi_disc, sample_from_disc_kl)
        return kl_norm, kl_dis

    def compute_discrete_kl(self, mu_disc, xi_disc, sample_from_disc_kl):
        mu_disc_prior, xi_disc_prior = self.update_prior_values()
        if sample_from_disc_kl:
            kl_dis = self.compute_sampled_discrete_kl(mu_disc, xi_disc,
                                                      mu_disc_prior, xi_disc_prior)
        else:
            kl_dis = calculate_general_closed_form_gauss_kl(mean_q=mu_disc,
                                                            log_var_q=2. * xi_disc,
                                                            mean_p=mu_disc_prior,
                                                            log_var_p=2. * xi_disc_prior,
                                                            axis=(1, 3))
        return kl_dis

    def compute_sampled_discrete_kl(self, mu_disc, xi_disc,
                                    mu_disc_prior, xi_disc_prior):
        log_qz_x = compute_log_normal_pdf(self.dist.kappa,
                                          mean=mu_disc, log_var=2. * xi_disc)
        log_pz = compute_log_normal_pdf(self.dist.kappa,
                                        mean=mu_disc_prior, log_var=2. * xi_disc_prior)
        kl_dis = tf.reduce_sum(log_qz_x - log_pz, axis=2)
        return kl_dis

    def update_prior_values(self):
        current_batch_n = self.dist.lam.shape[0]
        mu_disc_prior = self.mu_0[:current_batch_n, :, :]
        xi_disc_prior = self.xi_0[:current_batch_n, :, :]
        return mu_disc_prior, xi_disc_prior

    def load_prior_values(self):
        with open(file=self.prior_file, mode='rb') as f:
            parameters = pickle.load(f)

        mu_0 = tf.constant(parameters['mu'], dtype=tf.float32)
        xi_0 = tf.constant(parameters['xi'], dtype=tf.float32)
        categories_n = mu_0.shape[1]
        # TODO: check if this idea is valid
        prior_shape = mu_0.shape
        # mu_0 = tf.math.reduce_mean(mu_0 - 0.075, keepdims=True)
        mu_0 = tf.math.reduce_mean(mu_0 + 0.05, keepdims=True)
        mu_0 = tf.broadcast_to(mu_0, shape=prior_shape)
        # xi_0 = tf.math.reduce_mean(xi_0 - 0.05, keepdims=True)
        xi_0 = tf.math.reduce_mean(xi_0 + 0.05, keepdims=True)
        xi_0 = tf.broadcast_to(xi_0, shape=prior_shape)

        self.mu_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=mu_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)
        self.xi_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=xi_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)


class OptIGRDis(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.use_continuous = False
        self.load_prior_values()

    def reparameterize(self, params_broad):
        mu, xi = params_broad
        # mu += self.mu_0
        # xi += self.xi_0
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        z_discrete = [self.dist.psi]
        return z_discrete


class OptPlanarNF(OptIGR):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)

    def select_distribution(self, mu, xi):
        self.dist = IGR_Planar(mu=mu, xi=xi, planar_flow=self.nets.planar_flow,
                               temp=self.temp, sample_size=self.sample_size)

    def compute_sampled_discrete_kl(self, mu_disc, xi_disc, mu_disc_prior, xi_disc_prior):
        log_qz_x = compute_log_normal_pdf(self.dist.kappa,
                                          mean=mu_disc, log_var=2. * xi_disc)
        log_pz = compute_log_normal_pdf(self.dist.lam,
                                        mean=mu_disc_prior, log_var=2. * xi_disc_prior)
        kl_dis = tf.reduce_sum(log_qz_x - log_pz, axis=2)
        pf_log_jac_det = calculate_planar_flow_log_determinant(self.dist.kappa,
                                                               self.nets.planar_flow)
        kl_dis = kl_dis + pf_log_jac_det
        return kl_dis


class OptPlanarNFDis(OptIGRDis, OptPlanarNF):

    def __init__(self, nets, optimizer, hyper):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)


class OptSBFinite(OptIGR):

    def __init__(self, nets, optimizer, hyper, use_continuous):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper)
        self.prior_file = hyper['prior_file']
        self.use_continuous = use_continuous

    def reparameterize(self, params_broad):
        z = []
        self.load_prior_values()
        if self.use_continuous:
            mean, log_var, mu, xi = params_broad
            z_norm = sample_normal(mean=mean, log_var=log_var)
            z.append(z_norm)
        else:
            mu, xi = params_broad
        self.select_distribution(mu, xi)
        self.dist.generate_sample()
        self.n_required = self.dist.psi.shape[1]
        z_discrete = self.complete_discrete_vector()

        z.append(z_discrete)
        return z

    def select_distribution(self, mu, xi):
        self.dist = IGR_SB_Finite(mu, xi, self.temp, self.sample_size)

    def complete_discrete_vector(self):
        z_discrete = self.dist.psi
        return z_discrete

    def load_prior_values(self):
        with open(file=self.prior_file, mode='rb') as f:
            parameters = pickle.load(f)

        mu_0 = tf.constant(parameters['mu'], dtype=tf.float32)
        xi_0 = tf.constant(parameters['xi'], dtype=tf.float32)
        categories_n = mu_0.shape[1]

        self.mu_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=mu_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)
        self.xi_0 = shape_prior_to_sample_size_and_discrete_var_num(
            prior_param=xi_0, batch_size=self.batch_size, categories_n=categories_n,
            sample_size=self.sample_size, discrete_var_num=self.nets.disc_var_num)


class OptSB(OptSBFinite):

    def __init__(self, nets, optimizer, hyper, use_continuous):
        super().__init__(nets=nets, optimizer=optimizer, hyper=hyper,
                         use_continuous=use_continuous)
        self.max_categories = hyper['latent_discrete_n']
        self.threshold = hyper['threshold']
        self.truncation_option = hyper['truncation_option']
        self.prior_file = hyper['prior_file']
        self.quantile = 50
        self.use_continuous = use_continuous

    def select_distribution(self, mu, xi):
        self.dist = IGR_SB(mu, xi, sample_size=self.sample_size,
                           temp=self.temp, threshold=self.threshold)
        self.dist.truncation_option = self.truncation_option
        self.dist.quantile = self.quantile

    def complete_discrete_vector(self):
        batch_size, n_required = self.dist.psi.shape[0], self.dist.psi.shape[1]
        missing = self.max_categories - n_required
        zeros = tf.constant(value=0., dtype=tf.float32,
                            shape=(batch_size, missing, self.sample_size, self.num_of_vars))
        z_discrete = tf.concat([self.dist.psi, zeros], axis=1)
        return z_discrete


# =================================================================================================
def compute_loss(log_px_z, kl_norm, kl_dis, run_jv=False,
                 gamma=tf.constant(1.), discrete_c=tf.constant(0.), continuous_c=tf.constant(0.)):
    if run_jv:
        loss = -tf.reduce_mean(log_px_z - gamma * tf.math.abs(kl_norm - continuous_c)
                               - gamma * tf.math.abs(kl_dis - discrete_c))
    else:
        kl = kl_norm + kl_dis
        elbo = log_px_z - kl
        # sample_size = log_px_z.shape[1]
        sample_size = 1
        elbo_iwae = tf.math.reduce_logsumexp(elbo, axis=1)
        loss = -tf.math.reduce_mean(elbo_iwae, axis=0)
        loss += tf.math.log(tf.constant(sample_size, dtype=tf.float32))
    return loss


def compute_log_bernoulli_pdf(x, x_logit):
    x_broad = infer_shape(fromm=x_logit, tto=x)
    cross_ent = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad, logits=x_logit)
    log_px_z = tf.reduce_sum(cross_ent, axis=(1, 2, 3))
    return log_px_z


def compute_log_categorical_pmf(d, log_alpha):
    log_normalized = log_alpha - tf.reduce_logsumexp(log_alpha, axis=1, keepdims=True)
    log_categorical_pmf = tf.math.reduce_sum(d * log_normalized, axis=1)
    log_categorical_pmf = tf.math.reduce_sum(log_categorical_pmf, axis=(1, 2))
    # log_categorical_pmf = tf.math.reduce_mean(log_categorical_pmf)
    return log_categorical_pmf


def compute_log_categorical_pmf_grad(d, log_alpha):
    normalized = tf.math.softmax(log_alpha, axis=1)
    grad = d - normalized
    return grad


def compute_log_gaussian_pdf(x, x_logit):
    mu, xi = x_logit
    mu = tf.math.sigmoid(mu)
    xi = 1.e-6 + tf.math.softplus(xi)
    pi = 3.141592653589793

    x_broad = infer_shape(fromm=mu, tto=x)

    log_pixel = (- 0.5 * ((x_broad - mu) / xi) ** 2. -
                 0.5 * tf.math.log(2 * pi) - tf.math.log(1.e-8 + xi))
    log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])
    return log_px_z


def infer_shape(fromm, tto):
    # batch_size, image_size, sample_size = fromm.shape[0], fromm.shape[1:4], fromm.shape[4]
    batch_size, image_size, sample_size = fromm.shape[0], fromm.shape[1:4], 1
    x_w_extra_col = tf.reshape(tto, shape=(batch_size,) + image_size + (1,))
    x_broad = tf.broadcast_to(x_w_extra_col, shape=(batch_size,) + image_size + (sample_size,))
    return x_broad


def sample_z_tilde(one_hot, log_alpha):
    offset = 1.e-20
    bool_one_hot = tf.cast(one_hot, dtype=tf.bool)
    theta = tf.math.softmax(log_alpha, axis=1)
    v = tf.random.uniform(shape=log_alpha.shape)
    v_b = tf.where(bool_one_hot, v, 0.)
    v_b = tf.math.reduce_max(v_b, axis=1, keepdims=True)
    v_b = tf.broadcast_to(v_b, shape=v.shape)

    # The theta is where the problem is coming from
    aux1 = -tf.math.log(v + offset) / tf.clip_by_value(theta, 1.e-5, 1.)
    aux2 = tf.math.log(v_b + offset)
    aux = aux1 - aux2
    # tf.print(tf.linalg.norm(aux1))
    z_other = -tf.math.log(aux + offset)
    z_b = -tf.math.log(-tf.math.log(v_b + offset) + offset)
    z_tilde = tf.where(bool_one_hot, z_b, z_other)
    return z_tilde


def sample_normal(mean, log_var):
    epsilon = tf.random.normal(shape=mean.shape)
    z_norm = mean + tf.math.exp(log_var * 0.5) * epsilon
    return z_norm


def sample_kl_norm(z_norm, mean, log_var):
    log_pz = compute_log_normal_pdf(sample=z_norm, mean=0., log_var=0.)
    log_qz_x = compute_log_normal_pdf(sample=z_norm, mean=mean, log_var=log_var)
    kl_norm = log_qz_x - log_pz
    return kl_norm


def calculate_simple_closed_gauss_kl(mean, log_var):
    kl_norm = 0.5 * tf.reduce_sum(tf.math.exp(log_var) + tf.math.pow(mean, 2) -
                                  log_var - tf.constant(1.),
                                  axis=1)
    return kl_norm


def calculate_general_closed_form_gauss_kl(mean_q, log_var_q, mean_p, log_var_p, axis=(1,)):
    var_q = tf.math.exp(log_var_q)
    var_p = tf.math.exp(log_var_p)

    trace_term = tf.reduce_sum(var_q / var_p - 1., axis=axis)
    means_term = tf.reduce_sum(tf.math.pow(mean_q - mean_p, 2) / var_p, axis=axis)
    log_det_term = tf.reduce_sum(log_var_p - log_var_q, axis=axis)
    kl_norm = 0.5 * (trace_term + means_term + log_det_term)
    return kl_norm


def calculate_planar_flow_log_determinant(z, planar_flow):
    log_det = tf.constant(0., dtype=tf.float32)
    nested_layers = int(len(planar_flow.weights) / 3)
    zl = z
    for l in range(nested_layers):
        pf_layer = planar_flow.get_layer(index=l)
        w, b, _ = pf_layer.weights
        u = pf_layer.get_u_tilde()
        uTw = tf.math.reduce_sum(u * w, axis=1)
        wTz = tf.math.reduce_sum(w * zl, axis=1)
        h_prime = 1. - tf.math.tanh(wTz + b[0, :, :, :]) ** 2
        log_det -= tf.math.log(tf.math.abs(1 + h_prime * uTw))
        zl = pf_layer.call(zl)
    log_det = tf.reduce_sum(log_det, axis=[-1])
    return log_det


def compute_log_normal_pdf(sample, mean, log_var):
    pi = 3.141592653589793
    log2pi = -0.5 * tf.math.log(2 * pi)
    log_exp_sum = -0.5 * (sample - mean) ** 2 * tf.math.exp(-log_var)
    log_normal_pdf = tf.reduce_sum(log2pi + -0.5 * log_var + log_exp_sum, axis=1)
    return log_normal_pdf


def calculate_categorical_closed_kl(log_alpha, normalize=True):
    offset = 1.e-20
    categories_n = tf.constant(log_alpha.shape[1], dtype=tf.float32)
    log_uniform_inv = tf.math.log(categories_n)
    pi = tf.math.softmax(log_alpha, axis=1) if normalize else log_alpha
    kl_discrete = tf.reduce_sum(pi * (tf.math.log(pi + offset) + log_uniform_inv), axis=(1, 3))
    return kl_discrete


def sample_kl_exp_gs(log_psi, log_pi, temp):
    uniform_probs = get_broadcasted_uniform_probs(shape=log_psi.shape)
    # log_pz = compute_log_exp_gs_dist(log_psi=log_psi, logits=tf.math.log(uniform_probs), temp=temp)
    temp_prior = tf.constant(0.5, dtype=tf.float32)
    log_pz = compute_log_exp_gs_dist(log_psi=log_psi, logits=tf.math.log(uniform_probs),
                                     temp=temp_prior)
    log_qz_x = compute_log_exp_gs_dist(log_psi=log_psi, logits=log_pi, temp=temp)
    kl_discrete = tf.math.reduce_sum(log_qz_x - log_pz, axis=2)
    return kl_discrete


def get_broadcasted_uniform_probs(shape):
    batch_n, categories_n, sample_size, disc_var_num = shape
    uniform_probs = tf.constant([1 / categories_n for _ in range(categories_n)], dtype=tf.float32,
                                shape=(1, categories_n, 1, 1))
    uniform_probs = shape_prior_to_sample_size_and_discrete_var_num(uniform_probs, batch_n,
                                                                    categories_n, sample_size,
                                                                    disc_var_num)
    return uniform_probs


def shape_prior_to_sample_size_and_discrete_var_num(prior_param, batch_size, categories_n,
                                                    sample_size, discrete_var_num):
    prior_param = tf.reshape(prior_param, shape=(1, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, 1, 1))
    prior_param = tf.broadcast_to(prior_param, shape=(batch_size, categories_n, sample_size, 1))
    prior_param = tf.broadcast_to(prior_param,
                                  shape=(batch_size, categories_n, sample_size, discrete_var_num))
    return prior_param


def reshape_and_stack_z(z):
    if len(z) > 1:
        z = tf.concat(z, axis=1)
        z = flatten_discrete_variables(original_z=z)
    else:
        z = flatten_discrete_variables(original_z=z[0])
    return z


def flatten_discrete_variables(original_z):
    batch_n, disc_latent_n, sample_size, disc_var_num = original_z.shape
    z_discrete = tf.TensorArray(dtype=tf.float32, size=sample_size,
                                element_shape=(batch_n, disc_var_num * disc_latent_n))
    for i in tf.range(sample_size):
        value = tf.reshape(original_z[:, :, i, :],
                           shape=(batch_n, disc_var_num * disc_latent_n))
        z_discrete = z_discrete.write(index=i, value=value)
    z_discrete = tf.transpose(z_discrete.stack(), perm=[1, 2, 0])
    return z_discrete


def make_one_hot(z_dis):
    categories_n = z_dis.shape[1]
    idx = tf.argmax(z_dis, axis=1)
    one_hot = tf.transpose(tf.one_hot(idx, depth=categories_n), perm=[0, 3, 1, 2])
    return one_hot
