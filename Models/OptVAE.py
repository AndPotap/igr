import pickle
import tensorflow as tf
from os import environ as os_env
from Utils.Distributions import IGR_I, IGR_Planar, IGR_SB, IGR_SB_Finite
from Utils.Distributions import GS, compute_log_exp_gs_dist
from Utils.Distributions import project_to_vertices_via_softmax_pp
from Utils.Distributions import compute_igr_log_probs
from Utils.Distributions import compute_log_gauss_grad
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
        batch_n, _ = z[0].shape[0], z[0].shape[2]
        z = reshape_and_stack_z(z=z)
        x_logit = tf.TensorArray(dtype=tf.float32, size=self.sample_size,
                                 element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(self.sample_size):
            x_logit = x_logit.write(index=i, value=self.nets.decode(z[:, :, i])[0])
        x_logit = tf.transpose(x_logit.stack(), perm=[1, 2, 3, 4, 0])
        return x_logit

    def decode_gaussian(self, z):
        z = reshape_and_stack_z(z=z)
        batch_n, _ = z.shape[0], z.shape[2]
        mu = tf.TensorArray(dtype=tf.float32, size=self.sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        xi = tf.TensorArray(dtype=tf.float32, size=self.sample_size,
                            element_shape=(batch_n,) + self.nets.image_shape)
        for i in tf.range(self.sample_size):
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

    def compute_loss(self, x, x_logit, z, params_broad,
                     sample_from_cont_kl, sample_from_disc_kl, test_with_one_hot):
        if self.dataset_name == 'celeb_a' or self.dataset_name == 'fmnist':
            log_px_z = compute_log_gaussian_pdf(x=x, x_logit=x_logit, sample_size=self.sample_size)
        else:
            log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit, sample_size=self.sample_size)
        kl_norm, kl_dis = self.compute_kl_elements(z=z, params_broad=params_broad,
                                                   sample_from_cont_kl=sample_from_cont_kl,
                                                   sample_from_disc_kl=sample_from_disc_kl,
                                                   test_with_one_hot=test_with_one_hot)
        loss = compute_loss(log_px_z=log_px_z, kl_norm=kl_norm, kl_dis=kl_dis,
                            sample_size=self.sample_size,
                            run_jv=self.run_jv, gamma=self.gamma,
                            discrete_c=self.discrete_c, continuous_c=self.continuous_c)
        return loss

    @tf.function()
    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl, sample_from_disc_kl):
        z, x_logit, params_broad = self.perform_fwd_pass(x=x,
                                                         test_with_one_hot=self.test_with_one_hot)
        loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                 sample_from_cont_kl=sample_from_cont_kl,
                                 sample_from_disc_kl=sample_from_disc_kl,
                                 test_with_one_hot=self.test_with_one_hot)
        return loss

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape:
            z, x_logit, params_broad = self.perform_fwd_pass(x=x, test_with_one_hot=False)
            loss = self.compute_loss(x=x, x_logit=x_logit, z=z, params_broad=params_broad,
                                     sample_from_cont_kl=self.sample_from_cont_kl,
                                     sample_from_disc_kl=self.sample_from_disc_kl,
                                     test_with_one_hot=False)
        gradients = tape.gradient(target=loss, sources=self.nets.trainable_variables)
        return gradients, loss

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


class OptRELAX(OptVAE):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizer=optimizers[0], hyper=hyper)
        self.optimizer_encoder = self.optimizer
        self.optimizer_decoder = optimizers[1]
        self.optimizer_var = optimizers[2]
        cov_net_shape = (self.n_required, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape)
        num_latents = self.n_required * self.num_of_vars
        shape = (1, self.n_required, self.sample_size, self.num_of_vars)
        initial_log_temp = tf.constant([1.6093 for _ in range(num_latents)],
                                       shape=shape)
        initial_eta = tf.constant([1. for _ in range(num_latents)],
                                  shape=shape)
        self.log_temp = tf.Variable(initial_log_temp, name='log_temp', trainable=True)
        self.eta = tf.Variable(initial_eta, name='eta', trainable=True)
        self.decoder_vars = [v for v in self.nets.trainable_variables if 'decoder' in v.name]
        self.encoder_vars = [v for v in self.nets.trainable_variables if 'encoder' in v.name]
        self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp] + [self.eta]

    def compute_loss(self, z, params, x,
                     sample_from_cont_kl=None, sample_from_disc_kl=None,
                     test_with_one_hot=False):
        categories_n = tf.cast(z.shape[1], dtype=tf.float32)
        x_logit = self.decode([z])
        log_px_z = compute_log_bernoulli_pdf(x=x, x_logit=x_logit, sample_size=self.sample_size)
        log_probs = self.transform_params_into_log_probs(params)
        log_unif_probs = - tf.math.log(categories_n * tf.ones_like(z))
        log_p = self.compute_log_pmf(z=z, log_probs=log_unif_probs)
        log_qz_x = self.compute_log_pmf(z=z, log_probs=log_probs)
        kl = tf.reduce_mean(log_p - log_qz_x)
        # kl = tf.reduce_mean(calculate_categorical_closed_kl(log_alpha=log_alpha, normalize=True))
        loss = -tf.math.reduce_mean(log_px_z) - kl
        return loss

    @staticmethod
    def transform_params_into_log_probs(params):
        log_probs = params[0]
        return log_probs

    def compute_log_pmf(self, z, log_probs):
        log_pmf = compute_log_categorical_pmf(z, log_probs)
        return log_pmf

    def compute_log_pmf_grad(self, z, params):
        log_probs = self.transform_params_into_log_probs(params)
        grad = compute_log_categorical_pmf_grad(z, log_probs)
        return grad

    def compute_losses_from_x_wo_gradients(self, x, sample_from_cont_kl, sample_from_disc_kl):
        params = self.nets.encode(x)
        one_hot = self.get_relax_variables_from_params(x, params)[-1]
        loss = self.compute_loss(z=one_hot, x=x, params=params)
        return loss

    @tf.function()
    def compute_gradients(self, x):
        params = self.nets.encode(x)
        c_phi, c_phi_tilde, one_hot = self.get_relax_variables_from_params(x, params)
        loss = self.compute_loss(x=x, params=params, z=one_hot)
        c_diff = tf.reduce_mean(c_phi - c_phi_tilde)

        c_phi_diff_grad_theta = tf.gradients(c_diff, params[0])[0]
        log_qz_x_grad_theta = self.compute_log_pmf_grad(z=one_hot, params=params)
        relax_grad_theta = self.compute_relax_grad(loss, c_phi_tilde, log_qz_x_grad_theta,
                                                   c_phi_diff_grad_theta)
        encoder_grads = tf.gradients(params[0], self.encoder_vars, grad_ys=relax_grad_theta)
        decoder_grads = tf.gradients(loss, self.decoder_vars)

        variance = compute_grad_var_over_batch(relax_grad_theta)
        cov_net_grad = tf.gradients(variance, self.con_net_vars)

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        return gradients, loss, relax_grad_theta, params

    def compute_relax_grad(self, loss, c_phi_tilde, log_qz_x_grad, c_phi_diff_grad_theta):
        diff = loss - self.eta * c_phi_tilde
        relax = diff * log_qz_x_grad
        relax += self.eta * c_phi_diff_grad_theta
        relax += log_qz_x_grad  # since f depends on theta
        return relax

    def compute_c_phi(self, z, x, params):
        r = tf.math.reduce_mean(self.relax_cov.net(z))
        c_phi = self.compute_loss(x=x, z=z, params=params) + r
        return c_phi

    def apply_gradients(self, gradients):
        encoder_grads, decoder_grads, cov_net_grad = gradients
        self.optimizer_encoder.apply_gradients(zip(encoder_grads, self.encoder_vars))
        self.optimizer_decoder.apply_gradients(zip(decoder_grads, self.decoder_vars))
        self.optimizer_var.apply_gradients(zip(cov_net_grad, self.con_net_vars))


class OptRELAXIGR(OptRELAX):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)
        num_latents = self.n_required * self.num_of_vars
        shape = (1, self.n_required, self.sample_size, self.num_of_vars)
        initial_log_temp = tf.constant([-1.8972 for _ in range(num_latents)],
                                       shape=shape)
        self.log_temp = tf.Variable(initial_log_temp, name='log_temp', trainable=True)
        cov_net_shape = (self.n_required + 1, self.sample_size, self.num_of_vars)
        self.relax_cov = RelaxCovNet(cov_net_shape)
        # self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp] + [self.eta]
        self.con_net_vars = self.relax_cov.net.trainable_variables + [self.log_temp]

    @staticmethod
    def transform_params_into_log_probs(params):
        mu, xi = params
        log_probs = compute_igr_log_probs(mu, tf.math.exp(tf.clip_by_value(xi, -50., 50.)))
        return log_probs

    def compute_log_pmf(self, z, log_probs):
        log_categorical_pmf = tf.math.reduce_sum(z * log_probs, axis=1)
        log_categorical_pmf = tf.math.reduce_sum(log_categorical_pmf, axis=(1, 2))
        return log_categorical_pmf

    #  def compute_log_pmf_grad(self, z, params):
    #      log_probs = self.transform_params_into_log_probs(params)
    #      normalized = tf.math.exp(tf.clip_by_value(log_probs, -50., 50.))
    #      grad = z - normalized
    #      return grad

    def get_relax_variables_from_params(self, x, params):
        mu, xi = params
        z_un = mu + tf.math.exp(tf.clip_by_value(xi, -50., 50.)) * tf.random.normal(shape=mu.shape)
        z = project_to_vertices_via_softmax_pp(z_un / tf.math.exp(self.log_temp))
        one_hot = tf.transpose(tf.one_hot(tf.argmax(tf.stop_gradient(z), axis=1),
                                          depth=self.n_required + 1), perm=[0, 3, 1, 2])
        c_phi = self.compute_c_phi(z=z, x=x, params=params)
        return c_phi, z_un, one_hot

    @tf.function()
    def compute_gradients(self, x):
        with tf.GradientTape() as tape_cov:
            with tf.GradientTape(persistent=True) as tape:
                params = self.nets.encode(x)
                tape.watch(params)
                c_phi, z_un, one_hot = self.get_relax_variables_from_params(x, params)
                loss = self.compute_loss(x=x, params=params, z=one_hot)
                log_gauss_grad = compute_log_gauss_grad(z_un, params)
                log_cat_grad = self.compute_log_pmf_grad(z=one_hot, params=params)

            c_phi_g = tape.gradient(target=c_phi, sources=params)
            log_cat_g = tape.gradient(target=log_cat_grad, sources=params)
            lax_grad = self.compute_lax_grad(loss, c_phi, log_cat_g, log_gauss_grad, c_phi_g)

            c_phi_grad = tape.gradient(target=c_phi, sources=self.encoder_vars)
            log_qz_x_grad = tape.gradient(target=log_cat_grad, sources=self.encoder_vars)
            log_qc_x_grad = tape.gradient(target=log_gauss_grad, sources=self.encoder_vars)
            encoder_grads = self.compute_lax_grad(loss, c_phi, log_qz_x_grad,
                                                  log_qc_x_grad, c_phi_grad)
            decoder_grads = tape.gradient(target=loss, sources=self.decoder_vars)

            variance = compute_grad_var_over_batch(lax_grad[0])
        cov_net_grad = tape_cov.gradient(target=variance, sources=self.con_net_vars)

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        return gradients, loss, lax_grad, params

    def compute_lax_grad(self, loss, c_phi, log_qz_x_grad, log_qc_grad, c_phi_grad):
        lax_grads = []
        for i in range(len(log_qz_x_grad)):
            lax = loss * log_qz_x_grad[i]
            lax -= c_phi * log_qc_grad[i]
            lax += c_phi_grad[i]
            lax += log_qz_x_grad[i]
            lax_grads.append(lax)
        return lax_grads

    @tf.function()
    def compute_gradients_(self, x):
        params = self.nets.encode(x)
        c_phi, z_un, one_hot = self.get_relax_variables_from_params(x, params)
        loss = self.compute_loss(x=x, params=params, z=one_hot)

        c_phi_grad = tf.gradients(c_phi, params)
        log_gauss_grad = compute_log_gauss_grad(z_un, params)
        log_probs = self.transform_params_into_log_probs(params)
        log_qz_x_grad = tf.gradients(log_probs, params, grad_ys=one_hot)
        lax_grad = self.compute_lax_grad(loss, c_phi, log_qz_x_grad, log_gauss_grad, c_phi_grad)

        encoder_grads = tf.gradients(params, self.encoder_vars, grad_ys=lax_grad)
        decoder_grads = tf.gradients(loss, self.decoder_vars)

        variance = compute_grad_var_over_batch(lax_grad[0])
        cov_net_grad = tf.gradients(variance, self.con_net_vars)

        gradients = (encoder_grads, decoder_grads, cov_net_grad)
        return gradients, loss, lax_grad, params


class OptRELAXGSDis(OptRELAX):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)

    def get_relax_variables_from_params(self, x, params):
        log_alpha = params[0]
        offset = 1.e-20
        u = tf.random.uniform(shape=log_alpha.shape)
        z_un = log_alpha - tf.math.log(-tf.math.log(u + offset) + offset)
        one_hot = tf.transpose(tf.one_hot(tf.argmax(z_un, axis=1), depth=self.n_required),
                               perm=[0, 3, 1, 2])
        z_tilde_un = sample_z_tilde_cat(one_hot, log_alpha)

        # z = tf.math.softmax(z_un / tf.math.exp(self.log_temp) + log_alpha, axis=1)
        # z_tilde = tf.math.softmax(z_tilde_un / tf.math.exp(self.log_temp) + log_alpha, axis=1)
        z = tf.math.softmax(z_un / tf.math.exp(self.log_temp), axis=1)
        z_tilde = tf.math.softmax(z_tilde_un / tf.math.exp(self.log_temp), axis=1)

        c_phi = self.compute_c_phi(z=z, x=x, params=params)
        c_phi_tilde = self.compute_c_phi(z=z_tilde, x=x, params=params)
        return c_phi, c_phi_tilde, one_hot


class OptRELAXBerDis(OptRELAXGSDis):
    def __init__(self, nets, optimizers, hyper):
        super().__init__(nets=nets, optimizers=optimizers, hyper=hyper)

    def compute_log_pmf(self, z, log_probs):
        log_pmf = bernoulli_loglikelihood(b=z, log_alpha=log_probs)
        log_pmf = tf.math.reduce_sum(log_pmf, axis=(1, 2, 3))
        return log_pmf

    def compute_log_pmf_grad(self, z, params):
        log_alpha = self.transform_params_into_log_probs(params)
        grad = bernoulli_loglikelihood_grad(z, log_alpha)
        return grad

    def get_relax_variables_from_params(self, x, params):
        log_alpha = params[0]
        u = tf.random.uniform(shape=log_alpha.shape)
        z_un = log_alpha + safe_log_prob(u) - safe_log_prob(1 - u)
        one_hot = tf.cast(tf.stop_gradient(z_un > 0), dtype=tf.float32)
        z_tilde_un = sample_z_tilde_ber(log_alpha=log_alpha, u=u)

        z = tf.math.sigmoid(z_un / tf.math.exp(self.log_temp) + log_alpha)
        z_tilde = tf.math.sigmoid(z_tilde_un / tf.math.exp(self.log_temp) + log_alpha)
        c_phi = self.compute_c_phi(z=z, x=x, params=params)
        c_phi_tilde = self.compute_c_phi(z=z_tilde, x=x, params=params)
        return c_phi, c_phi_tilde, one_hot


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
def compute_loss(log_px_z, kl_norm, kl_dis, sample_size=1, run_jv=False,
                 gamma=tf.constant(1.), discrete_c=tf.constant(0.), continuous_c=tf.constant(0.)):
    if run_jv:
        loss = -tf.reduce_mean(log_px_z - gamma * tf.math.abs(kl_norm - continuous_c)
                               - gamma * tf.math.abs(kl_dis - discrete_c))
    else:
        kl = kl_norm + kl_dis
        elbo = log_px_z - kl
        elbo_iwae = tf.math.reduce_logsumexp(elbo, axis=1)
        loss = -tf.math.reduce_mean(elbo_iwae, axis=0)
        loss += tf.math.log(tf.constant(sample_size, dtype=tf.float32))
    return loss


def compute_log_bernoulli_pdf(x, x_logit, sample_size):
    x_broad = tf.broadcast_to(tf.expand_dims(x, 4), shape=x.shape + (sample_size,))
    cross_ent = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_broad, logits=x_logit)
    log_px_z = tf.reduce_sum(cross_ent, axis=(1, 2, 3))
    return log_px_z


def compute_log_categorical_pmf(d, log_alpha):
    log_normalized = log_alpha - tf.reduce_logsumexp(log_alpha, axis=1, keepdims=True)
    log_categorical_pmf = tf.math.reduce_sum(d * log_normalized, axis=1)
    log_categorical_pmf = tf.math.reduce_sum(log_categorical_pmf, axis=(1, 2))
    return log_categorical_pmf


def compute_log_categorical_pmf_grad(d, log_alpha):
    normalized = tf.math.softmax(log_alpha, axis=1)
    grad = d - normalized
    return grad


def softplus(x):
    m = tf.maximum(tf.zeros_like(x), x)
    return m + tf.math.log(tf.exp(-m) + tf.math.exp(x - m))


def bernoulli_loglikelihood(b, log_alpha):
    output = b * (-softplus(-log_alpha)) + (1. - b) * (-log_alpha - softplus(-log_alpha))
    return output


def bernoulli_loglikelihood_grad(b, log_alpha):
    sna = tf.math.sigmoid(-log_alpha)
    return b * sna - (1 - b) * (1 - sna)


def compute_log_gaussian_pdf(x, x_logit, sample_size):
    mu, xi = x_logit
    mu = tf.math.sigmoid(mu)
    xi = 1.e-6 + tf.math.softplus(xi)
    pi = 3.141592653589793

    x_broad = tf.broadcast_to(tf.expand_dims(x, 4), shape=x.shape + (sample_size,))

    log_pixel = (- 0.5 * ((x_broad - mu) / xi) ** 2. -
                 0.5 * tf.math.log(2 * pi) - tf.math.log(1.e-8 + xi))
    log_px_z = tf.reduce_sum(log_pixel, axis=[1, 2, 3])
    return log_px_z


def safe_log_prob(x, eps=1.e-8):
    return tf.math.log(tf.clip_by_value(x, eps, 1.0))


# def sample_z_tilde_ber(log_alpha, one_hot):
#     # TODO: add testing for this function
#     v = tf.random.uniform(shape=log_alpha.shape)
#     theta = tf.math.sigmoid(log_alpha)
#     v_0 = v * (1 - theta)
#     v_1 = v * theta + (1 - theta)
#     v_tilde = tf.where(one_hot == 1., v_1, v_0)
#
#     z_tilde_un = log_alpha + safe_log_prob(v_tilde) - safe_log_prob(1 - v_tilde)
#     return z_tilde_un
#

def sample_z_tilde_ber(log_alpha, u, eps=1.e-8):
    u_prime = tf.math.sigmoid(-log_alpha)
    v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1.0)
    v_1 = tf.clip_by_value(v_1, 0, 1)
    v_1 = tf.stop_gradient(v_1)
    v_1 = v_1 * (1 - u_prime) + u_prime
    v_0 = u / tf.clip_by_value(u_prime, eps, 1.0)
    v_0 = tf.clip_by_value(v_0, 0, 1)
    v_0 = tf.stop_gradient(v_0)
    v_0 = v_0 * u_prime

    v = tf.where(u > u_prime, v_1, v_0)
    v = v + tf.stop_gradient(u - v)
    z_tilde_un = log_alpha + safe_log_prob(v) - safe_log_prob(1 - v)
    return z_tilde_un


def sample_z_tilde_cat(one_hot, log_alpha):
    offset = 1.e-20
    bool_one_hot = tf.cast(one_hot, dtype=tf.bool)
    theta = tf.math.softmax(log_alpha, axis=1)
    v = tf.random.uniform(shape=log_alpha.shape)
    v_b = tf.where(bool_one_hot, v, 0.)
    v_b = tf.math.reduce_max(v_b, axis=1, keepdims=True)
    v_b = tf.broadcast_to(v_b, shape=v.shape)

    aux1 = -tf.math.log(v + offset) / tf.clip_by_value(theta, 1.e-5, 1.)
    aux2 = tf.math.log(v_b + offset)
    aux = aux1 - aux2
    z_other = -tf.math.log(aux + offset)
    z_b = -tf.math.log(-tf.math.log(v_b + offset) + offset)
    z_tilde = tf.where(bool_one_hot, z_b, z_other)
    return z_tilde


def compute_grad_var_over_batch(relax_grad):
    variance = tf.math.square(relax_grad)
    variance = tf.math.reduce_sum(variance, axis=(1, 2, 3))
    variance = tf.math.reduce_mean(variance)
    return variance


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
