import unittest
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from Utils.Distributions import iterative_sb
from Tests.TestDistributions import calculate_eta_from_kappa
from Models.OptVAE import OptDLGMM


class TestDLGMM(unittest.TestCase):

    def test_sb(self):
        test_tolerance = 1.e-6
        z_kumar = tf.constant([[0.1, 0.2, 0.3, 0.4],
                               [0.2, 0.3, 0.4, 0.1]])
        z_kumar = tf.expand_dims(tf.expand_dims(z_kumar, axis=-1), axis=-1)
        approx = iterative_sb(z_kumar)
        ans = tf.constant(calculate_eta_from_kappa(z_kumar.numpy()), dtype=z_kumar.dtype)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Stick-Break Kumar')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def setUp(self):
        self.hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
                      'save_every': 50, 'sample_size_testing': 1 * int(1.e0),
                      'dtype': tf.float32, 'sample_size': 1, 'check_every': 50,
                      'epochs': 300, 'learning_rate': 1 * 1.e-4, 'batch_n': 100,
                      'num_of_discrete_var': 20, 'sample_from_disc_kl': True,
                      'stick_the_landing': True, 'test_with_one_hot': False,
                      'dataset_name': 'mnist',
                      'model_type': 'linear', 'temp': 1.0,
                      'sample_from_cont_kl': True}

    def test_loss(self):
        pass

    def test_log_px_z(self):
        test_tolerance = 1.e-6
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        x = tf.random.uniform(shape=(1, 4, 4, 1))
        x_logit = tf.random.normal(shape=(1, 4, 4, 1, n_required))
        self.hyper['n_required'] = n_required
        nets, optimizer = [], []
        optvae = OptDLGMM(nets, optimizer, self.hyper)
        approx = optvae.compute_log_px_z(x, x_logit, pi)
        x_broad = tf.repeat(tf.expand_dims(x, 4), axis=4, repeats=n_required)
        x_broad = tf.reshape(x_broad, shape=(16, 4))
        x_logit = tf.reshape(x_logit, shape=(16, 4))
        dist = tfpd.Bernoulli(logits=x_logit)
        ans = dist.log_prob(x_broad)
        ans = tf.reduce_sum(dist.log_prob(x_broad), axis=0)
        ans = tf.reduce_sum(pi[0, :, 0, 0] * ans)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Reconstruction')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_log_pz(self):
        test_tolerance = 1.e-6
        batch_n, n_required, sample_size, dim = 2, 4, 1, 3
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        z = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        self.hyper['n_required'] = n_required
        nets, optimizer = [], []
        optvae = OptDLGMM(nets, optimizer, self.hyper)
        optvae.mu_prior = tf.zeros(shape=(batch_n, 1, sample_size, dim))
        mult = 1. / tf.sqrt(tf.constant(dim, dtype=pi.dtype))
        for k in range(n_required - 1):
            mu_prior = tf.ones(shape=(batch_n, 1, sample_size, dim))
            u = tf.random.uniform(shape=mu_prior.shape)
            mu_prior = tf.where(u < 0.5, -1.0, 1.0)
            mu_prior = mult * mu_prior
            optvae.mu_prior = tf.concat([optvae.mu_prior, mu_prior],
                                        axis=1)
        optvae.log_var_prior = tf.zeros_like(z)
        approx = optvae.compute_log_pz(z, pi)

        loc = optvae.mu_prior
        scale = tf.math.exp(0.5 * optvae.log_var_prior)
        dist = tfpd.Normal(loc, scale)
        ans = pi * tf.reduce_sum(dist.log_prob(z), axis=3, keepdims=True)
        ans = tf.reduce_mean(tf.reduce_sum(ans, axis=(1, 2, 3)))
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Normal Prior')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_kld(self):
        test_tolerance = 1.e-6
        log_a = tf.constant([[0., -1., 1., 2.0, -2.0]])
        log_a = tf.expand_dims(tf.expand_dims(log_a, axis=-1), axis=-1)
        log_b = tf.constant([[0., -1., 1., 2.0, -2.0]])
        log_b = tf.expand_dims(tf.expand_dims(log_b, axis=-1), axis=-1)
        a, b = tf.math.exp(log_a), tf.math.exp(log_b)
        n_required = log_a.shape[1]
        self.hyper['n_required'] = n_required
        nets, optimizer = [], []
        optvae = OptDLGMM(nets, optimizer, self.hyper)
        ans = 0.
        for k in range(n_required):
            dist = tfpd.Kumaraswamy(concentration0=a[:, k, 0, 0],
                                    concentration1=b[:, k, 0, 0])
            ans += dist.entropy()
        approx = optvae.compute_kld(log_a, log_b)
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Kumaraswamy KL')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)

    def test_log_qz_x(self):
        test_tolerance = 1.e-6
        batch_n, n_required, sample_size, dim = 2, 4, 1, 3
        pi = tf.constant([[0.1, 0.2, 0.3, 0.4]])
        pi = tf.expand_dims(tf.expand_dims(pi, axis=-1), axis=-1)
        n_required = pi.shape[1]
        z = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        self.hyper['n_required'] = n_required
        nets, optimizer = [], []
        optvae = OptDLGMM(nets, optimizer, self.hyper)
        mean = tf.random.normal(shape=(batch_n, n_required, sample_size, dim))
        log_var = tf.zeros_like(z)
        approx = optvae.compute_log_qz_x(z, pi, mean, log_var)

        ans = 0
        for k in range(n_required):
            loc = mean[:, k, 0, :]
            scale = tf.math.exp(0.5 * log_var[:, k, 0, :])
            dist = tfpd.Normal(loc=loc, scale=scale)
            aux = tf.reduce_prod(tf.math.exp(dist.log_prob(z[:, k, 0, :])), axis=1)
            ans += pi[0, k, 0, 0] * aux
        ans = tf.reduce_mean(tf.math.log(ans))
        diff = tf.linalg.norm(approx - ans) / tf.linalg.norm(ans)
        print('\nTEST: Normal A Posterior')
        print(f'Diff {diff:1.3e}')
        self.assertTrue(expr=diff < test_tolerance)


if __name__ == '__main__':
    unittest.main()
