import unittest
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
from Models.OptVAE import OptDLGMM


class TestDLGMM(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
