import time
import numpy as np
import tensorflow as tf
from Utils.Distributions import compute_gradients, apply_gradients
from Utils.initializations import initialize_mu_and_xi_for_logistic, initialize_mu_and_xi_equally
from Utils.general import setup_logger
logger = setup_logger(log_file_name='./Log/discrete.log')


class MinimizeEmpiricalLoss:

    def __init__(self, learning_rate: float, temp_init: float, temp_final: float, moments_diff: float = 9.2,
                 sample_size: int = int(1.e3), max_iterations: int = int(1.e4),
                 run_kl: bool = True, tolerance: float = 1.e-5, model_type: str = 'IGR_I'):

        self.learning_rate = learning_rate
        self.moments_diff = moments_diff
        self.temp_init = temp_init
        self.temp_final = temp_final
        self.sample_size = sample_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.model_type = model_type
        self.run_kl = run_kl

        self.iteration = 0
        self.training_time = 0
        self.iter_time = 0
        self.total_samples_for_moment_evaluation = 1
        self.mean_loss = 10
        self.mean_n_required = 0
        self.check_every = 10
        self.threshold = 0.9
        self.temp_schedule = np.linspace(temp_final, temp_init, num=max_iterations)
        self.temp = tf.constant(value=temp_init, dtype=tf.float32)
        self.categories_n = 0
        self.params = []
        self.loss_iter = np.zeros(shape=max_iterations)
        self.n_required_iter = np.zeros(shape=max_iterations)
        self.run_iteratively = False

    def set_variables(self, params):
        self.params = params
        self.categories_n = params[0].shape[1]

    def optimize_model(self, mean_p, var_p, probs, p_samples):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        continue_training = True
        self.total_samples_for_moment_evaluation = 100
        t0 = time.time()
        while continue_training:
            current_iter_time = time.time()
            loss, n_required = self.take_gradient_step_and_compute_loss(optimizer=optimizer, probs=probs)
            self.iter_time += time.time() - current_iter_time

            self.loss_iter[self.iteration] = loss.numpy()
            self.n_required_iter[self.iteration] = n_required

            self.evaluate_progress(loss_iter=self.loss_iter, n_required_iter=self.n_required_iter)
            self.iteration += 1
            continue_training = self.determine_continuation()
        self.training_time = time.time() - t0
        logger.info(f'\nTraining took: {self.training_time:6.1f} sec')

    def take_gradient_step_and_compute_loss(self, optimizer, probs):
        grad, loss, n_required = compute_gradients(params=self.params, temp=self.temp,
                                                   probs=probs, dist_type=self.model_type,
                                                   sample_size=self.sample_size, threshold=self.threshold,
                                                   run_iteratively=self.run_iteratively,
                                                   run_kl=self.run_kl)
        apply_gradients(optimizer=optimizer, gradients=grad, variables=self.params)
        return loss, n_required

    def evaluate_progress(self, loss_iter, n_required_iter):
        if (self.iteration % self.check_every == 0) & (self.iteration > self.check_every):
            self.check_mean_loss_to_previous_iterations(loss_iter=loss_iter, n_required_iter=n_required_iter)
            # self.check_mean_progress_to_previous_iterations(loss_iter=loss_iter)
            # self.check_moments_convergence(mean_p=mean_p, var_p=var_p)

            logger.info(f'Iter {self.iteration:4d} || '
                        f'Loss {self.mean_loss:2.3e} || '
                        f'N Required {int(self.mean_n_required):4d}')

    def check_mean_loss_to_previous_iterations(self, loss_iter, n_required_iter):
        new_window = self.iteration - self.check_every
        self.mean_loss = np.mean(loss_iter[new_window: self.iteration])
        self.mean_n_required = np.mean(n_required_iter[new_window: self.iteration])

    def determine_continuation(self):
        continue_training = ((self.iteration < self.max_iterations) &
                             (np.abs(self.mean_loss) > self.tolerance))
        return continue_training


def get_initial_params_for_model_type(model_type, shape):
    batch_size, categories_n, sample_size, num_of_vars = shape
    if model_type == 'GS':
        uniform_probs = np.array([1 / categories_n for _ in range(categories_n)])
        pi = tf.constant(value=np.log(uniform_probs), dtype=tf.float32,
                         shape=(batch_size, categories_n, 1, 1))
        pi_init = tf.constant(pi.numpy().copy(), dtype=tf.float32,
                              shape=(batch_size, categories_n, 1, 1))
        params = [tf.Variable(initial_value=pi)]
        params_init = [tf.Variable(initial_value=pi_init)]
    elif model_type == 'IGR_SB' or model_type == 'IGR_SB_Finite':
        shape_igr = (batch_size, categories_n - 1, sample_size, num_of_vars)
        mu, xi = initialize_mu_and_xi_for_logistic(shape_igr, seed=21)
        mu_init, xi_init = tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())
        params = [mu, xi]
        params_init = [mu_init, xi_init]
    elif model_type == 'IGR_I':
        shape_igr = (batch_size, categories_n - 1, sample_size, num_of_vars)
        mu, xi = initialize_mu_and_xi_equally(shape_igr)
        mu_init, xi_init = tf.constant(mu.numpy().copy()), tf.constant(xi.numpy().copy())
        params = [mu, xi]
        params_init = [mu_init, xi_init]
    else:
        raise RuntimeError
    return params, params_init


def obtain_results_from_minimizer(minimizer):
    iteration = minimizer.iteration
    time_taken = minimizer.iter_time
    n_required = minimizer.mean_n_required
    return iteration, time_taken, n_required


def update_results(results_to_update, minimizer, dist_case, run_case, cat_case: int = 1):
    tracks = obtain_results_from_minimizer(minimizer=minimizer)
    for idx, var in enumerate(results_to_update):
        if minimizer.model_type == 'GS':
            var[cat_case, dist_case, run_case] = tracks[idx]
        else:
            var[dist_case, run_case] = tracks[idx]
