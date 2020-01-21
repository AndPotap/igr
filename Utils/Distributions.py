# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
import numpy as np
import tensorflow as tf
from typing import Tuple, List
from os import environ as os_env
os_env['TF_CPP_MIN_LOG_LEVEL'] = '2'
# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================


class Distributions:

    def __init__(self, batch_size: int, categories_n: int, sample_size: int = 1, num_of_vars: int = 1,
                 noise_type: str = 'normal', temp: tf.Tensor = tf.constant(0.1, dtype=tf.float32)):

        self.noise_type = noise_type
        self.temp = temp
        self.batch_size = batch_size
        self.categories_n = categories_n
        self.sample_size = sample_size
        self.num_of_vars = num_of_vars

        self.n_required = categories_n
        self.lam = tf.constant(0., dtype=tf.float32)
        self.log_psi = tf.constant(0., dtype=tf.float32)
        self.psi = tf.constant(0., dtype=tf.float32)

    def broadcast_params_to_sample_size(self, params: list):
        params_broad = []
        for param in params:
            shape = (self.batch_size, self.categories_n, self.sample_size, self.num_of_vars)
            param_w_samples = tf.broadcast_to(input=param, shape=shape)
            params_broad.append(param_w_samples)
        return params_broad

    def sample_noise(self, shape) -> tf.Tensor:
        if self.noise_type == 'normal':
            epsilon = tf.random.normal(shape=shape)
        elif self.noise_type == 'trunc_normal':
            epsilon = tf.random.truncated_normal(shape=shape)
        else:
            raise RuntimeError
        return epsilon


class IGR_I(Distributions):
    def __init__(self, mu, xi, temp, sample_size=1, noise_type='normal'):
        super().__init__(batch_size=mu.shape[0], categories_n=mu.shape[1], sample_size=sample_size,
                         noise_type=noise_type, temp=temp, num_of_vars=mu.shape[3])

        self.mu = mu
        self.xi = xi

    def generate_sample(self):
        mu_broad, xi_broad = self.broadcast_params_to_sample_size(params=[self.mu, self.xi])
        epsilon = self.sample_noise(shape=mu_broad.shape)
        sigma_broad = tf.math.exp(xi_broad)
        self.lam = self.transform(mu_broad, sigma_broad, epsilon)
        self.log_psi = self.lam - tf.math.reduce_logsumexp(self.lam, axis=1, keepdims=True)
        self.psi = self.project_to_vertices()

    def transform(self, mu_broad, sigma_broad, epsilon):
        lam = (mu_broad + sigma_broad * epsilon) / self.temp
        return lam

    def project_to_vertices(self):
        psi = project_to_vertices_via_softmax_pp(self.lam)
        return psi


class IGR_Planar(IGR_I):
    def __init__(self, mu, xi, temp, planar_flow, sample_size=1, noise_type='normal'):
        super().__init__(mu, xi, temp, sample_size, noise_type)
        self.planar_flow = planar_flow

    def transform(self, mu_broad, sigma_broad, epsilon):
        lam = (self.planar_flow(mu_broad + sigma_broad * epsilon)) / self.temp
        return lam


class IGR_SB(IGR_I):

    def __init__(self, mu, xi, temp, sample_size=1, noise_type='normal', threshold=0.99, run_iteratively=False):
        super().__init__(mu, xi, temp, sample_size, noise_type)

        self.threshold = threshold
        self.truncation_option = 'quantile'
        self.quantile = 70
        self.run_iteratively = run_iteratively
        self.lower = np.zeros(shape=(self.categories_n - 1, self.categories_n - 1))
        self.upper = np.zeros(shape=(self.categories_n - 1, self.categories_n - 1))

    def transform(self, mu_broad, sigma_broad, epsilon):
        kappa = tf.math.sigmoid(mu_broad + sigma_broad * epsilon)
        eta = self.get_eta(kappa)
        self.perform_truncation_via_threshold(vector=eta)
        lam = eta[:, :self.n_required, :] / self.temp
        return lam

    def get_eta(self, kappa):
        if self.run_iteratively:
            eta = iterative_sb(kappa)
        else:
            self.lower, self.upper = generate_lower_and_upper_triangular_matrices_for_sb(
                categories_n=self.categories_n, lower=self.lower, upper=self.upper,
                batch_size=self.batch_size, sample_size=self.sample_size)
            eta = self.perform_stick_break(kappa)
        return eta

    def perform_stick_break(self, kappa):
        accumulated_prods = accumulate_one_minus_kappa_prods(kappa, self.lower, self.upper)
        eta = kappa * accumulated_prods
        return eta

    def perform_truncation_via_threshold(self, vector):
        vector_cumsum = tf.math.cumsum(x=vector, axis=1)
        larger_than_threshold = tf.where(condition=vector_cumsum <= self.threshold)
        if self.truncation_option == 'quantile':
            self.n_required = int((np.percentile(larger_than_threshold[:, 1] + 1, q=self.quantile)))
        elif self.truncation_option == 'max':
            self.n_required = (tf.math.reduce_max(larger_than_threshold[:, 1]) + 1).numpy()
        else:
            self.n_required = (tf.math.reduce_mean(larger_than_threshold[:, 1]) + 1).numpy()


class GS(Distributions):

    def __init__(self, log_pi, temp, sample_size: int = 1):
        super().__init__(batch_size=log_pi.shape[0], categories_n=log_pi.shape[1], sample_size=sample_size,
                         temp=temp, num_of_vars=log_pi.shape[3])
        self.log_pi = log_pi

    def generate_sample(self):
        ς = 1.e-20
        log_pi_broad = self.broadcast_params_to_sample_size(params=[self.log_pi])[0]
        uniform = tf.random.uniform(shape=log_pi_broad.shape)
        gumbel_sample = -tf.math.log(-tf.math.log(uniform + ς) + ς)
        y = (log_pi_broad + gumbel_sample) / self.temp
        self.log_psi = y - tf.math.reduce_logsumexp(y, axis=1, keepdims=True)
        self.psi = tf.math.softmax(logits=y, axis=1)


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Distribution functions
# ===========================================================================================================
def compute_log_gs_dist(psi: tf.Tensor, logits: tf.Tensor, temp: tf.Tensor) -> tf.Tensor:
    n_required = tf.constant(value=psi.shape[1], dtype=tf.float32)
    ς = tf.constant(1.e-20)

    log_const = tf.math.lgamma(n_required) + (n_required - 1) * tf.math.log(temp)
    log_sum = tf.reduce_sum(logits - (temp + tf.constant(1.)) * tf.math.log(psi + ς), axis=1)
    log_norm = - n_required * tf.math.log(tf.reduce_sum(tf.math.exp(logits) / psi ** temp, axis=1) + ς)

    log_p_concrete = log_const + log_sum + log_norm
    return log_p_concrete


def compute_log_exp_gs_dist(log_psi: tf.Tensor, logits: tf.Tensor, temp: tf.Tensor) -> tf.Tensor:
    categories_n = tf.constant(log_psi.shape[1], dtype=tf.float32)
    log_cons = tf.math.lgamma(categories_n) + (categories_n - 1) * tf.math.log(temp)
    aux = logits - temp * log_psi
    log_sums = tf.math.reduce_sum(aux, axis=1) - categories_n * tf.math.reduce_logsumexp(aux, axis=1)
    log_exp_gs_dist = log_cons + log_sums
    return log_exp_gs_dist


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Optimization functions for the Expectation Minimization Loss
# ===========================================================================================================
def compute_loss(params: List[tf.Tensor], temp: tf.Tensor, probs: tf.Tensor, dist_type: str = 'sb',
                 sample_size: int = 1, threshold: float = 0.99, run_iteratively=False, run_kl=True):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, params=params, temp=temp,
                                             sample_size=sample_size, threshold=threshold,
                                             run_iteratively=run_iteratively)

    chosen_dist.generate_sample()
    psi_mean = tf.reduce_mean(chosen_dist.psi, axis=[0, 2, 3])
    if run_kl:
        loss = psi_mean * (tf.math.log(psi_mean) - tf.math.log(probs[:chosen_dist.n_required] + 1.e-20))
        loss = tf.reduce_sum(loss)
    else:
        loss = tf.reduce_sum((psi_mean - probs[:chosen_dist.n_required]) ** 2)
    return loss, chosen_dist.n_required


def compute_gradients(params, temp: tf.Tensor, probs: tf.Tensor, run_kl=True,
                      dist_type: str = 'sb', sample_size: int = 1, run_iteratively=False,
                      threshold: float = 0.99) -> Tuple[tf.Tensor, tf.Tensor, int]:
    with tf.GradientTape() as tape:
        loss, n_required = compute_loss(params=params, temp=temp, probs=probs, sample_size=sample_size,
                                        threshold=threshold, dist_type=dist_type, run_kl=run_kl,
                                        run_iteratively=run_iteratively)
        gradient = tape.gradient(target=loss, sources=params)
    return gradient, loss, n_required


def apply_gradients(optimizer: tf.keras.optimizers, gradients: tf.Tensor, variables):
    optimizer.apply_gradients(zip(gradients, variables))


# ===========================================================================================================
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Utils
# ===========================================================================================================
@tf.function
def project_to_vertices_via_softmax_pp(lam):
    offset = 1.
    lam_i_lam_max = lam - tf.math.reduce_max(lam, axis=1, keepdims=True)
    exp_lam = tf.math.exp(lam_i_lam_max)
    sum_exp_lam = tf.math.reduce_sum(exp_lam, axis=1, keepdims=True)
    psi = exp_lam / (sum_exp_lam + offset)

    extra_cat = (1 - tf.math.reduce_sum(psi, axis=1, keepdims=True))
    psi = tf.concat(values=[psi, extra_cat], axis=1)

    return psi


def accumulate_one_minus_kappa_prods(kappa, lower, upper):
    forget_last = -1
    one = tf.constant(value=1., dtype=tf.float32)

    diagonal_kappa = tf.linalg.diag(tf.transpose(one - kappa[:, :forget_last, :, :], perm=[0, 2, 3, 1]))
    accumulation = tf.transpose(tf.tensordot(lower, diagonal_kappa, axes=[[1], [3]]), perm=[1, 0, 4, 2, 3])
    accumulation_w_ones = accumulation + upper
    cumprod = tf.math.reduce_prod(input_tensor=accumulation_w_ones, axis=2)
    return cumprod


def generate_lower_and_upper_triangular_matrices_for_sb(categories_n, lower, upper,
                                                        batch_size, sample_size, num_of_vars=1):
    zeros_row = np.zeros(shape=categories_n - 1)

    for i in range(categories_n - 1):
        for j in range(categories_n - 1):
            if i > j:
                lower[i, j] = 1
            elif i == j:
                lower[i, j] = 1
                upper[i, j] = 1
            else:
                upper[i, j] = 1

    lower = np.vstack([zeros_row, lower])
    upper = np.vstack([upper, zeros_row])

    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1))
    upper = np.reshape(upper, newshape=(batch_size, categories_n, categories_n - 1, 1, 1))
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1, sample_size, 1))
    upper = np.broadcast_to(upper, shape=(batch_size, categories_n, categories_n - 1, sample_size, num_of_vars))

    lower = tf.constant(value=lower, dtype=tf.float32)  # no reshape needed
    upper = tf.constant(value=upper, dtype=tf.float32)
    return lower, upper


@tf.function
def iterative_sb(kappa):
    batch_size, max_size, sample_size, num_of_vars = kappa.shape
    η = tf.TensorArray(dtype=tf.float32, size=max_size, clear_after_read=True,
                       element_shape=(batch_size, sample_size, num_of_vars))
    η = η.write(index=0, value=kappa[:, 0, :, :])
    cumsum = tf.identity(kappa[:, 0, :, :])
    next_cumsum = tf.identity(kappa[:, 1, :, :] * (1 - kappa[:, 0, :, :]) + kappa[:, 0, :, :])
    max_iter = tf.constant(value=max_size - 1, dtype=tf.int32)
    for i in tf.range(1, max_iter):
        η = η.write(index=i, value=kappa[:, i, :, :] * (1. - cumsum))
        cumsum += kappa[:, i, :, :] * (1. - cumsum)
        next_cumsum += kappa[:, i + 1, :, :] * (1. - next_cumsum)

    η = η.write(index=max_size - 1, value=kappa[:, max_size - 1, :, :] * (1. - cumsum))
    return tf.transpose(η.stack(), perm=[1, 0, 2, 3])


def generate_sample(sample_size: int, params, dist_type: str, temp, threshold: float = 0.99,
                    output_one_hot=False):
    chosen_dist = select_chosen_distribution(dist_type=dist_type, threshold=threshold,
                                             params=params, temp=temp, sample_size=sample_size)
    categories_n = params[0].shape[1]
    chosen_dist.generate_sample()
    if output_one_hot:
        vector = np.zeros(shape=(1, categories_n, sample_size, 1))
        n_required = chosen_dist.psi.shape[1]
        vector[:, :n_required, :, :] = chosen_dist.psi.numpy()
        return vector
    else:
        sample = np.argmax(chosen_dist.psi.numpy(), axis=1)
        return sample


def select_chosen_distribution(dist_type: str, params, temp=tf.constant(0.1, dtype=tf.float32),
                               sample_size: int = 1, threshold: float = 0.99, run_iteratively=False):
    if dist_type == 'IGR_SB':
        mu, xi = params
        chosen_dist = IGR_SB(mu=mu, xi=xi, temp=temp, sample_size=sample_size, threshold=threshold)
        if run_iteratively:
            chosen_dist.run_iteratively = True
    elif dist_type == 'GS':
        pi = params[0]
        chosen_dist = GS(log_pi=pi, temp=temp, sample_size=sample_size)
    elif dist_type == 'IGR_I':
        mu, xi, = params
        chosen_dist = IGR_I(mu=mu, xi=xi, temp=temp, sample_size=sample_size)
    else:
        raise RuntimeError

    return chosen_dist
