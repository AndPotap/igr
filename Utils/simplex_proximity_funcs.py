import time
import pickle
from matplotlib import pyplot as plt
import numba
import seaborn as sns
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import tensorflow as tf
from Utils.Distributions import GS, IGR_I, generate_sample


def load_parameter_values(prior_file):
    with open(file=prior_file, mode='rb') as f:
        parameters = pickle.load(f)
    μ = tf.constant(parameters['mu'], dtype=tf.float32)
    ξ = tf.constant(parameters['xi'], dtype=tf.float32)
    return μ, ξ


def plot_boxplots(model: str, results, τ_grid, mult=5):
    obs_n, samples_n = results[model]['sample'].shape
    rows_list = []
    plt.figure(dpi=150)
    plt.style.use(style='ggplot')
    # plt.title(model + ' Simplex Proximity Distribution')
    for i in range(obs_n // mult):
        for s in range(samples_n):
            entry = {'tau': τ_grid[mult * i], 'distance': results[model]['sample'][mult * i, s]}
            rows_list.append(entry)
    df = pd.DataFrame(rows_list)
    ax = sns.boxplot(x='tau', y='distance', data=df, color='royalblue', boxprops={'alpha': 0.5})
    ax.set_xticklabels([f'{tau:0.2f}' for tau in [τ_grid[mult * i] for i in range(obs_n // mult)]])
    ax.tick_params(labelrotation=90)
    plt.xlabel('τ')
    plt.ylabel('Distribution of Distance to Simplex Vertex')
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{model}_boxplot.png')
    plt.show()


def plot_grad_norm_stat_result(stat: str, models, results, τ_grid):
    colors = ['blue', 'red', 'orange', 'purple', 'green']
    plt.figure(dpi=150)
    plt.title(stat.title() + ' Gradient Norm')

    for idx, model in enumerate(models.keys()):
        plt.plot(τ_grid, results[model][stat], label=model, color=colors[idx], alpha=0.5)
    plt.xlabel(r'$\tau$')
    plt.ylabel('Grad Norm')
    plt.annotate(s='Using a Uniform Distribution with 50 categories', xy=(1, -40),
                 xycoords='axes points', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{stat}_grad_norm.png')
    plt.show()


def run_gradient_norm_analysis(models, stats, temps):
    results = create_placeholders_for_statistical_results(stats=stats, models=models,
                                                          samples_per_stat=temps.shape[0])
    for model_name, params in models.items():
        model = get_model_from_name(model_name=model_name, params=params)
        for stat in stats:
            results[model_name][stat] = get_gradient_norm_stat_of_model(
                model=model, parameter=params[0], temps=temps, stat=stat)
    return results


def get_gradient_norm_stat_of_model(model, parameter, temps, stat='mean'):
    grad_norm_stat = np.zeros(shape=temps.shape[0])
    for i in range(temps.shape[0]):
        τ = tf.constant(temps[i], dtype=tf.float32)
        model.temp = τ
        grad = compute_gradients_with_respect_to_tensor(model=model, parameter=parameter)
        grad_norm_stat[i] = compute_norm_stat_across_batches(grad, stat=stat)
    return grad_norm_stat


def compute_gradients_with_respect_to_tensor(model, parameter):
    with tf.GradientTape() as tape:
        tape.watch(tensor=parameter)
        model.do_reparameterization_trick()
    grad = tape.gradient(target=model.psi, sources=parameter)
    return grad


def compute_norm_stat_across_batches(grad, stat):
    grad_norm = tf.linalg.norm(grad, axis=1)
    if stat == 'mean':
        grad_norm_stat = tf.math.reduce_mean(grad_norm)
    elif stat == 'std':
        grad_norm_stat = tf.math.reduce_std(grad_norm)
    elif stat == 'max':
        grad_norm_stat = tf.math.reduce_max(grad_norm)
    elif stat == 'count_zeros':
        total_n = grad_norm.shape[0]
        mask = grad_norm[:, 0, 0] == 0.
        grad_norm_stat = np.sum(mask) / total_n
    else:
        raise RuntimeError
    return grad_norm_stat


def plot_stat_result(stat: str, models, results, τ_grid):
    colors = ['blue', 'red', 'orange', 'purple', 'green']
    plt.figure(dpi=150)
    plt.title(stat.title() + ' Distance to Simplex Vertex')

    for idx, model in enumerate(models.keys()):
        curve = fit_curve_to_result_along_grid(result=results[model][stat], τ_grid=τ_grid)
        plt.plot(τ_grid, results[model][stat], label=model, color=colors[idx], alpha=0.5)
        plt.plot(τ_grid, curve.predict(τ_grid.reshape(-1, 1)),
                 label=model + ' smooth', color=colors[idx], alpha=0.25, linestyle='--')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Euclidean Distance')
    plt.annotate(s='Using a Uniform Distribution with 50 categories', xy=(1, -40),
                 xycoords='axes points', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{stat}_distance_to_simplex.png')
    plt.show()


def fit_curve_to_result_along_grid(result: np.ndarray, τ_grid: np.ndarray, gamma: float = 10.):
    curve = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
    curve.fit(τ_grid.reshape(-1, 1), result.reshape(-1, 1))
    return curve


def run_simulation(samples, τ_grid, models, threshold, stats2run):
    results = create_placeholders_for_statistical_results(stats=stats2run, models=models,
                                                          samples_per_stat=τ_grid.shape[0])
    tic = time.time()
    for model, params in models.items():
        results[model].update({'sample': np.zeros(shape=(τ_grid.shape[0], samples))})
        results[model].update({'sample_var': np.zeros(shape=(τ_grid.shape[0], params[0].shape[1]))})
        for i in range(τ_grid.shape[0]):
            τ = tf.constant(τ_grid[i], dtype=tf.float32)
            ψ = generate_sample(sample_size=samples, params=params, dist_type=model, temp=τ,
                                threshold=threshold, output_one_hot=True)[0, :, :, 0]
            # ψ = generate_sample(sample_size=samples * 100, params=params, dist_type=model, temp=τ,
            #                     threshold=threshold, output_one_hot=True)[0, :, :, 0]
            diff = calculate_distance_to_simplex(ψ=ψ, argmax_locs=np.argmax(ψ, axis=0))
            results[model]['sample_var'][i, :] = tf.math.reduce_variance(ψ, axis=1)
            results[model]['sample'][i, :] = diff[np.argsort(diff)][:samples]
            for stat in stats2run:
                results[model][stat][i] = compute_statistic(stat=stat, samples=diff)
    print(f'\nExperiment took: {time.time() - tic:6.1f} sec')
    return results


@numba.jit(nopython=True, parallel=True)
def calculate_distance_to_simplex(ψ, argmax_locs):
    samples_n = ψ.shape[1]
    categories_n = ψ.shape[0]
    diffs = np.zeros(shape=samples_n)
    for s in numba.prange(samples_n):
        zeros = np.zeros(shape=categories_n)
        zeros[argmax_locs[s]] = 1
        diffs[s] = np.sqrt(np.sum((zeros - ψ[:, s]) ** 2))
    return diffs


def compute_statistic(stat, samples):
    if stat[0] == 'p':
        return np.percentile(samples, q=int(stat[1:]))
    else:
        stat2numpy = {'mean': np.mean, 'median': np.median, 'max': np.max, 'min': np.min,
                      'std': np.std}
        return stat2numpy[stat](samples)


def create_placeholders_for_statistical_results(stats, models, samples_per_stat):
    placeholders = {model: {} for model in models.keys()}
    for model in models:
        placeholders[model].update({stat: np.zeros(shape=samples_per_stat) for stat in stats})

    return placeholders


def get_model_from_name(model_name, params):
    default_temp = tf.constant(0.1)
    if model_name == 'GS':
        model = GS(log_pi=params[0], temp=default_temp)
    elif model_name == 'IGR_I':
        model = IGR_I(mu=params[0], xi=params[1], temp=default_temp)
    else:
        raise RuntimeError
    return model


def broadcast_prior_to_batch_size(parameter, batch_size):
    categories_n, sample_size = parameter.shape[1], parameter.shape[2]
    parameter = tf.broadcast_to(parameter, shape=(batch_size, categories_n, sample_size))
    return parameter
