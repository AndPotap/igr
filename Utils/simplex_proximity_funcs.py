import time
import pickle
from matplotlib import pyplot as plt
import numba
import seaborn as sns
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import tensorflow as tf
from Utils.Distributions import generate_sample


def load_parameter_values(prior_file):
    with open(file=prior_file, mode='rb') as f:
        parameters = pickle.load(f)
    mu = tf.constant(parameters['mu'], dtype=tf.float32)
    xi = tf.constant(parameters['xi'], dtype=tf.float32)
    return mu, xi


def plot_boxplots(model: str, results, temp_grid, mult=5):
    obs_n, samples_n = results[model]['sample'].shape
    rows_list = []
    plt.figure(dpi=150)
    plt.style.use(style='ggplot')
    for i in range(obs_n // mult):
        for s in range(samples_n):
            entry = {'tau': temp_grid[mult * i], 'distance': results[model]['sample'][mult * i, s]}
            rows_list.append(entry)
    df = pd.DataFrame(rows_list)
    ax = sns.boxplot(x='tau', y='distance', data=df, color='royalblue', boxprops={'alpha': 0.5})
    ax.set_xticklabels([f'{tau:0.2f}' for tau in [temp_grid[mult * i] for i in range(obs_n // mult)]])
    ax.tick_params(labelrotation=90)
    plt.xlabel('τ')
    plt.ylabel('Distribution of Distance to Simplex Vertex')
    plt.ylim([0., 1.])
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{model}_boxplot.png')
    plt.show()


def plot_stat_result(stat: str, models, results, temp_grid):
    colors = ['blue', 'red', 'orange', 'purple', 'green']
    plt.figure(dpi=150)
    plt.title(stat.title() + ' Distance to Simplex Vertex')

    for idx, model in enumerate(models.keys()):
        curve = fit_curve_to_result_along_grid(result=results[model][stat], temp_grid=temp_grid)
        plt.plot(temp_grid, results[model][stat], label=model, color=colors[idx], alpha=0.5)
        plt.plot(temp_grid, curve.predict(temp_grid.reshape(-1, 1)),
                 label=model + ' smooth', color=colors[idx], alpha=0.25, linestyle='--')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Euclidean Distance')
    plt.annotate(s='Using a Uniform Distribution with 50 categories', xy=(1, -40),
                 xycoords='axes points', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./Results/Outputs/{stat}_distance_to_simplex.png')
    plt.show()


def fit_curve_to_result_along_grid(result: np.ndarray, temp_grid: np.ndarray, gamma: float = 10.):
    curve = KernelRidge(alpha=1.0, kernel='rbf', gamma=gamma)
    curve.fit(temp_grid.reshape(-1, 1), result.reshape(-1, 1))
    return curve


def run_simulation(samples, temp_grid, models, threshold, stats2run):
    results = create_placeholders_for_statistical_results(stats=stats2run, models=models,
                                                          samples_per_stat=temp_grid.shape[0])
    tic = time.time()
    for model, params in models.items():
        results[model].update({'sample': np.zeros(shape=(temp_grid.shape[0], samples))})
        categories_n = params[0].shape[1] if model == 'GS' else params[0].shape[1] + 1
        results[model].update({'sample_var': np.zeros(shape=(temp_grid.shape[0], categories_n))})
        for i in range(temp_grid.shape[0]):
            temp = tf.constant(temp_grid[i], dtype=tf.float32)
            psi = generate_sample(sample_size=samples, params=params, dist_type=model, temp=temp,
                                  threshold=threshold, output_one_hot=True)[0, :, :, 0]
            diff = calculate_distance_to_simplex(psi=psi, argmax_locs=np.argmax(psi, axis=0))
            results[model]['sample_var'][i, :] = tf.math.reduce_variance(psi, axis=1)
            results[model]['sample'][i, :] = diff[np.argsort(diff)][:samples]
            for stat in stats2run:
                results[model][stat][i] = compute_statistic(stat=stat, samples=diff)
    print(f'\nExperiment took: {time.time() - tic:6.1f} sec')
    return results


@numba.jit(nopython=True, parallel=True)
def calculate_distance_to_simplex(psi, argmax_locs):
    samples_n = psi.shape[1]
    categories_n = psi.shape[0]
    diffs = np.zeros(shape=samples_n)
    for s in numba.prange(samples_n):
        zeros = np.zeros(shape=categories_n)
        zeros[argmax_locs[s]] = 1
        diffs[s] = np.sqrt(np.sum((zeros - psi[:, s]) ** 2))
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
