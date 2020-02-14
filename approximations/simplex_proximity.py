import numpy as np
import tensorflow as tf
# from approximations.simplex_proximity_funcs import load_parameter_values, run_simulation, plot_stat_result
from Utils.simplex_proximity_funcs import run_simulation, plot_stat_result
from Utils.simplex_proximity_funcs import fit_curve_to_result_along_grid
from Utils.simplex_proximity_funcs import plot_boxplots

K, N, S = 50, 10, 8
# K, N, S = 50, 100, 10_000

pi = tf.constant(value=[1 / K for _ in range(K)], dtype=tf.float32, shape=(1, K, 1, 1))
# mu, xi = load_parameter_values(prior_file='./Results/mu_xi_unif_50_ng.pkl')


all_temps = np.linspace(start=0.05, stop=1.0, num=N)
threshold = 0.99

models = {'GS': [pi]}
# models = {'IGR_I': [mu, xi], 'GS': [pi]}

stats2run = ['median', 'p10', 'p20', 'std']

results = run_simulation(samples=S, τ_grid=all_temps, models=models, threshold=threshold, stats2run=stats2run)

plot_boxplots(model='GS', results=results, τ_grid=all_temps)
plot_stat_result(stat='median', models=models, results=results, τ_grid=all_temps)

# plot_boxplots(model='IGR_I', results=results, τ_grid=all_temps)
# igr_curve = fit_curve_to_result_along_grid(result=results['IGR_I']['median'], τ_grid=all_temps)
gs_curve = fit_curve_to_result_along_grid(result=results['GS']['median'], τ_grid=all_temps)
