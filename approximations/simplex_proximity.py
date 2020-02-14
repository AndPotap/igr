import numpy as np
import tensorflow as tf
# from approximations.simplex_proximity_funcs import load_parameter_values, run_simulation, plot_stat_result
from approximations.simplex_proximity_funcs import run_simulation, plot_stat_result
from approximations.simplex_proximity_funcs import fit_curve_to_result_along_grid
from approximations.simplex_proximity_funcs import plot_boxplots

# K, N, S = 50, 10, 8
K, N, S = 50, 100, 10_000

π = tf.constant(value=[1 / K for _ in range(K)], dtype=tf.float32, shape=(1, K, 1, 1))
# μ_ng, ξ_ng = load_parameter_values(prior_file='./Results/mu_xi_unif_50_ng.pkl')


all_temps = np.linspace(start=0.05, stop=1.0, num=N)
threshold = 0.99
models = {'GS': [π]}
# models = {'GauSoftMax': [μ_ng, ξ_ng], 'ExpGS': [π]}
# models = {'ExpGS': [π], 'GauSoftPlus': [μ_ng, ξ_ng], 'GauSoftMax': [μ_ng, ξ_ng], 'Cauchy': [μ_ng, ξ_ng]}
stats2run = ['median', 'p10', 'p20', 'std']

results = run_simulation(samples=S, τ_grid=all_temps, models=models, threshold=threshold, stats2run=stats2run)

plot_boxplots(model='GS', results=results, τ_grid=all_temps)
plot_stat_result(stat='median', models=models, results=results, τ_grid=all_temps)
# plot_boxplots(model='IGR_I', results=results, τ_grid=all_temps)
# ng_curve = fit_curve_to_result_along_grid(result=results['GauSoftMax']['median'], τ_grid=all_temps)
gs_curve = fit_curve_to_result_along_grid(result=results['GS']['median'], τ_grid=all_temps)
