from Utils.estimate_loglike import estimate_log_likelihood

dataset_name = 'mnist'
# dataset_name = 'fmnist'
# dataset_name = 'omniglot'
path_to_trained_models = './Results/trained_models/' + dataset_name + '/'
models = {
    1: {'model_dir': 'igr', 'model_type': 'IGR_I_Dis'},
    2: {'model_dir': 'gs', 'model_type': 'GS_Dis'},
    3: {'model_dir': 'pf', 'model_type': 'IGR_Planar_Dis'},
    4: {'model_dir': 'sb', 'model_type': 'IGR_SB_Finite_Dis'},
}
select_case = 2
run_with_sample = True
samples_n = 1 * int(1.e3)

model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

estimate_log_likelihood(path_to_trained_models, dataset_name, samples_n, model_type, run_with_sample)
