import time
from Utils.estimate_loglike import calculate_test_log_likelihood
from Utils.estimate_loglike import setup_logger
from Utils.estimate_loglike import load_hyper_and_data
from Utils.estimate_loglike import setup_optimizer

tic = time.time()
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

weights_file = 'vae.h5'
model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

test_dataset, hyper, epoch = load_hyper_and_data(path_to_trained_models, dataset_name,
                                                 samples_n, run_with_sample)
vae_opt = setup_optimizer(path_to_trained_models, hyper, model_type)
logger = setup_logger(log_file_name='./Log/nll.txt', logger_name='nll')

calculate_test_log_likelihood(logger, vae_opt, test_dataset, epoch, model_type, tic)
