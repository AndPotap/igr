from Utils.estimate_loglike import estimate_log_likelihood
from Utils.estimate_loglike import get_available_logs
from Utils.estimate_loglike import manage_files
from Utils.estimate_loglike import setup_logger

run_with_sample = True
samples_n = 1 * int(1.e3)
# datasets = ['mnist', 'fmnist']
datasets = ['mnist']
architectures = ['nonlinear']
models = {
    # 1: {'model_dir': 'igr', 'model_type': 'IGR_I_Dis'},
    2: {'model_dir': 'gs', 'model_type': 'GS_Dis'},
    # 3: {'model_dir': 'pf', 'model_type': 'IGR_Planar_Dis'},
    # 4: {'model_dir': 'sb', 'model_type': 'IGR_SB_Finite_Dis'},
}
logger = setup_logger(log_file_name='./Log/nll.txt', logger_name='nll')
for dataset in datasets:
    for arch in architectures:
        for _, v in models.items():
            path_to_trained_models = './Results/trained_models/' + dataset + '/'
            path_to_trained_models += v['model_dir'] + '/'
            path_to_trained_models += arch + '/'
            logs = get_available_logs(path_to_trained_models)
            for l in logs:
                current_path = path_to_trained_models + l + '/'
                checks, weights_file = manage_files(path_to_trained_models + l + '/')
                logger.info(current_path)
                estimate_log_likelihood(current_path, dataset, weights_file, logger,
                                        samples_n, v['model_type'], run_with_sample)
