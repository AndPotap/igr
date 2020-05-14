import tensorflow as tf
from Models.train_vae import run_vae_for_all_cases

run_with_sample = True
num_of_repetitions = 1
temps = [0.15]
# temps = [0.03, 0.07, 0.10, 0.15, 0.25, 0.50, 0.67]
model_cases = {
    1: {'model_type': 'IGR_I_Dis', 'n_required': 9,
        'prior_file': './Results/mu_xi_unif_' + str(round(temps[0] * 100)) + '_IGR_I.pkl'},
    # 2: {'model_type': 'IGR_Planar_Dis', 'n_required': 9},
    # 3: {'model_type': 'IGR_SB_Finite_Dis', 'n_required': 9,
    #     'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl'},
    # 4: {'model_type': 'IGR_SB_Dis', 'n_required': 49,
    #     'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl',
    #     'threshold': 0.9, 'truncation_option': 'quantile'},
    # 5: {'model_type': 'GS_Dis', 'n_required': 10},
}
dataset_cases = {
    1: {'dataset_name': 'mnist', 'architecture': 'dense'},
    # 2: {'dataset_name': 'fmnist', 'architecture': 'dense'},
    # 3: {'dataset_name': 'celeb_a', 'architecture': 'conv_jointvae'},
}
hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
         'test_with_one_hot': True,
         'sample_from_disc_kl': False,
         'check_every': 10,
         'sample_size_testing': 1 * int(1.e2),
         'learning_rate': 3 * 1.e-4,
         'batch_n': 100, 'epochs': 100,
         'sample_from_cont_kl': True, 'num_of_discrete_var': 20,
         'sample_size': 1, 'stick_the_landing': False,
         'run_jv': False, 'gamma': tf.constant(30.),
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}

run_vae_for_all_cases(hyper, model_cases, dataset_cases, temps,
                      num_of_repetitions, run_with_sample)
