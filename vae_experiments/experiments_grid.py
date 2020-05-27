import tensorflow as tf
from Models.train_vae import run_vae_for_all_cases

run_with_sample = False
temps = [0.15]
# temps = [5.00]
# temps = [0.03, 0.07, 0.10, 0.15, 0.25, 0.50, 0.67]

# import numpy as np
# np.random.seed(21)
# seeds = np.random.randint(low=1, high=int(1.e4), size=5)
# seeds = [5328, 5945, 8965, 49, 9337]
seeds = [5328]

model_cases = {
    1: {'model_type': 'IGR_I_Dis', 'n_required': 9,
        'prior_file': './Results/mu_xi_unif_10_IGR_I.pkl'},
    # 2: {'model_type': 'IGR_Planar_Dis', 'n_required': 9,
    #     'prior_file': './Results/mu_xi_unif_10_IGR_I.pkl'},
    # 3: {'model_type': 'IGR_SB_Finite_Dis', 'n_required': 9,
    #     'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl'},
    # 4: {'model_type': 'IGR_SB_Dis', 'n_required': 49,
    #     'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl',
    #     'threshold': 0.9, 'truncation_option': 'quantile'},
    # 5: {'model_type': 'GS_Dis', 'n_required': 10},
    # 6: {'model_type': 'Relax_GS_Dis', 'n_required': 10},
    # 7: {'model_type': 'Relax_Ber_Dis', 'n_required': 200},
}
dataset_cases = {
    # 1: {'dataset_name': 'mnist', 'architecture': 'dense'},
    # 1: {'dataset_name': 'mnist', 'architecture': 'dense_nonlinear'},
    # 2: {'dataset_name': 'fmnist', 'architecture': 'dense'},
    3: {'dataset_name': 'omniglot', 'architecture': 'dense'},
}
hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
         'sample_size_testing': 1 * int(1.e0),
         'check_every': 10,
         'sample_from_disc_kl': True,
         'test_with_one_hot': True,
         'epochs': 1000,
         'learning_rate': 3 * 1.e-4,
         'batch_n': 100,
         'num_of_discrete_var': 20,
         # 'num_of_discrete_var': 1,  # for bernoulli experiment
         # 'batch_n': 64,
         # 'num_of_discrete_var': 30,
         'stick_the_landing': True,
         'sample_size': 1, 'sample_from_cont_kl': True,
         'run_jv': False, 'gamma': tf.constant(30.),
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}

run_vae_for_all_cases(hyper, model_cases, dataset_cases, temps,
                      seeds, run_with_sample)
