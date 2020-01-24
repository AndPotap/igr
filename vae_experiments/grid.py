import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
model_type = 'IGR_Planar_Dis'

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 9, 'num_of_discrete_var': 30,
         'latent_norm_n': 0, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'dense',
         'run_jv': False, 'γ': tf.constant(30.),
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': hyper['n_required'] + 1})
hyper.update({'model_type': model_type, 'temp': 0.10,
              'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl', 'num_of_discrete_param': 2,
              'run_closed_form_kl': True})

experiment = {
    1: {'dataset_name': 'mnist', 'temp': 0.10},
    2: {'dataset_name': 'mnist', 'temp': 0.50},
    3: {'dataset_name': 'fmnist', 'temp': 0.10},
    4: {'dataset_name': 'fmnist', 'temp': 0.50},
    5: {'dataset_name': 'celeb_a', 'temp': 0.10, 'architecture': 'conv_jointvae'},
    6: {'dataset_name': 'celeb_a', 'temp': 0.50, 'architecture': 'conv_jointvae'},
}

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)
