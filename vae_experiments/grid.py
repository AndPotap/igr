import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
epochs = 5
model_type = 'IGR_SB'
# n_required = 9
n_required = 49
latent_discrete_n = n_required + 1
run_closed_form_kl = True
num_of_discrete_param = 2
# model_type = 'GS'
# n_required = 10
# latent_discrete_n = n_required
# run_closed_form_kl = False
# num_of_discrete_param = 1

temps = [0.1, 0.25, 0.5]
# temps = [0.67, 0.85, 1.0]

hyper = {'model_type': model_type, 'architecture': 'conv_jointvae',
         'n_required': n_required, 'latent_discrete_n': latent_discrete_n,
         'num_of_discrete_param': num_of_discrete_param, 'num_of_discrete_var': 1,
         'num_of_norm_param': 2, 'num_of_norm_var': 1,
         'epochs': epochs, 'learning_rate': 0.0005, 'batch_n': 64, 'sample_size': 1,
         'run_jv': True, 'run_closed_form_kl': run_closed_form_kl,
         'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl', 'threshold': 0.99,
         'truncation_option': 'quantile'}

cases = {
    1: {'dataset_name': 'mnist', 'γ': tf.constant(30.), 'latent_norm_n': 10,
        'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)},
    2: {'dataset_name': 'fmnist', 'γ': tf.constant(30.), 'latent_norm_n': 10,
        'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)},
    3: {'dataset_name': 'celeb_a', 'γ': tf.constant(100.), 'latent_norm_n': 32,
        'cont_c_linspace': (0., 50., 100_000), 'disc_c_linspace': (0., 10., 100_000)}
}
i = 0
experiment = {}
for _, c in cases.items():
    for t in temps:
        i += 1
        c.update({'temp': t})
        print(c)
        experiment.update({i: c})

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)
