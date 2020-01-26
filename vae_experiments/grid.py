import tensorflow as tf
from collections import namedtuple
from Models.train_vae import run_vae
run_with_sample = True
epochs = 50
model_type = 'IGR_Planar_Dis'
n_required = 9
latent_discrete_n = n_required + 1
# model_type = 'GS_Dis'
# n_required = 10
# latent_discrete_n = n_required

temps = [0.1, 0.25, 0.5]
# temps = [0.67, 0.85, 1.0]

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': n_required, 'num_of_discrete_var': 30,
         'latent_norm_n': 0, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': epochs, 'architecture': 'dense',
         'run_jv': False, 'γ': tf.constant(30.),
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': latent_discrete_n})
hyper.update({'model_type': model_type, 'temp': 0.10,
              'prior_file': './Results/mu_xi_unif_10_IGR_SB_Finite.pkl', 'num_of_discrete_param': 2,
              'run_closed_form_kl': True})

Case = namedtuple('Case', 'dataset arch')
datasets = [Case(dataset='mnist', arch='dense'),
            Case(dataset='fmnist', arch='dense'),
            Case(dataset='celeb_a', arch='conv_jointvae')]
i = 0
experiment = {}
for d in datasets:
    for t in temps:
        i += 1
        exp = {'dataset_name': d.dataset, 'architecture': d.arch, 'temp': t}
        print(exp)
        experiment.update({i: exp})

for _, d in experiment.items():
    for key, value in d.items():
        hyper[key] = value
    run_vae(hyper=hyper, run_with_sample=run_with_sample)
