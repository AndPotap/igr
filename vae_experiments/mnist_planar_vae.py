# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True

hyper = {'dataset_name': 'mnist', 'sample_size': 1, 'n_required': 9, 'num_of_discrete_var': 30,
         'model_type': 'IGR_Planar_Dis', 'temp': 0.10, 'num_of_discrete_param': 2,
         'latent_norm_n': 0, 'num_of_norm_var': 0, 'num_of_norm_param': 0,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'dense_nf',
         'run_jv': False, 'γ': tf.constant(30.), 'run_closed_form_kl': True,
         'cont_c_linspace': (0., 5., 25_000), 'disc_c_linspace': (0., 5., 25_000)}
hyper.update({'latent_discrete_n': hyper['n_required'] + 1})

run_vae(hyper=hyper, run_with_sample=run_with_sample)
# ===========================================================================================================
