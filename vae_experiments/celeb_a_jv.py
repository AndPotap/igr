# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Load Imports and  data
# ===========================================================================================================
import tensorflow as tf
from Models.train_vae import run_vae
run_with_sample = True
# model_type = 'IGR_Planar'
model_type = 'IGR_SB'

hyper = {'dataset_name': 'celeb_a', 'sample_size': 1, 'n_required': 49, 'num_of_discrete_var': 1,
         'latent_norm_n': 32, 'num_of_norm_var': 1, 'num_of_norm_param': 2,
         'learning_rate': 0.001, 'batch_n': 64, 'epochs': 100, 'architecture': 'conv_jointvae',
         'run_jv': True, 'γ': tf.constant(100.),
         'cont_c_linspace': (0., 50., 100_000), 'disc_c_linspace': (0., 10., 100_000)}
hyper.update({'latent_discrete_n': hyper['n_required'] + 1})

if model_type in ['GS', 'GS_Dis']:
    hyper.update({'model_type': model_type, 'temp': 0.25, 'num_of_discrete_param': 1,
                  'run_closed_form_kl': False, 'n_required': hyper['n_required'] + 1})

elif model_type in ['IGR_I', 'IGR_Planar']:
    hyper.update({'model_type': model_type, 'temp': 0.50, 'num_of_discrete_param': 2,
                  'run_closed_form_kl': True})
elif model_type in ['IGR_SB', 'IGR_SB_Finite']:
    hyper.update({'model_type': model_type, 'temp': 0.50, 'num_of_discrete_param': 2,
                  'run_closed_form_kl': True, 'threshold': 0.99, 'truncation_option': 'quantile',
                  'prior_file': './Results/mu_xi_unif_50_IGR_SB_Finite.pkl'})
else:
    raise RuntimeError
run_vae(hyper=hyper, run_with_sample=run_with_sample)
# ===========================================================================================================
