import tensorflow as tf
from Models.train_vae import run_vae

run_with_sample = True
# seeds = [5328, 5945, 8965, 49, 9337]
hyper = {'latent_norm_n': 0, 'num_of_norm_param': 0, 'num_of_norm_var': 0,
         'temp': 1.0, 'sample_from_disc_kl': True, 'stick_the_landing': True,
         'test_with_one_hot': False, 'sample_from_cont_kl': True,
         'sample_size_testing': 1 * int(1.e0),
         'dataset_name': 'mnist',
         'seed': 9335,
         'model_type': 'DLGMM_IGR_SB',
         # 'model_type': 'DLGMM_IGR',
         'architecture': 'dlgmm_dense',
         'batch_n': 100,
         'n_required': 50,
         'latent_discrete_n': 50,
         # 'n_required': 5,
         # 'latent_discrete_n': 5,
         'num_of_discrete_param': 4,
         'sample_size': 1,
         'num_of_discrete_var': 50,
         'epochs': 300,
         'learning_rate': 3 * 1.e-4,
         'save_every': 50,
         'check_every': 50,
         'dtype': tf.float32}
run_vae(hyper, run_with_sample=run_with_sample)
