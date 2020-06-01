import time
import pickle
import tensorflow as tf
from Models.train_vae import construct_nets_and_optimizer
from Utils.load_data import load_vae_dataset

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
select_case = 4
run_with_sample = False
samples_n = 1 * int(1.e3)

hyper_file = 'hyper.pkl'
weights_file = 'vae.h5'
model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

with open(file=path_to_trained_models + hyper_file, mode='rb') as f:
    hyper = pickle.load(f)

batch_n = hyper['batch_n']
# batch_n = int(1.e4)
hyper['sample_size_testing'] = samples_n
tf.random.set_seed(seed=hyper['seed'])
data = load_vae_dataset(dataset_name=dataset_name, batch_n=batch_n, epochs=hyper['epochs'],
                        run_with_sample=run_with_sample,
                        architecture=hyper['architecture'], hyper=hyper)
train_dataset, test_dataset, np_test_images, hyper = data
epoch = hyper['epochs']
vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=model_type)
vae_opt.nets.load_weights(filepath=path_to_trained_models + weights_file)
vae_opt.test_with_one_hot = True

test_loss_mean = tf.keras.metrics.Mean()
for x_test in test_dataset:
    loss = vae_opt.compute_losses_from_x_wo_gradients(x_test,
                                                      sample_from_cont_kl=False,
                                                      sample_from_disc_kl=False)
    test_loss_mean(loss)
# train_loss_mean = tf.keras.metrics.Mean()
# for x_train in train_dataset:
#     loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x_train,
#                                                           sample_from_cont_kl=False,
#                                                           sample_from_disc_kl=False)
#     train_loss_mean(loss)

evaluation_print = f'Epoch {epoch:4d} || '
evaluation_print += f'TeELBOC {-test_loss_mean.result():2.5e} || '
# evaluation_print += f'Train Loss {train_loss_mean.result():2.5e} || '
evaluation_print += f'{model_type} || '
toc = time.time()
evaluation_print += f'Time: {toc - tic:2.2e} sec'
print(evaluation_print)
