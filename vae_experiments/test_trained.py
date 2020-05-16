import pickle
import tensorflow as tf
from Models.train_vae import construct_nets_and_optimizer
from Utils.load_data import load_vae_dataset

dataset_name = 'mnist'
# path_to_trained_models = './Results/trained_models/' + dataset_name + '/'
path_to_trained_models = './Results/trained_models/' + '/'
models = {
    1: {'model_dir': 'igr_i', 'label': 'IGR-I(0.15)', 'model_type': 'IGR_I'},
    2: {'model_dir': 'gs', 'label': 'GS(0.67)', 'model_type': 'GS_Dis'},
}
select_case = 2
run_with_sample = False
samples_n = 1
hyper_file = 'hyper.pkl'
weights_file = 'vae.h5'
model_type = models[select_case]['model_type']
path_to_trained_models += models[select_case]['model_dir'] + '/'

with open(file=path_to_trained_models + hyper_file, mode='rb') as f:
    hyper = pickle.load(f)
batch_n = hyper['batch_n']
batch_n = int(1.e4)
data = load_vae_dataset(dataset_name=dataset_name, batch_n=batch_n, epochs=hyper['epochs'],
                        run_with_sample=run_with_sample,
                        architecture=hyper['architecture'], hyper=hyper)
train_dataset, test_dataset, np_test_images, hyper = data
epoch = hyper['epochs']
vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=model_type)
vae_opt.nets.load_weights(filepath=path_to_trained_models + weights_file)

test_loss_mean = tf.keras.metrics.Mean()
for x_test in test_dataset:
    loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x_test,
                                                          sample_from_cont_kl=False,
                                                          sample_from_disc_kl=False)
    test_loss_mean(loss)
test_print = f'Epoch {epoch:4d} || '
test_print += f'TeELBOC{-test_loss_mean.result():2.5e} || '
print(test_print)
