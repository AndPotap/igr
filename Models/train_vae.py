import time
import pickle
import tensorflow as tf
from Utils.load_data import load_vae_dataset
from Models.VAENet import construct_networks, determine_path_to_save_results
from Models.OptVAE import OptVAE, OptGauSoftMax, OptSBVAE, OptExpGS
from Models.OptVAE import OptGauSoftMaxDis, OptExpGSDis, OptPlanarNFDis
from Utils.viz_vae import plot_originals, plot_reconstructions_samples_and_traversals
from Utils.general import setup_logger, append_timestamp_to_file
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Train VAE
# ===========================================================================================================


def run_vae(hyper, run_with_sample):
    data = load_vae_dataset(dataset_name=hyper['dataset_name'], batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'], run_with_sample=run_with_sample,
                            architecture=hyper['architecture'], hyper=hyper)
    train_dataset, test_dataset, test_images, hyper = data

    vae_opt = construct_nets_and_optimizer(hyper=hyper, model_type=hyper['model_type'])

    train_vae(vae_opt=vae_opt, hyper=hyper, train_dataset=train_dataset,
              test_dataset=test_dataset, test_images=test_images)


def construct_nets_and_optimizer(hyper, model_type):
    nets = construct_networks(hyper=hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    if model_type == 'VAE':
        vae_opt = OptVAE(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS':
        vae_opt = OptExpGS(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS_Dis':
        vae_opt = OptExpGSDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I':
        vae_opt = OptGauSoftMax(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I_Dis':
        vae_opt = OptGauSoftMaxDis(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_SB_Dis':
        vae_opt = OptSBVAE(nets=nets, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_Planar_Dis':
        vae_opt = OptPlanarNFDis(nets=nets, optimizer=optimizer, hyper=hyper)
    else:
        raise RuntimeError
    return vae_opt


def train_vae(vae_opt, hyper, train_dataset, test_dataset, test_images, monitor_gradients=False):
    writer, logger, results_path = start_all_logging_instruments(hyper=hyper, test_images=test_images)
    init_vars = run_initialization_procedure(hyper, test_images, results_path)
    (hyper_file, iteration_counter, results_file, cont_c_linspace, disc_c_linspace, grad_monitor_dict,
     grad_norm) = init_vars

    with writer.as_default():
        initial_time = time.time()
        for epoch in range(1, hyper['epochs'] + 1):
            t0 = time.time()
            train_loss_mean = tf.keras.metrics.Mean()
            for x_train in train_dataset.take(hyper['iter_per_epoch']):
                vae_opt, iteration_counter = perform_train_step(x_train, vae_opt, train_loss_mean,
                                                                iteration_counter, disc_c_linspace, cont_c_linspace)
            t1 = time.time()
            monitor_vanishing_grads(monitor_gradients, x_train, vae_opt,
                                    iteration_counter, grad_monitor_dict, epoch)

            evaluate_progress_in_test_set(epoch=epoch, test_dataset=test_dataset, vae_opt=vae_opt,
                                          hyper=hyper, logger=logger, iteration_counter=iteration_counter,
                                          train_loss_mean=train_loss_mean, time_taken=t1 - t0,
                                          grad_norm=grad_norm)

            save_intermediate_results(epoch, vae_opt, test_images,
                                      hyper, results_file, results_path, writer)

        save_final_results(vae_opt.nets, logger, results_file, initial_time, temp=vae_opt.temp.numpy())


def start_all_logging_instruments(hyper, test_images):
    results_path = determine_path_to_save_results(model_type=hyper['model_type'],
                                                  dataset_name=hyper['dataset_name'])
    writer = tf.summary.create_file_writer(logdir=results_path)
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=results_path + '/loss.log',
                                                                 termination='.log'),
                          logger_name=append_timestamp_to_file('logger', termination=''))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    plot_originals(test_images=test_images, results_path=results_path)
    return writer, logger, results_path


def log_all_hyperparameters(hyper, logger):
    logger.info(f"GPU Available: {tf.test.is_gpu_available()}")
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def run_initialization_procedure(hyper, test_images, results_path):
    init_vars = initialize_vae_variables(results_path=results_path, hyper=hyper)
    hyper_file, *_ = init_vars

    with open(file=hyper_file, mode='wb') as f:
        pickle.dump(obj=hyper, file=f)

    return init_vars


def initialize_vae_variables(results_path, hyper):
    iteration_counter = 0
    results_file = results_path + '/vae.h5'
    hyper_file = results_path + '/hyper.pkl'
    cont_c_linspace = convert_into_linspace(hyper['continuous_c_linspace'])
    disc_c_linspace = convert_into_linspace(hyper['discrete_c_linspace'])
    grad_monitor_dict = {}
    grad_norm = tf.constant(0., dtype=tf.float32)
    init_vars = (hyper_file, iteration_counter, results_file, cont_c_linspace, disc_c_linspace, grad_monitor_dict,
                 grad_norm)
    return init_vars


def convert_into_linspace(limits_tuple):
    var_linspace = tf.linspace(start=limits_tuple[0], stop=limits_tuple[1], num=limits_tuple[2])
    return var_linspace


def perform_train_step(x_train, vae_opt, train_loss_mean, iteration_counter, disc_c_linspace, cont_c_linspace):
    output = vae_opt.compute_gradients(x=x_train)
    gradients, loss, log_px_z, kl, kl_n, kl_d = output
    vae_opt.apply_gradients(gradients=gradients)
    iteration_counter += 1
    train_loss_mean(loss)
    append_train_summaries(tracking_losses=output, iteration_counter=iteration_counter)
    update_regularization_channels(vae_opt=vae_opt, iteration_counter=iteration_counter,
                                   disc_c_linspace=disc_c_linspace, cont_c_linspace=cont_c_linspace)
    return vae_opt, iteration_counter


def append_train_summaries(tracking_losses, iteration_counter):
    gradients, loss, log_px_z, kl, kl_n, kl_d = tracking_losses
    tf.summary.scalar(name='Train ELBO', data=-loss, step=iteration_counter)
    tf.summary.scalar(name='Train Recon', data=log_px_z, step=iteration_counter)
    tf.summary.scalar(name='Train KL', data=kl, step=iteration_counter)
    tf.summary.scalar(name='Train KL Norm', data=kl_n, step=iteration_counter)
    tf.summary.scalar(name='Train KL Dis', data=kl_d, step=iteration_counter)


def update_regularization_channels(vae_opt, iteration_counter, disc_c_linspace, cont_c_linspace):
    if iteration_counter < disc_c_linspace.shape[0]:
        vae_opt.continuous_c = cont_c_linspace[iteration_counter]
        vae_opt.discrete_c = disc_c_linspace[iteration_counter]


def monitor_vanishing_grads(monitor_gradients, x_train, vae_opt, iteration_counter, grad_monitor_dict, epoch):
    if monitor_gradients:
        grad_norm = vae_opt.monitor_parameter_gradients_at_psi(x=x_train)
        grad_monitor_dict.update({iteration_counter: grad_norm.numpy()})
        with open(file='./Results/gradients_' + str(epoch) + '.pkl', mode='wb') as f:
            pickle.dump(obj=grad_monitor_dict, file=f)


def evaluate_progress_in_test_set(epoch, test_dataset, vae_opt, hyper, logger, iteration_counter,
                                  time_taken, train_loss_mean):
    test_progress = create_test_progress_tracker()
    for x_test in test_dataset.take(hyper['iter_per_epoch']):
        test_progress = update_test_progress(test_progress)
    log_test_progress()


def create_test_progress_tracker():
    vars_to_track = {'elbo': (False, False), 'elbo_closed': (False, True), 'jv': (True, False),
                     'jv_closed': (True, True), 'n_mean': ()}
    test_track = {'vars_to_track': vars_to_track}
    for k, _ in vars_to_track.items():
        test_track[k] = tf.keras.metrics.Mean()
    return test_track


def update_test_progress(x_test, vae_opt, test_progress):
    test_progress['n_mean'](vae_opt.n_required)
    for k, v in test_progress['vars_to_track'].items():
        loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_jv=v[0], run_closed_form_kl=v[1])
        test_progress[k](loss)
    return test_progress


def log_test_progress(logger, test_progress, epoch, time_taken, iteration_counter, temp):
    test_print = f'Epoch {epoch:4d} || '
    for k, _ in test_progress['vars_to_track'].items():
        loss = test_progress[k].result().numpy()
        test_print += f'{k} {-loss:2.5e} || '
    test_print +=
    logger.info(f'TeELBO {-elbo.result().numpy():2.5e} || '
                f'TeELBOC {-elbo_closed.result().numpy():2.5e} || '
                f'TeJV {jv.result().numpy():2.5e} || '
                f'TeJVC {jv_closed.result().numpy():2.5e} || '
                f'TrL {train_loss_mean.result().numpy():2.5e} || '
                f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || '
                f'N: {n_mean.result():4.1f}')
    logger.info(test_print)
    tf.summary.scalar(name='Test ELBO', data=-test_progress['elbo'].result(), step=epoch)
    tf.summary.scalar(name='N Required', data=test_progress['n_mean'].result(), step=epoch)
    tf.summary.scalar(name='Temp', data=temp, step=epoch)


def save_intermediate_results(epoch, vae_opt, test_images, hyper, results_file, results_path, writer):
    if epoch % 10 == 0:
        vae_opt.nets.save_weights(filepath=append_timestamp_to_file(results_file, '.h5'))
        plot_reconstructions_samples_and_traversals(hyper=hyper, epoch=epoch, results_path=results_path,
                                                    test_images=test_images, vae_opt=vae_opt)
    writer.flush()


def save_final_results(nets, logger, results_file, initial_time, temp):
    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    logger.info(f'Final temp {temp: 4.5f}')
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    nets.save_weights(filepath=results_file)

# ===========================================================================================================
