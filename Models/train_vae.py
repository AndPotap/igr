import time
import pickle
import tensorflow as tf
from Utils.load_data import load_vae_dataset
from Models.VAENet import setup_model, determine_path_to_save_results
from Models.OptVAE import OptVAE, OptGauSoftMax, OptSBVAE, OptExpGS
from Models.OptVAE import OptGauSoftMaxDis, OptExpGSDis
from Models.OptVAE import OptPlanarNFDis
from Utils.viz_vae import plot_originals
from Utils.viz_vae import plot_reconstructions_samples_and_traversals
from Utils.general import setup_logger
from Utils.general import append_timestamp_to_file
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# ===========================================================================================================
# Train VAE
# ===========================================================================================================


def run_vae(hyper, run_with_sample):
    data = load_vae_dataset(dataset_name=hyper['dataset_name'], batch_n=hyper['batch_n'],
                            epochs=hyper['epochs'], run_with_sample=run_with_sample,
                            architecture=hyper['architecture'], hyper=hyper)
    train_dataset, test_dataset, test_images, hyper = data

    model, vae_opt = get_model_and_optimizer(hyper=hyper, model_type=hyper['model_type'])

    train_vae_model(vae_opt=vae_opt, model=model, hyper=hyper, train_dataset=train_dataset,
                    test_dataset=test_dataset, test_images=test_images)


def get_model_and_optimizer(hyper, model_type):
    model = setup_model(hyper=hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    if model_type == 'VAE':
        vae_opt = OptVAE(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS':
        vae_opt = OptExpGS(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'GS_Dis':
        vae_opt = OptExpGSDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I':
        vae_opt = OptGauSoftMax(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_I_Dis':
        vae_opt = OptGauSoftMaxDis(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_SB_Dis':
        vae_opt = OptSBVAE(model=model, optimizer=optimizer, hyper=hyper)
    elif model_type == 'IGR_Planar_Dis':
        vae_opt = OptPlanarNFDis(model=model, optimizer=optimizer, hyper=hyper)
    else:
        raise RuntimeError
    return model, vae_opt


def train_vae_model(vae_opt, model, hyper, train_dataset, test_dataset, test_images, monitor_gradients=False):
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

            save_intermediate_results(epoch, model, vae_opt, test_images,
                                      hyper, results_file, results_path, writer)

        save_final_results(model, logger, results_file, initial_time, temp=vae_opt.temp.numpy())


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
                                  time_taken, train_loss_mean, grad_norm):
    n_mean = tf.keras.metrics.Mean()
    recon_mean, kl_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    kl_n_mean, kl_d_mean = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    jv_closed, jv = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    elbo_closed, elbo = tf.keras.metrics.Mean(), tf.keras.metrics.Mean()
    use_analytical = hyper['use_analytical_in_test']
    for x_test in test_dataset.take(hyper['iter_per_epoch']):
        jv_closed_loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=True,
                                                                        run_analytical_kl=True)
        jv_loss, *_ = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=True,
                                                                 run_analytical_kl=False)
        output_closed = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=False,
                                                                   run_analytical_kl=True)
        output = vae_opt.compute_losses_from_x_wo_gradients(x=x_test, run_ccβvae=False,
                                                            run_analytical_kl=False)
        elbo_loss, log_px_z, kl, kl_n, kl_d = output
        elbo_closed_loss, log_px_z_closed, kl_closed, kl_n_closed, kl_d_closed = output_closed
        jv_closed(jv_closed_loss)
        jv(jv_loss)
        elbo(elbo_loss)
        elbo_closed(elbo_closed_loss)
        n_mean(vae_opt.n_required)
        recon_mean(log_px_z)
        if use_analytical:
            kl_mean(kl_closed)
            kl_n_mean(kl_n_closed)
            kl_d_mean(kl_d_closed)
        else:
            kl_mean(kl)
            kl_n_mean(kl_n)
            kl_d_mean(kl_d)
    logger.info(f'Epoch {epoch:4d} || TeELBO {-elbo.result().numpy():2.5e} || '
                f'TeELBOC {-elbo_closed.result().numpy():2.5e} || '
                f'TeJV {jv.result().numpy():2.5e} || '
                f'TeJVC {jv_closed.result().numpy():2.5e} || '
                f'TrL {train_loss_mean.result().numpy():2.5e} || '
                f'{time_taken:4.1f} sec || i: {iteration_counter:6,d} || '
                f'N: {n_mean.result():4.1f}')
    if use_analytical:
        tf.summary.scalar(name='Test ELBO', data=-elbo_closed.result(), step=epoch)
    else:
        tf.summary.scalar(name='Test ELBO', data=-elbo.result(), step=epoch)
    tf.summary.scalar(name='N Required', data=n_mean.result(), step=epoch)
    tf.summary.scalar(name='Temp', data=vae_opt.temp, step=epoch)
    tf.summary.scalar(name='Test Recon', data=recon_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL', data=kl_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL Norm', data=kl_n_mean.result(), step=epoch)
    tf.summary.scalar(name='Test KL Dis', data=kl_d_mean.result(), step=epoch)


def save_intermediate_results(epoch, model, vae_opt, test_images, hyper, results_file, results_path, writer):
    if epoch % 10 == 0:
        model.save_weights(filepath=append_timestamp_to_file(results_file, '.h5'))
        plot_reconstructions_samples_and_traversals(model=model, hyper=hyper, epoch=epoch,
                                                    results_path=results_path,
                                                    test_images=test_images, vae_opt=vae_opt)
    writer.flush()


def save_final_results(model, logger, results_file, initial_time, temp):
    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    logger.info(f'Final temp {temp: 4.5f}')
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    model.save_weights(filepath=results_file)

# ===========================================================================================================
