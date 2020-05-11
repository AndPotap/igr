import time
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from Models.SOP import SOP, brodcast_to_sample_size
from Utils.general import setup_logger
from Utils.general import append_timestamp_to_file


class SOPOptimizer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @tf.function()
    def perform_fwd_pass(self, x_upper, use_one_hot, sample_size=1):
        logits = self.model.call(x_upper=x_upper, use_one_hot=use_one_hot, sample_size=sample_size)
        return logits

    @tf.function()
    def compute_gradients_and_loss(self, x_upper, x_lower):
        with tf.GradientTape() as tape:
            logits = self.perform_fwd_pass(x_upper=x_upper, use_one_hot=False)
            loss = compute_loss(x_lower=x_lower, logits=logits, sample_size=1)
        gradients = tape.gradient(target=loss, sources=self.model.trainable_variables)
        return gradients, loss

    @tf.function()
    def compute_loss_for_testing(self, x_upper, x_lower, use_one_hot, sample_size):
        logits = self.perform_fwd_pass(x_upper=x_upper,
                                       use_one_hot=use_one_hot, sample_size=sample_size)
        loss = compute_loss(x_lower=x_lower, logits=logits, sample_size=sample_size)
        return loss

    @tf.function()
    def apply_gradients(self, gradients):
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


@tf.function()
def compute_loss(x_lower, logits, sample_size):
    width_loc, height_loc, rgb_loc = 1, 2, 3
    x_lower_broad = brodcast_to_sample_size(x_lower, sample_size)
    log_pxl_z_broad = -tf.nn.sigmoid_cross_entropy_with_logits(labels=x_lower_broad, logits=logits)
    log_pxl_z = tf.math.reduce_sum(log_pxl_z_broad, axis=[width_loc, height_loc, rgb_loc])
    loss = -tf.math.reduce_logsumexp(log_pxl_z, axis=1)
    loss = tf.math.reduce_mean(loss, axis=0) + tf.math.log(tf.constant(sample_size, dtype=tf.float32))
    return loss


def run_sop(hyper, results_path, data):
    train_dataset, test_dataset = data

    sop_optimizer = setup_sop_optimizer(hyper=hyper)

    log_path = results_path + f'/loss_{sop_optimizer.model.model_type}.log'
    logger = setup_logger(log_file_name=append_timestamp_to_file(file_name=log_path,
                                                                 termination='.log'))
    log_all_hyperparameters(hyper=hyper, logger=logger)
    train_sop(sop_optimizer=sop_optimizer, hyper=hyper, train_dataset=train_dataset,
              test_dataset=test_dataset, logger=logger)


def setup_sop_optimizer(hyper):
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'],
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         decay=1.e-3)
    model = SOP(hyper=hyper)
    sop_optimizer = SOPOptimizer(model=model, optimizer=optimizer)
    return sop_optimizer


def log_all_hyperparameters(hyper, logger):
    for key, value in hyper.items():
        logger.info(f'Hyper: {key}: {value}')


def train_sop(sop_optimizer, hyper, train_dataset, test_dataset, logger):
    initial_time = time.time()
    iteration_counter = 0
    for epoch in range(1, hyper['epochs'] + 1):
        train_mean_loss = tf.keras.metrics.Mean()
        tic = time.time()
        for x_train in train_dataset.take(hyper['iter_per_epoch']):
            x_train_lower = x_train[:, 14:, :, :]
            x_train_upper = x_train[:, :14, :, :]

            gradients, loss = sop_optimizer.compute_gradients_and_loss(x_upper=x_train_upper,
                                                                       x_lower=x_train_lower)
            sop_optimizer.apply_gradients(gradients=gradients)
            train_mean_loss(loss)
            iteration_counter += 1
        toc = time.time()
        evaluate_progress_in_test_set(epoch=epoch, sop_optimizer=sop_optimizer,
                                      test_dataset=test_dataset,
                                      logger=logger, hyper=hyper,
                                      iteration_counter=iteration_counter,
                                      time_taken=toc - tic,
                                      train_mean_loss=train_mean_loss)

    final_time = time.time()
    logger.info(f'Total training time {final_time - initial_time: 4.1f} secs')
    results_file = f'./Log/model_weights_{sop_optimizer.model.model_type}.h5'
    results_file = append_timestamp_to_file(file_name=results_file, termination='.h5')
    sop_optimizer.model.save_weights(filepath=results_file)


def evaluate_progress_in_test_set(epoch, sop_optimizer, test_dataset, logger, hyper, iteration_counter,
                                  time_taken, train_mean_loss):
    test_mean_loss = tf.keras.metrics.Mean()
    for x_test in test_dataset.take(hyper['iter_per_epoch']):
        x_test_lower = x_test[:, 14:, :, :]
        x_test_upper = x_test[:, :14, :, :]
        loss = sop_optimizer.compute_loss_for_testing(x_upper=x_test_upper,
                                                      x_lower=x_test_lower, use_one_hot=True,
                                                      sample_size=100)
        test_mean_loss(loss)
    logger.info(f'Epoch {epoch:4d} || Test_Recon {test_mean_loss.result().numpy():2.5e} || '
                f'Train_Recon {train_mean_loss.result().numpy():2.5e} || '
                f'Temp {sop_optimizer.model.temp:2.1e} || '
                f'{sop_optimizer.model.model_type} || '
                f'Time {time_taken:4.1f} sec || i: {iteration_counter:6,d}')


def viz_reconstruction(test_image, model):
    x_test_upper = test_image[:, :14, :, :]
    logits = model.call(x_upper=x_test_upper)
    reconstruction = tf.math.round(tf.math.sigmoid(logits))
    image = np.concatenate((x_test_upper.numpy(), reconstruction), axis=1)
    selected_list = [10, 23, 2, 5, 20, 11, 0, 12, 33, 52]
    for selected in selected_list:
        plt.figure()
        plt.imshow(image[selected, :, :, 0])
        plt.savefig(f'./Results/recon_{selected}.png')

        plt.figure()
        plt.imshow(test_image.numpy()[selected, :, :, 0], cmap='gray')
        plt.savefig(f'./Results/recon_original_{selected}.png')
