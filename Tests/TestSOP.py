import numpy as np
import tensorflow as tf
from Utils.load_data import load_mnist_sop_data
from Models.SOP import SOP
from Models.SOPOptimizer import SOPOptimizer
from Models.SOPOptimizer import run_sop


def test_fwd_pass_connections_and_gradient():
    batch_size, width, height = 64, 14, 28
    hyper = {'width_height': (width, height, 1),
             'model_type': 'GS',
             'batch_size': batch_size,
             'units_per_layer': 240,
             'temp': tf.constant(0.1)}
    shape = (batch_size, width, height, 1)
    x_upper, x_lower = create_upper_and_lower_dummy_data(shape=shape)
    sop = SOP(hyper=hyper)
    with tf.GradientTape() as tape:
        logits = sop.call(x_upper=x_upper)
        breakpoint()
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_lower, logits=logits)
    grad = tape.gradient(sources=sop.trainable_variables, target=loss)
    print('\nTEST: Forward pass and gradient computation')
    assert grad is not None


def test_count_of_network_parameters():
    batch_size, width, height, rgb = 64, 14, 28, 1
    units_per_layer = 240
    hyper = {'width_height': (width, height, rgb),
             'model_type': 'GS',
             'batch_size': batch_size,
             'units_per_layer': units_per_layer,
             'temp': tf.constant(0.1)}
    shape = (batch_size, width, height, rgb)
    sop = SOP(hyper=hyper)
    sop.build(input_shape=shape)
    print('\nTEST: Number of parameters in the network')
    assert sop.h1_dense.count_params() == (392 + 1) * units_per_layer
    assert sop.h2_dense.count_params() == (units_per_layer + 1) * units_per_layer
    assert sop.out_dense.count_params() == (units_per_layer + 1) * 392


def test_optimizer_step():
    batch_size = 64
    width = 14
    height = 28
    hyper = {'width_height': (width, height, 1), 'model_type': 'IGR_Planar',
             'batch_size': batch_size, 'learning_rate': 0.001,
             'temp': tf.constant(0.1)}
    shape = (batch_size, width, height, 1)
    x_upper, x_lower = create_upper_and_lower_dummy_data(shape=shape)
    sop = SOP(hyper=hyper)
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyper['learning_rate'])
    sop_opt = SOPOptimizer(model=sop, optimizer=optimizer)
    gradients, loss = sop_opt.compute_gradients_and_loss(x_upper=x_upper, x_lower=x_lower)
    sop_opt.apply_gradients(gradients=gradients)
    assert gradients is not None


def test_in_mnist_sample():
    batch_n = 6
    epochs = 5
    width = 14
    height = 28
    hyper = {'width_height': (width, height, 1), 'model_type': 'GS',
             'batch_size': batch_n, 'learning_rate': 0.001,
             'epochs': epochs, 'iter_per_epoch': 10, 'temp': tf.constant(0.1)}
    results_path = './Log/'
    data = load_mnist_sop_data(batch_n=batch_n, run_with_sample=True)
    run_sop(hyper=hyper, results_path=results_path, data=data)
    hyper['model_type'] = 'IGR_Planar'
    run_sop(hyper=hyper, results_path=results_path, data=data)
    hyper['model_type'] = 'IGR_I'
    run_sop(hyper=hyper, results_path=results_path, data=data)
    hyper['model_type'] = 'IGR_SB'
    run_sop(hyper=hyper, results_path=results_path, data=data)


def create_upper_and_lower_dummy_data(shape):
    x_upper = generate_square(position='upper', shape=shape)
    x_lower = generate_square(position='lower', shape=shape)
    return x_upper, x_lower


def generate_square(shape, position):
    v = np.zeros(shape=shape)
    if position == 'lower':
        v[:, 0:5, 10:20] = 1
    else:
        v[:, 10:, 10:20] = 1
    v = tf.constant(value=v, dtype=tf.float32)
    return v
