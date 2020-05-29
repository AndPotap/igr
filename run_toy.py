import tensorflow as tf
from rebar_toy import RELAX
from rebar_toy import toy_loss

lr = 1 * 1.e-2
batch_n, categories_n, sample_size, num_of_vars = 1, 1, 1, 1
shape = (batch_n, categories_n, sample_size, num_of_vars)
# max_iter = 1 * int(1.e4)
# check_every = int(1.e3)
max_iter = 1 * int(1.e3)
check_every = int(1.e2)

relax = RELAX(toy_loss, lr, shape)

for idx in range(max_iter):
    grads, loss = relax.compute_gradients()
    relax.apply_gradients(grads)
    theta = tf.math.sigmoid(relax.log_alpha)
    eta = relax.eta.numpy()
    temp = tf.math.exp(relax.log_temp).numpy()
    if idx % check_every == 0:
        print(f'Loss {loss.numpy(): 2.5e} || '
              f'Theta {theta.numpy()[0, 0, 0, 0]: 2.5e} || '
              f'Temp {temp:2.5e} || Eta {eta:2.5e} || '
              f'i {idx: 4d}')
