import tensorflow as tf
from rebar_toy import RELAX
from rebar_toy import toy_loss

lr = 1 * 1.e-2
shape = (1, 1, 1, 1)
max_iter = 1 * int(1.e3)

relax = RELAX(toy_loss, lr, shape)

for idx in range(max_iter):
    grads, loss = relax.compute_gradients()
    relax.apply_gradients(grads)
    theta = tf.math.sigmoid(relax.log_alpha)
    if idx % 100 == 0:
        print(f'Loss {loss.numpy(): 2.5e} | |'
              f'Theta {theta.numpy()[0, 0, 0, 0]: 2.5e} | | i {idx: 4d}')
