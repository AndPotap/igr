from Utils.load_data import load_mnist_sop_data
from Models.SOPOptimizer import run_sop
import tensorflow as tf

model_type = 'GS'
# model_type = 'IGR_I'
# model_type = 'IGR_Planar'
hyper = {'width_height': (14, 28, 1),
         'units_per_layer': 240,
         'model_type': model_type,
         'batch_size': 100,
         'learning_rate': 0.01,
         'weight_decay': 1.e-3,
         'min_learning_rate': 1.e-4,
         'epochs': 1 * int(1.e2),
         'iter_per_epoch': 937,
         'test_sample_size': 100,
         'temp': tf.constant(1.0)}
data = load_mnist_sop_data(batch_n=hyper['batch_size'])
run_sop(hyper=hyper, results_path='./Log/', data=data)
