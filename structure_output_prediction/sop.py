from Utils.load_data import load_mnist_sop_data
from Models.SOPOptimizer import run_sop_for_all_cases
import tensorflow as tf

model_type = 'GS'
# model_type = 'IGR_I'
# model_type = 'IGR_Planar'

# num_of_repetitions = 1
# architectures = ['double_linear']
# sample_sizes = [1]
num_of_repetitions = 3
architectures = ['double_linear', 'triple_linear', 'nonlinear']
sample_sizes = [1, 5, 50]
baseline_hyper = {'width_height': (14, 28, 1),
                  'units_per_layer': 240,
                  'model_type': model_type,
                  'batch_size': 100,
                  # 'learning_rate': 0.01,
                  'learning_rate': 0.001,
                  'weight_decay': 1.e-3,
                  'min_learning_rate': 1.e-4,
                  'epochs': 5 * int(1.e2),
                  'check_every': 50,
                  'iter_per_epoch': 937,
                  'test_sample_size': 100,
                  'architecture': 'nonlinear',
                  'sample_size': 1,
                  'temp': tf.constant(1.0)}
idx = 0
variant_hyper = {}
for arch in architectures:
    for sample_size in sample_sizes:
        variant_hyper.update({idx: {'sample_size': sample_size, 'architecture': arch}})
        idx += 1
data = load_mnist_sop_data(batch_n=baseline_hyper['batch_size'])
run_sop_for_all_cases(baseline_hyper, variant_hyper, data, num_of_repetitions)
