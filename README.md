# Invertible Gaussian Reparameterization: Revisting the Gumbel-Softmax

This repo contains a TensorFlow 2.0 implementation of the Invertible Gaussian Reparameterization.

<br>**Abstract**<br>
*The Gumbel-Softmax is a continuous distribution over the simplex that is often used as a relaxation of discrete
distributions. Because it can be readily interpreted and easily reparameterized, it enjoys widespread use. Unfortunately, we
show that the cost of this aesthetic interpretability is material: the temperature hyperparameter must be set too high, KL
estimates are noisy, and as a result, performance suffers. We circumvent the previous issues by proposing a much simpler and
more flexible reparameterizable family of distributions that transforms Gaussian noise into a one-hot approximation through an
invertible function. This invertible function is composed of a modified softmax and can incorporate diverse transformations
that serve different specific purposes. For example, the stick-breaking procedure allows us to extend the reparameterization
trick to distributions with countably infinite support, or normalizing flows let us increase the flexibility of the
distribution. Our construction improves numerical stability and outperforms the Gumbel-Softmax in a variety of experiments
while generating samples that are closer to their discrete counterparts and achieving lower-variance gradients.*

## Overview
The goal of this documentation is to clarify the structure of the repo and to provide a guide to replicate the
results from the paper. To avoid reading all the repo details, I will recommend that you find the files that
are linked to the experiment that you want to run and only then check the information about the folder of
interest.

### Requirements
Briefly, the requirements for the project are to install Python >= 3.6 (since we use printing syntax
only available after 3.6) and to `pip` install the packages from `requirements.txt` (they will fetch all the dependencies
needed). This repo was develop using Tensorflow 2.0.1, but it runs for 2.1.0 as well. The only
package that requires a specific version is Tensorflow Datasets 1.3.0 since the features in CelebA
changed. Moreover, we also added a Singulariy definition file `igr_singularity.def` if you want to create an
image to run the project in a HPC cluster (it will only require that the host has the 10.0 CUDA drivers available).
Finally, check that the installation was successful by running
```
cd ./igr/
python3 vae_experiments/mnist_vae.py
```
from wherever you cloned the repo. It should successfuly run an experiment with an small sample that should
take 5 - 10 seconds.

## General Information

### Structure of the Repository

* `Log`: is the directory where the outputs from the experiments are logged and saved (weights of the NNs).
  The contents of this directory are disposable, move the results that you want to save into the
  `Results` directory for further analysis (see below).
* `Models`: contains the training functions (`train_vae.py`, `SOPOptimizer,py`), the optimizer classes
(`OptVAE`, `SOPOptimizer`), and the neural network architectures (`VAENet.py` `SOP.py`)
for both the VAE and for the SOP experiments.
* `Results`: this directory serves two purposes. First, it holds all the figures created by the scripts in the
  repo and then it contains the outputs from the experiments runs that are used as input for some figures /
  tables.
* `Tests`: contains various tests for relevant classes in the repo. The name indicates which class is being tested.
* `Utils`: has key functions / classes used throughout the repo: `Distributions.py` contains the GS and
  IGR samplers, `MinimizeEmpiricalLoss.py` contains the optimizer class for learning the parameters that
  approximate a discrete distribution, `general.py` contains the functions used for approximating discrete
  distributions, to evaluate simplex proximity and to initializing the distribution's parameters,
  `load_data.py` contains all the utils needed to load the data, `posterior_sampling_funcs.py`, contains the
  functions to sample from the posterior distribution and `viz_vae.py` contains the functions to
  plot the performance over time.
* `approximations`: contains all the scripts to approximate discrete distributions and to learn the IGR
  priors (more details in the next section).
* `vae_experiments`: contains all the scripts to run the VAE experiments to
replicate the paper results (more details in the next section).
* `structure_output_prediction`: contains all the scripts to run the SOP experiments (more details in the next section).

### Conventions

This experiment is run by a function named `run_vae` this function takes as arguments two parameters. (1) is a
dictionary that contains all the hyperparameter specifications of the model `hyper` and (2) a flag to test
that the code is running properly `run_with_sample` (set to False in order to run the experiment with all the data).
The contents of that the hyperparameter dictionary expects are detailed below:

| Key                               | Value (Example)                            | Description     |
| :-------------------------------- | :------------------------------: | :-----------------------: |
| `dataset_name` | `<str> ('mnist')`  | The name of the dataset to run the model. |
| `model_type` | `<str> ('ExpGSDis')`  | The name of the model to use. Look at `./Models/train_vae.py`  for all the model options. |
| `temp` | `<float> (0.25)`  | The value of the temperature hyperparameter.|
| `sample_size`  | `<int> (1)`  | The number of samples that are taken from the noise distribution at each iteration. |
| `n_required` | `<int> ('10')`  | The number of categories needed for each discrete variable. |
| `num_of_discrete_param` | `<int> (1)`  | The number of parameters for the discrete variables. |
| `num_of_discrete_var` | `<int> (2)`  | The number of discrete variables in the model. |
| `num_of_norm_param` | `<int> (0)`  | The number of parameters for the continuous variables. |
| `num_of_norm_var` | `<int> (0)`  | The number of continuous variables in the model. |
| `latent_norm_n` | `<int> (0)`  | The dimensionality of the continuous variables in the model. |
| `architecture` | `<str> ('dense')`  | The neural network architecture employed. All the options are in `./Models/VAENet.py`.|
| `learning_rate` | `<float> (0.001)`  | The learning rate used when training. |
| `batch_n` | `<int> (64)`  | The batch size taken per iteration. |
| `epochs` | `<int> (100)`  | The number of epochs used when training. |
| `run_jv` | `<bool> (False)`  | Whether to run the JointVAE model or not. |
| `gamma` | `<int> (100)`  | The number of epochs used when training. |
| `cont_c_linspace` | `<tuple> ((0., 5., 25000))`  | The lower bound, upper bound, and how many iters to get from lower to upper. |
| `disc_c_linspace` | `<tuple> ((0., 5., 25000))`  | The lower bound, upper bound, and how many iters to get from lower to upper. |
| `check_every` | `<int> (1)`  | How often (in terms of epochs) the test loss is evaluated. |


## Replicating Figures / Tables in the paper

### Implementation Nuances

The IGR-SB extends the reparameterization trick to distributions with an countably infinite
support. However, to implement the IGR-SB with finite resources, there is a threshold imposed on the
stick breaking procedure that determines a finite number of categories that are needed for a proper
estimation. Additionally, the number of categories has a maximum that is set as a
reference (this is called `latent_discrete_n` in the `hyper` dictionary). This maximum can be moved
accordingly to fit the problem. For example, for CelebA we set a maximum number of categories to 50
but the thresholding procedure ended up selecting 20-30. Moving the maximum beyond 50 would have
resulted in waste of memory allocated but would have not yield any quantitive difference. However,
setting the maximum to 10 would have truncated the stick-breaking procedure too soon and would have
resulted in a loss of performance. To avoid this situation, we recommend monitoring if the threshold
is met. If not, then increasing the maximum would be needed.
