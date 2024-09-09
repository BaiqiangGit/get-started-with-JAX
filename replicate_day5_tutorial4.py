# Side Note:
# Another highlevel intro video of jax from google deepmind: 
# https://www.youtube.com/watch?v=1RcORri2ZJg

"""
Tutorial 4: Flax - From Zero to Hero
https://www.youtube.com/watch?v=5eUSmJvK8WA&t=1567s

Official Flax Docs and examples:
1. https://flax.readthedocs.io/en/latest/
2. https://github.com/google/flax/tree/main/examples

Install Flax and Haiku (Haiku just for comparison purposes):

pip install --upgrade -q git+https://github.com/google/flax.git
pip install --upgrade -q git+https://github.com/deepmind/dm-haiku  

"""

def start_new_block(msg=''):
    print()
    print(">>>>------------------------------------------------")
    if msg: print(msg)
    print()

# ----------------------------------------
start_new_block("import libs: jax, haiku, pytorch, python, etc")
import jax
from jax import lax # lax is a library of primitives, akin xla operations 
from jax import random, numpy as jnp

# Flax - (Flexibility + Jax ==> Flax)
# NN lib built on top of Jax, developed by Google Research (Brain Team)
# Flax was "designed for flexibility"

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn # nn notation also used in Pytroch, and flax's oder api
from flax.training import train_state # a useful dataclass to keep train state

# DeepMind's NN Jax Lib - for comparison purposes, we are not learning Haiku here
import haiku as hk

# Jax optimizers - a separate lib developed by DeepMind
import optax
msg = "In Jax eco-system, every lib takes care of a specific vertical\n\
      - flax/haiku: for neural network building and construction\n\
      - optax: for optimizer, gradient processing and optimization, https://github.com/google-deepmind/optax\n\
      - chex: debugging and testing https://github.com/google-deepmind/chex\n\
      - RLax - Library for implementing reinforcement learning agents\n\
      - EasyLM - LLMs made easy:\n\
      - Scenic - A Jax Library for Computer Vision Research and Beyond\n\
      - Jraph - Lightweight graph neural network library "

print(msg)

# Flax does not re-invent the wheel - we use Pytorch dataloader
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Python libs
import functools
from typing import Any, Callable, Sequence, Optional

# other 3rd party libs
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------
start_new_block("Build NN : explore a simple ffwd model")
model = nn.Dense(features=5)
# all Flax NN layers inherit from the Module class (similar to Pytorch)
print("base class of nn.Dense", nn.Dense.__base__)

# ------------------------------------------
start_new_block("Build NN : Inference with simple model")
seed = 23
key1, key2 = jax.random.split(jax.random.PRNGKey(seed=seed))
x = jax.random.normal(key1, (10,)) # create a dummy input, a 10-d random vector

# initialization call - this gives us the actual model weights
# (remember Jax handles state externally!)
y, params = model.init_with_output(key2, x)
params = model.init(key2, x)

print('y:', y)
print('params type', type(params))
print(jax.tree.map(lambda x:x.shape, params)) # return same pytree object

# Note1: automatic shape inference: 10 is inferred from x
# Note2: this is not any more true: now return params as dict,
#        https://flax.readthedocs.io/en/latest/guides/converting_and_upgrading/regular_dict_upgrade_guide.html
#        previously return immutable structure (hence FrozenDict) 
#        https://github.com/google/flax/issues/1223
# Note3: init_with_output if you care, or simply init: 
#        https://flax.readthedocs.io/en/latest/api_reference/flax.linen/init_apply.html

# ------------------------------------------
start_new_block("Build NN : Predict (simply apply, state is external in Flax)")
y = model.apply(params, x)
print(y)

print("Pytorch Syntax model(x) does not work anymore.")
try:
    y = model(x)
except Exception as e:
    print(e)

# ------------------------------------------
msg = "Code exersize: Haiku vs Flax, Haiku is clearner, Flax is powerful with 9k+ models)"
start_new_block(msg)

print("Haiku needs to explicitly init via hk.tranform fn")
model = hk.transform(lambda x: hk.Linear(5)(x))
seed = 23
key1, key2 = jax.random.split(jax.random.PRNGKey(seed=seed))
x = jax.random.normal(key1, (10,))
params = model.init(key2, x)
y = model.apply(params, None, x) # 
print(y)
print(hk.Linear.__base__)

# Note: flax is training efficient: takes 80% of pytorch runtime
# https://github.com/huggingface/transformers/tree/main/examples/flax/text-classification

# ------------------------------------------
msg = "A toy example: training a linear regression model"
start_new_block(msg)

# ------------------------------------------
msg = "First implement a pure-JAX approch, \nthen in the next do it in the Flax-way"
start_new_block(msg)

# Defining a toy dataset
n_samples = 1500
x_dim = 2 # choose small number here, e.g. 2d, which is easy to visulize
y_dim = 1
noise_amplitude = 0.1

# Generate ground truth W and b
# Note: we could get W,b from a randomly innitialized nn.Dense, 
# while we init via jax.random here to be closer to JAX for now
key, w_key, b_key = jax.random.split(jax.random.PRNGKey(seed=seed), num=3)
W = jax.random.normal(w_key, (x_dim, y_dim))
b = jax.random.normal(b_key, (y_dim, ))

# this is the structure Flax Expects (nested frozen dict)
true_params = {'params':{"bias":b, 'kernel':W}}

# Generate samples with additional noise
key, x_key, noise_key = jax.random.split(key, num=3)

xs = random.normal(x_key, (n_samples, x_dim))
ns = noise_amplitude * random.normal(noise_key, (n_samples, y_dim)) # noise
ys = jnp.dot(xs, W) + b + ns # noisy targets, b is broadcasted in batch dimension

print(f"xs shape = {xs.shape}; ys shape = {ys.shape}")

# ------------------------------------------
msg = "Visualize data in 3D, via scatter plot (looks not working properly ...)"
start_new_block(msg)
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
assert xs.shape[-1] == 2 and ys.shape[-1] == 1
ax.scatter(xs[:, 0], xs[:, 1], zs = ys)

# plt.show()

# ------------------------------------------
msg = "Make Loss function"
start_new_block(msg)

def make_mse_loss(xs, ys):

    def mse_loss(params):
        """
        Given model params, return loss on (xs, ys)
        """
        def squared_error(x,y):
            pred = model.apply(params, x)
            # Inner product becuase 'y' should have in general more than 1 dims
            loss = jnp.inner(y-pred, y-pred) / 2.0
            return loss
        # batched version via vmap
        return jnp.mean(jax.vmap(squared_error)(xs, ys), axis = 0)
    
    return jax.jit(mse_loss) # jit the result
    
mse_loss = make_mse_loss(xs, ys)
value_and_grad_fn = jax.value_and_grad(mse_loss) # this is only the function

# ------------------------------------------
msg = "Let's reuse the simple feed forward layer, \
       since it trivially implements linear regression"
start_new_block(msg)

model = nn.Dense(features = y_dim) # automatic shape inference
params = model.init(key, xs)
print(f"Initial Params = {params}")

# training hyper parameters
lr = 0.3
epochs = 20
log_period_epoch = 5

# start trianing
for epoch in range(epochs):
    loss, grads = value_and_grad_fn(params)
    # SGD (pure jax way)
    params = jax.tree.map(lambda p,g: p-lr*g, params, grads)
    if epoch % log_period_epoch == 0:
        print(f"epoch {epoch}, loss = {loss}")
print(f"learned params = {params}")
print(f"gt params = {true_params}")





        


