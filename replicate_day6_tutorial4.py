
#-----------------------------------
"要自律，要自律，要自律，默念三遍"

"https://flax.readthedocs.io/en/v0.5.3/notebooks/annotated_mnist.html"

#-----------------------------------
import os

import flax.training.train_state
os.system('clear') 

import flax.training
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, grad, vmap, pmap, tree 
from jax import random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import train_state
import optax
import haiku as hk

from typing import Sequence, NamedTuple 
from inspect import signature

#-----------------------------------
def start_new_block(msg=''):
    print("\n>>> ------------------------------------")
    print(msg +'\n' if msg else '')

#-----------------------------------
start_new_block('Start Deepminds optax')

print('- Set training hyper params:')
lr = 1e-2
n_epoch = 10
log_period_epoch = 1

print('- Prepare training inputs/targets:')
seed = 23
batch_size = 1500
x_dim = 10
y_dim = 1
noise_scale = 0.1

key = jax.random.PRNGKey(seed=seed)
key, x_key, m_key, w_key, b_key, n_key = jax.random.split(key, num=6)

xs = jax.random.normal(x_key, (batch_size, x_dim))
W  = jax.random.normal(w_key, (x_dim, y_dim))
b  = jax.random.normal(b_key, (y_dim, ))
n  = jax.random.normal(n_key, (batch_size, y_dim)) * noise_scale
ys = jnp.dot(xs, W) + b + n

print('- Create mlp model:')
model = nn.Dense(features=5) # not yet init, not actually weights/bias
params = model.init(m_key, xs) # init here, weights/bias available

print("- Create mse Loss fn:")
def make_mse_loss(xs, ys):
    """ given params, return mse """
    def mse_loss(params):
        def square_error(x, y):
            pred = model.apply(params, x) # run inference in Flax
            loss = jnp.inner(y-pred, y-pred)
            return loss
        return jnp.mean(jax.vmap(square_error)(xs, ys), axis = 0)
    return jax.jit(mse_loss)

mse_loss = make_mse_loss(xs, ys)
value_and_grad_fn = jax.value_and_grad(mse_loss)

print('- Set optimizer:')
opt_sgd = optax.sgd(learning_rate=lr)
opt_state = opt_sgd.init(params)
print(opt_state)

#-----------------------------------
start_new_block('Start Training:')
for epoch in range(n_epoch):
    loss, grads = value_and_grad_fn(params)
    updates, opt_state = opt_sgd.update(grads, opt_state) # return state
    params = optax.apply_updates(params, updates)
    if epoch % log_period_epoch == 0:
        print(f'epoch: {epoch}, loss: {loss}')

#-----------------------------------
start_new_block('Optax is very powerful:')
print('- create arbitrary optimizers')
print('- with arbitrary hyperparams, chaining, param freezing')
print('- official doc: https://optax.readthedocs.io/en/latest/ ')

# Example from Flax (ImageNet example)
# https://github.com/google/flax/blob/main/examples/imagenet/train.py#L88
"""def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn

tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
)
"""
# Example from Haiku (ImageNet example)
# https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/train.py#L116
"""def make_optimizer() -> optax.GradientTransformation:
  # SGD with nesterov momentum and a custom lr schedule
  return optax.chain(
      optax.trace(
          decay=FLAGS.optimizer_momentum,
          nesterov=FLAGS.optimizer_use_nesterov),
      optax.scale_by_schedule(lr_schedule), optax.scale(-1))"""

#-----------------------------------
start_new_block('Create Custom Models: advanced Optax examples')
class MLP(nn.Module): 
    # nn.Module is Python's dataclass
    # https://docs.python.org/3/library/dataclasses.html
    num_neurons_per_layer: Sequence[int]
    def setup(self): # as dataclass implicitly uses the __init__ function
        self.layers = [nn.Dense(n) for n in self.num_neurons_per_layer]
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers)-1 :
                x = nn.relu(x)
        return x
key = jax.random.PRNGKey(seed=23)
x_key, init_key = jax.random.split(key)
model = MLP(num_neurons_per_layer=[16, 8, 1]) # define mlp
x = jax.random.uniform(x_key, (4,4)) # input x -> (batch size, input dim)
params = model.init(init_key, x) # init params
y = model.apply(params, x) # one forward pass 

print(jax.tree.map(jnp.shape, params)) # jnp.shape is a fn
print(f'Output: {y}')

#-----------------------------------
start_new_block('Another way, init inline with nn.compact pattern')
print('https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/setup_or_nncompact.html')

class MLP(nn.Module): 
    # nn.Module is Python's dataclass
    # https://docs.python.org/3/library/dataclasses.html
    num_neurons_per_layer: Sequence[int]
    @nn.compact
    def __call__(self, x):
        for i, num_neurons in enumerate(self.num_neurons_per_layer):
            x = nn.Dense(num_neurons)(x)
            if i != len(self.num_neurons_per_layer)-1 :
                x = nn.relu(x)
        return x
key = jax.random.PRNGKey(seed=23)
x_key, init_key = jax.random.split(key)
model = MLP(num_neurons_per_layer=[16, 8, 1]) # define mlp
x = jax.random.uniform(x_key, (4,4)) # input x -> (batch size, input dim)
params = model.init(init_key, x) # init params
y = model.apply(params, x) # one forward pass, apply triggers __call__() fn 

print(jax.tree.map(jnp.shape, params)) # jnp.shape is a fn
print(f'Output: {y}')
# Note: this time y is different

#-----------------------------------
msg = "Dive deeper and understand how nn.Dense module designs itself\n\
      Introducing self.params of nn.Module (Trainable parameters)"
start_new_block(msg)

class MyDenseImp(nn.Module):
    num_neurons: int
    weight_init: callable = nn.initializers.lecun_normal()
    bias_init: callable = nn.initializers.zeros
    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', # param name
                            self.weight_init, # init fn, RNG passed implicity via init fn
                            (x.shape[-1], self.num_neurons)) # shape info
        bias = self.param('bias', self.bias_init, (self.num_neurons, ))
        return jnp.dot(x, weight) + bias
    
x_key, init_key = jax.random.split(jax.random.PRNGKey(seed=23))
model = MyDenseImp(num_neurons=3) # init the model
x = jax.random.uniform(x_key, (4,4)) # preprae batched input
params = model.init(init_key, x) # init params
y = model.apply(params, x) # one forward pass

print(jax.tree.map(jnp.shape, params))
print(f"Output: {y}")

print('Discrepency explained: lecun_normal() and zeros both return init fn:')
print(signature(nn.initializers.lecun_normal()))
print(signature(nn.initializers.zeros))



#-----------------------------------
msg = "Introducinng variable (not optimized via gradient descent, but may change in fwd pass)"
start_new_block(msg)
class BiasAdderWithRunningMean(nn.Module):
    decay:float = 0.99
    @nn.compact
    def __call__(self, x):

        is_initialized = self.has_variable('batch_stats', 'ema')

        # 'batch_stats': 
        # - variable collection name, reserved in Flax implementation of BatchNorm
        # 'ema': 
        # - variable name
        ema = self.variable('batch_stats', 'ema', lambda shape: jnp.zeros(shape), x.shape[1:])

        # self.param will by default add this variable to "params" collection, 
        # not 'batch_stats' collection

        # again some idiosyncrasies here, we need to pass a key even though 
        # we don't actually use it..
        
        bias = self.param('bias', lambda key, shape: jnp.zeros(shape), x.shape[1:])

        if is_initialized:
            # self.variable returns a reference hence .value
            ema.value = self.decay * ema.value + (1.0-self.decay) * jnp.mean(x, axis = 0, keepdims = True)
        
        return x - ema.value + bias
    
# ----------------  
x_key, init_key = jax.random.split(jax.random.PRNGKey(seed = 23))
model = BiasAdderWithRunningMean()
x = jax.random.uniform(x_key, (10, 4)) # batched input
variables = model.init(init_key, x, mutable=True)

print(f"Multiple collections = {variables}")

# we have to use mutable since regular params are not modified during the forward pass, 
# but these variables are. We can't keep state internally (in jax) so we have to return it
y, non_trainable_variable_updates = model.apply(variables, x, mutable = ['batch_stats'])

print(y)
print(non_trainable_variable_updates)


#-----------------------------------
msg = "Let's train such model (with both trainable/non-trainable parameters):"
start_new_block(msg)

def update_step(opt, apply_fn, x, opt_state, params, non_trainable_params):

    def loss_fn(params):
        y, non_trainable_variable_updates = apply_fn(
                {"params": params, **non_trainable_params}, # add to params
                x,
                mutable = list(non_trainable_params.keys())) # set mutable via key
        loss = ((x-y) ** 2).sum() # not doing anything, jsut for demo purpose
        return loss, non_trainable_variable_updates
    
    (loss, non_trainable_params), grads = jax.value_and_grad(loss_fn, has_aux = True)(params)
    
    updates, opt_state = opt.update(grads, opt_state) # calculate and apply updates
    params = optax.apply_updates(params, updates)

    return opt_state, params, non_trainable_params # all of these represent the state - ugly

# init model/input
model = BiasAdderWithRunningMean()
x = jnp.ones((10,4))
variables = model.init(jax.random.PRNGKey(seed = 23), x, mutable=True)
print(f"Multiple collections = {variables}")

# get trainable/non-trainables
non_trainable_params = {"batch_stats":variables.pop('batch_stats')}
params = variables.pop('params')

print(non_trainable_params)
print(params)

del variables # free resource

# set optimizer
sgd_opt = optax.sgd(learning_rate=0.1) 
opt_state = sgd_opt.init(params)

# training loop
for i in range(5):
    # we will later see how TrainState abstraction make this step much more elegant
    opt_state, params, non_trainable_params = update_step(sgd_opt, model.apply, x, opt_state, params, non_trainable_params)
    print(f'batchnorm iteration {i}, non trainable params:', non_trainable_params)
    print(f'batchnorm iteration {i}, trainable params:', params)
    

#-----------------------------------
msg = "Let's go one level up ** with abstraction ** \n"
msg += "- certain layers like BatchNorm will use variables in the background\n"
msg += "- a last and complicated example in Flax idiosyncrasies"
start_new_block(msg)

class DDNBlock(nn.Module):
    """
    Dense + Dropout + BatchNorm Combo
    """
    num_neurons: int
    training: bool

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_neurons)(x)
        x = nn.Dropout(rate=0.5, deterministic=not self.training)(x)
        x = nn.BatchNorm(use_running_average= not self.training)(x)
        return x
    
key1, key2, key3, key4 = jax.random.split(jax.random.PRNGKey(seed=23), num=4)

model = DDNBlock(num_neurons=3, training=True)
x = jax.random.uniform(key1, (3,4,4))

# New: because of dropout, we now have to include its unique key
# kinda wierd, but you will get used to it
variables = model.init({'params':key2, 'dropout':key3}, x)
print(variables)

# and same here, everything else remain the same as previous example
y, non_trainable_params = model.apply(variables, x, 
                                      rngs={'dropout':key4}, 
                                      mutable=['batch_stats'])

# let's run these model variables during "evaluation"
eval_model = DDNBlock(num_neurons=3, training = False)
# as training = False, we do not have stochasticity in fwd pass
# and as well no need to update the non-trainable variables within 'batch_states'
y = eval_model.apply(variables, x)
print('y\n', y)

#-----------------------------------
msg = "Finnally, a fully fledged CNN on MNIST, in Flax"
start_new_block(msg)

# define the cnn
class CNN(nn.Module): # lots of hardcoding, for demo purpose
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2,2), strides=(2,2))
        x = x.reshape((x.shape[0], -1)) # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x

#-----------------------------------
msg = "Reuse pytorch data loaders and create data pipeline"
start_new_block(msg)

def custom_transform(x):
    return np.expand_dims(np.array(x, dtype=np.float32), axis=2) / 255.
def custom_collate_fn(batch):
    transposed_data = list(zip(*batch))
    labels = np.array(transposed_data[1])
    imgs = np.stack(transposed_data[0])
    return imgs, labels

mnist_img_size = (28,28,1)
batch_size = 128

from torchvision.datasets import MNIST 
from torch.utils.data import DataLoader
train_dataset = MNIST(root = './train_mnist', 
                               train = True, 
                               download=True,
                               transform=custom_transform)
test_dataset =  MNIST(root = './train_mnist', 
                               train = False, 
                               download=True,
                               transform=custom_transform)

train_loader = DataLoader(train_dataset,
                              batch_size=128,
                              shuffle=True,
                              collate_fn=custom_collate_fn,
                              drop_last=True)

test_loader = DataLoader(test_dataset,
                              batch_size=128,
                              shuffle=False,
                              collate_fn=custom_collate_fn,
                              drop_last=True)

#-----------------------------------
msg = "Speed optimization - loading whole dataset into memory"
start_new_block(msg)

train_images = jnp.array(train_dataset.data)
train_labels = jnp.array(train_dataset.targets)

# np.expand_dims is to convert shape from (10000, 28, 28) -> (10000, 28, 28, 1)
# We don't have to do this for training images because custom_transform does it for us.
test_images = np.expand_dims(jnp.array(test_dataset.data), axis=3)
test_labels = jnp.array(test_dataset.targets)

#-----------------------------------
msg = "Define the evaluation metrics"
start_new_block(msg)

# asterisk and slash in function parameter
# https://realpython.com/python-asterisk-and-slash-special-parameters/
# Left side                     	Divider	      Right side
# Positional-only arguments  	    /	          Positional or keyword arguments
# Positional or keyword arguments	*	          Keyword-only arguments

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'acc': accuracy,
  }
  return metrics
 
def cross_entropy_loss(*, logits, labels):
  labels_onehot = jax.nn.one_hot(labels, num_classes=10)
  return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()

#-----------------------------------
msg = "Define the training functions "
start_new_block(msg)

@jax.jit
def train_step(state, imgs, gt_labels):
    def loss_fn(params):
        logits = CNN().apply({'params': params}, imgs)
        one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes= len(MNIST.classes))
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis = -1))
        return loss, logits
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads) # this is whole update now, concise!
    metrics = compute_metrics(logits=logits, labels=gt_labels)
    return state, metrics

def train_one_epoch(state, dataloader, epoch):
    "train for 1 epoch on the training set"
    batch_metrics = []
    for cnt, (imgs, labels) in enumerate(dataloader):
        state, metrics = train_step(state, imgs, labels)
        batch_metrics.append(metrics)
    # aggregate the metrics
    batch_metric_np = jax.device_get(batch_metrics) # pull from accelerator to host (CPU)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metric_np]) for k in batch_metric_np[0]
        }
    return state, epoch_metrics_np

#-----------------------------------
msg = "Define the evaluation functions "
start_new_block(msg)

@jax.jit
def eval_step(state, imgs, gt_labels):
    logits = CNN().apply({'params': state.params}, imgs)
    return compute_metrics(logits=logits, labels=gt_labels)

def evaluate_model(state, test_imgs, test_lbls):
    """Evaluate on validation set. """
    metrics = eval_step(state, test_imgs, test_lbls)
    metrics = jax.device_get(metrics) # pull from gpu, to cpu
    metrics = jax.tree.map(lambda x:x.item(), metrics) # np.ndarray -> scalar
    return metrics

#-----------------------------------
msg = "Define the train state object:"
start_new_block(msg)

# TrainState object: a nice and tidy way to manage things in training
def create_train_state(key, learning_rate, momentum):
    cnn = CNN()
    params = cnn.init(key, jnp.ones([1, *mnist_img_size]))['params']
    sdg_opt = optax.sgd(learning_rate=learning_rate, momentum=momentum)
    # TrainState is a simple built-in wrapper class
    # that makes things a bit cleaner
    return flax.training.train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=sdg_opt)

#-----------------------------------
msg = "Launch Training:"
start_new_block(msg)

seed = 0
learning_rate = 0.1
momentum = 0.9
num_epochs = 10
batch_size = 32

train_state = create_train_state(jax.random.PRNGKey(seed), learning_rate, momentum)

for epoch in range(1, num_epochs+1):
    
    train_state, train_metrics = train_one_epoch(train_state, train_loader, epoch)
    print(f"Train epoch:{epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['acc']}")

    test_metrics = evaluate_model(train_state, test_images, test_labels)
    print(f"Test epoch:{epoch}, loss: {test_metrics['loss']}, accuracy: {test_metrics['acc']}")

#-----------------------------------
msg = "To add dropout/batchnorm, mind the init and train_step:"
start_new_block(msg)


