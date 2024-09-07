
import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, vmap, pmap
from jax import make_jaxpr
import matplotlib.pyplot as plt
from typing import NamedTuple

"""
Parallelism in Jax: with pmap
"""

def start_new_block(msg = ''):
    if msg:
        print(f'\n>>{'-'*50}\n{msg}\n')
    else:
        print(f'\n>>{'-'*50}\n')

#-------------------------------------------
msg = 'Parallelism in Jax: with pmap'
start_new_block(msg)
# pmap basics
print(jax.devices()) 

#-------------------------------------------
start_new_block()

# let's run a simply example
x = np.arange(5)
w = np.array([2.,3.,4.])

def convolve(w, x): # implememntation of 1D conv
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    return jnp.array(output)

result = convolve(w,x)
print(repr(result))

#-------------------------------------------
start_new_block()

n_devices = jax.local_device_count()
print(f'Number of available devices: {n_devices}')

# let's now work with a heavier load
xs = np.arange(5 * n_devices).reshape(-1, 5)
ws = np.stack([w] * n_devices)

# Note
print(xs.shape, ws.shape)

#-------------------------------------------
msg = 'First way of optimizing this: using vmap'
start_new_block(msg)

vmap_result = jax.vmap(convolve, in_axes=(0,0))(ws, xs)
print(result) # call __str__()
print(repr(vmap_result)) # call __repr__()

#-------------------------------------------
msg = 'Amazingly: simply swap vmap for pmap, you are now running on multiple devices'
start_new_block(msg)
pmap_result = jax.pmap(convolve, in_axes=(0,0))(ws, xs)
print(pmap_result)
print(repr(pmap_result))

#-------------------------------------------
msg = 'No cross-device communication costs. \n Computations are done independently on each dev'
start_new_block(msg)
print('pmap ref:\nhttps://www.tensorflow.org/probability/examples/Distributed_Inference_with_JAX?hl=zh-cn')
double_pmap_result = jax.pmap(convolve)(jax.pmap(convolve)(ws, xs), xs)
print(repr(double_pmap_result))

#-------------------------------------------
msg = 'Same but with a broadcast w (recall: same as for vmap!)'
start_new_block(msg)
pmap_smarter_result = jax.pmap(convolve, in_axes=(None, 0))(w, xs)
print('in_axes = (None, 0) tells: \nfirst argument shoud be broadcasted \nand 2nd argument 0th dimension is batch dimension')
print(repr(pmap_smarter_result))

#-------------------------------------------
msg = 'Communication between devices: with jax.pmap'
start_new_block(msg)

# a normalization fn
def normalized_convolution(w, x):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    output = jnp.array(output)
    normalizer = jax.lax.psum(output, axis_name='batch_dm') # here cross device communication happens
    return output / normalizer

res_pmap = jax.pmap(normalized_convolution, 
                    axis_name='batch_dm',
                    in_axes=(None,0))(w, xs)
res_vmap = jax.vmap(normalized_convolution,
                    axis_name='batch_dm',
                    in_axes=(None, 0))(w, xs)
print(repr(res_pmap))
print(repr(res_vmap))

print(f"Verify the output is normalized: {sum(res_pmap[:,0])}")

#-------------------------------------------
msg = 'Repeat and explore with only pmap code snippet'
start_new_block(msg)
print('Note:\npmap doing computation on multiple pysical device \nvmap do same thing on single device\n')

# a normalization fn
def normalized_convolution(w, x):
    output = []
    for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
    output = jnp.array(output)
    print(output) #  pmap or vmap call jit implicitly
    normalizer = jax.lax.psum(output, axis_name='batch_dm') # here cross device communication happens
    return output / normalizer

res_pmap = jax.vmap(normalized_convolution, 
                    axis_name='batch_dm',
                    in_axes=(None,0))(w, xs) # broadcast

pmap_result = jax.pmap(convolve)(ws, xs) # no broadcast

print(repr(pmap_result))
print(repr(res_pmap))

print(f"Verify the output is normalized: {sum(res_pmap[:,2])}")

#-------------------------------------------
msg = 'Before training another ML model, some useful fns'
start_new_block(msg)

def sum_squared_error(x, y):
    return sum((x-y)**2)

x = jnp.arange(4, dtype = jnp.float32)
y = x + 0.1

print("An efficient way to return both grads and loss value,")
print('via jax.value_and_grad:')
loss, grads = jax.value_and_grad(sum_squared_error)(x, y)
print('loss : ', loss)
print('grads: ', grads)

#-------------------------------------------
start_new_block()
print('And sometimes the loss function needs to return intermediate results')
print('via jax.grad, by adding has_aux=True:')
def sum_squared_erro_with_aux(x, y):
    return sum((x-y)**2), x-y

grads, aux = jax.grad(sum_squared_erro_with_aux, has_aux=True)(x, y)
print('grads:', grads)
print('auxillary:', aux)

#-------------------------------------------
msg = 'Create a very simple model in parallel'
start_new_block(msg)

class Params(NamedTuple):
    weight: jnp.ndarray
    bias: jnp.ndarray

lr = 5e-3

def init_model(rng):
    weights_key, bias_key = jax.random.split(rng)
    weight = jax.random.normal(weights_key, ())
    bias = jax.random.normal(bias_key, ())
    return Params(weight, bias)

def forward(params:Params, xs:jnp.array):
    return params.weight * xs + params.bias

def loss_fn(params, x, y):
    y_hat = forward(params, x)
    return jnp.mean((y_hat-y)**2) # MSE loss

import functools
@functools.partial(jax.pmap, axis_name='batch')
def update(params, xs, ys):

    # compute the gradient on given batch (individually on each device)
    loss, grads = jax.value_and_grad(loss_fn)(params, xs, ys)

    # combine the gradient across all devices (all reduce called)
    grads = jax.lax.pmean(grads, axis_name='batch')

    # combine loss from all devices (for logging purpose only)
    loss = jax.lax.pmean(loss, axis_name='batch')

    # each device performs its own SGD update, but since we start with same params
    # and synchronise gradients, the params stsy in sync on each device
    new_params = jax.tree.map(lambda param, grad: param - grad * lr, params, grads)

    # If we were using adam of another stateful optimizer
    # we would also do something like:
    # updates, new_optimizer_state = optimizer(grads, optimizer_state)
    # and then use updates instead of grads to actually update the params
    # (and we would include the new_optimizer_state in the output, naturally)

    return new_params, loss

#------------------------------------------
msg = 'Generate true data from y = w*x + b + noise'
start_new_block(msg)

true_w, true_b = 2, -1
shape = (128, 1)
xs = np.random.normal(size=shape)
noise = 0.5*np.random.normal(size=shape)
ys = xs*true_w + true_b + noise

#plt.scatter(xs, ys)
#plt.show()

#-------------------------------------------
msg = 'Initialise params and replicate across devices'
start_new_block(msg)

params = init_model(jax.random.PRNGKey(seed=0))
n_devices = jax.local_device_count()
replicated_params = jax.tree.map(lambda x:jnp.array([x] * n_devices), params)
print('replicated_params')
print(replicated_params)

#-------------------------------------------
msg = 'Prepare the data'
start_new_block(msg)
def reshape_for_pmap(data, n_devices):
    return data.reshape(n_devices, data.shape[0]//n_devices, * data.shape[1:])

x_parallel = reshape_for_pmap(xs, n_devices)
y_parallel = reshape_for_pmap(ys, n_devices)

print(x_parallel.shape, y_parallel.shape)

#-------------------------------------------
msg = 'Start training'
start_new_block(msg)

def type_after_update(name, obj):
    print(f"after first update(), {name} is a {type(obj)}")

# training loop
num_epoches = 10001
for epoch in range(num_epoches):

    # here params and data get communicateed to devices
    replicated_params, loss = update(replicated_params, x_parallel, y_parallel)

    # Note:
    # replicated_params and loss are on device, sharded
    # x_parallel and y_parallel remain as numpy assy, on the host 

    if epoch == 0:
        type_after_update('replicated_params.weight', replicated_params.weight)
        type_after_update('loss', loss)
        type_after_update('x_parallel', x_parallel)
    if epoch %1000 == 0:
        print(loss.shape)
        print(f"Step{epoch:3d}, loss: {loss[0]:.3f}")


#-------------------------------------------
msg = 'View param on each device'
start_new_block(msg)

# Like the loss, the leaves of params have an extra leading dimension
# so we take the params from the first device

params = jax.device_get(jax.tree.map(lambda x: x[0], replicated_params))
print('replicated_params for all devices:')
print(repr(replicated_params))
print('params on device [0]:')
print(repr(params))


#-------------------------------------------
msg = 'Plot the training results'
start_new_block(msg)

plt.scatter(xs, ys, c = 'g', label = 'y_t')
plt.scatter(xs, forward(params, xs), c = 'r', label = 'y_hat')
plt.legend()
# plt.show()

#-------------------------------------------
msg = 'More to learn in the next: Advanced Autodiff\n'
msg += "1. how to freeze/unfreeze part of trained model for finetune?\n"
msg += "2. how to get per sample gradients instead of batch grad? \n"
msg += "jax.lax.stop_gradient is the primitive for this purpose.\n"
start_new_block(msg)


#-------------------------------------------
msg = 'RL Learning Example: refer to original video for RL TD loss/pseudo loss'
start_new_block(msg)

# Value fn (simple linear fn) and init parameter
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])

# An exampmle transition
s_tm1 = jnp.array([1., 2., -1.]) # status at time t-1
r_t = jnp.array(1.) # reward at time t
s_t = jnp.array([2., 1., 0.]) # state at time t

def td_loss(theta, s_tm1, r_t, s_t):
    y_tm1 = value_fn(theta, s_tm1) 
    target = r_t + value_fn(theta, s_t)
    return (jax.lax.stop_gradient(target) - y_tm1) ** 2

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

print(delta_theta)

#-------------------------------------------
msg = 'Straight through estimator in VQ-VAE (constant 1s in gradient) \n'
msg += 'https://arxiv.org/pdf/2209.04836\n'
msg += 'https://www.geeksforgeeks.org/how-does-pytorch-backprop-through-argmax/'
start_new_block(msg)
print("when a network has cerntain simple non-differentiable fn in network")

def f(x):
    return jnp.round(x) # non differentiable

def straight_through_f(x):
    return x + jax.lax.stop_gradient(f(x) - x)

x = 5.6
print('f(x)', f(x))
print('straight through f(x)', straight_through_f(x)) # same value in the forward pass

print("jax.grad(f)(x):", jax.grad(f)(x)) # non diff so just return 0
print('grad(straight_through_f)(x):', jax.grad(straight_through_f)(x)) # gradient always be 1

#-------------------------------------------
msg = 'Fetch per sample gradient, instead of batch grad'
start_new_block(msg)

# in jax
# batch the data
batched_s_sm1 = jnp.stack([s_tm1] * 2)
batched_r_t = jnp.stack([r_t] * 2)
batched_s_t = jnp.stack([s_t] * 2)

perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))
print(repr(perex_grads(theta, batched_s_sm1, batched_r_t, batched_s_t)))

#-------------------------------------------
msg = 'Jax Autodiff engine is super powerful\n'
msg += "https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html"
start_new_block(msg)

print('MAML in a few lines (model agnositic meta learning, \nmulti-task loss minimization):')

def meta_loss_fn(params, data):
    """ calculates the loss after one step os SGD """
    grads = jax.grad(loss_fn)(params, data)
    return loss_fn(params - lr*grads, data)

# meta_grads = jax.grad(meta_loss_fn)(params, data)

