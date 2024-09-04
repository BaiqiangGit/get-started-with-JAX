
# jax: jit, autograde, accelerate
# - numpy style syntax
# - ai accelerator agnostic

# jax syntax akin numpy
import jax
import jax.numpy as jnp
import numpy as np

# four basic transform functions
from jax import grad, jit, vmap, pmap

# jax low level api lax (lax is anagram for xl) 
from jax import lax

from jax import make_jaxpr
from jax import random
from jax import device_put
from matplotlib import pyplot as plt

# Fact 1: jax syntax is remarkably similar to numpy's
x_np = np.linspace(0,10,1000)
y_np = 2 * np.sin(x_np) * np.cos(x_np)
fig = plt.figure()
timer = fig.canvas.new_timer(interval = 20)
timer.add_callback(plt.close) # use a call back 
plt.plot(x_np, y_np)
timer.start()
plt.show()
del timer

# Fact 2: jax arrays are immutable (embrace the functinal programming paradim)
size = 10
index = 0
value = 23

# numpy: mutable array, can modify values in place
x = np.arange(size)
print(x)
x[index] = value
print(x)

# jax: immutable array
try:
    x = jnp.arange(size)
    print(x)
    x[index] = value
    print(x)
except Exception as e:
    print('[Info: error caught: ------------------]')
    print(e)
    print('set value only possible via creating another object')
    y = x.at[index].set(value)
    print(f"object x id: {id(x)} \nobject y id: {id(y)}")
    print(x, y)


# Fact 3: jax handles random numbers differently
seed = 0 
key = jax.random.key(seed) # Create a pseudo-random number generator (PRNG) key given an integer seed.
x = jax.random.normal(key, (10,))
print(type(x)) # <class 'jaxlib.xla_extension.ArrayImpl'>, no need to specify device, like in pytorch
print(x)


# Fact 4: Jax is AI accelerator agnostic, same code runs everywhere
# Data is automagically pushed to the AI accelerator, no need for '.to(device)'
size = 3000
x_jnp = jax.random.normal(key, (size, size), dtype = jnp.float32)
x_np  = np.random.normal(size = (size,size)).astype(np.float32)

from timeit import Timer 
t1 = Timer(lambda: jnp.dot(x_jnp, x_jnp.T).block_until_ready()) # 1) on GPU - fast
t2 = Timer(lambda: np.dot(x_np, x_np.T)) # 2) on CPU - slow, numpy only works with cpus
t3 = Timer(lambda: jnp.dot(x_np, x_np.T).block_until_ready()) # 3) on GPU with transfer overhead
print('---------- time profiling with timeit')
print(t1.timeit(number=10), t2.timeit(number=10), t3.timeit(number=10))

# ref - asynchronous dispatch & block_until_ready: https://jax.readthedocs.io/en/latest/async_dispatch.html
# Jax adopts asynchronous dispatch, allows Python code to “run ahead” of an accelerator device
# asynchronous dispatch is misleading us and we are not timing the execution 
# of the matrix multiplication, only the time to dispatch the work. To measure 
# the true cost of the operation we must either read the value on the host 
# (e.g., convert it to a plain old host-side numpy array), 
# or use the block_until_ready() method on a jax.Array value to wait
# for the computation that produced it to complete.

print(jax.devices())
print(jax.default_backend())

# JAX Transform Functions

# jit compiles your functions using XLA and caches them --> speedy

# simple helper visualization function
def visualize_fn(fn, l = -10, r = 10, n = 1000):
    x = np.linspace(l, r, num=n)
    y = fn(x)
    fig = plt.figure()
    timer = fig.canvas.new_timer(interval = 20)
    timer.add_callback(plt.close) # use a call back 
    plt.plot(x, y)
    timer.start()
    plt.show()
    del timer

# define a function
def selu(x, alpha = 1.67, lmbda = 1.05): # selu is an activation function
    return lmbda * jnp.where(x>0, x, alpha*jnp.exp(x) - alpha)

# compile it
selu_jit = jax.jit(selu)

# visualize it
visualize_fn(selu_jit)

data = jax.random.normal(key, (1000000, ))

print('non-jit version:')
print(Timer(lambda: selu(data).block_until_ready()).timeit(number=10))

print('jit version:')
print(Timer(lambda: selu_jit(data).block_until_ready()).timeit(number=10))

# grad: automatic back-propagation
# differentiation can be: manual, symbolic, numeric, or automatic

# first example: automatic diff
def sum_logistic(x):
    return jnp.sum(1.0/(1.0+jnp.exp(-x)))
x = jnp.arange(3.)
loss = sum_logistic # rename it to get some semantics

# by default grad calculates the derivative of a fn w.r.t 1st parameter
# here first parameter is the only one, so dose not matter

grad_loss = jax.grad(loss)
print(grad_loss(x))

# Numeric diff (to double check that autodiff works correctly)
def finite_differenes(f, x): # delta_f/delta_x
    eps  = 1e-5
    eye  = jnp.eye(len(x)) # each row in one-hot mask to isolate each element
    diff = jnp.array([(f(x+eps*v) - f(x-eps*v)) / (2*eps) for v in eye])
    return diff

print('this is the same result than jax.grad(f,x)')
print(finite_differenes(loss,x))

# Second example : automatic diff
x = 1. # example input
f = lambda x : x**2 + x + 4 # simple 2nd order polynomial fn
visualize_fn(f, l=-1, r=2, n=100)

dfdx  = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)

print(f(x), dfdx(x), d2fdx(x), d3fdx(x))
print(type(dfdx))

# Task: what if we had 2 inputs
# Note1: closer to math
# Note2: more power ful compared to .backward( (pytorch syntax))
x = 1.
y = 1.
f = lambda x, y : x**2 + x + 4 + y**2

# dx
dfdx = jax.grad(f)
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
print(f(x,y), dfdx(x,y), d2fdx(x,y), d3fdx(x,y))

# dy
dfdy = grad(f, argnums = (1,)) # 2*y
print(type(dfdy))
print(dfdy(x,y))

# JAX autodiff engine is very powerful ('advance' example)
# jacobians

from jax import jacfwd, jacrev

f = lambda x, y: x**2 + y**2 # simple paraboloid
# symbolic: 
# Jacobians are 1st order partial derivatives
# df/dx = 2x
# df/dy = 2y
# J = [df/dx, df/dy] 
# Hessians are 2nd order partial derivatives
# d2f/dxdx = 2, d2f/dydy = 2
# d2f/dxdy = 0, d2f/dydx = 0
# H = [[d2f/dxdx, d2f/dxdy],
#      [d2f/dydx, d2f/dydy]]

# ref: https://bobondemon.github.io/2024/02/07/%E9%AB%98%E6%95%88%E7%8E%87%E8%A8%88%E7%AE%97-Jacobian-Hessian-VJP-JVP-HVP/

def hessian(f):
    return jit(jacfwd(jacrev(f, argnums=(0,1)), argnums =(0,1)))

print(f"Jacobian = {jacrev(f, argnums = (0,1))(1.,1.)}")

# Edge Case: abs(x)

f = lambda x: abs(x)
visualize_fn(f)
print(f(-1), f(1))
dfdx = jax.grad(f)
print(dfdx(1.), dfdx(0.)) # convention dfdx(0.) = 1

# jax.vmap()
# write your functions as if you were dealing with a single datapoint
W = jax.random.normal(key, (150, 100)) # linear layer weights
batched_x = jax.random.normal(key, (10, 100)) # a batch of 10 flattened images

@jit
def apply_matrix(W, x):
    return jnp.dot(W, x)

@jit
def naively_batched_apply_matrix(W, batched_x):
    return jnp.stack([apply_matrix(W, x) for x in batched_x])

print('Naively batched:')
t = Timer(lambda:naively_batched_apply_matrix(W, batched_x).block_until_ready()).timeit(number=10)
print(t)

print('Jit Naively batched:')
t = Timer(lambda:naively_batched_apply_matrix(W, batched_x).block_until_ready()).timeit(number=10)
print(t)


@jit
def batched_apply_matrix(W, batched_x):
    return jnp.dot(batched_x, W.T) # [10,100] @ [100, 150] => [10, 150]
print("Manually Batched:")
t = Timer(lambda:batched_apply_matrix(W, batched_x).block_until_ready()).timeit(number=10)
print(t)


@jit
def vmap_batched_apply_matrix(W, batched_x):
    return jax.vmap(apply_matrix, in_axes=(None, 0))(W, batched_x)
print("Auto vectorized with vmap:")
t = Timer(lambda:vmap_batched_apply_matrix(W, batched_x).block_until_ready()).timeit(number=10)
print(t)

# Jax layers
# high to low api: Numpy -> lax -> xla 
# lax is more stricter and more powerful

x = jnp.array([1,2,1])
y = jnp.ones(10)

print(jnp.add(1,1.0)) # jax.numpy API implicitly promotes mixed types
try:
    print(lax.add(1,1.0)) # jax.lax api requires explicit type promotion (trigger error)
except Exception as e:
    print(e)

# Numpy API
result1 = jnp.convolve(x, y)
# lax API : more powerful/customization flexibility
result2 = lax.conv_general_dilated(
        x.reshape(1,1,3).astype(float), 
        y.reshape(1,1,10),
        window_strides = (1,),
        padding = [(len(y)-1, len(y)-1)],
        )
print(result1)
print(result2[0][0])
assert np.allclose(result1, result2[0][0], atol = 1e-6)




