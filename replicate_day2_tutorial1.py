import jax
import numpy as np
from timeit import Timer

from jax import random
from jax import make_jaxpr # jax expression
from jax import numpy as jnp

from jax import jacfwd
from jax import jacrev
from jax import jit
from jax import grad
from jax import vmap
from jax import pmap
from jax import lax 

"""
pmap: The purpose of pmap() is to express single-program multiple-data (SPMD) programs. 
      Applying pmap() to a function will compile the function with XLA (similarly to jit()), 
      then execute it in parallel on XLA devices, such as multiple GPUs or multiple TPU cores. 
      Semantically it is comparable to vmap() because both transformations map a function over array axes, 
      but where
      vmap() vectorizes functions by pushing the mapped axis down into primitive operations, 
      pmap() instead replicates the function and executes each replica on its own XLA device in parallel.
      The mapped axis size must be less than or equal to the number of local XLA devices available
      , as returned by jax.local_device_count()
      https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html

vmap: Vectorizing map. Creates a function which maps fun over argument axes.
      https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html#jax.vmap1


PyTree: JAX arrays contained in nested tuples, lists, dictionaries. JAX functional conventionally store data in pytrees

"""

""""
Lax: jax.lax is a library of primitives operations that underpins libraries such as jax.numpy. Transformation rules, such as JVP and batching rules, are typically defined as transformations on jax.lax primitives.
     Many of the primitives are thin wrappers around equivalent XLA operations, described by the XLA operation semantics documentation. In a few cases JAX diverges from XLA, usually to ensure that the set of operations is closed under the operation of JVP and transpose rules.
     Where possible, prefer to use libraries such as jax.numpy instead of using jax.lax directly. The jax.numpy API follows NumPy, and is therefore more stable and less likely to change than the jax.lax API.
     https://jax.readthedocs.io/en/latest/jax.lax.html

Flax: A neural network library and ecosystem for JAX designed for flexibility
      https://github.com/google/flax

Haiku: (already in maintainance mode, ignore this) 
       Haiku is a library built on top of JAX designed to provide simple, composable abstractions for machine learning research.
       https://dm-haiku.readthedocs.io/en/latest/
"""
#----------------------------------

# timer
def benchmark(fn, x, number = 10):
    t1 = Timer(lambda: fn(*x).block_until_ready()).timeit(number=number)
    print(f"[Benchmark] Fn {fn.__name__} takes {t1} to run {number} times")
    print()
    return t1

# exception handling
def try_fn(fn, x):
    print(f"[Try Fn] {fn.__name__} --------------- ")
    try:
        r = fn(*x)
        print(f"Run Fn {fn.__name__}: Successful\n")
    except Exception as e:
        print(f"Run Fn {fn.__name__}: Exception Caught with error message:")
        print(e)
    print()

    return    

#----------------------------------

# how does jit actually work
def norm(x):
    x = x - x.mean()
    return x/x.std(0)
# jit version
norm_compiled = jit(norm)

# prepare input x
key = jax.random.PRNGKey(seed=0)
size = (10000, 100)
x = jax.random.normal(key, size, dtype=jnp.float32)

# verify: should be close
assert np.allclose(norm(x), norm_compiled(x), atol=1e5)

# benchmark
benchmark(norm, (x,), 1000)
benchmark(norm_compiled, (x,), 1000)
print(type(norm))
print(type(norm_compiled))

#----------------------------------

# Jit: example of failure: array shapes must be static
def get_negatives(x):
    # in runtime, depends on the actual entires of x, 
    # the return will change in shape, **not tolerated by jit**
    return x[x<0] 

x = jax.random.normal(key, (10, ), dtype=jnp.float32)

# try w/o jit
try_fn(get_negatives, (x, ))
try_fn(jax.jit(get_negatives), (x, )) # error caught, x[x>0] varies in shape, when varying x


#----------------------------------


# so how does it work in the background ?
# tracing different levels of abstraction
# Note: anytime we get the same shapes and types we just call the compiled fun

@jit
def f(x, y):
    print("Running f():")
    print(f"x = {x}")
    print(f"y = {y}")
    result = jnp.dot(x+1, y+1)
    print(f"result = {result}")
    return result

x = np.random.randn(3, 4)
y = np.random.randn(4)
print("First Call:")   
# will see the internal prints
benchmark(f, (x, y))

x2 = np.random.randn(3,4)
y2 = np.random.randn(4)
print("Second Call:") 
# side effect: won't see any internal prints
benchmark(f, (x2, y2))

x3 = np.random.randn(3,5)
y3 = np.random.randn(5)
print("Third Call:") 
# will see internal prints as compilation not done 
# for  this shapes/types 
print(f(x3, y3)) 

x4 = np.random.randn(3,5)
y4 = np.random.randn(5)
print("Fourth Call:") 
# will see internal prints as compilation not done 
# for  this shapes/types 
print(f(x4, y4)) 

#----------------------------------

# Same function as above just without print
def f(x,y):
    return jnp.dot(x+1, y+1)

# background workflow
# make_jaxpr: is jax expression, 
# a flow model that jit creates in the background
# which does its tracing procedure
# the resulted flow can be compiled by xla

print(make_jaxpr(f)(x, y))
"""
{ lambda ; a:f32[3,4] b:f32[4]. let
    c:f32[3,4] = add a 1.0
    d:f32[4] = add b 1.0
    e:f32[3] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] c d
  in (e,) }
"""


#----------------------------------

# 2nd example of failure

# # remember: tracer cares about shapes and types
# the first time we call jit function, it is going to input abstract shape and datatype
# no info about the value !! -- that is why it will fail here:
@jit
def f(x, neg): 
    return -x if neg else x # return depends on the value - not tolerated

try_fn(f, (1, True)) 

# workaround: the 'static' argument
# use functools partial, to set some arguments as constant
# and return a new version of function 
# https://docs.python.org/3/library/functools.html#functools.partial

from functools import partial

@partial(jax.jit, static_argnums=(1, )) # first argument
def f(x, neg):
    print(x, neg)
    return -x if neg else x

print('------ static args')
print(f(1, True))
print(f(2, True))
print(f(3, False))
print(f(23, False))

# print the jax expressions

def f(x, neg):
    print(x, neg)
    return -x if neg else x

print('\nPrint jit expression 1 - no abstract shape/type traced:')
make_jaxpr_partial = partial(make_jaxpr,  static_argnums=(0, 1))
print(make_jaxpr_partial(f)(1, False))
"""
1 True
{ lambda ; . let  in (-1,) }
"""

print('\nPrint jit expression 2 - abstract x is traced:')
make_jaxpr_partial = partial(make_jaxpr,  static_argnums=(1, ))
print(make_jaxpr_partial(f)(1, True))
"""
Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/0)> True
{ lambda ; a:i32[]. let b:i32[] = neg a in (b,) }
"""

#----------------------------------
# 3rd example of failure

print('\n# 3rd example of failure')
@jax.jit
def f(x):
    print(x)
    print(x.shape)
    print(jnp.array(x.shape).prod())
    return x.reshape(jnp.array(x.shape).prod()) # error: reshape expects a concrete value

x = jnp.ones((2,3))

try_fn(f, (x,))

# Work around: using numpy instead of jax.numpy
# Why? probably jit is nested compilation
# and internal jit compilables needs constant shape/type
# Task: add some print statements !
print('fix 3rd failure')
@jax.jit
def f(x):
    return x.reshape(np.prod(x.shape))
f(x)
print(f._cache_size())

#----------------------------------

# Gotcha #1: pure functions
# jax is designed to work only on pure functions

# Pure functions ? informal defintion here:
# 1. all input data is passed through the function arguments, 
#    all the results are output through the function results
# 2. a pure function will always return the same result, ]
#    if invoked with the same inputs

#----------------------------------

# Example 1
def impure_print_side_effect(x):
    print('Executing function') # violating #1, via print than return
    return x
# the side effect appear during the first run
print("First Call:", jax.jit(impure_print_side_effect)(4.))

# subsequent runs with parameters of some type and shape 
# may not show the side-effect
# this is because JAX now invokes a cached compiled version 
# of the function for the underlying type and shape 
print("Second Call:", jax.jit(impure_print_side_effect)(5.))

# JAX re-runs the python function when the type/shape of argument changes
print("Third Call:", jax.jit(impure_print_side_effect)(5.))

#----------------------------------

# Example 2: 
# it is bad and wrong idea to use global variables
# jit will cache the global variable, and reuse it, even it's updated already
g = 0.
def impure_uses_globals(x):
    return x + g # violating both #1 and #2

# JAX captures the value of global parameter during the first run
print("First Call:", jax.jit(impure_uses_globals)(4.))

# Let's updated the global
g = 10.

# Subsequent runs may silently use the cached value of the globals
print("Second Call:", jax.jit(impure_uses_globals)(4.))

# JAX re-runs the python function when the type/shape of the arguments changes
# This will end up reading the latest value of the global

# let's change type:

print("Third Call:", jax.jit(impure_uses_globals)(jnp.array([4,])))

""" Outputs: 
First Call: 4.0
Second Call: 4.0
Third Call: [14.]
"""

#----------------------------------
# Example 3: very important! Haiku/Flax are basically built upon this idea

def pure_uses_internal_state(x):
    state = dict(even=0, odd=0)
    for i in range(10): # not accessing any global variables
        state['even' if i%2 == 0 else 'odd'] += x
    return state['even'] + state['odd']

# this compiles without error
print(jax.jit(pure_uses_internal_state)(5.)) 

#----------------------------------
# Example 4: iterators are a no no !
# lax.fori_loop: smarter way of for loops, which works with xla
# https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html
# similarly for lax.scan, lax.cond, etc

# this works
array = jnp.arange(10)
print(jax.lax.fori_loop(lower=0, upper=10,body_fun= lambda i,x : x+array[i], init_val=0)) # expects result 45

# this does not work: iterators are stateful (value changes), violating purity constraint
iterator = iter(jnp.arange(10)) # only cache the first run value: 0
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # expected result 0

"""
output:
45
0
"""

#----------------------------------
# Gotcha #2: In-Place Updates
jax_array = jnp.zeros((3,3), dtype=jnp.float32)
updated_array = jax_array.at[1,:].set(1.0)

print('original array unchanged:\n', jax_array)
print(id(jax_array))
print('updated array:\n', updated_array)
print(id(updated_array))

# Note: If input x is not reused, the xla compiler will optimize 
# the array update to occur in-place

# The expressiveness of numpy is stil there!

jax_array = jnp.ones((5,6))
new_jax_array = jax_array.at[::2,3:].add(2) # very easy for matrix manipulations
print("original array:\n", jax_array)
print("new array post-additions\n", new_jax_array)


#----------------------------------
# Gotcha #3: Out-of_Bounds Indexing (Very Confusing)
# As JAX is accelerator agnostic, Jax had to make a non-error behaviour for out of bounds indexing
# similar to how invidi fp arighmetic resutls in NaNs and not an exception

# Numpy behavior
try:
    np.arange(10)[11]
except Exception as e:
    print(f'Exception {e}')

"""
Output:
Exception index 11 is out of bounds for axis 0 with size 10
"""

# JAX behavior: jax tries to abstract the accelerator info from you, as consequence:
# 1) updates at out-of-bounds indices are skipped
# 2) retrievals results in oob index being clamped to exceeded bound (allow neg index)
# in generaal there are currently some bugs so just consider the behavior undefined!
print(jnp.arange(10).at[11].add(23)) # example of 1), update with index oob
print(jnp.arange(10)[-11])
print(jnp.arange(10)[11])

#----------------------------------
# Gotcha #4: Non-array inputs, 
# forced to have ndarray or scalar as input
# not a bug: added by design, for performance

# Numpy behaviour
print(np.sum([1,2,3]))
# Jax behaviour
try:
    jnp.sum([1,2,3])
except TypeError as e:
    print(f"TypeError: {e}")

"""
Output: TypeError: sum requires ndarray or scalar arguments, got <class 'list'> at position 0.
"""

# why? view jaxpr to the rescue! -> inefficient optimization
def permissive_sum(x):
    return jnp.sum(jnp.array(x))
x = list(np.arange(10))
print(make_jaxpr(permissive_sum)(x))

"""
{ lambda ; a:i32[] b:i32[] c:i32[] d:i32[] e:i32[] f:i32[] g:i32[] h:i32[] i:i32[]
    j:i32[]. let
    k:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] a
    l:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] b
    m:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] c
    n:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] d
    o:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] e
    p:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] f
    q:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] g
    r:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] h
    s:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] i
    t:i32[1] = broadcast_in_dim[broadcast_dimensions=() shape=(1,)] j
    u:i32[10] = concatenate[dimension=0] k l m n o p q r s t
    v:i32[] = reduce_sum[axes=(0,)] u
  in (v,) }
"""
print(jnp.array([1,2,3]))
print('Benchmark:')
benchmark(jax.jit(permissive_sum), (jnp.arange(1000), ))


#----------------------------------
# Gotcha #5: Random Numbers 
# - Mersenne Twister PRNG is know to have a number of problems
# - Numpy PRNG is stateful (violating pure functions rule)
# - PRNG internal state advances, each time you sample, you consume its entropy
# - https://numpy.org/doc/stable/reference/random/generated/numpy.random.get_state.html

print('>>> Gotcha #5: Random Numbers')
print('numpy rng behaviour')

# Let's sample calling the same function twice
print(np.random.random())
print(np.random.random())

print('rng state')
np.random.seed(seed=0)
rng_state = np.random.get_state()
print(rng_state[2:])

print('rng state changes with sampling')
_ = np.random.uniform()
rng_state = np.random.get_state()
print(rng_state[2:])

_ = np.random.uniform()
rng_state = np.random.get_state()
print(rng_state[2:])


"""
Output:
0.7324710224089364
0.880221882299888
(624, 0, 0.0)
(2, 0, 0.0) 
(4, 0, 0.0) # prng state changes, internally advanced
"""

# Jax's random functiosn can't modify PRNG state
key = random.PRNGKey(seed=0)
print(key) # key defines the state (2 unsigned int32s)

# let's again sample calling the same function twice
print('jax random prng behaviour')
print(jax.random.normal(key, shape=(1,)))
print(key) # verify that the key is not changed
print(jax.random.normal(key, shape=(1,))) # sample again, but always same result ...
print(key)

# Solution? -> split every time you need a pseudo random number
print("old key", key) 
key, subkey = jax.random.split(key)
normal_prn = jax.random.normal(subkey, shape=(1,))
print(f"   \----- split --- > new key {key}")
print(f"     \---> new subkey {subkey} ---> normal {normal_prn}")

# Note 1: you can also split into more subkeys and not just 1
# Note 2: key, subkey no difference, it is only a convention
# Note 3: can divide subkey again

# why this design ?
# 1) reproducible 
# 2) parallelizable  
# 3) vectorizable

np.random.seed(seed=0)
def bar(): return np.random.uniform()
def baz(): return np.random.uniform()
def foo(): return bar() + 2*baz()

# not reproducible result, 1) is violated
print(foo())
print(foo())

"""
Output:
1.9791922366721637
1.6925297420654375
"""
# What if we want to parallelize this code ?
# numpy assumes single thread environment
# and python gurantees left to right execution
# numpy assume too much, 2) is violated

# numpy behaviour
np.random.seed(seed=0)
print('numpy individually:', np.stack([np.random.uniform() for _ in range(3)]))
np.random.seed(seed=0)
print('numpy all at once:', np.random.uniform(size=3))

# Jax behaviour
key = jax.random.PRNGKey(seed=0)
subkeys = jax.random.split(key, 3)
sequence = np.stack([jax.random.normal(subkey) for subkey in subkeys])
print('jax individually:', sequence)
key = jax.random.PRNGKey(seed=0)
print('jax all at once:', jax.random.normal(key, shape=(3,)))

# generating individually and all-at-once return same results
# with numpy, but different results in jax, numpy violates 3)
"""
Output:
numpy individually: [0.5488135  0.71518937 0.60276338]
numpy all at once: [0.5488135  0.71518937 0.60276338]
jax individually: [1.1188384 0.5781488 0.8535516]
jax all at once: [ 1.8160863  -0.48262316  0.33988908]
"""

# Gotcha #6: control flow
import matplotlib.pyplot as plt
# python control flow + grad() -> everything is ok
def f(x):
    if x<3:
        return 3. * x ** 2
    else:
        return -4 * x
x = np.linspace(-10,10,1000)
y = [f(e) for e in x]
plt.plot(x, y)
plt.close()

print('python control flow + grad() works w.o issue:')
print(jax.grad(f)(2.)) # ok
print(jax.grad(f)(4.)) # ok


# Python control flow + jit() -> problems in pradise

# the tradeoff is that with higher levels of abstraction, we gain a more general view
# of the python code, and thus save on re-compilations
# but we require more constraints on the python code, to complete the trace

# Example 1: conditioning on value, with static argnums
f_jit = jax.jit(f, static_argnums=(0,))
x = 2.
print(make_jaxpr(f, static_argnums=(0,))(x))
print(make_jaxpr(f_jit, static_argnums=(0,))(x))
print(f_jit(x))

# Example 2:  range depends on value again
def f(x,n):
    y = 0
    for i in range(n):  
        y = y + x[i]
    return y

f_jit = jax.jit(f, static_argnums=(1,))
x = (jnp.array([2.,3., 4.]), 15)

print(f_jit(*x))
print(jax.make_jaxpr(f_jit, static_argnums=(1,))(*x))

# Note: there is a catch - static args should not change too often
# otherwise it will trigger recompilation, again and again
# the overhead may be detrimental to your application

# Using lower level api is better (at the cost of being less readable)
def f_fori(x, n):
    body_fun = lambda i,val: val+x[i]
    return lax.fori_loop(0,n,body_fun,0)

f_fori_jit = jax.jit(f_fori)
print(make_jaxpr(f_fori_jit)(*x))
print(f_fori_jit(*x))

# Example 3: 
# You can condition on the dimension of the input data
# This is not problematic (it will only cache a single branch)
def log2_if_rank_2(x):
    if x.ndim == 2:
        ln_x = jnp.log(x)
        ln_2 = jnp.log(2.0)
        return ln_x/ln_2
    else:
        return x
print(make_jaxpr(log2_if_rank_2)(jnp.array([1,2,3])))

# Gotcha 7: NaNs handling

# the default non error behavior will only simply return a NaN
print(jnp.divide(0., 0.))
# if you want to debug where the NaNs are coming from, there are multpipe ways:
# one example
from jax import config
config.update("jax_debug_nans", True)
try:
    print(jnp.divide(0., 0.))
except Exception as e:
    print(e)
# Note: by default, jax enforces single precision
# work around is simpl like in numpy
config.update('jax_enable_x64', True)
x = jax.random.uniform(key, (1000, ), dtype = jnp.float64)
print(x.dtype)


