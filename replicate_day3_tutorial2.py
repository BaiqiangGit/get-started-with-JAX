"""
磨刀不误砍柴工
养心性不误做事
"""
import jax
import jax.interpreters
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, pmap, vmap, random, make_jaxpr, config
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, NamedTuple
import functools

"""
Purpose of tutorial:
Build complex ML model
Parallel computation on multiple devices

Note:
The problem of state vs pure functions
Jax does not love **state**, which tends to trigger impure functions

Concepts Recap:
- pure functions
- stateful variable
"""

def start_new_section():
    print()
    print('-'*80)
    # print('     \---------------------------------------------------------/')
    # print('      \------------------- [New Section] ---------------------/')
    # print("       \-----------------------------------------------------/")
    # print("        \___________________________________________________/")
        
# --------------------------------------- 

# 1) Previous Example, with globals， 
# cached version depends on 1st call

# init global
g = 0.

def impure_uses_globals(x):
    return x+g

print("First Call:", jit(impure_uses_globals)(4,)) # cached g=0.

# update now
g = 10.
print("Second Call:", jit(impure_uses_globals)(4,))


# ---------------------------------
start_new_section()

# 2）Jax PRNG state: is not stateful 
# but sample diversity is desirable

state = jax.random.PRNGKey(seed=0)
state1, state2, state3 = jax.random.split(state, 3) # 
print(state, state1, state2)

"""
Lets now explicitly address anmd understand the problem of the state!
Why?
Well, NNs love statefullness: model params, optimizer params, batchnorm, etc
and we have seen that Jax seems to have a problem with it.
"""

# ---------------------------------
start_new_section()

# 3）impure member function in python classes
class Counter:
    """ A simple Counter"""
    def __init__(self)->None:
        self.n = 0
    def count(self)->int:
        """ Increments the counter and returns the new value"""
        self.n += 1
        return self.n
    def reset(self)->None:
        self.n = 0

counter = Counter()
for _ in range(3):
    print('expected counting', counter.count())
        
counter.reset()

# so far works like a charm

# ---------------------------------
start_new_section()

# Here comes the problem:
# when jit the member function, triggers the 
# ** Impure Function Problem ** 
# self.n is external to self.count() function

fast_count = jax.jit(counter.count)
for _ in range(3):
    print('jit fn counting:', fast_count()) # cached first self.n = 0

# use jaxpr to understand
counter.reset()

# Note: suspicious leaking behaviour
# careful about jit/make_jaxpr, it creates the same cache, e.g.
# 1) first created jit fn, then make_jaxpr will simply reuse the cache 
# 2) called make_jaxpr, then jit will simply resue the existing cache

print('>> LOOKS cache shared between jit and make_jaxpr, no recompilatioin')
counter.count() 
print("current val of counter.n: ", counter.count()) 
print("this is cache, n is still 1:", make_jaxpr(counter.count)()) 

fast_count._clear_cache()

counter.count() 
counter.count() 
print("current val of counter.n: ", counter.count()) 
print("this is cache, n is still 1:", make_jaxpr(counter.count)()) 

# Note: need to know how to workaround, to delete cached jit fn

# ---------------------------------
start_new_section()

# Solution for impure member function: 
# General idea: pass the state as fn input, and return in output

CounterState = int # implemented simply as an integer 

class CounterV2:

    def count(self, n:CounterState) -> Tuple[int, CounterState]:
        """
        you could just return n+1, but here we separate its role as 
        the output and as the counter state for didactic purposes,
        as the output may be some arbitrary fn of state in general case.
        """
        return n+1, n+1
    
    def reset(self)->CounterState:
        return 0
    
counter = CounterV2()
state = counter.reset() # notic how reset now returns state (external vs internal imp)

for _ in range(3): # works like a charm pre-jit
    value, state = counter.count(state) # looks familiar ?? split random keys ?
    print('pre-jit value:', value)

# ---------------------------------------
state = counter.reset()
fast_count = jax.jit(counter.count)
for _ in range(3):
    value, state = fast_count(state)
    print('jit value:', value)


"""
Summary: 

How to convert a stateful fn, into a stateless fn:

Class StatefulClass:
    state: State
    def stateful_method(self, *args, **kwargs) -> Output

Class StatelessClass:
    def stateless_method(state:State, *args, **kwargs) -> (Output, State)

"""

# ---------------------------------
start_new_section()

# one more step, to build fully fledged neural networks

# Enter PyTree
# Question: why gradients are a problem in the first place ?

## Pytree basics

f = lambda x,y,z,w: x**2 + y**2 + z**2 + w**2

# jax: torch.backward() is not that great, also jax
x,y,z,w = [1.]*4
dfdx, dfdy,dfdz, dfdw = grad(f, argnums=(0,1,2,3))(x, y, z, w)

print(dfdx, dfdy, dfdz, dfdw)

# we can now update the params, by
# x -= lr*dfdx
# y -= lr*dfdy
# z -= lr*dfdz
# ...
# w -= lr*dfdw

# ---------------------------------
start_new_section()

# this is too awkward: we do have a better way
# we want to, more natually, wrap our params in some data structures like dicts, etc
# In jax, this data structure is PyTree: 
# https://jax.readthedocs.io/en/latest/pytrees.html
# jax solutioin for finding gradients for arbitrarily nested data structures

pytree_example = [
    [1, '1', object()],
    [1, (2, 3), ()],
    [1, {'k1':2, 'k2':(3,4)}, 5],
    [{'a':2, 'b': (2,3) }, jnp.array([1,2,3])],
]

# Let's see hwo many leaves we have
for pytree in pytree_example:
    leaves = jax.tree.leaves(pytree) # jax's handy little fn
    print(f"{repr(pytree):<45} has {len(leaves)} leaves: {leaves}" )
    
# How do we manipulate PyTree?
list_of_lists = [
    {'a':3},
    [1,2,3], 
    [1,2], 
    [1,2,3,4]
    ]

# For single/multi arg functions use tree_map
# tree_map iterates through leaves and applies the lambda fn
print("Single arg fn:")
print(jax.tree.map(lambda x: x*2, list_of_lists))

print("Multi Arg Fn:")
another_lol = deepcopy(list_of_lists)
print(jax.tree.map(lambda x,y: x+y, list_of_lists, another_lol)) 

# structure needs to be same
another_lol.append([23])
try:
    print(jax.tree.map(lambda x,y: x+y, list_of_lists, another_lol)) 
except Exception as e:
    print(e)
    
"""
Output:
List arity mismatch: 5 != 4; list: [{'a': 3}, [1, 2, 3], [1, 2], [1, 2, 3, 4], [23]].
"""
# ---------------------------------
start_new_section()

# ---------------------------------------------
# Now we are moving to NNs, Finallly 
# multi-layer perceptron

def init_mlp_params(layer_widths):
    params = []
    # allocate weights and biases (model params)
    # Notice: we are not using jax prng here - does not matter for this simple example
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(weights = np.random.normal(size=(n_in, n_out))*np.sqrt(2/n_in),
                 biases = np.ones(shape=(n_out,)))
        )
    return params

# Instantiate a single input - single output, 3 layer deep MLP
params = init_mlp_params([1, 128, 128, 1])
# another example with PyTree
jax_params = jax.tree.map(lambda x: x.shape, params)
print('use pytree to verify shape:\n', jax_params)

# ---------------------------------
start_new_section()

def forward(params, x):
    *hidden,last = params # cool python asterisk, return unpacked list
    for layer in hidden:
        x = jnp.dot(x, layer['weights']) + layer['biases']
        x = jax.nn.relu(x)
    return jnp.dot(x, last['weights']) + last['biases']

def loss_fn(params, x, y):
    y_hat = forward(params, x)
    return jnp.mean((y_hat-y)**2) # MSE loss

@jit # we do jit at highest level, to give xla plenty of space to optimize
def update(params, x, y):

    # Note: grads is a pytree with same structure as params
    # grad is one of the many Jax fns that has builtin support for pytrees

    # grad() returns same structure to 1st argument
    grads = jax.grad(loss_fn)(params, x, y)
    
    # grads and params have same structure
    assert repr(params) == repr(grads) 

    # SGD update
    return jax.tree.map(lambda p,g: p-lr*g, params, grads)

## now prepare the input x
xs = np.random.normal(size=(128,1))
ys = jnp.sin(3*xs)

## train
lr = 1e-3
num_epoches = 1000
for _ in range(num_epoches):
    params = update(params, xs, ys) # update params (not in place)

plt.scatter(xs, ys, label = 'Y')
plt.scatter(xs, forward(params, xs), label="Y_hat")
plt.legend()
# plt.show()

# ---------------------------------
start_new_section()

# Custom PyTree
class MyContainer: # this could be a linear layer, a conv layer, or whatever module
    """ a named container """
    def __init__(self, name: str, a: int, b: int, c: int):
        self.name = name
        self.a = a
        self.b = b
        self.c = c

# create example pytree, with intention to have 8 leaves
example_pytree = [MyContainer('Alice', 1, 2, 3), MyContainer('Bob', 4, 5, 6)] 

leaves = jax.tree.leaves(example_pytree)
print(f"{repr(example_pytree):<45}\n has {len(leaves)} leaves:\n {leaves}")

"""
Output
[<__main__.MyContainer object at 0x7fe8cc11ed50>, <__main__.MyContainer object at 0x7fe8cc15fec0>] 
has 2 leaves: 
[<__main__.MyContainer object at 0x7fe8cc11ed50>, <__main__.MyContainer object at 0x7fe8cc15fec0>]
"""

# This will not work
try:
    jax.tree_map(lambda x: x+1, example_pytree)
except Exception as e:
    print(e)

"""
unsupported operand type(s) for +: 'MyContainer' and 'int'
"""

# ---------------------------------
start_new_section()
# work around, we need to define 2 functions (flatten/unflatten)

def flatten_MyContainer(container):
    """ Returns an iterable over container contents, and aux data """
    flat_content = [container.a, container.b, container.c]
    aux_data = container.name
    return flat_content, aux_data

def unflatten_MyContainer(aux_data, flat_content):
    """  Converts back to MyContainer """
    return MyContainer(aux_data, *flat_content)

# Register a custom PyTree node
jax.tree_util.register_pytree_node(MyContainer, flatten_MyContainer, unflatten_MyContainer)

# let's try again print number of leaves
leaves = jax.tree.leaves(example_pytree)
print(f"{repr(example_pytree):<45} \nhas {len(leaves)} leaves: \n{leaves}")

# let's try again apply tree map
result = jax.tree.map(lambda x: x+1, example_pytree)
print(jax.tree.leaves(result)) # works as expected now.


# ---------------------------------
start_new_section()

# Finally: a common gotcha working with PyTrees
# Mistake nodes for leaves

# make a simple zeros pytree
zeros_pytree = [jnp.zeros((2,3)), jnp.zeros((3,4))]
print(zeros_pytree)

# make another tree with ones instead of zeros
shapes = jax.tree.map(lambda x: x.shape, zeros_pytree)
print(shapes)

# this shapes in pytree, with 4 leaves
# tree map will operate on each leaf
ones_tree = jax.tree.map(jnp.ones, shapes)
print(ones_tree)

# fix: change shapes to array
shapes = jax.tree.map(lambda x: jnp.array(x.shape), zeros_pytree)
print(shapes)
ones_tree = jax.tree.map(jnp.ones, shapes)
print(ones_tree)


# ---------------------------------
start_new_section()


















































