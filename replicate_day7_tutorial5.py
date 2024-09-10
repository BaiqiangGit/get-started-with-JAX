"""
凝神闭关，用心一处
"""

"""
10/09/2024
Flax Linen Module Tutorial
https://colab.research.google.com/github/google/flax/blob/main/docs/linen_intro.ipynb#scrollTo=gP1LlMikgrig
"""

import functools
from typing import Any, Callable, Sequence, Optional
import jax
from jax import lax, random, numpy as jnp
import flax, flax.linen as nn

# block fn to make cmd outputs more structured
class new_block:
    cnt: int = 0
    def msg(self, msg=''):
        print("\n>>>>--------------------------------------------------")
        if msg: 
            self.cnt += 1
            print(f"{self.cnt}) " + msg, end='\n')
        print('----------')

start_new_block = new_block()


# -------------------------------------------------
start_new_block.msg("Invoke Module: instantiate a Dense layer")

# Instantiate a Dense layer: features is number of neurons
model = nn.Dense(features=3)


# -------------------------------------------------
msg = "Init the model variables: pass PRNG key, and input to __call__()"
start_new_block.msg(msg)

key_x, key_m = jax.random.split(jax.random.PRNGKey(seed=2), num=2)
x = jax.random.normal(key_x, (4, 4)) # fake input
init_variables = model.init(key_m, x)

# -------------------------------------------------
msg = "Check init_variabes device: \nNote the complicated way of handling <class 'dict_values'>"
start_new_block.msg(msg)

##
device = next(iter(next(iter(init_variables.values())).values())).device
print("init_variables:")
print(type(init_variables))
print(type(init_variables.values()))
print(device)
print(init_variables)

# -------------------------------------------------
msg = "Inspect init_varibles"
start_new_block.msg(msg)

if 'params' in init_variables:
    for param_name, param_value in init_variables['params'].items():
        print(f"{param_name}:")
        print(f"  Shape: {param_value.shape}")
        print(f"  Device: {param_value.device}")
        print(f"  Data type: {param_value.dtype}")
        print(f"  First few values: {param_value.flatten()[:5]}")  # Show first 5 values
        print()
else:
    print("No 'params' found in init_variables.")


# -------------------------------------------------
msg = "Inspect model attributes"
start_new_block.msg(msg)

print("\nModel Attributes:")
print(f"Name: {model.__class__.__name__}")
print(f"Features: {model.features}")
print(f"Use bias: {model.use_bias}")
print(f"Dtype: {model.dtype}")
print(f"Param dtype: {model.param_dtype}")
print(f"Precision: {model.precision}")
print(f"Kernel init: {model.kernel_init}")
print(f"Bias init: {model.bias_init}")
print(f"Model param: {model.param}")

# Get information about expected bias parameter
"""
Remember that Flax models are designed to be stateless, 
so the actual parameter values (including bias) are not stored in the model object itself. 
They are created and managed separately when you initialize and use the model.
"""

# -------------------------------------------------
msg = "Check Available GPUs and CPUs"
start_new_block.msg(msg)
# Check if init_variables contents are on GPU or CPU
print(f"Available Jax cpu device: {jax.devices('cpu')}")
print(f"Available Jax gpu device: {jax.devices('gpu')}")


# -------------------------------------------------
msg = "Move init_variables to CPU/GPU device"
start_new_block.msg(msg)

devices = jax.devices('gpu')
if devices:
    device = devices[0]  # Get the first CPU device
    init_variables = jax.device_put(init_variables, device)
    print(f"Moved init_variables to {device}")
else:
    print("No CPU devices available. This is unexpected.")

# -------------------------------------------------
msg = "Check if init_variables on CPU or GPU device"
start_new_block.msg(msg)

def get_device(x):
    if isinstance(x, dict):
        return get_device(next(iter(x.values())))
    elif hasattr(x, 'device'):
        return x.device
    else:
        return None

device = get_device(init_variables)
print(f"init_variables contents are on: {device}")
if device: 
    print(f"Device type: {device.platform.upper()}")

# -------------------------------------------------
msg = "Call apply method of model, implicitly calling __call__()"
msg += '\n- can replace default __call__ method with custom fn, via method = xx, for init and apply'
msg += '\n- module can also accept RNGS and MUTABLEKINDS, to support custom rng key (dropout), or non-traibable params (batchnorm) '
start_new_block.msg(msg)

y = model.apply(init_variables, x)
print("Output of: \ny = model.apply(init_variables, x)\n", y)

# -------------------------------------------------
msg = "Composing Submodules: declare in separate setup() method"
start_new_block.msg(msg)

class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]
    
    def __call__(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

"""
jax.random.PRNGKey(seed, *, impl=None)
This function produces old-style legacy PRNG keys, which are arrays of dtype uint32. 
For more, see the note in the PRNG keys section. When possible, jax.random.key() is 
recommended for use instead.
"""

key1, key2 = jax.random.split(jax.random.key(seed = 0), num=2)
x = jax.random.uniform(key1, (4,4))

model = ExplicitMLP(features = [3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

# jax.tree_util.tree_map
# Alias of jax.tree.map()
# https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.tree_map.html

print('initialized parameter shapes')
print(jax.tree.map(jnp.shape, init_variables))  
print("output:\n", y)


# -------------------------------------------------
msg = "Composing Submodules: Compact form, declares submodules inline (not init)"
start_new_block.msg(msg)

class CompactMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layer_{i}')(x) # init in place, default name Dense_0, etc
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x

key_x, key_m = jax.random.split(jax.random.key(seed=0), num=2)
model = CompactMLP(features = [3,4,5])
x = jax.random.uniform(key_x,(4,4))
init_variables = model.init(key_m, x)
y = model.apply(init_variables, x)

print('init_variables shape:')
print(jax.tree.map(jnp.shape, init_variables))
print('y:\n', y)
            
# -------------------------------------------------
msg = "Declare and use Variables: add trainable to self.param inline, use:"
msg += '\nself.param(parameter_name, parameter_init_fn, *init_args, **init_kwargs)'
start_new_block.msg(msg)

# code a dense layer
class SimpleDense(nn.Module):
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros # or .zero_init()

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel',
                            self.kernel_init, # RNG passed implicitly
                            (x.shape[-1], self.features)) # shape info 
        bias = self.param('bias',
                            self.bias_init,
                            (self.features, ) )
        y = jax.lax.dot_general(x, kernel, (((x.ndim-1,), (0,)),  ((), ())),)
        y = y + bias
        return y

key_x, key_m = jax.random.split(jax.random.key(seed=0), num=2)
x = jax.random.uniform(key_x, (4, 4))
model = SimpleDense(features=3)
init_variables = model.init(key_m, x)
y = model.apply(init_variables, x)
print('initialized variables:')
print(init_variables)

print('output:')
print(y)

# -------------------------------------------------
msg = "Declare and use Variables: declare non-trainable variables, via setup():"
start_new_block.msg(msg)

class ExplicitDense(nn.Module):
    features_in: int # <<--- here need explicit input shape, no auto shape inference during init
    features: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros_init()

    def setup(self):
        self.kernel = self.param('kernel',
                                  self.kernel_init,
                                  (self.features_in, self.features))
        self.bias = self.param('bias',
                                self.bias_init,
                                (self.features, ))
    def __call__(self, x):
        y = jax.lax.dot_general(x, self.kernel,
                                (((x.ndim-1,), (0,)), ((), ())),)
        y = y + self.bias
        return y

key_x, key_m = jax.random.split(jax.random.key(seed=0), num=2)
x = jax.random.uniform(key_x, (4,4))
model = ExplicitDense(features_in = 4, features=3)
init_variables = model.init(key_m, x)
y = model.apply(init_variables, x)

print('Init_variables:')
print(init_variables)
print('Output:')
print(y)

# -------------------------------------------------
msg = "Declare and use Variables: Declare Generally Mutable Variables, use:"
msg += '\nself.variable(variable_kind, variable_name, variable_init_fn, *init_args, **init_kwargs)'
msg += '\n the name of the nested-dict collection that this will be stored in inside the top Modules variables'
msg += 'e.g. batch_states for batchnorm, cache for autoregressive cache, param for trainable parameters'
start_new_block.msg(msg)

class Counter(nn.Module):
    @nn.compact
    def __call__(self):
        # easy pattern to detect if we are initializing
        is_initialized = self.has_variable('counter', 'count')
        counter = self.variable('counter', 'count', lambda: jnp.zeros((), dtype = jnp.int32))
        if is_initialized:
            counter.value += 1
        return counter.value

key1 = jax.random.key(seed=0)
model = Counter()
init_variables = model.init(key1)
print('init_variables:')
print(init_variables)

# -------------------------------------------------
msg = "Check mutated variables"
start_new_block.msg(msg)

y, mutated_variables = model.apply(init_variables, mutable = ['counter'])
print('mutated_variables:', mutated_variables)
print('Output:', y)

# -------------------------------------------------
msg = "Example: model with differentiable parameters, stochastic layers, and mutable variables"
start_new_block.msg(msg)

class DDBBlock(nn.Module):
    """ Differentiable, Dropout, BatchNorm """
    features: int
    training: bool
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.Dropout(rate=0.5)(x, deterministic = not self.training)
        x = nn.BatchNorm(use_running_average = not self.training)(x)
        x = nn.relu(x)
        return x

key1, key2, key3, key4 = jax.random.split(jax.random.key(seed=0), 4)
x = jax.random.uniform(key1, (3,4,4))
model = DDBBlock(features=3, training=True)
init_variables = model.init({'params': key2, 'dropout': key3}, x) # specify key for different variable kind 
_, init_params = flax.core.pop(init_variables, 'params')

print('init_variables:')
print(init_variables)
print()

print('init_params')
print(init_params)


# -------------------------------------------------
msg = "When calling apply with mutable kinds, return a pair of (output, mutated_variables)"
start_new_block.msg(msg)

y, mutated_variables = model.apply(init_variables, x, 
                        rngs = {'dropout': key4}, 
                        mutable = ['batch_stats'])

# now re-assemble the full updated variables
updated_variables = dict(params=init_params, **mutated_variables)

print('updated variables:')
print(updated_variables)
print()

print('intialized variable shapes:')
print(jax.tree.map(jnp.shape, init_variables))
print()

print('Output:')
print(y)

# -------------------------------------------------
msg = "Let's run evaluation mode"
start_new_block.msg(msg)
eval_model = DDBBlock(features=3, training=False)
y = eval_model.apply(updated_variables, x) # nothing mutable/no-variable-change, only return value
print('Output:')
print(y)

# -------------------------------------------------
msg = "JAX transformations inside modules, e.g jax.jit(nn.Dense(features = xxx))"
msg += '\nor jax.jit(model.apply)'
start_new_block.msg(msg)
jit_eval = jax.jit(eval_model.apply)
y = jit_eval(updated_variables, x) # nothing mutable/no-variable-change, only return value
print('Output:')
print(y)


# -------------------------------------------------
msg = "For memory-expensive computations, we can remat our method to recompute a Module's output during a backwards pass."
msg += "\nKnown Gotcha: at the moment, the decorator changes the RNG stream slightly, so comparing remat'd and undecorated initializations will look different."
start_new_block.msg(msg)


class RematMLP(nn.Module):
  features: Sequence[int]
  # For all transforms, we can annotate a method, or wrap an existing
  # Module class. Here we annotate the method.
  @nn.remat
  @nn.compact
  def __call__(self, inputs):
    x = inputs
    for i, feat in enumerate(self.features):
      x = nn.Dense(feat, name=f'layers_{i}')(x)
      if i != len(self.features) - 1:
        x = nn.relu(x)
    return x

key1, key2 = random.split(random.key(3), 2)
x = random.uniform(key1, (4,4))

model = RematMLP(features=[3,4,5])
init_variables = model.init(key2, x)
y = model.apply(init_variables, x)

print('initialized parameter shapes:\n', jax.tree.map(jnp.shape, init_variables))
print('output:\n', y)

# -------------------------------------------------
msg = "Vmap Modules inside, via usual jax vmap arags: in_axes, output_axes, axis_size"
msg += "\nObtain Batched Multihead Attention, from single head unbatched attention, via vmap"
start_new_block.msg(msg)

"""
You can now vmap Modules inside. The transform has a lot of arguments, they have the usual jax vmap args:

in_axes - an integer or None for each input argument
out_axes - an integer or None for each output argument
axis_size - the axis size if you need to give it explicitly
In addition, we provide for each kind of variable it's axis rules:

variable_in_axes - a dict from kinds to a single integer or None specifying the input axes to map
variable_out_axes - a dict from kinds to a single integer or None specifying the output axes to map
split_rngs - a dict from RNG-kinds to a bool, specifying whether to split the rng along the axis.
"""

# -------------------------------------------------
msg = "Implement Unbatched Single Head Attention"
start_new_block.msg(msg)

class SingleHeadAttention(nn.Module):
    attn_dropout_rate: float = 0.1
    train: bool = False

    @nn.compact
    def __call__(self, query, key, value, bias=None, dtype= jnp.float32):
        assert key.ndim == query.ndim
        assert key.ndim == value.ndim
        n = key.ndim
        attn_weights = lax.dog_general(query, key, (((n-1,), (n-1,)),  ((),())))
        if bias is not None: attn_weights += bias
        norm_dims = tuple(range(attn_weights.ndim//2,  attn_weights.ndim))
        attn_weights = nn.softmax(attn_weights, axis=norm_dims)
        attn_weights = nn.Dropout(self.attn_dropout_rate)(attn_weights, deterministic = not self.train)
        attn_weights = attn_weights.astype(dtype)

        contact_dims(tuple(range(n-1, attn_weights.ndim)), tupe(range(0, n-1)))
        y = lax.dot_general(attn_weights, value, (contract_dims, ((), ())))
        return y

