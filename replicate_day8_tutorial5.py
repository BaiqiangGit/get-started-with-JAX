"""
要专注，要专注，要专注
https://flax.readthedocs.io/en/latest/quick_start.html

"""

"""
Continue Coding Batched MultiHead attention
"""

import os
import jax, numpy as jnp
import numpy as np
import flax.linen as nn
import functools
from typing import Any, Callable, Sequence, Optional
import matplotlib.pyplot as plt


# print function
os.system('clear')
class start_new_block(nn.Module):
    cnt:int= 0
    def msg(self, msg = ''):
        print(">>>>---------------------------------------------------")
        if msg: print(f'{self.cnt})', msg)
        print('-------')

new_block = start_new_block()


# start coding
# -------------------------------------------------
msg = 'Coding Single Head Attention'
new_block.msg(msg)

class RawDotProductAttention(nn.Module):

    attn_dropout_rate:float=0.1
    train: bool = False
    
    @nn.compact
    def __call__(self, query, key, value, bias= None, dtype = jnp.float32):
        assert key.ndim == query.ndim == value.ndim
        
        n = key.ndim

        # a tuple of tuples of sequences of ints of the form 
        # ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims))
        lhs_contracting_dims = (n-1,)
        rhs_contracting_dims = (n-1,)
        lhs_batch_dims = ()
        rhs_batch_dims = ()
        dimension_numbers = ((lhs_contracting_dims, rhs_contracting_dims), 
                               (lhs_batch_dims, rhs_batch_dims))
        
        # Q @ K_T
        attn_weights = jax.lax.dot_general(query, key, dimension_numbers)
        if bias is not None: attn_weights += bias
        norm_dims = tuple(range(attn_weights.ndim//2, attn_weights.ndim))
        attn_weights = jax.nn.softmax(attn_weights, axis = norm_dims)
        attn_weights = nn.Dropout(self.attn_dropout_rate)(attn_weights, deterministic=not self.train)

        attn_weights = attn_weights.astype(dtype)

        contract_dims = (tuple(range(n-1, attn_weights.ndim)),
                         tuple(range(0, n-1))
                         )
        # sm(q@k_t/sqrt(k)) @ V
        y = jax.lax.dot_general(attn_weights, value, (contract_dims, ((),())))

        return y

class DotProductAttention(nn.Module):
    qkv_dim: Optional[int] = None
    out_dim: Optional[int] = None
    train: bool = False

    @nn.compact
    def __call__(self, q, kv, bias=None, dtype=jnp.float32):
        qkv_dim = self.qkv_dim or q.shape[-1]
        out_dim = self.out_dim or q.shape[-1]

        # create qkv weights partial template fn (some args filled)
        QKVDense = functools.partial(nn.Dense,
                                     features = qkv_dim,
                                     use_bias=False,
                                     dtype=dtype)
        query = QKVDense(name='query')(q)
        key = QKVDense(name='key')(kv)
        value = QKVDense(name='value')(kv)

        y = RawDotProductAttention(train=self.train)(
                    query, key, value, bias=bias, dtype=dtype)
        y = nn.Dense(out_dim, dtype=dtype, name='out')(y)

        return y

# -------------------------------------------------
msg = 'Coding Multi Head Attention'
new_block.msg(msg)
class MultiHeadAttention(nn.Module):
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    batch_axis:Sequence[int] = (0,)
    num_heads: int=2
    broadcast_dropout: bool=False
    train: bool = False

    @nn.compact
    def __call__(self, q, kv, bias= None, dtype=jnp.float32):
        qkv_dim = self.qkv_features or q.shape[-1]
        out_dim = self.out_features or q.shape[-1]

        # make multihead attention from single head attention
        Attn = nn.vmap(DotProductAttention,
                       in_axes=(None, None, None),
                       out_axes=2,
                       axis_size=self.num_heads,
                       variable_axes={'params':0},
                       split_rngs={'params':True, 'dropout': not self.broadcast_dropout})

        # vmap across batch dimensions
        for axis in reversed(sorted(self.batch_axis)):
            Attn = nn.vmap(Attn, 
                           in_axes=(axis, axis, axis),
                           out_axes=axis,
                           variable_axes={'params':None},
                           split_rngs={'params':False, 'dropout':False})
            
        # run the vmap'd clas on inputs
        y = Attn(qkv_dim=qkv_dim//self.num_heads, 
                 out_dim=out_dim,
                 train=self.train,
                 name='attention')(q, kv, bias)
        
        return y.mean(axis=-2)

# -------------------------------------------------
msg = 'Run Batched Multihead Attention'
new_block.msg(msg)

# keys, x, model, variables
key1, key2, key3, key4 = jax.random.split(jax.random.key(0), num=4)
x = jax.random.uniform(key1, (3,13,64))
model = functools.partial(MultiHeadAttention,
                          broadcast_dropout=False,
                          num_heads=2,
                          batch_axis=(0,))

init_variables = model(train=False).init({'params':key2}, x, x)
print('initial parameters:')
print(jax.tree.map(jnp.shape, init_variables))

y = model(train=True).apply(init_variables, x, x, rngs = dict(dropout=key4))
print('output:')
print(y)


# -------------------------------------------------
msg = 'Apply nn.scale (A lifted version of jax.lax.scan) to Modules, \nincluding immutable parameters and mutable variables'
new_block.msg(msg)

"""
Scan allows us to apply lax.scan to Modules, including their parameters and mutable variables. To use it we have to specify how we want each "kind" of variable to be transformed. For scanned variables we specify similar to vmap via in variable_in_axes, variable_out_axes:

nn.broadcast broadcast the variable kind across the scan steps as a constant
<axis:int> scan along axis for e.g. unique parameters at each step
OR we specify that the variable kind is to be treated like a "carry" by passing to the variable_carry argument.

Further, for scan'd variable kinds, we further specify whether or not to split the rng at each step.
"""

class SimpleScan(nn.Module):
    features:int
    @nn.compact
    def __call__(self, xs):
        LSTM = nn.scan(nn.LSTMCell,
                       in_axes=1,
                       out_axes=1,
                       variable_broadcast='params',
                       split_rngs=dict(params=False))
        lstm = LSTM(self.features, name = 'lstm_cell')
        dummy_rng = jax.random.key(seed=0)
        input_shape = xs[:,0].shape
        init_carry = lstm.initialize_carry(dummy_rng, input_shape)

        return lstm(init_carry, xs)
    
key1, key2 = jax.random.split(jax.random.key(0), 2)
xs = jax.random.uniform(key1, (1,5,2))
model = SimpleScan(2)
init_variables = model.init(key2, xs)

print('init variables:')
print(init_variables)

print('output:')
print(y)