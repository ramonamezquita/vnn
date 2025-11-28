import jax
import jax.numpy as jnp
from flax import nnx


def variance_activation(x: jax.Array) -> jax.Array:
    eps = 1e-6
    mean = x[..., 0]
    var = nnx.softplus(x[..., 1]) + eps
    return jnp.stack([mean, var], axis=-1)
