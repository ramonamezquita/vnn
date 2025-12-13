from typing import Callable

import jax
from flax import nnx


class Dense(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *,
        rngs: nnx.Rngs,
        activation_fn: Callable[[jax.Array], jax.Array] | None = nnx.relu,
    ):
        """Dense layer.

        Combines affine transformation + activation function.

        Parameters
        ----------
        in_features : int
            The number of input features.

        out_features : int
            The number of output features.

        rngs : nnx.Rngs
            RNGS key.

        activation_fn : Callable, default=nnx.relu
            Activation function to use. Default is relu.
            Use None for no activation (identity function).
        """
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)
        self.activation_fn = activation_fn or nnx.identity

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.activation_fn(self.linear(x))
