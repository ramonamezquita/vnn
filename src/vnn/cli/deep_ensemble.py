from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from vbnn.activations import variance_activation
from vbnn.criteria import normal_negative_log_likelihood
from vbnn.xy import xy_factory


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


def build_dense_nn(
    input_size: int,
    output_size: int,
    *,
    rngs: nnx.Rngs,
    hidden_sizes: tuple[int, ...] = (),
    hidden_activation_fn: Callable[[jax.Array], jax.Array] | None = nnx.relu,
    output_activation_fn: Callable[[jax.Array], jax.Array] | None = None,
) -> nnx.Sequential:
    """Build dense neural network (i.e., a multilayer-perceptron).

    Parameters
    ----------
    input_size: int
        The number of expected features in the input x.

    output_size: int
        The number of output features.

    rngs : nnx.Rngs
        RNGS key.

    hidden_sizes : tuple, default=()
        The number of features on each hidden layer.
        Default is empty (i.e., no hidden layers).

    hidden_activation_fn : Callable, default=nnx.relu
        Activation function applied to hidden layers. Default is relu.

    output_activation_fn : Callable or None, default=None
        Activation function applied to output. Default is None.

    Returns
    -------
    nnx.Sequential
    """
    sizes = (input_size,) + hidden_sizes + (output_size,)
    n_layers = len(sizes) - 1  # Total number of layers.
    layers: list[Dense] = []

    for i in range(n_layers):
        dense = Dense(
            in_features=sizes[i],
            out_features=sizes[i + 1],
            rngs=rngs,
            activation_fn=output_activation_fn
            if i == n_layers - 1
            else hidden_activation_fn,
        )
        layers.append(dense)

    return nnx.Sequential(*layers)


@nnx.jit  # automatic state propagation
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    def loss_fn(model: nnx.Module) -> jax.Array:
        return normal_negative_log_likelihood(y_true=y, y_pred=model(x))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # in-place updates
    return loss


if __name__ == "__main__":
    seed = 42
    rngs = nnx.Rngs(seed)
    key = jax.random.PRNGKey(seed)
    x, y = xy_factory("polynomial")()

    y += jnp.sqrt(0.01) * jax.random.normal(key, shape=y.shape)

    model = build_dense_nn(
        input_size=1,
        output_size=2,
        hidden_sizes=(50,),
        output_activation_fn=variance_activation,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)
    loss = train_step(model, optimizer, x, y)

    n_epochs = 15000

    for i in range(n_epochs):
        # Run the optimization for one step and make a stateful update to the following:
        # - The train state's model parameters
        # - The optimizer state
        loss = train_step(model, optimizer, x, y)
        print(f"[{i}] Loss: {loss}")

    params = model(x)
    plt.plot(x, y)
    plt.plot(x, params[:, 0])
    plt.show()
