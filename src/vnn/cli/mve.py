import argparse
import time
from typing import Callable

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from flax import nnx

from vnn.activations import variance_activation
from vnn.criteria import normal_negative_log_likelihood
from vnn.data import ArrayDataset, MiniBatchIterator
from vnn.layers import Dense
from vnn.xy import xy_factory


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
        Size for each hidden layer.
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
    n_layers = len(sizes) - 1  # Total number of layers ignoring input.
    layers: tuple[Dense, ...] = tuple(
        Dense(
            in_features=sizes[i],
            out_features=sizes[i + 1],
            rngs=rngs,
            activation_fn=output_activation_fn
            if i == n_layers - 1
            else hidden_activation_fn,
        )
        for i in range(n_layers)
    )

    return nnx.Sequential(*layers)


def zero_param_column(
    state: nnx.State,
    path: tuple[str | int, ...],
    position: int = 0,
) -> nnx.State:
    """Zero out a specific column of a parameter in a state.

    Parameters
    ----------
    state: nnx.State
        The neural network state containing parameters.

    path: tuple[str | int, ...]
        The path to the target parameter. If None, no parameter is modified.

    position: int, default=0
        The column index to set to zero.

    Returns
    -------
    nnx.State
        New state.
    """

    def freezer(
        variable_path: tuple[str, ...], variable_param: nnx.Param
    ) -> tuple[tuple[str, ...], nnx.Param]:
        if variable_path != path:
            return variable_param

        # Get parameter value and set its column at `position` to 0.
        new_value: jax.Array = variable_param[...].at[..., position].set(0)
        variable_param.set_value(new_value)

        return variable_param

    return nnx.map_state(freezer, state)


@nnx.jit
def update_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Train for a single step.

    This function takes a Module, an Optimizer, and a batch of data, and returns
    the loss for that step. The loss and the gradients are computed using the
    `nnx.value_and_grad` transform over the `loss_fn`. The gradients are passed to
    the optimizer's update method to update the model's parameters.

    Parameters
    ----------
    model : nnx.Module
        Model to train.

    optimizer: nnx.Optimizer
        Optimizer for updating the model state.

    x: jax.Array
        Input values.

    y: jax.Array
        Target values. Used for computing the loss.


    Returns
    -------
    loss : float
    """

    def loss_fn(model: nnx.Module) -> jax.Array:
        return normal_negative_log_likelihood(y_true=y, y_pred=model(x))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def update_state_with_frozen_variance(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Same as `train_step`, but with adittional zeroing of variance gradient."""
    print("update_state_with_frozen_variance")

    def loss_fn(model: nnx.Module) -> jax.Array:
        return normal_negative_log_likelihood(y_true=y, y_pred=model(x))

    loss, grads = nnx.value_and_grad(loss_fn)(model)

    # Freeze variance-related params.
    grads = zero_param_column(grads, ("layers", 2, "linear", "kernel"), position=-1)
    grads = zero_param_column(grads, ("layers", 2, "linear", "bias"), position=-1)

    # Update.
    optimizer.update(model, grads)
    return loss


def train_model(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    iterator: MiniBatchIterator,
    n_epochs: int = 100,
    n_warmup_epochs: int = 0,
) -> nnx.Module:
    # Store initial variance parameters
    _, state = nnx.split(model)
    initial_var_kernel = state["layers"][2]["linear"]["kernel"].value[:, -1].copy()
    initial_var_bias = state["layers"][2]["linear"]["bias"].value[-1].copy()

    for epoch in range(n_epochs):
        start_time = time.time()
        num_batches: int = 0
        running_loss: float = 0.0

        update_fn = (
            update_state_with_frozen_variance
            if epoch < n_warmup_epochs
            else update_state
        )

        for x, y in iterator:
            num_batches += 1
            loss = update_fn(model, optimizer, x, y)
            running_loss += loss

        print(
            f"Epoch {epoch + 1} in {time.time() - start_time:.2f} sec: "
            f"Train Loss: {running_loss / num_batches:.4f}"
        )

        # Check variance params during warmup
        if epoch < n_warmup_epochs:
            _, state = nnx.split(model)
            current_var_kernel = state["layers"][2]["linear"]["kernel"].value[:, -1]
            current_var_bias = state["layers"][2]["linear"]["bias"].value[-1]

            assert jnp.allclose(current_var_kernel, initial_var_kernel), (
                f"Variance kernel changed at epoch {epoch}!"
            )
            assert jnp.allclose(current_var_bias, initial_var_bias), (
                f"Variance bias changed at epoch {epoch}!"
            )

    return model


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "-d", "--dataset", default="polynomial", type=str, help="Dataset to use."
    )
    parser.add_argument(
        "-e", "--epochs", default=1000, type=int, help="Number of epochs."
    )

    parser.add_argument(
        "-w", "--warmup", default=100, type=int, help="Number of warmup epochs."
    )
    return parser


def main(args: argparse.Namespace) -> None:
    rngs = nnx.Rngs(args.seed)
    key = jax.random.PRNGKey(args.seed)
    x, y = xy_factory(args.dataset)(num=1024)
    y += jnp.sqrt(0.001) * jax.random.normal(key, shape=y.shape)

    dataset = ArrayDataset(x, y)
    iterator = MiniBatchIterator(dataset, key=key, batch_size=64)

    model = build_dense_nn(
        input_size=1,
        output_size=2,
        hidden_sizes=(100, 200),
        hidden_activation_fn=nnx.relu,
        output_activation_fn=variance_activation,
        rngs=rngs,
    )
    optimizer = nnx.Optimizer(model, optax.sgd(0.001), wrt=nnx.Param)
    model = train_model(
        model, optimizer, iterator, n_epochs=args.epochs, n_warmup_epochs=args.warmup
    )
    params = model(x)

    if args.plot:
        plt.plot(x, y)
        plt.plot(x, params[:, 0])
        plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(vars(args))
    model, optimizer, x, y = main(args)
