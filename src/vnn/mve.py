"""
This module estimates the mean and variance of a noisy observation model by
following the two steps approach described in [1].

- Stage 1: Train keeping variance parameters fixed (warm-up stage).
- Stage 2: Train in full.

References
----------
[1] David A. Nix and Andreas S. Weigend. Estimating the mean and variance of the target probability distribution.
"""

import argparse
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import trange


def negative_log_likelihood(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    """Negative log likelihood for Normal distribution.

    Parameters
    ----------
    y_true : jax.Array
        Ground truth.

    y_pred : jax.Array
        Predicted values.
    """
    mu = y_pred[:, [0]]
    sigma2 = sigma2_transformation(y_pred[:, [1]])
    return jnp.mean(jnp.log(sigma2) + ((y_true - mu) ** 2 / sigma2))


def sigma2_transformation(x: jax.Array) -> jax.Array:
    return nnx.softplus(x) + 1e-6


class MVENetwork(nnx.Module):
    """Mean-Variance Estimation network.

    Parameters
    ----------
    rngs: nnx.Rngs
        RNGS key.
    """

    def __init__(self, *, rngs: nnx.Rngs):
        self.mu_net = nnx.Sequential(
            nnx.Linear(1, 100, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(100, 1, rngs=rngs),
        )

        self.sigma2_net = nnx.Sequential(
            nnx.Linear(1, 100, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(100, 1, rngs=rngs),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.concat((self.mu_net(x), self.sigma2_net(x)), axis=1)


@nnx.jit
def update_full_state(
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

    def loss_fn(model: MVENetwork) -> jax.Array:
        return negative_log_likelihood(y_true=y, y_pred=model(x))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def update_mean_state_only(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Updates the mean state only.

    Similar logic as `update_full_state`, but here gradients are taken wrt to mean
    related params only.
    """

    def loss_fn(model: MVENetwork) -> jax.Array:
        return negative_log_likelihood(y_true=y, y_pred=model(x))

    mu_net_params = nnx.All(nnx.Param, nnx.PathContains("mu_net"))
    diff_state = nnx.DiffState(0, mu_net_params)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
    optimizer.update(model, grads)
    return loss


def train_loop(
    model: nnx.Module,
    x: jax.Array,
    y: jax.Array,
    optimizer: nnx.Optimizer,
    n_epochs: int,
    update_fn: Callable,
    desc: str = "Training"
) -> nnx.Module:
    """Generic training loop."""

    pbar = trange(n_epochs, desc=desc)
    for _ in pbar:
        loss = update_fn(model, optimizer, x, y)
        pbar.set_postfix(loss=f"{loss:.4f}")

    return model


def train_mve(
    seed: int = 42,
    n_epochs: int = 100,
    n_warmup_epochs: int = 50,
    learning_rate: float = 0.001,
):
    # X,y
    n_samples = 500
    x = jnp.linspace(0, 2, n_samples).reshape(-1, 1)
    y = 2 * x**3 - 5 * x**2 + 3 * x + 7

    # Mean-Variance Model.
    new_model = MVENetwork(rngs=nnx.Rngs(seed))

    # Stage 1: Train keeping variance parameters fixed (warm-up stage).
    # This is achieved by differentiating wrt to the mean-related params.
    # For more details see the following GitHub discussion:
    # https://github.com/google/flax/issues/4167
    mu_net_params = nnx.All(nnx.Param, nnx.PathContains("mu_net"))
    optimizer = nnx.Optimizer(new_model, optax.adam(learning_rate), wrt=mu_net_params)
    warmed_up_model = train_loop(
        new_model,
        x,
        y,
        optimizer=optimizer,
        n_epochs=n_warmup_epochs,
        update_fn=update_mean_state_only,
        desc="Stage 1 (warming up)"
    )
    

    # Stage 2: Train in full.
    optimizer = nnx.Optimizer(warmed_up_model, optax.adam(learning_rate), wrt=nnx.Param)
    finished_model = train_loop(
        warmed_up_model,
        x,
        y,
        optimizer=optimizer,
        n_epochs=n_epochs - n_warmup_epochs,
        update_fn=update_full_state,
        desc="Stage 2 (full training)"
    )

    return finished_model, x, y


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-s", "--seed", default=42, type=int, help="Random seed.")
    parser.add_argument(
        "-w",
        "--n_warmup_epochs",
        default=5_000,
        type=int,
        help="Number of warmup epochs.",
    )
    parser.add_argument(
        "-e",
        "--n_total_epochs",
        default=10_000,
        type=int,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "-l", "--learning_rate", default=1e-3, type=float, help="Learning rate."
    )

    return parser


def main(args: argparse.Namespace) -> None:
    trained_model, x, y = train_mve(
        seed=args.seed,
        n_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        learning_rate=args.learning_rate,
    )

    if args.plot:
        import matplotlib.pyplot as plt

        y_pred = trained_model(x)
        mu_pred = y_pred[:, 0]
        sigma2_pred = sigma2_transformation(y_pred[:, 1])

        lb = mu_pred - 1.96 * jnp.sqrt(sigma2_pred)
        ub = mu_pred + 1.96 * jnp.sqrt(sigma2_pred)

        ax = plt.subplot()
        ax.plot(x, y, label="True values")
        ax.plot(x, mu_pred, label="Predicted values")
        ax.plot(x, lb, ls="--", c="gray", label="95% CI")
        ax.plot(x, ub, ls="--", c="gray")
        plt.legend()

        plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
