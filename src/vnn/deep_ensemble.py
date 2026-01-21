"""


References
----------
[1] Sklearn bagging implementation:
https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/ensemble/_bagging.py

[2] 

"""

import itertools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from joblib import Parallel, delayed, effective_n_jobs
from tqdm import trange

from vnn.datasets import get_dataset
from vnn.main import RunParams

LEARNING_RATE = 1e-3


def make_model_factory(
    activation_fn: Callable[[jax.Array], jax.Array] = nnx.sigmoid,
    hidden_size: int = 100,
) -> Callable[[nnx.Rngs], nnx.Sequential]:
    def model_factory(rngs: nnx.Rngs) -> nnx.Sequential:
        return nnx.Sequential(
            nnx.Linear(1, hidden_size, rngs=rngs),
            activation_fn,
            nnx.Linear(hidden_size, 1, rngs=rngs),
        )

    return model_factory


def partition_models(n_models, n_jobs) -> list[int]:
    """Private function used to partition models between jobs."""
    # Compute the number of jobs.
    n_jobs = min(effective_n_jobs(n_jobs), n_models)

    # Partition models between jobs.
    n_models_per_job = np.full(n_jobs, n_models // n_jobs, dtype=int)
    n_models_per_job[: n_models % n_jobs] += 1

    return n_models_per_job.tolist()


def get_sample_size(total_size: int, sample_size: int | float | None) -> int:
    if sample_size is None:
        return total_size
    elif isinstance(sample_size, int):
        return sample_size

    # `sample_size` is fractional value relative to `total_size`.
    return max(int(sample_size * total_size), 1)


def train_n_models(
    model_factory: Callable[[nnx.Rngs], nnx.Module],
    n_models: int,
    X: jax.Array,
    y: jax.Array,
    sample_size: int | float | None,
    key: jax.Array,
    n_epochs: int,
):
    """Trains n distinct models.

    Parameters
    ----------
    model_factory : Callable[[nnx.Rngs], nnx.Module]
        Returns new model instances from a random key.

    n_models : int
        Number of models to train.

    X : jax.Array
        Feature matrix.

    y : jax.Array
        Target values.

    sample_size : int or float
        The number of samples to draw from X to train each model (with
        replacement).

        - If None, then draw `X.shape[0]` samples.
        - If int, then draw `sample_size` samples.
        - If float, then draw `sample_size * X.shape[0]` samples.

    key: jax.Array
        A PRNG key (from key, split, fold_in).

    n_epochs : int
        Number of epochs.
    """
    trained_models: list[nnx.Module] = []
    total_size = X.shape[0]

    # Create a distinct random key for each model.
    keys = jax.random.split(key, n_models)

    sample_size = get_sample_size(total_size, sample_size)

    for i in range(n_models):
        random_key = keys[i]

        # Randomly draw sample indices (`replace=True` for independence).
        indices = jax.random.choice(
            random_key, total_size, (sample_size,), replace=True
        )

        # Create new model using the given `model_factory`.
        # Each model is initialized with its own independent/distinct random key.
        model = model_factory(nnx.Rngs(random_key))

        # Create new optimizer.
        optimizer = nnx.Optimizer(model, optax.adam(LEARNING_RATE), wrt=nnx.Param)

        X_sampled = X[indices, :]
        y_sampled = y[indices, :]

        # Train loop.
        pbar = trange(n_epochs, desc="Training")
        for _ in pbar:
            loss = update_state(model, optimizer, X_sampled, y_sampled)
            pbar.set_postfix(loss=f"{loss:.4f}")

        trained_models.append(model)

    return trained_models


@nnx.jit
def update_state(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    x: jax.Array,
    y: jax.Array,
) -> jax.Array:
    def loss_fn(model: nnx.Module) -> jax.Array:
        return jnp.mean(jnp.square(y - model(x)))

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


def run(params: RunParams):
    n_jobs = params.n_jobs
    n_models = params.n_models
    n_epochs = params.n_epochs
    sample_size = params.sample_size

    X, y = get_dataset(params.dataset)
    n_models_per_job = partition_models(n_models, n_jobs)

    # Create a distinct random key for each job.
    random_keys = jax.random.split(jax.random.key(params.seed), n_jobs)

    # Parallel loop.
    # Each job trains multiple models (unless we have 1 model per job).
    all_models = Parallel(n_jobs=n_jobs)(
        delayed(train_n_models)(
            make_model_factory(hidden_size=params.hidden_size),
            n_models_per_job[i],
            X,
            y,
            sample_size,
            random_keys[i],
            n_epochs,
        )
        for i in range(n_jobs)
    )

    all_models = list(itertools.chain.from_iterable(all_models))

    # Collect predictions
    Y = [model(X) for model in all_models]
    Y = jnp.stack(Y, axis=1).squeeze(-1)  # (n_points, n_models)

    # Ensemble statistics
    mean = jnp.mean(Y, axis=1)
    std = jnp.std(Y, axis=1)

    if params.plot:
        import matplotlib.pyplot as plt

        ax = plt.subplot()

        # Individual prediction.
        for i in range(params.n_models):
            ax.plot(X[:, 0], Y[:, i], alpha=0.2, c="black")

        # Uncertainty prediction.
        ax.fill_between(
            X[:, 0],
            mean - 1.96 * std,
            mean + 1.96 * std,
            alpha=0.3,
        )

        # Mean prediction.
        ax.plot(X[:, 0], mean, label="Network output", c="black")

        # Actual observations.
        ax.scatter(X[:, 0], y, label="Observations", s=2, c="black", alpha=0.1)

        ax.set_xlabel("X")
        ax.set_ylabel("y")
        plt.legend()
        plt.title("Deep Ensemble Regression")
        plt.show()
