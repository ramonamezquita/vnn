"""
This module estimates the mean and variance of a noisy observation model by
following the two steps approach described in [1].

- Stage 1: Train keeping variance parameters fixed (warm-up stage).
- Stage 2: Train in full.

References
----------
[1] Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
target probability distribution", Proceedings of 1994 IEEE International
Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
vol.1, doi: 10.1109/ICNN.1994.374138.
"""

from typing import Callable, Literal, Self

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def gaussian_nll_loss(
    mean: jax.Array,
    target: jax.Array,
    var: jax.Array,
    full: bool = False,
) -> jax.Array:
    """Compute the Gaussian negative log likelihood loss.
    
    References
    ----------
    [1] PyTorch GaussianNLLLoss:
    https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """

    # Calculate the loss
    loss = 0.5 * (jnp.log(var) + (mean - target) ** 2 / var)
    if full:
        loss += 0.5 * jnp.log(2 * jnp.pi)
    return loss.mean()


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, X: jax.Array, y: jax.Array
) -> float:
    """Performs a single training step on a batch.

    Computes the loss, evaluates gradients, updates the model and
    returns the scalar loss.

    Parameters
    ----------
    model : nnx.Module
        Model to be trained.

    optimizer : nnx.Optimizer
        Optimizer used to update the model parameters.

    X : jax.Array
        Input batch.

    y : jax.Array
        Target batch.

    Returns
    -------
    jax.Array
        Scalar array.
    """

    def loss_fn(model: nnx.Module):
        output = model(X)
        mean = output[:, 0]
        var = output[:, 1]
        return gaussian_nll_loss(mean, y, var)

    argnums = nnx.DiffState(0, optimizer.wrt)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=argnums)(model)
    optimizer.update(model, grads)
    return loss


class MLP(nnx.Module):
    """One-hidden-layer feedforward network.

    Parameters
    ----------
    n_features_in : int
        Input feature dimension.

    n_hidden_units : int
        Hidden layer dimension.

    n_features_out : int
        Output feature dimension.

    hidden_activation_fn : Callable, default=nnx.sigmoid
        Activation function for hidden layer.

    output_activation_fn: Callable, default=nnx.identity,
        Output transformation function.

    rngs : nnx.Rngs
        Random number generators used for parameter initialization and
        stochastic layers (e.g., dropout).

    """

    def __init__(
        self,
        n_features_in: int,
        n_hidden_units: int,
        n_features_out: int,
        hidden_activation_fn: Callable[[jax.Array], jax.Array] = nnx.sigmoid,
        output_activation_fn: Callable[[jax.Array], jax.Array] = nnx.identity,
        *,
        rngs: nnx.Rngs,
    ):
        self.sequential = nnx.Sequential(
            nnx.Linear(n_features_in, n_hidden_units, rngs=rngs),
            hidden_activation_fn,
            nnx.Linear(n_hidden_units, n_features_out, rngs=rngs),
            output_activation_fn,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.sequential(x)


class MVEModule(nnx.Module):
    """Mean-Variance-Estimation network.

    Predicts the mean and variance of a scalar target.
    The network outputs a two-dimensional array containing [mean, variance].

    Parameters
    ----------
    n_hidden_units : int, optional
        Number of hidden units in each subnetwork.

    hidden_activation_fn : Callable, default=nnx.sigmoid
        Activation function for hidden layer.

    rngs : nnx.Rngs
        Random number generator.

    References
    ----------
    [1] Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
    target probability distribution", Proceedings of 1994 IEEE International
    Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
    vol.1, doi: 10.1109/ICNN.1994.374138.
    """

    def __init__(
        self,
        n_hidden_units: int = 10,
        hidden_activation_fn: Callable[[jax.Array], jax.Array] = nnx.sigmoid,
        *,
        rngs: nnx.Rngs,
    ):
        self.mean = MLP(
            n_features_in=1,
            n_hidden_units=n_hidden_units,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            rngs=rngs,
        )
        self.sigma2 = MLP(
            n_features_in=1,
            n_hidden_units=n_hidden_units,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            output_activation_fn=nnx.softplus,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.concat((self.mean(x), self.sigma2(x)), axis=1)


class MVE(BaseEstimator, TransformerMixin):
    """Sklearn compatible Mean-Variance-Estimation network.

    Training is performed in two stages: a warm-up phase where only the mean
    parameters are updated, followed by joint optimization of both mean and
    variance parameters.

    Scikit-learn compatibility allows the estimator to be used with ensemble
    methods such as bagging.

    Parameters
    ----------
    n_hidden_units : int, default=20
        Number of hidden units.

    n_total_epochs: int = 10000
        Number of total epochs.

    n_warmup_epochs: int = 5000
        Number of warmup epochs.

    learning_rate: float = 1e-3
        Learning rate.

    activation_fn : str, {"sigmoid", "relu"}, default="sigmoid"
        Activation function for hidden layer.

    rngs : nnx.Rngs
        Random number generator.


    Example
    -------
    >>> from vnn.datasets import get_dataset
    >>> from vnn.mve import MVE
    >>> X, y = get_dataset("sinusoidal")
    >>> BaggingRegressor(MVE(), n_estimators=10, random_state=0).fit(X, y)
    """

    name_to_function = {"sigmoid": nnx.sigmoid, "relu": nnx.relu}

    def __init__(
        self,
        n_hidden_units: int = 10,
        n_total_epochs: int = 5000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: Literal["sigmoid", "relu"] = "sigmoid",
        *,
        random_state: int = 42,
    ):
        self.n_hidden_units = n_hidden_units
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.random_state = random_state

    def fit(self, X: jax.Array, y: jax.Array) -> Self:
        # Create new MVE network.
        model = MVEModule(
            n_hidden_units=self.n_hidden_units,
            hidden_activation_fn=self.name_to_function[self.activation_fn],
            rngs=nnx.Rngs(self.random_state),
        )

        # Stage 1: Train keeping variance parameters fixed (warm-up stage).
        # This is achieved by differentiating wrt to the mean-related params only.
        # For more details on how to do this in flax see the following GitHub discussion:
        # https://github.com/google/flax/issues/4167
        mean_optimizer = nnx.Optimizer(
            model,
            optax.adam(self.learning_rate),
            wrt=nnx.All(nnx.Param, nnx.PathContains("mean")),
        )
        self._fit_loop(
            model,
            X,
            y,
            mean_optimizer,
            n_epochs=self.n_warmup_epochs,
        )

        # Stage 2: Train in full.
        # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
        # are updated until the total number of epochs is reached.
        # NOTE: Training is resumed using a new `Optimizer` instance.
        optimizer = nnx.Optimizer(model, optax.adam(self.learning_rate), wrt=nnx.Param)
        self._fit_loop(
            model,
            X,
            y,
            optimizer,
            n_epochs=self.n_total_epochs - self.n_warmup_epochs,
        )

        self.model_ = model

        return self

    def predict(self, X: jax.Array) -> jax.Array:
        check_is_fitted(self)
        return self.model_(X)

    def _fit_loop(
        self,
        model: MVEModule,
        X: jax.Array,
        y: jax.Array,
        optimizer: nnx.Optimizer,
        n_epochs: int,
    ) -> None:
        for _ in range(n_epochs):
            train_step(model, optimizer, X, y)
