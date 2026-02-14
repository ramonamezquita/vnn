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

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
CRITERION = nn.GaussianNLLLoss()


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Performs a single training step on a batch."""
    optimizer.zero_grad()
    outputs = model(X)
    loss = CRITERION(outputs[:, 0], y, outputs[:, 1])
    loss.backward()
    optimizer.step()
    return loss


class MLP(nn.Module):
    """One-hidden-layer feedforward network."""

    def __init__(
        self,
        n_features_in: int,
        n_hidden_units: int,
        n_features_out: int,
        hidden_activation_fn: ActivationFunction = nn.Sigmoid(),
        output_activation_fn: ActivationFunction = nn.Identity(),
    ):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Linear(n_features_in, n_hidden_units),
            hidden_activation_fn,
            nn.Linear(n_hidden_units, n_features_out),
            output_activation_fn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class MVEModule(nn.Module):
    """Mean-Variance-Estimation network."""

    def __init__(
        self,
        n_hidden_units: int = 10,
        hidden_activation_fn: ActivationFunction = nn.Sigmoid(),
    ):
        super().__init__()
        self.mean = MLP(
            n_features_in=1,
            n_hidden_units=n_hidden_units,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
        )
        self.sigma2 = MLP(
            n_features_in=1,
            n_hidden_units=n_hidden_units,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            output_activation_fn=nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat((self.mean(x), self.sigma2(x)), axis=1)


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
    """

    name_to_function = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU()}

    def __init__(
        self,
        n_hidden_units: int = 10,
        n_total_epochs: int = 5000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: Literal["sigmoid", "relu"] = "sigmoid",
    ):
        self.n_hidden_units = n_hidden_units
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Self:
        X, y = map(torch.tensor, (X, y))

        # Create new MVE network.
        model = MVEModule(
            n_hidden_units=self.n_hidden_units,
            hidden_activation_fn=self.name_to_function[self.activation_fn],
        )

        optimizer = optim.Adam(model.parameters())

        # Stage 1: Train keeping variance parameters fixed (warm-up stage).
        model.sigma2.requires_grad_(False)
        self.fit_loop(model, X, y, optimizer, n_epochs=self.n_warmup_epochs)

        # Stage 2: Train in full.
        # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
        # are updated until the total number of epochs is reached.
        model.sigma2.requires_grad_(True)
        self.fit_loop(
            model, X, y, optimizer, n_epochs=self.n_total_epochs - self.n_warmup_epochs
        )

        self.model_ = model

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        check_is_fitted(self)
        self.model_.eval()
        return self.model_(X)

    def fit_loop(
        self,
        model: MVEModule,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: optim.Optimizer,
        n_epochs: int,
    ) -> None:
        model.train(True)
        for _ in range(n_epochs):
            train_step(model, optimizer, X, y)
