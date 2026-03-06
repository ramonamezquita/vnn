from typing import Iterator, Literal, Self

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from vnn.regularizers import (
    Regularizer,
    RegularizerOptions,
    build_regularizer,
)

from ._modules import MVE


class History:
    def __init__(self, metrics: list[str]):
        self.metrics = metrics
        self._history: list[dict[str, float]] = list()

    def __len__(self) -> int:
        return len(self._history)

    def __iter__(self) -> Iterator[dict[str, float]]:
        return iter(self._history)

    def __getitem__(self, idx: int) -> dict[str, float]:
        return self._history[idx]

    def save(self, **metrics: float) -> None:
        """Save metrics."""
        if set(metrics) != set(self.metrics):
            raise ValueError(f"Expected metrics {self.metrics}, got {list(metrics)}")

        self._history.append(metrics)

    def last(self) -> dict[str, float]:
        """Return last saved metrics."""
        return self._history[-1].copy()


class MVERegressor(BaseEstimator, TransformerMixin):
    """Sklearn compatible Mean Variance Estimator (MVE) network.

    Scikit-learn compatibility allows the estimator to be used with ensemble
    methods such as bagging.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        Number of units in each hidden layer.

    n_total_epochs: int = 10000
        Number of total epochs.

    n_warmup_epochs: int = 5000
        Number of warmup epochs.

    learning_rate: float = 1e-3
        Learning rate.

    activation_fn : str, {"sigmoid", "relu", "tanh"}, default="sigmoid"
        Activation function for hidden layer.

    l2_penalty : float, default=0.0
        L2 regularization factor.

    l1_penalty : float, default=0.0
        L1 regularization factor.

    cauchy_scale: float, default=1.0
        Cauchy regualization scale.

    metrics: tuple of str, default=()
        Metrics to keep record during training.

    References
    ----------
    [1] Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
    target probability distribution", Proceedings of 1994 IEEE International
    Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
    vol.1, doi: 10.1109/ICNN.1994.374138.
    """

    default_metrics: tuple[str] = ("loss",)
    name_to_function: dict[str, nn.Module] = {
        "sigmoid": nn.Sigmoid(),
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
    }

    def __init__(
        self,
        hidden_layer_sizes: tuple[int] = (100,),
        n_total_epochs: int = 10000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: Literal["sigmoid", "relu", "tanh"] = "sigmoid",
        l2_penalty: float = 0.0,
        l1_penalty: float = 0.0,
        cauchy_scale: float = 0.0,
        metrics: tuple[str] = (),
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.cauchy_scale = cauchy_scale
        self.metrics = metrics

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.history_ = History(self.metrics + self.default_metrics)

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        train_dataloader = DataLoader(
            TensorDataset(X, y),
            batch_size=X.shape[0],
            shuffle=True,
        )

        model = MVE(
            hidden_layer_sizes=self.hidden_layer_sizes,
            hidden_activation_fn=self.name_to_function[self.activation_fn],
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        # NOTE: Both `regularizer` and `criterion` must be reduced using the
        # same aggregating function (e.g., "mean"), so that they remain on
        # the same scale.
        reg_options = RegularizerOptions(
            l1_penalty=self.l1_penalty,
            l2_penalty=self.l2_penalty,
            cauchy_scale=self.cauchy_scale,
        )
        regularizer = build_regularizer(reg_options)

        # Stage 1: Train keeping variance parameters fixed (warm-up stage).
        # Save computations by not updating variance subnetwork weights until
        # y(x) is somewhat close to f(x).
        model.sigma2.requires_grad_(False)
        fit_mve(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            mean_regularizer=regularizer,
            var_regularizer=regularizer,
            n_epochs=self.n_warmup_epochs,
            history=self.history_,
        )

        # Stage 2: Train in full.
        # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
        # are updated until the total number of epochs is reached.
        model.sigma2.requires_grad_(True)
        fit_mve(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            mean_regularizer=regularizer,
            var_regularizer=regularizer,
            n_epochs=self.n_total_epochs - self.n_warmup_epochs,
            history=self.history_,
        )

        self.model_ = model

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        self.model_.eval()
        X = torch.as_tensor(X, dtype=torch.float32)
        output: torch.Tensor = self.model_(X)
        return output.detach().numpy()


def fit_mve(
    model: MVE,
    n_epochs: int,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    mean_regularizer: Regularizer,
    var_regularizer: Regularizer,
    history: History,
) -> np.ndarray:
    for _ in range(n_epochs):
        model.train()
        running_loss: float = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()
            outputs = model(X)
            mean = outputs[:, 0]
            var = outputs[:, 1]
            nll = F.gaussian_nll_loss(mean, y, var, reduction="mean")
            loss = nll + mean_regularizer(model.mean) + var_regularizer(model.sigma2)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        history.save(loss=running_loss)
