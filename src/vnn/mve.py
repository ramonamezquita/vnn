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

from typing import Callable, Iterator, Literal, Protocol, Self

import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim

from vnn.datasets import TORCH_DTYPE
from vnn.metrics import get_metric
from vnn.regularizers import ModelRegularizer

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]
CRITERION = nn.GaussianNLLLoss()


class Metric(Protocol):
    def __call__(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        return ...


class Regularizer(Protocol):
    def __call__(self, model: nn.Module) -> torch.Tensor: ...


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


def train_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    X: torch.Tensor,
    y: torch.Tensor,
    regularizer: Regularizer,
) -> float:
    """Performs a single training step on a batch."""
    optimizer.zero_grad()
    outputs = model(X)
    loss = CRITERION(outputs[:, 0], y, outputs[:, 1])
    loss += regularizer(model)
    loss.backward()
    optimizer.step()
    return loss.item()


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

    l2_penalty : float, default=0.0
        L2 regularization factor.

    l1_penalty : float, default=0.0
        L1 regularization factor.
    """

    default_metrics = (
        "loss",
        "mean_weight_l1_norm",
        "mean_weight_l2_norm",
        "sigma2_weight_l1_norm",
        "sigma2_weight_l2_norm",
    )
    name_to_function = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU()}

    def __init__(
        self,
        n_hidden_units: int = 10,
        n_total_epochs: int = 5000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: Literal["sigmoid", "relu"] = "sigmoid",
        l2_penalty: float = 0.0,
        l1_penalty: float = 0.0,
        cauchy_penalty: float = 0.0,
        cauchy_scale: float = 0.1,
        metrics: tuple[str] = (),
        calc_input_gradient_at: tuple[float] = (),
    ):
        self.n_hidden_units = n_hidden_units
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.cauchy_penalty = cauchy_penalty
        self.cauchy_scale = cauchy_scale
        self.metrics = metrics
        self.calc_input_gradient_at = calc_input_gradient_at

    def get_metric_names(self) -> tuple[str]:
        metric_names = self.metrics + self.default_metrics
        gradient_names = tuple(f"grad_at_{x}" for x in self.calc_input_gradient_at)
        return metric_names + gradient_names

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> Self:
        self.history_ = History(self.get_metric_names())

        X = torch.as_tensor(X, dtype=TORCH_DTYPE)
        y = torch.as_tensor(y, dtype=TORCH_DTYPE)

        # Create new MVE network.
        model = MVEModule(
            n_hidden_units=self.n_hidden_units,
            hidden_activation_fn=self.name_to_function[self.activation_fn],
        )

        optimizer = optim.Adam(model.parameters())
        regularizer = self.get_regularizer()

        # Stage 1: Train keeping variance parameters fixed (warm-up stage).
        model.sigma2.requires_grad_(False)
        self.fit_loop(
            model, X, y, optimizer, regularizer, n_epochs=self.n_warmup_epochs
        )

        # Stage 2: Train in full.
        # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
        # are updated until the total number of epochs is reached.
        model.sigma2.requires_grad_(True)
        self.fit_loop(
            model,
            X,
            y,
            optimizer,
            regularizer,
            n_epochs=self.n_total_epochs - self.n_warmup_epochs,
        )

        self.model_ = model

        return self

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        check_is_fitted(self)
        self.model_.eval()
        return self.model_(X)

    def get_regularizer(self) -> ModelRegularizer:
        regularizer = ModelRegularizer()
        if self.l1_penalty > 0.0:
            regularizer.add_l1_regularizer(self.l1_penalty)
        if self.l2_penalty > 0.0:
            regularizer.add_l2_regularizer(self.l2_penalty)
        if self.cauchy_penalty > 0.0:
            regularizer.add_cauchy_regularizer(self.cauchy_penalty, self.cauchy_scale)
        return regularizer

    def calc_metrics(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> dict[str, float]:
        metrics = {}
        for metric_name in self.metrics:
            metrics[metric_name] = get_metric(metric_name)(y_true, y_pred[:, 0])
        return metrics

    def calc_input_gradients(self, model: MVEModule) -> dict[str, float]:
        grads = {}
        for x in self.calc_input_gradient_at:
            X = torch.tensor([[x]], dtype=TORCH_DTYPE, requires_grad=True)
            model.mean(X).backward()
            grads[f"grad_at_{x}"] = X.grad.item()
        return grads

    def calc_weight_norms(self, model: MVEModule) -> dict[str, float]:
        result = {}
        for name in ("mean", "sigma2"):
            subnetwork = getattr(model, name)
            params = torch.cat([p.detach().flatten() for p in subnetwork.parameters()])
            result[f"{name}_weight_l1_norm"] = params.abs().sum().item()
            result[f"{name}_weight_l2_norm"] = params.pow(2).sum().sqrt().item()
        return result

    def fit_loop(
        self,
        model: MVEModule,
        X: torch.Tensor,
        y: torch.Tensor,
        optimizer: optim.Optimizer,
        regularizer: Regularizer,
        n_epochs: int,
    ) -> None:
        model.train(True)
        for _ in range(n_epochs):
            loss = train_step(model, optimizer, X, y, regularizer=regularizer)
            with torch.no_grad():
                y_pred = model(X)

            epoch_metrics = {}
            epoch_metrics.update(self.calc_metrics(y_true=y, y_pred=y_pred))
            epoch_metrics.update(self.calc_weight_norms(model))
            epoch_metrics.update(self.calc_input_gradients(model))
            epoch_metrics["loss"] = loss
            self.history_.save(**epoch_metrics)
