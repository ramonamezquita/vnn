from typing import Iterator, Self

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vnn.regularizers import Regularizer, ZeroRegularizer

from ._modules import MVE, WeightsInitializer, make_sigma2_bias_init


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

    activation_fn : nn.Module, default=nn.Sigmoid()
        Activation function for hidden layer.

    weights_initializer : WeightsInitializer | None, default=None
        Weights initializer for mean subnetwork.

    regularizer : Regularizer | None = None
        Weights regularizer during training.

    grad_max_norm : float, default=1.0
        Grad max norm for gradient clipping.

    metrics: tuple of str, default=()
        Metrics to keep record during training.

    disable_pbar: bool, default=False
        If True, progress bar is not displayed.

    References
    ----------
    [1] Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
    target probability distribution", Proceedings of 1994 IEEE International
    Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
    vol.1, doi: 10.1109/ICNN.1994.374138.
    """

    default_metrics: tuple[str] = ("loss",)

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        n_total_epochs: int = 10000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: nn.Module = nn.Sigmoid,
        weights_initializer: WeightsInitializer | None = None,
        regularizer: Regularizer | None = None,
        grad_max_norm: float = 1.0,
        metrics: tuple[str, ...] = (),
        disable_pbar: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.weights_initializer = weights_initializer
        self.regularizer = regularizer
        self.grad_max_norm = grad_max_norm
        self.metrics = metrics
        self.disable_pbar = disable_pbar

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        self.history_ = History(self.metrics + self.default_metrics)

        model = MVE(
            hidden_layer_sizes=self.hidden_layer_sizes,
            hidden_activation_fn=self.activation_fn,
            weights_initializer=self.weights_initializer,
        )
        model = torch.compile(model)
        regularizer = self.regularizer or ZeroRegularizer()

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        train_dataloader = DataLoader(
            TensorDataset(X, y),
            batch_size=X.shape[0],
            shuffle=True,
        )

        # Stage 1: Train keeping variance parameters fixed (warm-up stage).
        # Save computations by not updating variance subnetwork weights until
        # y(x) is somewhat close to f(x).
        model.sigma2.requires_grad_(False)
        warmup_optimizer = optim.Adam(model.mean.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(warmup_optimizer, T_max=self.n_warmup_epochs)
        fit_mve(
            model=model,
            dataloader=train_dataloader,
            optimizer=warmup_optimizer,
            mean_regularizer=regularizer,
            var_regularizer=ZeroRegularizer(),
            n_epochs=self.n_warmup_epochs,
            grad_max_norm=self.grad_max_norm,
            history=self.history_,
            scheduler=scheduler,
            disable_pbar=self.disable_pbar,
        )

        # Set the bias of the output variance to the logmse.
        with torch.no_grad():
            mean_pred = model(X)[:, 0].squeeze()
            logmse: float = torch.log(torch.mean((y - mean_pred) ** 2)).item()

        model.sigma2.apply(make_sigma2_bias_init(logmse))

        # Stage 2: Train in full.
        # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
        # are updated until the total number of epochs is reached.
        n_remaining_epochs = self.n_total_epochs - self.n_warmup_epochs
        model.sigma2.requires_grad_(True)
        full_optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(full_optimizer, T_max=n_remaining_epochs)
        fit_mve(
            model=model,
            dataloader=train_dataloader,
            optimizer=full_optimizer,
            mean_regularizer=regularizer,
            var_regularizer=ZeroRegularizer(),
            n_epochs=n_remaining_epochs,
            grad_max_norm=self.grad_max_norm,
            history=self.history_,
            scheduler=scheduler,
            disable_pbar=self.disable_pbar,
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


def mve_loss(
    model: MVE,
    X: torch.Tensor,
    y: torch.Tensor,
    mean_regularizer: Regularizer,
    var_regularizer: Regularizer,
) -> torch.Tensor:
    outputs = model(X)
    mean = outputs[:, 0]
    var = outputs[:, 1].clamp_min(1e-4)
    nll = F.gaussian_nll_loss(mean, y, var, reduction="mean")
    loss = nll + mean_regularizer(model.mean) + var_regularizer(model.sigma2)
    return loss


def fit_mve(
    model: MVE,
    n_epochs: int,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    mean_regularizer: Regularizer,
    var_regularizer: Regularizer,
    grad_max_norm: float,
    history: History,
    scheduler: optim.lr_scheduler.LRScheduler,
    disable_pbar: bool = False,
) -> np.ndarray:
    pbar = tqdm(range(n_epochs), desc="Epoch", disable=disable_pbar)

    for epoch in pbar:
        model.train()
        running_loss: float = 0.0
        n_batches = 0

        for X, y in dataloader:
            optimizer.zero_grad()
            loss = mve_loss(model, X, y, mean_regularizer, var_regularizer)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        mean_lr = optimizer.param_groups[0]["lr"]
        var_lr = (
            optimizer.param_groups[1]["lr"]
            if len(optimizer.param_groups) > 1
            else mean_lr
        )
        pbar.set_postfix(
            loss=f"{avg_loss:.4f}", mean_lr=f"{mean_lr:.1e}", var_lr=f"{var_lr:.1e}"
        )
        history.save(loss=avg_loss)
        scheduler.step()
