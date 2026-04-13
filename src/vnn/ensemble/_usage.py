import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from torch import nn

from vnn.datasets import Dataset, get_dataset
from vnn.regularizers import Regularizer

from ._modules import WeightsInitializer
from ._sklearn import MVERegressor


@dataclass
class TrainingIO:
    X: np.ndarray
    y: np.ndarray
    mean: np.ndarray
    var: np.ndarray


@dataclass
class RunResult:
    tr_io: TrainingIO
    te_io: TrainingIO
    ensemble: BaggingRegressor
    metrics: list[dict[str, float]]
    weights: np.ndarray


def get_default_args(**kwargs) -> dict[str, Any]:
    signature = inspect.signature(run)
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_args.update(kwargs)
    return default_args


def run(
    dataset: str,
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_jobs: int = 4,
    n_samples: int = 100,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    activation_fn: nn.Module = nn.Sigmoid(),
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer | None = None,
    grad_max_norm: float = 1.0,
    metrics: tuple[str] = (),
    disable_pbar: bool = False,
) -> RunResult:
    """Train and evaluate ensemble.

    Parameters
    ----------
    dataset : str, {"piecewise", "cubic", "sinusoidal"}, default="piecewise"

    n_estimators : int, default=10
        Number of estimators in the ensemble

    n_total_epochs: int = 10000
        Number of total epochs.

    n_warmup_epochs: int = 5000
        Number of warmup epochs.

    n_jobs: int, default=4
        The number of jobs to run in parallel for both fit and predict.

    n_samples: int, default=100
        Number of samples to train with.

    hidden_layer_sizes : tuple of int, default=(100,)
        Number of units in each hidden layer.

    test_size : float, default=0.2
        Test size.

    random_state : int, default=42
        Random seed.

    verbose: int, default=2
        Controls verbosity levels.

    learning_rate: float = 1e-3
        Learning rate.

    activation_fn : nn.Module, default=nn.Sigmoid()
        Activation function for hidden layers.

    weights_initializer: WeightsInitializer or None, default=None
        Weights initializer for mean subnetwork.

    regularizer : Regularizer or None, default=None
        Regularizer during training.

    grad_max_norm : float, default=1.0
        Grad max norm for gradient clipping.

    metrics : tuple of str, default=()
        Metric to track during training.

    disable_pbar: bool, default=False
        If True, progress bar is not displayed.
    """
    dataset: Dataset = get_dataset(dataset)
    x, y = dataset.sample(n_samples, seed=random_state)
    x, y = map(torch.from_numpy, (x, y))
    X = x.reshape(-1, 1)

    if test_size > 0.0:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    else:
        X_tr = X
        y_tr = y
        X_te = None
        y_te = None

    ensemble = fit(
        X_tr,
        y_tr,
        n_estimators=n_estimators,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        n_jobs=n_jobs,
        hidden_layer_sizes=hidden_layer_sizes,
        random_state=random_state,
        verbose=verbose,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        weights_initializer=weights_initializer,
        regularizer=regularizer,
        grad_max_norm=grad_max_norm,
        metrics=metrics,
        disable_pbar=disable_pbar,
    )
    metrics = None
    weights = None

    mean_tr, var_tr = predict(ensemble, X_tr)
    tr_io = TrainingIO(X_tr.flatten(), y_tr, mean_tr, var_tr)

    if X_te is not None:
        X_te: np.ndarray
        mean_te, var_te = predict(ensemble, X_te)
        te_io = TrainingIO(X_te.flatten(), y_te, mean_te, var_te)
    else:
        te_io = None

    return RunResult(tr_io, te_io, ensemble, metrics, weights)


def fit(
    X: torch.Tensor,
    y: torch.Tensor,
    n_estimators: int,
    n_total_epochs: int,
    n_warmup_epochs: int,
    n_jobs: int,
    hidden_layer_sizes: tuple[int, ...],
    random_state: int,
    verbose: int,
    learning_rate: float,
    activation_fn: nn.Module,
    weights_initializer: WeightsInitializer,
    regularizer: Regularizer,
    grad_max_norm: float,
    metrics: tuple[str],
    disable_pbar: bool,
) -> BaggingRegressor:

    base_regressor = MVERegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        weights_initializer=weights_initializer,
        regularizer=regularizer,
        grad_max_norm=grad_max_norm,
        metrics=metrics,
        disable_pbar=disable_pbar,
    )

    ensemble = BaggingRegressor(
        base_regressor,
        n_estimators,
        bootstrap=False,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    ensemble.fit(X, y)
    return ensemble


def predict(
    ensemble: BaggingRegressor, X: np.ndarray, index_to_ignore: list[int] | None = ()
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions from each estimator in a fitted ensemble.

    Parameters
    ----------
    ensemble : BaggingRegressor
        A fitted sklearn.ensemble.BaggingRegressor instance.

    X : np.array
        Input tensor.

    Returns
    -------
    Tuple of np.array.
        Mean and variance prediction.
    """
    check_is_fitted(ensemble)

    output = np.stack(
        [
            est.predict(X)
            for i, est in enumerate(ensemble.estimators_)
            if i not in index_to_ignore
        ],
        axis=0,
    )
    means = output[:, :, 0]
    vars_ = output[:, :, 1]
    mean = means.mean(axis=0)
    var = (vars_ + np.square(means)).mean(axis=0) - np.square(mean)
    return mean, var


# ---------------
# Public Helpers
# ---------------


def get_mean_weights(ensemble: BaggingRegressor) -> dict[str, np.ndarray]:
    weights: list[np.ndarray] = []
    for estimator in ensemble.estimators_:
        estimator: MVERegressor
        subnet = estimator.model_.mean
        w = torch.cat([p.detach().flatten() for p in subnet.parameters()]).numpy()
        weights.append(w)

    weights = np.concatenate(weights)
    return weights


def get_metrics(ensemble: BaggingRegressor) -> list[dict[str, float]]:
    check_is_fitted(ensemble)

    histories = [est.history_ for est in ensemble.estimators_]
    metric_names = histories[0].metrics
    n_epochs = len(histories[0])

    metrics: list[dict[str, float]] = []
    for epoch in range(n_epochs):
        epoch_dict: dict[str, float] = {}

        for metric in metric_names:
            values = [h[epoch][metric] for h in histories]
            epoch_dict[metric] = float(np.mean(values))

        metrics.append(epoch_dict)

    return metrics
