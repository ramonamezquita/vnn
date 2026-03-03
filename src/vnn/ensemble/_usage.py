from dataclasses import dataclass

import numpy as np
import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from vnn.datasets import Dataset, get_dataset

from ._modules import MVERegressor


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


def run(
    dataset: str,
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_jobs: int = 4,
    n_samples: int = 100,
    hidden_layer_sizes: tuple[int] = (100,),
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    l2_penalty: float = 0.0,
    l1_penalty: float = 0.0,
    cauchy_scale: float = 0.0,
    activation_fn: str = "sigmoid",
    metrics: tuple[str] = (),
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

    random_state : int, default=32
        Random seed.

    verbose: int, default=2
        Controls verbosity levels.

    learning_rate: float = 1e-3
        Learning rate.

    l2_penalty : float, default=0.0
        L2 regularization factor.

    l1_penalty : float, default=0.0
        L1 regularization factor.

    cauchy_scale: float, default=1.0
        Cauchy regualization scale.

    activation_fn : str, {"sigmoid", "relu"}, default="sigmoid"
        Activation function for hidden layer.

    metrics : tuple of str, default=()
        Metric to track during training.
    """

    dataset: Dataset = get_dataset(dataset)
    x, y = dataset.sample(n_samples)
    x, y = map(torch.from_numpy, (x, y))
    X = x.reshape(-1, 1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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
        l2_penalty=l2_penalty,
        l1_penalty=l1_penalty,
        cauchy_scale=cauchy_scale,
        activation_fn=activation_fn,
        metrics=metrics,
    )
    metrics = get_metrics(ensemble)

    mean_tr, var_tr = predict(ensemble, X_tr)
    mean_te, var_te = predict(ensemble, X_te)

    tr_io = TrainingIO(X_tr.flatten(), y_tr, mean_tr, var_tr)
    te_io = TrainingIO(X_te.flatten(), y_te, mean_te, var_te)
    return RunResult(tr_io, te_io, ensemble, metrics)


def fit(
    X: torch.Tensor,
    y: torch.Tensor,
    n_estimators: int,
    n_total_epochs: int,
    n_warmup_epochs: int,
    n_jobs: int,
    hidden_layer_sizes: tuple[int],
    random_state: int,
    verbose: int,
    learning_rate: float,
    l2_penalty: float,
    l1_penalty: float,
    cauchy_scale: float,
    activation_fn: str,
    metrics: tuple[str],
) -> BaggingRegressor:

    base_regressor = MVERegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        l2_penalty=l2_penalty,
        l1_penalty=l1_penalty,
        cauchy_scale=cauchy_scale,
        metrics=metrics,
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


def predict(ensemble: BaggingRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions from each estimator in a fitted ensemble.

    Parameters
    ----------
    ensemble : BaggingRegressor
        A fitted sklearn.ensemble.BaggingRegressor instance.

    X : torch.Tensor
        Input tensor.

    Returns
    -------
    Tuple of torch.Tensor.
        Mean and variance prediction.
    """
    check_is_fitted(ensemble)

    n_estimators = ensemble.n_estimators
    output = np.stack(
        [ensemble.estimators_[i].predict(X) for i in range(n_estimators)], axis=0
    )
    means = output[:, :, 0]
    vars_ = output[:, :, 1]
    mean = means.mean(axis=0)
    var = (vars_ + np.square(means)).mean(axis=0) - np.square(mean)
    return mean, var


# ---------------
# Public Helpers
# ---------------


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
