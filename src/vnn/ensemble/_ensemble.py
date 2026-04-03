import inspect
from typing import Any, Type

import numpy as np
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from torch import nn
from torch.distributions import Cauchy, Distribution, Laplace, Normal

from ._sklearn import MVESklearnRegressor


def get_default_args(**kwargs) -> dict[str, Any]:
    signature = inspect.signature(train_ensemble)
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_args.update(kwargs)
    return default_args


def get_activation_fn(name: str) -> Type[nn.Module]:
    name_to_function = {"tanh": nn.Tanh, "sigmoid": nn.Sigmoid, "relu": nn.ReLU}

    if name not in name_to_function:
        raise ValueError(f"Activation function {name} is not supported.")

    return name_to_function[name]


def get_prior_distr(name: str, loc: float = 0.0, scale: float = 1.0) -> Distribution:
    name_to_distr = {"normal": Normal, "cauchy": Cauchy, "laplace": Laplace}
    if name not in name_to_distr:
        raise ValueError(f"Prior distribution {name} not supported.")

    return name_to_distr[name](loc, scale)


def train_ensemble(
    X,
    y,
    *,
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_jobs: int = 4,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    activation_fn: str = "tanh",
    prior: str | None = None,
    prior_loc: float = 0.0,
    prior_scale: float = 1.0,
    grad_max_norm: float = 1.0,
    metrics: tuple[str] = (),
    disable_pbar: bool = False,
) -> BaggingRegressor:
    """Train and evaluate ensemble of Mean-Variance Estimator (MVE) networks.

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

    activation_fn : str, default="tanh"
        Activation function for hidden layers.

    prior : str or None, default=None
        Prior distribution. Used for regularization and weights initialization.
        If None, no regularization is applied and weights are initialized using
        PyTorch default mechanism.

    prior_loc : float, default=0.0
        Prior distribution location.

    prior_scale : float, default=1.0
        Prior distribution scale.

    grad_max_norm : float, default=1.0
        Grad max norm for gradient clipping.

    metrics : tuple of str, default=()
        Metric to track during training.

    disable_pbar: bool, default=False
        If True, progress bar is not displayed.
    """
    if prior is not None:
        prior = get_prior_distr(prior, prior_loc, prior_scale)

    base_regressor = MVESklearnRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=get_activation_fn(activation_fn),
        prior_distr=prior,
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


def predict_ensemble(
    ensemble: BaggingRegressor, X: np.ndarray
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

    output = np.stack([est.predict(X) for est in ensemble.estimators_], axis=0)
    means = output[:, :, 0]
    vars_ = output[:, :, 1]
    mean = means.mean(axis=0)
    var = (vars_ + np.square(means)).mean(axis=0) - np.square(mean)
    return mean, var
