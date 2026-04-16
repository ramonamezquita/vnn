import inspect
from typing import Any

import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from torch import nn

from vnn.regularizers import Regularizer

from ._modules import WeightsInitializer
from ._sklearn import MVERegressor


def get_default_args(**kwargs) -> dict[str, Any]:
    signature = inspect.signature(train_ensemble)
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_args.update(kwargs)
    return default_args


def train_ensemble(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_jobs: int = 4,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    activation_fn: nn.Module = nn.Sigmoid(),
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer | None = None,
    grad_max_norm: float = 1.0,
    disable_pbar: bool = False,
) -> BaggingRegressor:
    """Train and evaluate ensemble of MVE networks.

    Parameters
    ----------
    X : torch.Tensor
        Input data.

    y : torch.Tensor
        Target data.


    n_estimators : int, default=10
        Number of estimators in the ensemble

    n_total_epochs: int = 10000
        Number of total epochs.

    n_warmup_epochs: int = 5000
        Number of warmup epochs.

    n_jobs: int, default=4
        The number of jobs to run in parallel for both fit and predict.

    hidden_layer_sizes : tuple of int, default=(100,)
        Number of units in each hidden layer.

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


    disable_pbar: bool, default=False
        If True, progress bar is not displayed.
    """
    base_regressor = MVERegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        weights_initializer=weights_initializer,
        regularizer=regularizer,
        grad_max_norm=grad_max_norm,
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
    ensemble: BaggingRegressor, X: torch.Tensor, to_ignore: tuple[int, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate predictions from each estimator in a fitted ensemble.

    Parameters
    ----------
    ensemble : BaggingRegressor
        A fitted instance.

    X : np.array
        Input tensor.

    Returns
    -------
    Tuple of np.array.
        Mean and variance prediction.
    """
    check_is_fitted(ensemble)

    output = torch.stack(
        [
            torch.Tensor(est.predict(X))
            for i, est in enumerate(ensemble.estimators_)
            if i not in to_ignore
        ],
        axis=0,
    )
    means = output[:, :, 0]
    vars_ = output[:, :, 1]
    mean = means.mean(axis=0)
    var = (vars_ + torch.square(means)).mean(axis=0) - torch.square(mean)
    return mean, var
