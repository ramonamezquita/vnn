from typing import Type

import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor
from torch import nn

from vnn.regularizers import Regularizer

from ._modules import WeightsInitializer
from ._sklearn import SKMVERegressor


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
    activation_fn: Type[nn.Module] = nn.Tanh,
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer | None = None,
    grad_max_norm: float = 1.0,
    disable_pbar: bool = False,
) -> BaggingRegressor:
    """Train an ensemble of mean-variance estimation (MVE) neural networks.

    This function fits a bagging ensemble where each base estimator is an MVE model
    trained via the two-stage procedure implemented in `train_mve`. Each estimator
    learns to predict both the conditional mean and variance of the target.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input features.

    y : torch.Tensor of shape (n_samples,)
        Target values.

    n_estimators : int, default=10
        Number of MVE models in the ensemble.

    n_total_epochs : int, default=10000
        Total number of training epochs per estimator.

    n_warmup_epochs : int, default=5000
        Number of warm-up epochs per estimator (mean network only).
        Must satisfy 0 <= n_warmup_epochs <= n_total_epochs.

    n_jobs : int, default=4
        Number of parallel jobs used to train and evaluate estimators.

    hidden_layer_sizes : tuple of int, default=(100,)
        Sizes of hidden layers in each MVE model.

    random_state : int, default=42
        Seed controlling randomness in the ensemble construction.

    verbose : int, default=2
        Verbosity level passed to the underlying ensemble.

    learning_rate : float, default=1e-3
        Learning rate for each estimator.

    activation_fn : Type[nn.Module], default=nn.Tanh
        Activation function used in hidden layers.

    weights_initializer : WeightsInitializer or None, default=None
        Optional initializer applied to model weights.

    regularizer : Regularizer or None, default=None
        Regularization function applied during training.

    grad_max_norm : float, default=1.0
        Maximum gradient norm for clipping.

    disable_pbar : bool, default=False
        If True, disables progress bars during training.

    Returns
    -------
    ensemble : BaggingRegressor
        Fitted ensemble of MVE estimators. Each estimator's ``predict`` method
        returns an array of shape (n_samples, 2) containing mean and variance.
    """
    base_regressor = SKMVERegressor(
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
    ensemble: BaggingRegressor, X: torch.Tensor, index_to_ignore: tuple[int, ...] = ()
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute ensemble mean and variance predictions from a fitted ensemble model.

    Each estimator in the ensemble is assumed to return a 2D output of shape
    (n_samples, 2), where the first column corresponds to the predicted mean
    and the second column to the predicted variance.

    The final predictive mean is computed as the average of individual means.
    The predictive variance is computed using the law of total variance:
                Var(Y) = E[Var(Y | model)] + Var(E[Y | model])

    Parameters
    ----------
    ensemble : BaggingRegressor
        A fitted ensemble regressor with base estimators that output mean and variance.

    X : torch.Tensor of shape (n_samples, n_features)
        Input data for prediction.

    index_to_ignore : tuple of int, default=()
        Indices of estimators to exclude from the aggregation.

    Returns
    -------
    mean : torch.Tensor of shape (n_samples,)
        Ensemble predictive mean.

    var : torch.Tensor of shape (n_samples,)
        Ensemble predictive variance.
    """
    check_is_fitted(ensemble)

    all_outputs: list[torch.Tensor] = [
        torch.Tensor(est.predict(X))
        for i, est in enumerate(ensemble.estimators_)
        if i not in index_to_ignore
    ]
    stacked_outputs = torch.stack(all_outputs, axis=0)

    means: torch.Tensor = stacked_outputs[:, :, 0]
    vars_: torch.Tensor = stacked_outputs[:, :, 1]

    mean = means.mean(axis=0)
    var = (vars_ + torch.square(means)).mean(axis=0) - torch.square(mean)
    return mean, var
