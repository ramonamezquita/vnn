from typing import Self, Type

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import BaggingRegressor
from sklearn.utils.validation import check_is_fitted
from torch import nn

from vnn.regularizers import Regularizer, no_op_regularizer

from ._modules import WeightsInitializer
from ._train import train_mve


class _SKMVERegressor(BaseEstimator, TransformerMixin):
    """Sklearn compatible Mean Variance Estimator (MVE) network.

    Scikit-learn compatibility allows parallel ensemble training.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        hidden_sizes: tuple[int, ...] = (100,),
        hidden_activation: Type[nn.Module] = nn.Tanh,
        num_total_epochs: int = 10000,
        num_warmup_epochs: int = 5000,
        initializer: WeightsInitializer | None = None,
        regularizer: Regularizer = no_op_regularizer,
        grad_max_norm: float = 1.0,
    ):
        self.lr = lr
        self.hidden_sizes = hidden_sizes
        self.hidden_activation = hidden_activation
        self.num_total_epochs = num_total_epochs
        self.num_warmup_epochs = num_warmup_epochs
        self.initializer = initializer
        self.regularizer = regularizer
        self.grad_max_norm = grad_max_norm

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        self.model_ = train_mve(
            X,
            y,
            lr=self.lr,
            hidden_sizes=self.hidden_sizes,
            hidden_activation=self.hidden_activation,
            num_total_epochs=self.num_total_epochs,
            num_warmup_epochs=self.num_warmup_epochs,
            initializer=self.initializer,
            regularizer=self.regularizer,
            grad_max_norm=self.grad_max_norm,
        )

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        self.model_.eval()
        out: torch.Tensor = self.model_(torch.as_tensor(X))
        return out.detach().numpy()


def train_ensemble(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float = 1e-3,
    num_estimators: int = 10,
    num_total_epochs: int = 10000,
    num_warmup_epochs: int = 5000,
    num_jobs: int = 4,
    hidden_sizes: tuple[int, ...] = (100,),
    hidden_activation: Type[nn.Module] = nn.Tanh,
    initializer: WeightsInitializer | None = None,
    regularizer: Regularizer | None = None,
    grad_max_norm: float = 1.0,
    random_state: int = 42,
    verbose: int = 2,
) -> BaggingRegressor:
    """Train an ensemble of mean-variance estimation (MVE) neural networks.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input features.

    y : torch.Tensor of shape (n_samples,)
        Target values.

    lr : float, default=1e-3
        Learning rate for each estimator.

    num_estimators : int, default=10
        Number of MVE models in the ensemble.

    num_total_epochs : int, default=10000
        Total number of training epochs.

    num_warmup_epochs : int, default=5000
        Number of warm-up epochs.

    num_jobs : int, default=4
        Number of parallel jobs used to train and evaluate estimators.

    hidden_sizes : tuple of int, default=(100,)
        Sizes of hidden layers in each MVE model.

    hidden_activation : Type[nn.Module], default=nn.Tanh
        Activation function used in hidden layers.

    initializer : WeightsInitializer or None, default=None
        Optional initializer applied to model weights.


    regularizer : Regularizer or None, default=None
        Regularization function applied during training.

    grad_max_norm : float, default=1.0
        Maximum gradient norm for clipping.

    random_state : int, default=42
        Seed controlling randomness in the ensemble construction.

    verbose : int, default=2
        Verbosity level passed to the underlying ensemble.


    Returns
    -------
    ensemble : BaggingRegressor
        Fitted ensemble of MVE estimators.
    """
    base_regressor = _SKMVERegressor(
        lr=lr,
        hidden_sizes=hidden_sizes,
        hidden_activation=hidden_activation,
        num_total_epochs=num_total_epochs,
        num_warmup_epochs=num_warmup_epochs,
        initializer=initializer,
        regularizer=regularizer,
        grad_max_norm=grad_max_norm,
    )

    ensemble = BaggingRegressor(
        base_regressor,
        num_estimators,
        bootstrap=False,
        random_state=random_state,
        verbose=verbose,
        n_jobs=num_jobs,
    )

    ensemble.fit(X, y)
    return ensemble


def predict_ensemble(ensemble: BaggingRegressor, X: torch.Tensor) -> torch.Tensor:
    """Stacks predictions from a fitted ensemble model.

    Parameters
    ----------
    ensemble : BaggingRegressor
        A fitted ensemble regressor.

    X : torch.Tensor of shape (n_samples, n_features)
        Input data for prediction.

    Returns
    -------
    torch.Tensor
    """
    check_is_fitted(ensemble)

    ensemble_outputs: list[torch.Tensor] = [
        torch.Tensor(estimator.predict(X)) for estimator in ensemble.estimators_
    ]
    return torch.stack(ensemble_outputs, axis=0)
