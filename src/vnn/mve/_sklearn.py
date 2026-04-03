from typing import Self, Type

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.distributions import Distribution

from ._train import train


class MVESklearnRegressor(BaseEstimator, TransformerMixin):
    """Sklearn wrapper for Mean Variance Estimator (MVE) network.

    Scikit-learn compatibility allows the estimator to be used with ensemble
    methods such as bagging which allow parallel training.

    Notes
    -----
    This class accepts Numpy arrays for input/output.

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

    prior_distr : Distribution or None, default=None
        Prior distribution for weight initialization and regularization.
        If None, no regularization is applied and weights are initialized using
        PyTorch default mechanism.

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
        activation_fn: Type[nn.Module] = nn.Sigmoid,
        prior_distr: Distribution | None = None,
        grad_max_norm: float = 1.0,
        metrics: tuple[str, ...] = (),
        disable_pbar: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_total_epochs = n_total_epochs
        self.n_warmup_epochs = n_warmup_epochs
        self.learning_rate = learning_rate
        self.activation_fn = activation_fn
        self.prior_distr = prior_distr
        self.grad_max_norm = grad_max_norm
        self.metrics = metrics
        self.disable_pbar = disable_pbar

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        self.model_ = train(
            X,
            y,
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_total_epochs=self.n_total_epochs,
            n_warmup_epochs=self.n_warmup_epochs,
            learning_rate=self.learning_rate,
            activation_fn=self.activation_fn,
            prior_distr=self.prior_distr,
            grad_max_norm=self.grad_max_norm,
            disable_pbar=self.disable_pbar,
        )
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)

        self.model_.eval()
        X = torch.as_tensor(X, dtype=torch.float32)
        return self.model_(X).numpy()
