from typing import Self, Type

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from torch import nn

from ._modules import WeightsInitializer
from ._train import Regularizer, no_op_regularizer, train_mve


class SKMVERegressor(BaseEstimator, TransformerMixin):
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
    """

    default_metrics: tuple[str] = ("loss",)

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        n_total_epochs: int = 10000,
        n_warmup_epochs: int = 5000,
        learning_rate: float = 1e-3,
        activation_fn: Type[nn.Module] = nn.Tanh,
        weights_initializer: WeightsInitializer | None = None,
        regularizer: Regularizer = no_op_regularizer,
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

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        self.model_ = train_mve(
            X,
            y,
            hidden_layer_sizes=self.hidden_layer_sizes,
            n_total_epochs=self.n_total_epochs,
            n_warmup_epochs=self.n_warmup_epochs,
            learning_rate=self.learning_rate,
            activation_fn=self.activation_fn,
            weights_initializer=self.weights_initializer,
            regularizer=self.regularizer,
            grad_max_norm=self.grad_max_norm,
            disable_pbar=self.disable_pbar,
        )

        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        self.model_.eval()
        return self.model_(torch.as_tensor(X)).detach().numpy()
