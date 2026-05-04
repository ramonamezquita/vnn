from typing import Type

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from vnn.initializers import make_sigma2_bias_init
from vnn.regularizers import Regularizer, no_op_regularizer

from ._modules import MVE, MVELoss, WeightsInitializer


def _train_loop(
    model: MVE,
    dataloader: DataLoader,
    n_epochs: int,
    optimizer: Optimizer,
    mean_reg: Regularizer,
    var_reg: Regularizer = no_op_regularizer,
    grad_max_norm: float = 1.0,
) -> None:
    """Train a MVE model for a fixed number of epochs.

    See `train_mve` for description of parameters.
    """
    model.train()
    criterion = MVELoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    for _ in range(n_epochs):
        running_loss: float = 0.0
        for X, y in dataloader:
            optimizer.zero_grad()  # reset gradients
            # the loss is composed of:
            # - a data term (`MVELoss`)
            # - regularization applied to the mean and variance subnetworks.
            loss = criterion(model(X), y) + mean_reg(model.mean) + var_reg(model.sigma2)
            loss.backward()  # set gradients
            clip_grad_norm_(model.parameters(), grad_max_norm)  # clip gradients
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()


def train_mve(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float = 1e-3,
    hidden_sizes: tuple[int, ...] = (100,),
    hidden_activation: Type[nn.Module] = nn.Tanh,
    num_total_epochs: int = 10000,
    num_warmup_epochs: int = 5000,
    initializer: WeightsInitializer | None = None,
    regularizer: Regularizer = no_op_regularizer,
    grad_max_norm: float = 1.0,
) -> MVE:
    """Train a mean-variance estimation (MVE) neural network via two-stage optimization.

    1. Warm-up stage:
       Only the mean subnetwork is trained while the variance subnetwork is frozen.
       This stabilizes training by first learning a reasonable estimate of E[Y | X].

    2. Full training stage:
       Both mean and variance subnetworks are jointly optimized using the full loss.

    After the warm-up stage, the variance subnetwork bias is initialized using the
    log mean squared error (log-MSE) of the mean predictions to provide a reasonable
    starting point for learning the variance.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input features.

    y : torch.Tensor of shape (n_samples,)
        Target values.

    lr : float, default=1e-3
        Learning rate used by the Adam optimizer in both stages.

    hidden_sizes : tuple of int, default=(100,)
        Sizes of hidden layers shared by both subnetworks.

    hidden_activation : Type[nn.Module], default=nn.Tanh
        Activation function applied in hidden layers.

    num_total_epochs : int, default=10000
        Total number of training epochs (warm-up + full training).

    num_warmup_epochs : int, default=5000
        Number of epochs in the warm-up stage. Must satisfy
        num_total_epochs >= num_warmup_epochs >= 0.

    initializer : WeightsInitializer or None, default=None
        Optional callable used to initialize model weights.

    regularizer : Regularizer, default=no_op_regularizer
        Regularization function applied to the mean subnetwork during training.

    grad_max_norm : float, default=1.0
        Maximum gradient norm for clipping to improve training stability.

    Returns
    -------
    model : MVE
        Trained MVE model. The forward pass returns a tensor of shape
        (n_samples, 2), where the columns correspond to predicted mean and variance.

    References
    ----------
    [1] Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
    target probability distribution", Proceedings of 1994 IEEE International
    Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
    vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    assert num_total_epochs >= num_warmup_epochs >= 0, (
        "The condition `num_total_epochs >= num_warmup_epochs >= 0` is not satisfied."
    )

    model = MVE(
        hidden_layer_sizes=hidden_sizes,
        hidden_activation_fn=hidden_activation,
        weights_initializer=initializer,
    )
    model: MVE = torch.compile(model)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=X.shape[0], shuffle=True)
    mean_reg = regularizer or no_op_regularizer

    # warm-up stage: train keeping variance parameters frozen.
    warmup_optimizer = Adam(model.mean.parameters(), lr=lr)
    model.sigma2.requires_grad_(False)
    _train_loop(
        model,
        dataloader,
        n_epochs=num_warmup_epochs,
        mean_reg=mean_reg,
        grad_max_norm=grad_max_norm,
        optimizer=warmup_optimizer,
    )

    # set the bias of the output variance to the logmse.
    with torch.no_grad():
        mean_pred = model.mean(X).squeeze()
        logmse: float = torch.log(torch.mean((y - mean_pred) ** 2)).item()
    model.sigma2.apply(make_sigma2_bias_init(logmse))

    # full-train stage: all parameters (from both subnetworks) are updated.
    num_remaining_epochs = num_total_epochs - num_warmup_epochs
    full_optimizer = Adam(model.parameters(), lr=lr)
    model.sigma2.requires_grad_(True)
    _train_loop(
        model,
        dataloader,
        n_epochs=num_remaining_epochs,
        mean_reg=mean_reg,
        var_reg=no_op_regularizer,
        grad_max_norm=grad_max_norm,
        optimizer=full_optimizer,
    )

    return model
