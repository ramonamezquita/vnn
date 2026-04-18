from typing import Type

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vnn.initializers import make_sigma2_bias_init

from ._modules import MVE, MVELoss, WeightsInitializer
from ._regularizers import Regularizer, no_op_regularizer

CRITERION = MVELoss()


def _training_loop(
    model: MVE,
    dataloader: DataLoader,
    n_epochs: int,
    optimizer: optim.Optimizer,
    mean_reg: Regularizer,
    var_reg: Regularizer = no_op_regularizer,
    grad_max_norm: float = 1.0,
    scheduler: LRScheduler | None = None,
    disable_pbar: bool = False,
) -> None:
    """Standard training loop."""

    pbar = tqdm(range(n_epochs), desc="Epoch", disable=disable_pbar)
    model.train()
    for _ in pbar:
        running_loss: float = 0.0
        n_batches: int = 0
        for X, y in dataloader:
            # fmt: off
            optimizer.zero_grad()                                    # Reset gradients.
            dat_loss = CRITERION(model(X), y)                        # NLL.
            reg_loss = mean_reg(model.mean) + var_reg(model.sigma2)  # Regularizers.
            tot_loss = dat_loss + reg_loss                           # Total loss (NLL + Reg).
            tot_loss.backward()                                      # Set gradients.
            clip_grad_norm_(model.parameters(), grad_max_norm)       # Clip gradients.
            optimizer.step()                                         # Update weights.
            running_loss += tot_loss.item()                          # Record loss.
            n_batches += 1
            # fmt: on

        avg_loss = running_loss / n_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        scheduler.step()


def train_mve(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    learning_rate: float = 1e-3,
    activation_fn: Type[nn.Module] = nn.Tanh,
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer = no_op_regularizer,
    grad_max_norm: float = 1.0,
    disable_pbar: bool = False,
) -> MVE:
    """Train a mean-variance estimation (MVE) model using a two-stage procedure.

    The training is split into two stages:
    1. warm-up: stage where only the mean subnetwork is trained while the variance subnetwork is frozen.
    2. full training: stage where both subnetworks are jointly optimized.

    Parameters
    ----------
    X : torch.Tensor
        Input features of shape (n_samples, n_features).

    y : torch.Tensor
        Target values of shape (n_samples,).

    hidden_layer_sizes : tuple of int, optional
        Sizes of hidden layers in both subnetworks

    n_total_epochs : int, default=10000
        Total number of training epochs across both stages.

    n_warmup_epochs : int, default=5000
        Number of epochs for the warm-up stage (mean network only).
        Must be less than or equal to `n_total_epochs`.

    learning_rate : float, default=1e-3
        Learning rate for both training stages.

    activation_fn : Type[nn.Module], default=nn.Tanh
        Activation function used in hidden layers.

    prior_distr : Distribution or None, optional
        Prior distribution over model parameters used for regularization.
        If provided, a MAP-style regularization term is applied to the mean
        subnetwork. If None, no regularization is applied.

    reg_penalty : float = 1.0
        Regularization penalty factor.

    grad_max_norm : float, default=1.0
        Maximum norm for gradient clipping.

    disable_pbar : bool, default=False
        If True, disables the progress bar.
    """

    model = MVE(
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activation_fn=activation_fn,
        weights_initializer=weights_initializer,
    )
    model = torch.compile(model)
    mean_reg = regularizer or no_op_regularizer

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=X.shape[0], shuffle=True)

    # ----------------------
    # Stage 1: Warm-up
    # ----------------------
    # Train keeping variance parameters fixed.
    # Save computations by not updating variance subnetwork weights until
    # y(x) is somewhat close to f(x).
    warmup_optimizer = optim.Adam(model.mean.parameters(), lr=learning_rate)
    warmup_scheduler = CosineAnnealingLR(warmup_optimizer, T_max=n_warmup_epochs)
    model.sigma2.requires_grad_(False)
    _training_loop(
        model,
        dataloader,
        n_epochs=n_warmup_epochs,
        mean_reg=mean_reg,
        grad_max_norm=grad_max_norm,
        optimizer=warmup_optimizer,
        scheduler=warmup_scheduler,
        disable_pbar=disable_pbar,
    )

    # Set the bias of the output variance to the logmse.
    with torch.no_grad():
        mean_pred = model(X)[:, 0].squeeze()
        logmse: float = torch.log(torch.mean((y - mean_pred) ** 2)).item()
    model.sigma2.apply(make_sigma2_bias_init(logmse))

    # ----------------------
    # Stage 2: Full training
    # ----------------------
    # For subsequent training, all parameters (from both subnetworks)
    # are updated until the total number of epochs is reached.
    n_remaining_epochs = n_total_epochs - n_warmup_epochs
    full_optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    full_scheduler = CosineAnnealingLR(full_optimizer, T_max=n_remaining_epochs)
    model.sigma2.requires_grad_(True)
    _training_loop(
        model,
        dataloader,
        n_epochs=n_remaining_epochs,
        mean_reg=mean_reg,
        var_reg=no_op_regularizer,
        grad_max_norm=grad_max_norm,
        optimizer=full_optimizer,
        scheduler=full_scheduler,
        disable_pbar=disable_pbar,
    )

    return model
