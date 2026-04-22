from typing import Type

import torch
from torch import nn, optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vnn.initializers import make_sigma2_bias_init
from vnn.regularizers import Regularizer, no_op_regularizer

from ._modules import MVE, MVELoss, WeightsInitializer

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
    """Train an MVE model for a fixed number of epochs.

    The loss is composed of:
        - a data term (`CRITERION`)
        - regularization applied to the mean and variance subnetworks.

    See `train_mve` for description of parameters.
    """

    pbar = tqdm(range(n_epochs), desc="Epoch", disable=disable_pbar)
    model.train()
    for _ in pbar:
        running_loss: float = 0.0
        n_batches: int = 0
        for X, y in dataloader:
            optimizer.zero_grad()
            dat_loss = CRITERION(model(X), y)
            reg_loss = mean_reg(model.mean) + var_reg(model.sigma2)
            tot_loss = dat_loss + reg_loss
            tot_loss.backward()
            clip_grad_norm_(model.parameters(), grad_max_norm)
            optimizer.step()
            running_loss += tot_loss.item()
            n_batches += 1

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
    """Train a mean-variance estimation (MVE) neural network via two-stage optimization.

    The model learns to predict both the conditional mean and variance of the target.
    Training proceeds in two stages:

    1. Warm-up stage:
       Only the mean subnetwork is trained while the variance subnetwork is frozen.
       This stabilizes training by first learning a reasonable estimate of E[Y | X].

    2. Full training stage:
       Both mean and variance subnetworks are jointly optimized using the full loss.

    After the warm-up stage, the variance subnetwork bias is initialized using the
    log mean squared error (log-MSE) of the mean predictions to provide a sensible
    starting scale for variance learning.

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Input features.

    y : torch.Tensor of shape (n_samples,)
        Target values.

    hidden_layer_sizes : tuple of int, default=(100,)
        Sizes of hidden layers shared by both subnetworks.

    n_total_epochs : int, default=10000
        Total number of training epochs (warm-up + full training).

    n_warmup_epochs : int, default=5000
        Number of epochs in the warm-up stage. Must satisfy
        0 <= n_warmup_epochs <= n_total_epochs.

    learning_rate : float, default=1e-3
        Learning rate used by the Adam optimizer in both stages.

    activation_fn : Type[nn.Module], default=nn.Tanh
        Activation function applied in hidden layers.

    weights_initializer : WeightsInitializer or None, default=None
        Optional callable used to initialize model weights.

    regularizer : Regularizer, default=no_op_regularizer
        Regularization function applied to the mean subnetwork during training.

    grad_max_norm : float, default=1.0
        Maximum gradient norm for clipping to improve training stability.

    disable_pbar : bool, default=False
        If True, disables progress bar output during training.

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

    model = MVE(
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activation_fn=activation_fn,
        weights_initializer=weights_initializer,
    )
    model: MVE = torch.compile(model)
    dataloader = DataLoader(TensorDataset(X, y), batch_size=X.shape[0], shuffle=True)
    mean_reg = regularizer or no_op_regularizer

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
        mean_pred = model.mean(X).squeeze()
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
