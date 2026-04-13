from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Distribution
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ._modules import MVE, WeightsInitializer, make_sigma2_bias_init


class Regularizer(Protocol):
    def __call__(self, m: nn.Module) -> torch.Tensor: ...


def no_op_regularizer(m: nn.Module) -> torch.Tensor:
    return torch.tensor(0.0)


@dataclass
class TrainOptions:
    n_epochs: int
    optimizer: optim.Optimizer
    mean_reg: Regularizer
    var_reg: Regularizer = no_op_regularizer
    grad_max_norm: float = 1.0
    scheduler: LRScheduler | None = None
    disable_pbar: bool = False


def make_map_regularizer(distr: Distribution, penalty: float = 1.0) -> Regularizer:
    """Create a parameter regularizer from a probability distribution.

    This constructs a regularization function that penalizes model parameters
    according to the negative log-probability under a given distribution.

    Parameters
    ----------
    distr : Distribution
        A PyTorch distribution object.

    penalty : float, optional
        Scaling factor for the regularization term. Defaults to 1.0.

    Returns
    -------
    Regularizer
    """

    def map_regularizer(m: nn.Module) -> torch.Tensor:
        return -penalty * sum(
            distr.log_prob(p).sum() for p in m.parameters() if p.requires_grad
        )

    return map_regularizer


def total_loss(
    model: MVE,
    X: torch.Tensor,
    y: torch.Tensor,
    mean_reg: Regularizer,
    var_reg: Regularizer = no_op_regularizer,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Computes total loss: NLL + Regularization.

    Parameters
    ----------
    model : MVE
        MVE network.

    X : torch.Tensor
        Input features.

    y : torch.Tensor
        Target values.

    mean_reg : Regularizer
        Regularization function applied to the model's mean parameters
        (e.g., model.mean).

    var_reg : Regularizer, default=no_op_regularizer
        Regularization function applied to the model's variance parameters
        (e.g., model.sigma2). Default is no regularizer.

    eps : float, default=1e-4
        Value added to var, for stability


    Returns
    -------
    torch.Tensor
        Total loss
    """
    outputs = model(X)
    mean = outputs[:, 0]
    var = outputs[:, 1]
    dat_loss = F.gaussian_nll_loss(mean, y, var, reduction="mean", eps=eps)
    reg_loss = mean_reg(model.mean) + var_reg(model.sigma2)
    tot_loss = dat_loss + reg_loss
    return tot_loss


def train_loop(model: MVE, dataloader: DataLoader, options: TrainOptions) -> None:
    """Standard training loop.

    Parameters
    ---------
    model : MVE
        Model to train.

    dataloader: Dataloader
        Training data iterator.

    options : TrainOptions
        Training options.
    """

    pbar = tqdm(range(options.n_epochs), desc="Epoch", disable=options.disable_pbar)
    model.train()
    for _ in pbar:
        running_loss: float = 0.0
        n_batches: int = 0
        for X, y in dataloader:
            # fmt: off
            options.optimizer.zero_grad()                                       # Reset gradients.
            loss = total_loss(model, X, y, options.mean_reg, options.var_reg)   # Compute loss.
            loss.backward()                                                     # Set gradients.
            clip_grad_norm_(model.parameters(), options.grad_max_norm)          # Clip gradients.
            options.optimizer.step()                                            # Update weights.
            running_loss += loss.item()                                         # Record loss.
            n_batches += 1
            # fmt: on

        avg_loss = running_loss / n_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        options.scheduler.step()


def train_mve(
    X,
    y,
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    learning_rate: float = 1e-3,
    activation_fn: nn.Module = nn.Sigmoid,
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer = no_op_regularizer,
    grad_max_norm: float = 1.0,
    disable_pbar: bool = False,
) -> MVE:
    """Train a mean-variance estimation (MVE) model using a two-stage procedure.

    The training is split into two phases:
    (1) a warm-up stage where only the mean subnetwork is trained while the
        variance subnetwork is frozen, and
    (2) a full training stage where both subnetworks are jointly optimized.

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

    activation_fn : Type[nn.Module], default=nn.Sigmoid
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
    regularizer = regularizer or no_op_regularizer

    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    train_dataloader = DataLoader(
        TensorDataset(X, y),
        batch_size=X.shape[0],
        shuffle=True,
    )

    # ----------------------
    # Stage 1: Warm-up
    # ----------------------
    # Train keeping variance parameters fixed.
    # Save computations by not updating variance subnetwork weights until
    # y(x) is somewhat close to f(x).
    warmup_optimizer = optim.Adam(model.mean.parameters(), lr=learning_rate)
    warmup_scheduler = CosineAnnealingLR(warmup_optimizer, T_max=n_warmup_epochs)
    options = TrainOptions(
        n_epochs=n_warmup_epochs,
        optimizer=warmup_optimizer,
        mean_reg=regularizer,
        grad_max_norm=grad_max_norm,
        scheduler=warmup_scheduler,
        disable_pbar=disable_pbar,
    )
    model.sigma2.requires_grad_(False)
    train_loop(model, train_dataloader, options)

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
    options = TrainOptions(
        n_epochs=n_remaining_epochs,
        optimizer=full_optimizer,
        mean_reg=regularizer,
        var_reg=no_op_regularizer,
        grad_max_norm=grad_max_norm,
        scheduler=full_scheduler,
        disable_pbar=disable_pbar,
    )
    model.sigma2.requires_grad_(True)
    train_loop(model, train_dataloader, options)

    return model
