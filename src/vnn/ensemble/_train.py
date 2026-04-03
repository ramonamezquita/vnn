from typing import Protocol, Type

import torch
from torch import nn
from torch.distributions import Distribution
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from vnn.initializers import make_random_init, make_sigma2_bias_init
from ._modules import MVE, calc_mve_loss


class StepFn(Protocol):
    """Interface for step functions.

    Used for type hint.
    """

    def __call__(
        self, model: nn.Module, X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor: ...


class Regularizer(Protocol):
    """Interface for regularizers.

    Used for type hint.
    """

    def __call__(self, model: nn.Module) -> torch.Tensor: ...


def default_regularizer(model: nn.Module) -> torch.Tensor:
    return torch.tensor(0.0)


def make_step_fn(
    optimizer: Optimizer, regularizer: Regularizer, grad_max_norm: float = 1.0
) -> StepFn:
    """Creates step function.

    Parameters
    ----------
    optimizer : Optimizer
        Torch optimizer instance.

    regularizer : Regularizer
        Regularizer for mean subnetwork.

    grad_max_norm : float, default=1.0
        Clip the gradient norm.

    Returns
    -------
    step_fn : StepFn
    """

    def step_fn(model: MVE, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Updates the parameters given a batch of data and returns the loss."""
        optimizer.zero_grad()
        loss = calc_mve_loss(model(X), y) + regularizer(model.mean)
        loss.backward()
        clip_grad_norm_(model.mean.parameters(), max_norm=grad_max_norm)
        clip_grad_norm_(model.sigma2.parameters(), max_norm=grad_max_norm)
        optimizer.step()
        return loss

    return step_fn


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


def fit_loop(
    step_fn: StepFn,
    *,
    model: MVE,
    n_epochs: int,
    dataloader: DataLoader,
    scheduler: LRScheduler,
    disable_pbar: bool = False,
) -> None:
    """Generic training loop.

    Parameters
    ----------
    step_fn : StepFn
        A callable that performs a single optimization step. It must take
        `(model, X, y)` as input and return a scalar loss tensor.

    model : MVE
        The model to be trained. It is passed to `step_fn` at each batch.

    n_epochs : int
        Number of full passes over the dataset.

    dataloader : DataLoader
        An iterable that yields batches of input data `(X, y)`.

    scheduler : LRScheduler
        Learning rate scheduler to be stepped once per epoch.

    disable_pbar : bool, optional
        If True, disables the progress bar. Defaults to False.

    """
    pbar = tqdm(range(n_epochs), desc="Epoch", disable=disable_pbar)

    for _ in pbar:
        running_loss: float = 0.0
        n_batches = 0

        for X, y in dataloader:
            loss = step_fn(model, X, y)
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        scheduler.step()


def train(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    learning_rate: float = 1e-3,
    activation_fn: Type[nn.Module] = nn.Sigmoid,
    prior_distr: Distribution | None = None,
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

    grad_max_norm : float, default=1.0
        Maximum norm for gradient clipping.

    disable_pbar : bool, default=False
        If True, disables the progress bar.
    """

    weights_initializer = make_random_init(prior_distr)

    model = MVE(
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activation_fn=activation_fn,
        weights_initializer=weights_initializer,
    )

    if prior_distr is not None:
        regularizer = make_map_regularizer(prior_distr)
    else:
        regularizer = default_regularizer

    dataloader = DataLoader(TensorDataset(X, y), batch_size=X.shape[0], shuffle=True)

    # ----------------------
    # Stage 1: Warm-up
    # ----------------------
    # Train keeping variance parameters fixed.
    # Save computations by not updating variance subnetwork weights until
    # y(x) is somewhat close to f(x).
    model.sigma2.requires_grad_(False)
    warmup_optimizer = Adam(model.mean.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(warmup_optimizer, T_max=n_warmup_epochs)
    step_fn = make_step_fn(warmup_optimizer, regularizer, grad_max_norm)
    fit_loop(
        step_fn,
        model=model,
        n_epochs=n_warmup_epochs,
        dataloader=dataloader,
        scheduler=scheduler,
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
    model.sigma2.requires_grad_(True)
    full_optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(full_optimizer, T_max=n_remaining_epochs)
    step_fn = make_step_fn(full_optimizer, regularizer, grad_max_norm)
    fit_loop(
        step_fn,
        model=model,
        n_epochs=n_remaining_epochs,
        dataloader=dataloader,
        scheduler=scheduler,
        disable_pbar=disable_pbar,
    )

    return model
