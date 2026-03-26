import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from vnn.fit import StepFunction
from vnn.fit import fit as fit_loop
from vnn.initializers import WeightsInitializer
from vnn.regularizers import Regularizer, ZeroRegularizer

from ._modules import MVE, calc_mve_loss


def _make_sigma2_bias_init(val: float) -> WeightsInitializer:

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.constant_(m.bias, val)

    return init


def _make_step_fn(
    model: MVE, optimizer: Optimizer, grad_max_norm: float = 1.0
) -> StepFunction:

    def step_fn(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        optimizer.zero_grad()
        loss = calc_mve_loss(model(X), y)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)
        optimizer.step()
        return loss

    return step_fn


def fit(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    learning_rate: float = 1e-3,
    activation_fn: nn.Module = nn.Sigmoid,
    weights_initializer: WeightsInitializer | None = None,
    regularizer: Regularizer | None = None,
    grad_max_norm: float = 1.0,
    disable_pbar: bool = False,
) -> MVE:
    model = MVE(
        hidden_layer_sizes=hidden_layer_sizes,
        hidden_activation_fn=activation_fn,
        weights_initializer=weights_initializer,
    )
    regularizer = regularizer or ZeroRegularizer()

    dataloader = DataLoader(
        TensorDataset(X, y),
        batch_size=X.shape[0],
        shuffle=True,
    )

    # Stage 1: Train keeping variance parameters fixed (warm-up stage).
    # Save computations by not updating variance subnetwork weights until
    # y(x) is somewhat close to f(x).
    model.sigma2.requires_grad_(False)
    warmup_optimizer = Adam(model.mean.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(warmup_optimizer, T_max=n_warmup_epochs)
    fit_loop(
        _make_step_fn(model, warmup_optimizer, grad_max_norm),
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

    model.sigma2.apply(_make_sigma2_bias_init(logmse))

    # Stage 2: Train in full.
    # For subsequent training, all parameters (from both mean and sigma2 subnetworks)
    # are updated until the total number of epochs is reached.
    n_remaining_epochs = n_total_epochs - n_warmup_epochs
    model.sigma2.requires_grad_(True)
    full_optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(full_optimizer, T_max=n_remaining_epochs)
    fit_loop(
        _make_step_fn(model, full_optimizer, grad_max_norm),
        n_epochs=n_remaining_epochs,
        dataloader=dataloader,
        scheduler=scheduler,
        disable_pbar=disable_pbar,
    )

    return model
