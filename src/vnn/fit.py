from typing import Protocol

import torch
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm


class StepFunction(Protocol):
    """Updates the parameters given a batch of data and returns the loss."""

    def __call__(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor: ...


def fit(
    step_fn: StepFunction,
    n_epochs: int,
    dataloader: DataLoader,
    scheduler: LRScheduler,
    disable_pbar: bool = False,
) -> None:
    """Generic training loop."""
    pbar = tqdm(range(n_epochs), desc="Epoch", disable=disable_pbar)

    for _ in pbar:
        running_loss: float = 0.0
        n_batches = 0

        for X, y in dataloader:
            loss = step_fn(X, y)
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / n_batches
        pbar.set_postfix(loss=f"{avg_loss:.4f}")
        scheduler.step()
