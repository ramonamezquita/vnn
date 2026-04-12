from typing import Protocol

import torch
from torch import nn


class Regularizer(Protocol):
    def __call__(self, m: nn.Module) -> torch.Tensor: ...


def no_op_regularizer(m: nn.Module) -> torch.Tensor:
    return torch.tensor(0.0)
