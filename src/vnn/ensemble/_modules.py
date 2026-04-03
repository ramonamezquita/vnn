from __future__ import annotations

from typing import Type

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Distribution

from vnn.initializers import zeros_init
from vnn.mlp import MLP


def calc_mve_loss(
    input: torch.Tensor, target: torch.Tensor, eps: float = 1e-6, reduction="mean"
) -> torch.Tensor:
    return MVELoss(eps, reduction)(input, target)


class SoftplusWithEps(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.softplus = nn.Softplus()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softplus(x) + self.eps


class Exponential(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x.clamp(max=88.0)) + self.eps


class MVELoss(nn.Module):
    def __init__(self, eps: float = 1e-6, reduction="mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean = input[:, 0]
        var = input[:, 1]
        return F.gaussian_nll_loss(
            mean, target, var, eps=self.eps, reduction=self.reduction
        )


class MVE(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        hidden_activation_fn: Type[nn.Module] = nn.Sigmoid,
        weights_initializer: Distribution | None = None,
    ):
        super().__init__()
        self.mean = MLP(
            n_features_in=1,
            hidden_layer_sizes=hidden_layer_sizes,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            weights_initializer=weights_initializer,
        )
        self.sigma2 = MLP(
            n_features_in=1,
            hidden_layer_sizes=hidden_layer_sizes,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            output_activation_fn=Exponential,
            weights_initializer=zeros_init,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.mean(x), self.sigma2(x)), dim=1)
