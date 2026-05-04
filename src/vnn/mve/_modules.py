from __future__ import annotations

from typing import Type

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.modules.loss import _Loss

from vnn.initializers import WeightsInitializer, zeros_init


class Exponential(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x.clamp(max=88.0)) + self.eps


class MLP(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: tuple[int, ...],
        n_features_out: int,
        hidden_activation_fn: Type[nn.Module] = nn.Tanh,
        output_activation_fn: Type[nn.Module] = nn.Identity,
        weights_initializer: WeightsInitializer | None = None,
    ):
        super().__init__()
        layer_sizes = (n_features_in,) + tuple(hidden_layer_sizes)
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), hidden_activation_fn()]
        layers += [nn.Linear(layer_sizes[-1], n_features_out), output_activation_fn()]

        self.sequential = nn.Sequential(*layers)

        if weights_initializer is not None:
            self.sequential.apply(weights_initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class MVE(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        hidden_activation_fn: Type[nn.Module] = nn.Tanh,
        weights_initializer: WeightsInitializer | None = None,
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


class MVELoss(_Loss):
    def __init__(self, eps: float = 1e-4, reduction: str = "mean"):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> torch.Tensor:
        mean = input[:, 0]
        var = input[:, 1]
        return F.gaussian_nll_loss(
            mean, target, var, reduction=self.reduction, eps=self.eps
        )
