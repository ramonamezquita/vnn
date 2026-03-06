from typing import Callable

import torch
from torch import nn

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


class MLP(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: tuple[int, ...],
        n_features_out: int,
        hidden_activation_fn: ActivationFunction = nn.Sigmoid(),
        output_activation_fn: ActivationFunction = nn.Identity(),
    ):
        super().__init__()

        layer_sizes = (n_features_in,) + tuple(hidden_layer_sizes)
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), hidden_activation_fn]
        layers += [nn.Linear(layer_sizes[-1], n_features_out), output_activation_fn]

        self.sequential = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)


class MVE(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes: tuple[int] = (100,),
        hidden_activation_fn: ActivationFunction = nn.Sigmoid(),
    ):
        super().__init__()
        self.mean = MLP(
            n_features_in=1,
            hidden_layer_sizes=hidden_layer_sizes,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
        )
        self.sigma2 = MLP(
            n_features_in=1,
            hidden_layer_sizes=hidden_layer_sizes,
            n_features_out=1,
            hidden_activation_fn=hidden_activation_fn,
            output_activation_fn=nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat((self.mean(x), self.sigma2(x)), axis=1)
