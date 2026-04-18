from typing import Type

import torch
from torch import nn

from vnn.initializers import WeightsInitializer


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
