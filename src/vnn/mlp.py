from typing import Callable

import torch
from torch import nn
from torch.distributions import Distribution

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


def make_weights_initializer(
    distribution: Distribution,
) -> Callable[[nn.Module], None]:

    @torch.no_grad()
    def weights_initializer(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            m.weight.copy_(distribution.sample(sample_shape=m.weight.size()))
            m.bias.copy_(distribution.sample(sample_shape=m.bias.size()))

    return weights_initializer


class MLP(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: tuple[int, ...],
        n_features_out: int,
        hidden_activation_fn: Callable[[], ActivationFunction] = nn.Sigmoid,
        output_activation_fn: Callable[[], ActivationFunction] = nn.Identity,
        weights_distribution: Distribution | None = None,
    ):
        super().__init__()
        layer_sizes = (n_features_in,) + tuple(hidden_layer_sizes)
        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers += [nn.Linear(in_size, out_size), hidden_activation_fn()]
        layers += [nn.Linear(layer_sizes[-1], n_features_out), output_activation_fn()]

        self.sequential = nn.Sequential(*layers)

        if weights_distribution is not None:
            self.sequential.apply(make_weights_initializer(weights_distribution))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x)
