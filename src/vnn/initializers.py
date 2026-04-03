from typing import Callable

import torch
from torch import nn
from torch.distributions import Distribution


def zeros_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


def make_sigma2_bias_init(val: float) -> Callable[[nn.Module], None]:

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.constant_(m.bias, val)

    return init


def make_random_init(distr: Distribution) -> Callable[[nn.Module], None]:

    @torch.no_grad()
    def weights_initializer_(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            m.weight.copy_(distr.expand(m.weight.size()).sample())
            m.bias.copy_(distr.expand(m.bias.size()).sample())

    return weights_initializer_
