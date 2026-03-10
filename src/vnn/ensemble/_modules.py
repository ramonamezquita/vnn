from __future__ import annotations

from typing import Callable, Protocol

import torch
from torch import Tensor, nn

ActivationFunction = Callable[[torch.Tensor], torch.Tensor]


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


class MLP(nn.Module):
    def __init__(
        self,
        n_features_in: int,
        hidden_layer_sizes: tuple[int, ...],
        n_features_out: int,
        hidden_activation_fn: Callable[[], ActivationFunction] = nn.Sigmoid,
        output_activation_fn: Callable[[], ActivationFunction] = nn.Identity,
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
        hidden_activation_fn: Callable[[], ActivationFunction] = nn.Sigmoid,
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


###############################
# Custom Weights Initializers #
###############################


class WeightsInitializer(Protocol):
    def __call__(self, m: nn.Module) -> None: ...


class WeightsInitializerFactory(Protocol):
    def __call__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> WeightsInitializer: ...


def _no_grad_cauchy_(
    tensor: Tensor,
    median: float = 0,
    sigma: float = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    with torch.no_grad():
        return tensor.cauchy_(median, sigma, generator=generator)


def _no_grad_laplace_(tensor: Tensor, loc: float = 0.0, scale: float = 1.0) -> Tensor:
    dist = torch.distributions.Laplace(loc, scale)
    with torch.no_grad():
        tensor.copy_(dist.sample(tensor.shape))
    return tensor


def init_cauchy_(
    tensor: Tensor,
    median: float = 0,
    sigma: float = 1,
    generator: torch.Generator | None = None,
) -> Tensor:
    return _no_grad_cauchy_(tensor, median, sigma, generator)


def init_laplace_(
    tensor: Tensor,
    loc: float = 0.0,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    return _no_grad_laplace_(tensor, loc, scale)


def zeros_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        nn.init.zeros_(m.bias)


def make_gaussian_init(
    loc: float = 0.0, scale: float = 1.0, generator: torch.Generator | None = None
) -> WeightsInitializer:

    def init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, loc, scale, generator)
            nn.init.normal_(m.bias, loc, scale, generator)

    return init


def make_cauchy_init(
    loc: float = 0.0, scale: float = 1.0, generator: torch.Generator | None = None
) -> WeightsInitializer:
    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_cauchy_(m.weight, loc, scale, generator)
            init_cauchy_(m.bias, loc, scale, generator)

    return init


def make_laplace_init(
    loc: float = 0.0, scale: float = 1.0, generator: torch.Generator | None = None
) -> WeightsInitializer:
    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            init_laplace_(m.weight, loc, scale, generator)
            init_laplace_(m.bias, loc, scale, generator)

    return init


def make_sigma2_bias_init(val: float) -> WeightsInitializer:

    def init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear) and m.out_features == 1:
            nn.init.constant_(m.bias, val)

    return init


def get_weights_initializer(
    name: str,
    loc: float = 0.0,
    scale: float = 1.0,
    generator: torch.Generator | None = None,
) -> WeightsInitializer:
    try:
        return _NAME_TO_FACTORY[name](loc, scale, generator)
    except KeyError:
        raise ValueError(f"'{name}' weights initializer not known.")


_NAME_TO_FACTORY: dict[str, WeightsInitializerFactory] = {
    "gaussian": make_gaussian_init,
    "cauchy": make_cauchy_init,
    "laplace": make_laplace_init,
}
