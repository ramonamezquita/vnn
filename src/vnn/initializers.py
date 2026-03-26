from typing import Protocol

import torch
from torch import Tensor, nn


class WeightsInitializer(Protocol):
    def __call__(self, m: nn.Module) -> None: ...


class WeightsInitializerFactory(Protocol):
    def __call__(
        self,
        loc: float = 0.0,
        scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> WeightsInitializer: ...


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


_NAME_TO_FACTORY: dict[str, WeightsInitializerFactory] = {
    "gaussian": make_gaussian_init,
    "cauchy": make_cauchy_init,
    "laplace": make_laplace_init,
}
