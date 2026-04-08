"""
Synthetic datasets for regression experiments.
"""

from typing import Callable

import torch
from scipy.ndimage import gaussian_filter1d


def make_gaussian_noise(scale: float = 1.0):
    """Factory function for Gaussian noise."""

    def gaussian_noise(x: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        generator = torch.Generator()

        if seed is not None:
            generator.manual_seed(seed)

        return scale * torch.randn(x.size(), generator=generator)

    return gaussian_noise


def make_gaussian_filter(scale: float = 1.0) -> Callable[[torch.Tensor], torch.Tensor]:
    """Factory function for Gaussian filters."""

    def gaussian_filter(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().numpy().flatten()
        x = gaussian_filter1d(x, scale)
        x = torch.as_tensor(x)
        return x

    return gaussian_filter


def identity(x: torch.Tensor) -> torch.Tensor:
    """Identity function."""
    return x


class Dataset:
    """Generic class for toy datasets.

    Parameters
    ----------
    x : torch.Tensor
        Function domain.

    U : Callable
        Ground truth function.

    F : Callable
        Forward operator.

    eps: Callable
        Noise function.

    rng: np.random.Generator or None, default=None
        If None, default numpy random generator is used.
    """

    def __init__(
        self,
        x: torch.Tensor,
        U: Callable,
        F: Callable = identity,
        noise: Callable | None = None,
    ):
        self.x = x
        self.U = U
        self.F = F
        if noise is None:
            noise = make_gaussian_noise()
        self.noise = noise

    def true(self, x: torch.Tensor) -> torch.Tensor:
        return self.U(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.F(self.U(x))

    def y(self, x: torch.Tensor | None = None, seed: int | None = None) -> torch.Tensor:
        x = self.x if x is None else x
        return self.forward(x) + self.noise(x, seed)

    def x_y(self, seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x, self.y(seed=seed)


# -------------------
# Dataset factories #
# -------------------


def cubic_poly() -> Dataset:
    x = torch.linspace(-3, 3, steps=128)
    noise = make_gaussian_noise(scale=3.0)
    return Dataset(x, U=lambda x: x**3, noise=noise)


def one_block() -> Dataset:

    def U(x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        out[x < -0.5] = -0.5
        out[(x >= -0.5) & (x < 0.5)] = 0.5
        out[(x >= 0.5) & (x < 0.75)] = -1.5
        out[(x >= 0.75) & (x <= 1)] = 0.0
        return out

    x = torch.linspace(-1, 1, 128)
    F = make_gaussian_filter(scale=2.0)
    noise = make_gaussian_noise(scale=0.05)
    return Dataset(x, U, F, noise)


_NAME_TO_DATASET = {"cubic_poly": cubic_poly, "one_block": one_block}


def get_avaliable_datasets() -> tuple[str]:
    return tuple(_NAME_TO_DATASET)


def get_x_y(name: str, seed: int = None) -> Dataset:
    ds = _NAME_TO_DATASET[name]()
    return ds.x_y(seed=seed)
