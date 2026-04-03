"""
Synthetic datasets for regression experiments.
"""

from typing import Callable, Protocol

import numpy as np
from scipy.ndimage import gaussian_filter1d

DeterministicFunction = Callable[[np.ndarray], np.ndarray]
StochasticFunction = Callable[[np.ndarray, np.random.Generator], np.ndarray]
NUMPY_DTYPE = np.float32


__all__ = ("get_avaliable_datasets", "get_dataset")


def make_gaussian_noise(scale: float = 1.0) -> StochasticFunction:
    """Factory function for Gaussian noise."""

    def eps(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0, scale, size=len(x))

    return eps


def make_gaussian_filter(scale: float = 1.0) -> DeterministicFunction:
    """Factory function for Gaussian filters."""

    def gaussian_filter(x: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(x, scale)

    return gaussian_filter


def identity(x: np.ndarray) -> np.ndarray:
    """Identity function."""
    return x


class Dataset:
    """Generic class for toy datasets.

    Parameters
    ----------
    x : np.ndarray
        Function domain.

    U : DeterministicFunction
        Ground truth function.

    F : DeterministicFunction
        Forward operator.

    eps: StochasticFunction
        Noise function.

    rng: np.random.Generator or None, default=None
        If None, default numpy random generator is used.
    """

    def __init__(
        self,
        U: DeterministicFunction,
        F: DeterministicFunction = identity,
        sigma: float = 1.0,
    ):
        self.U = U
        self.F = F
        self.sigma = sigma

    def true(self, x: np.ndarray) -> np.ndarray:
        return self.U(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.F(self.U(x))

    def y(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        eps = make_gaussian_noise(self.sigma)
        return self.forward(x) + eps(x, rng)

    # def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    #     if n > len(self):
    #         raise ValueError(f"`n` is greater than Dataset size ({len(self)}).")

    #     rng = np.random.default_rng(seed=seed)
    #     indices = sorted(rng.choice(len(self.x), n, replace=False))
    #     x_obs = self.x[indices]
    #     y_obs = self.u_forward()[indices] + self.eps(x_obs, rng)
    #     return x_obs, y_obs


# -------------------
# Dataset factories #
# -------------------


class DatasetFactory(Protocol):
    def __call__(self) -> Dataset: ...


def make_cubic() -> Dataset:

    def U(x: np.ndarray) -> np.ndarray:
        return x**3

    return Dataset(U, sigma=3.0)


def make_1d_block() -> Dataset:

    def U(x):
        condlist = [
            x < -0.5,
            (-0.5 <= x) & (x < 0.5),
            (0.5 <= x) & (x < 0.75),
            (0.75 <= x) & (x <= 1),
        ]
        funclist = [-0.5, 0.5, -1.5, 0]

        return np.piecewise(x, condlist, funclist)

    return Dataset(U=U, F=make_gaussian_filter(scale=2.0), sigma=0.05)


def make_1d_many_blocks() -> Dataset:
    x = np.linspace(-1, 1, 128, dtype=NUMPY_DTYPE)

    knots = np.array([-1.0, -0.75, -0.40, -0.10, 0.20, 0.55, 0.80, 1.0])
    heights = np.array([-0.8, 0.5, -1.2, 0.9, -0.4, 1.1, 0.2])

    def U(x):
        condlist = [(knots[i] <= x) & (x < knots[i + 1]) for i in range(len(knots) - 2)]
        condlist.append((knots[-2] <= x) & (x <= knots[-1]))  # include right edge

        return np.piecewise(x, condlist, heights)

    return Dataset(x=x, U=U, F=make_gaussian_filter(scale=2.0), sigma=0.05)


_NAME_TO_FACTORY: dict[str, DatasetFactory] = {
    "cubic": make_cubic,
    "1d_block": make_1d_block,
    "1d_many_blocks": make_1d_many_blocks,
}


def get_avaliable_datasets() -> tuple[str]:
    return tuple(_NAME_TO_FACTORY)


def get_dataset(name: str) -> Dataset:
    try:
        return _NAME_TO_FACTORY[name]()
    except KeyError:
        raise KeyError("Dataset {name} not known.")
