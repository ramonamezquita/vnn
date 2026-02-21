"""
Synthetic datasets for regression experiments.
"""

from functools import cached_property
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

DeterministicFunction = Callable[[np.ndarray], np.ndarray]
StochasticFunction = Callable[[np.ndarray, np.random.Generator], np.ndarray]


NUMPY_DTYPE = np.float32
TORCH_DTYPE = torch.float32


def plot_dataset(x_full, x_obs, u_true, u_forward, y_obs, y_pred, lb, ub, ax=None):
    if ax is None:
        ax = plt.subplot()

    # Ground truth.
    ax.plot(x_full, u_true, linestyle="dotted", label="Truth u", color="maroon")

    # Forward model.
    ax.plot(
        x_full,
        u_forward,
        linestyle="solid",
        color="cornflowerblue",
        label="Blurred model",
    )

    # Observations.
    ax.scatter(x_obs, y_obs, marker="x", color="black", s=3)

    # Predictions.
    ax.plot(x_obs, y_pred, color="purple", label="Network output")

    # Uncertainty.
    ax.fill_between(
        x_obs,
        lb,
        ub,
        color="purple",
        alpha=0.2,
        label="1.96 Std. dev.",
    )

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Model vs Ground Truth", fontsize=13)

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True)
    ax.set_axisbelow(True)
    return ax


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
        x: np.ndarray,
        U: DeterministicFunction,
        F: DeterministicFunction = identity,
        eps: StochasticFunction = make_gaussian_noise(),
        rng: np.random.Generator | None = None,
    ):
        self.x = x
        self.U = U
        self.F = F
        self.eps = eps
        self.rng = rng or np.random.default_rng()

    def __len__(self) -> int:
        return len(self.x)

    @cached_property
    def u_true(self) -> np.ndarray:
        return self.U(self.x)

    @cached_property
    def u_forward(self) -> np.ndarray:
        return self.F(self.u_true)

    def get_indices(self, n: int) -> np.ndarray:
        return self.rng.choice(len(self.x), n, replace=False)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n > len(self):
            raise ValueError(f"`n` is greater than Dataset size ({len(self)}).")

        indices = sorted(self.get_indices(n))
        x_obs = self.x[indices]
        y_obs = self.u_forward[indices] + self.eps(x_obs, self.rng)
        return x_obs, y_obs


def gen_sinusoidal() -> Dataset:
    x = np.linspace(0, np.pi / 2, 128, dtype=NUMPY_DTYPE)

    w_c = 5
    w_m = 4

    def U(x: np.ndarray) -> np.ndarray:
        m_x = np.sin(w_m * x)
        return m_x * np.sin(w_c * x)

    def eps(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        m_x = np.sin(w_m * x)
        sigma2 = 0.02 + 0.02 * (1 - m_x) ** 2
        return rng.normal(0, np.sqrt(sigma2))

    return Dataset(x, U, eps=eps)


def gen_cubic() -> Dataset:
    x = np.linspace(-3, 3, 128, dtype=NUMPY_DTYPE)

    def U(x: np.ndarray) -> np.ndarray:
        return x**3

    return Dataset(x, U, eps=make_gaussian_noise(scale=3.0))


def gen_piecewise() -> Dataset:
    x = np.linspace(-1, 1, 128, dtype=NUMPY_DTYPE)

    def U(x):
        return np.piecewise(
            x,
            [
                x < -0.5,
                (-0.5 <= x) & (x < 0.5),
                (0.5 <= x) & (x < 0.75),
                (0.75 <= x) & (x <= 1),
            ],
            [-0.5, 0.5, -1, 0],
        )

    return Dataset(
        x, U, F=make_gaussian_filter(scale=2.0), eps=make_gaussian_noise(0.05)
    )


def get_dataset(name: str) -> Dataset:
    name_to_gen = {
        "cubic": gen_cubic,
        "sinusoidal": gen_sinusoidal,
        "piecewise": gen_piecewise,
    }
    try:
        return name_to_gen[name]()
    except KeyError:
        raise KeyError("Dataset {name} not known.")
