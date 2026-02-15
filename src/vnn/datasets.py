"""
Synthetic datasets for regression experiments.
"""

from functools import cached_property
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

DeterministicFunction = Callable[[np.ndarray], np.ndarray]
StochasticFunction = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def make_gaussian_noise(scale: float = 1.0) -> StochasticFunction:
    def eps(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0, scale, size=len(x))

    return eps


def make_gaussian_filter(scale: float = 1.0) -> DeterministicFunction:
    def gaussian_filter(x: np.ndarray) -> np.ndarray:
        return gaussian_filter1d(x, scale)

    return gaussian_filter


def no_op(x: np.ndarray) -> np.ndarray:
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
        F: DeterministicFunction = no_op,
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

        indices = self.get_indices(n)
        x_obs = self.x[indices]
        y_obs = self.u_forward[indices] + self.eps(x_obs, self.rng)
        return x_obs, y_obs

    def plot(self, n: int | None = None, ax: plt.Axes | None = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.x, self.u_true, label="U (ground truth)")
        ax.plot(self.x, self.u_forward, label="F(U)")

        if n is not None:
            x_obs, y_obs = self.sample(n)
            ax.scatter(x_obs, y_obs, label="Samples")

        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        return ax


def gen_sinusoidal() -> Dataset:
    x = np.linspace(0, np.pi / 2, 128)

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
    x = np.linspace(-3, 3, 128)

    def U(x: np.ndarray) -> np.ndarray:
        return x**3

    return Dataset(x, U, eps=make_gaussian_noise(scale=3.0))


def gen_piecewise() -> Dataset:
    x = np.linspace(-1, 1, 128)

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
        x, U, F=make_gaussian_filter(scale=3.0), eps=make_gaussian_noise(0.05)
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
