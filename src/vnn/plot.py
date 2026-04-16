"""
Plotting utilities.
"""

import matplotlib.pyplot as plt
import numpy as np

from vnn.datasets import Dataset


def plot_95_ci(
    x: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ensemble predictions.

    Returns
    -------
    ax : plt.Axes
    """
    # Sort by `x`.
    idx = np.argsort(x)
    x = x[idx]
    mean = mean[idx]
    var = var[idx]

    lb = mean - 1.96 * np.sqrt(var)
    ub = mean + 1.96 * np.sqrt(var)
    if ax is None:
        ax = plt.subplot()

    ax.plot(x, mean, color="purple", label="Mean")
    ax.fill_between(x, lb, ub, color="purple", alpha=0.2, label="1.96 Std. dev.")
    return ax


def plot_dataset(
    dataset: Dataset, ax: plt.Axes | None = None, include_forward: bool = True
) -> plt.Axes:
    if ax is None:
        ax = plt.subplot()

    ax.plot(
        dataset.x,
        dataset.u_true(),
        linestyle="dotted",
        label="True u",
        color="red",
        linewidth=1.5,
    )

    if include_forward:
        ax.plot(
            dataset.x,
            dataset.u_forward(),
            linestyle="solid",
            linewidth=2.5,
            color="tab:blue",
            label="Blurred model",
        )
    return ax
