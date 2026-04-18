"""
Plotting utilities.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyro.infer.mcmc import MCMC

from vnn.datasets import Dataset


def update_mpl_params(**params) -> None:
    if not params:
        params = DEFAULT_MPL_PARAMS
    mpl.rcParams.update(params)


DEFAULT_MPL_PARAMS = {
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
}


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
    dataset: Dataset,
    ax: plt.Axes | None = None,
    include_forward: bool = True,
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


def plot_mcmc(
    mcmc: MCMC,
    X: torch.Tensor,
    Y: torch.Tensor,
    n_plot_samples: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:

    if ax is None:
        ax = plt.subplot()

    X = X.detach()
    Y = Y.detach()
    samples = mcmc.get_samples(n_plot_samples)

    ax.scatter(X, Y, alpha=0.4, label="data")

    for i in range(n_plot_samples):
        W = {k: v[i, ...] for k, v in samples.items()}
        mean = torch.func.functional_call(mcmc.kernel.model.fward, W, X)
        mean = mean.flatten().numpy()
        plt.plot(X, mean, alpha=1.0, color="blue")

    ax.set_ylabel("y")
    return ax
