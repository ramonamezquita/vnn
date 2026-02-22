import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor

from vnn.ensemble import extract_ensemble_weights


def plot_weight_distributions(
    ensemble: BaggingRegressor,
    n_bins: int = 50,
) -> plt.Figure:
    """Plot weight distributions per subnetwork and layer, aggregated across ensemble.

    Parameters
    ----------
    ensemble : BaggingRegressor
        Fitted ensemble of MVE estimators.
    n_bins : int, default=50
        Number of histogram bins.

    Returns
    -------
    fig : plt.Figure
    """
    check_is_fitted(ensemble)
    weights = extract_ensemble_weights(ensemble)

    n_plots = len(weights)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, (key, w) in zip(axes, weights.items()):
        ax.hist(
            w,
            bins=n_bins,
            density=True,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(key)
        ax.set_xlabel("weight value")
        ax.set_ylabel("density")

        # Annotate summary stats
        ax.text(
            0.97,
            0.95,
            f"μ={w.mean():.3f}\nσ={w.std():.3f}\nmax|w|={np.abs(w).max():.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    fig.tight_layout()
    return fig
