import inspect
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor

from vnn.mve import MVE, History, MVEModule


def get_default_args(**kwargs) -> dict[str, Any]:
    signature = inspect.signature(fit_ensemble)
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_args.update(kwargs)
    return default_args


def collect_ensemble_weights(ensemble: BaggingRegressor) -> dict[str, np.ndarray]:
    weights: dict[str, list[np.ndarray]] = {}
    for estimator in ensemble.estimators_:
        model: MVEModule = estimator.model_
        for subnet_name in ("mean", "sigma2"):
            subnetwork = getattr(model, subnet_name)
            w = torch.cat(
                [p.detach().flatten() for p in subnetwork.parameters()]
            ).numpy()
            weights.setdefault(subnet_name, []).append(w)

    # Concatenate across estimators
    weights = {k: np.concatenate(v) for k, v in weights.items()}
    return weights


def aggregate_ensemble_metrics(
    ensemble: BaggingRegressor,
) -> list[dict[str, float]]:
    check_is_fitted(ensemble)
    histories: list[History] = [est.history_ for est in ensemble.estimators_]
    metric_names = histories[0].metrics
    n_epochs = len(histories[0])

    aggregated: list[dict[str, float]] = []

    for epoch in range(n_epochs):
        epoch_dict: dict[str, float] = {}

        for metric in metric_names:
            values = [h[epoch][metric] for h in histories]
            epoch_dict[metric] = float(np.mean(values))

        aggregated.append(epoch_dict)

    return aggregated


def fit_ensemble(
    X: torch.Tensor,
    y: torch.Tensor,
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_jobs: int = 4,
    hidden_layer_sizes: tuple[int] = (100,),
    max_samples: float | int | None = 1.0,
    bootstrap: bool = False,
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    l2_penalty: float = 0.0,
    l1_penalty: float = 0.0,
    cauchy_penalty: float = 0.0,
    cauchy_scale: float = 1,
    activation_fn: str = "sigmoid",
    metrics: tuple[str] = (),
    calc_input_gradient_at: tuple[float] = (),
) -> BaggingRegressor:
    """Train and evaluate ensemble.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of estimators in the ensemble

    n_total_epochs: int = 10000
        Number of total epochs.

    n_warmup_epochs: int = 5000
        Number of warmup epochs.

    n_jobs: int, default=4
        The number of jobs to run in parallel for both fit and predict.

    hidden_layer_sizes : tuple of int, default=(100,)
        Number of units in each hidden layer.

    learning_rate: float = 1e-3
        Learning rate.

    activation_fn : str, {"sigmoid", "relu"}, default="sigmoid"
        Activation function for hidden layer.

    l2_penalty : float, default=0.0
        L2 regularization factor.

    l1_penalty : float, default=0.0
        L1 regularization factor.


    References
    ----------
    [1] Balaji L., Alexander P. and Charles B.,
    "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles".
    https://arxiv.org/abs/1612.01474
    """
    base_regressor = MVE(
        hidden_layer_sizes=hidden_layer_sizes,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        l2_penalty=l2_penalty,
        l1_penalty=l1_penalty,
        cauchy_penalty=cauchy_penalty,
        cauchy_scale=cauchy_scale,
        metrics=metrics,
        calc_input_gradient_at=calc_input_gradient_at,
    )

    ensemble = BaggingRegressor(
        base_regressor,
        n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    ensemble.fit(X, y)
    return ensemble


def predict_ensemble(ensemble: BaggingRegressor, X: torch.Tensor) -> np.ndarray:
    n_estimators = ensemble.n_estimators
    predictions = torch.stack(
        [ensemble.estimators_[i].predict(X) for i in range(n_estimators)], axis=0
    )
    predictions = predictions.detach().numpy()
    return predictions


def plot_ensemble(
    x_obs: np.ndarray,
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    means = y_pred[:, :, 0]
    mean = means.mean(axis=0)
    vars_ = y_pred[:, :, 1]
    var = (vars_ + np.square(means)).mean(axis=0) - np.square(mean)
    lb = mean - 1.96 * np.sqrt(var)
    ub = mean + 1.96 * np.sqrt(var)

    if ax is None:
        ax = plt.subplot()

    ax.scatter(x_obs, y_obs, marker="x", color="black", s=3, label="Observations")
    ax.plot(x_obs, mean, color="purple", label="Network output")
    ax.fill_between(x_obs, lb, ub, color="purple", alpha=0.2, label="1.96 Std. dev.")
    return ax


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
    weights = collect_ensemble_weights(ensemble)

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
