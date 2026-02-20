import inspect
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import check_is_fitted
from sklearn.ensemble import BaggingRegressor

from vnn.datasets import get_dataset, plot_dataset
from vnn.mve import MVE, History


def get_default_args(**kwargs) -> dict[str, Any]:
    signature = inspect.signature(run_ensemble)
    default_args = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    default_args.update(kwargs)
    return default_args


def fit_ensemble(
    X: torch.Tensor,
    y: torch.Tensor,
    n_estimators: int,
    n_total_epochs: int,
    n_warmup_epochs: int,
    n_hidden_units: int,
    n_jobs: int,
    max_samples: float | int | None,
    bootstrap: bool,
    random_state: int,
    verbose: int,
    learning_rate: float,
    l2_penalty: float,
    l1_penalty: float,
    cauchy_penalty: float,
    cauchy_scale: float,
    activation_fn: str,
    metrics: tuple[str],
) -> None:
    """Train a Deep Ensemble of mean-variance estimation (MVE) networks."""
    base_regressor = MVE(
        n_total_epochs=n_total_epochs,
        n_hidden_units=n_hidden_units,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
        l2_penalty=l2_penalty,
        l1_penalty=l1_penalty,
        cauchy_penalty=cauchy_penalty,
        cauchy_scale=cauchy_scale,
        metrics=metrics,
    )

    regr = BaggingRegressor(
        base_regressor,
        n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    regr.fit(X, y)
    return regr


def evaluate_ensemble(
    ensemble: BaggingRegressor, X: torch.Tensor
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Evaluates ensemble."""
    check_is_fitted(ensemble)
    n_estimators = ensemble.n_estimators
    predictions = torch.stack(
        [ensemble.estimators_[i].predict(X) for i in range(n_estimators)], axis=0
    )
    predictions = predictions.detach().numpy()
    means = predictions[:, :, 0]
    vars_ = predictions[:, :, 1]
    mean = means.mean(axis=0)
    var = (vars_ + np.square(means)).mean(axis=0) - np.square(mean)
    lb = mean - 1.96 * np.sqrt(var)
    ub = mean + 1.96 * np.sqrt(var)
    return predictions, {"mean": mean, "var": var, "lb": lb, "ub": ub}


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


def run_ensemble(
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_hidden_units: int = 100,
    n_jobs: int = 4,
    n_samples: int = 100,
    max_samples: float | int | None = 1.0,
    bootstrap: bool = False,
    random_state: int = 42,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    l2_penalty: float = 0.0,
    l1_penalty: float = 0.0,
    cauchy_penalty: float = 0.0,
    cauchy_scale: float = 0.1,
    activation_fn: str = "sigmoid",
    dataset: str = "piecewise",
    plot: bool = False,
    metrics: tuple[str] = (),
):
    """Train and evaluate ensemble.

    References
    ----------
    [1] Balaji L., Alexander P. and Charles B., "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles". https://arxiv.org/abs/1612.01474
    """

    ds = get_dataset(dataset)
    x_obs, y_obs = ds.sample(n_samples)
    x_obs, y_obs = map(torch.from_numpy, (x_obs, y_obs))
    x_obs_2d = x_obs.reshape(-1, 1)

    ensemble = fit_ensemble(
        x_obs_2d,
        y_obs,
        n_estimators=n_estimators,
        n_total_epochs=n_total_epochs,
        n_warmup_epochs=n_warmup_epochs,
        n_hidden_units=n_hidden_units,
        n_jobs=n_jobs,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=verbose,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        l1_penalty=l1_penalty,
        cauchy_penalty=cauchy_penalty,
        cauchy_scale=cauchy_scale,
        activation_fn=activation_fn,
        metrics=metrics,
    )

    predictions, statistics = evaluate_ensemble(ensemble, x_obs_2d)

    fig, ax = plt.subplots()
    ax = plot_dataset(
        x_full=ds.x,
        u_true=ds.u_true,
        u_forward=ds.u_forward,
        x_obs=x_obs,
        y_obs=y_obs,
        y_pred=statistics["mean"],
        lb=statistics["lb"],
        ub=statistics["ub"],
        ax=ax,
    )
    plt.tight_layout()
    plt.legend()
    if plot:
        plt.show()

    return ensemble, predictions, statistics, fig
