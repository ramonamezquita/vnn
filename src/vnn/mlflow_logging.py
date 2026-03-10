import io
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from vnn.ensemble import RunResult, TrainingIO


def log_to_mlflow(run_result: RunResult, params: dict[str, Any]) -> None:
    """Log all experiment artifacts, metrics and parameters to MLflow.

    Parameters
    ----------
    run_result : RunResult
        Result from `run()` containing training/test IO and model outputs.

    params : dict
        Parameters to log ,
    """
    _log_params(params)
    _log_io(run_result.tr_io, split="train")
    
    _log_epoch_metrics(run_result.metrics)
    _log_weights(run_result.weights)

    if run_result.te_io is not None:
        _log_io(run_result.te_io, split="test")
        _log_test_rmse(run_result.te_io)


# ----------------
# Private helpers
# ----------------


def _log_params(params: dict) -> None:
    mlflow.log_params(params)


def _log_io(io_data: TrainingIO, split: str) -> None:
    data = {
        "X": io_data.X,
        "y": io_data.y,
        "mean": io_data.mean,
        "var": io_data.var,
    }
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    mlflow.log_text(buffer.getvalue(), f"{split}_data.csv")


def _log_epoch_metrics(metrics: list[dict[str, float]]) -> None:
    for step, epoch_metrics in enumerate(metrics):
        mlflow.log_metrics(epoch_metrics, step=step)


def _log_test_rmse(te_io: TrainingIO) -> None:
    test_rmse = root_mean_squared_error(te_io.y, te_io.mean)
    mlflow.log_metric("test_rmse", test_rmse)


def _log_weights(weights: np.ndarray) -> None:
    thresholds = np.linspace(0.0, 15, 1500)
    w_abs = np.abs(weights)
    survival = np.array([(w_abs > t).mean() for t in thresholds])
    data = {"threshold": thresholds, "survival_prob": survival}
    df = pd.DataFrame(data)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    mlflow.log_text(buffer.getvalue(), "weight_survival.csv")
