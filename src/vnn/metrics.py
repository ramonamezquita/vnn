import torch
from sklearn import metrics as sk_metrics


def calc_metrics(
    y_true: torch.Tensor, y_pred: torch.Tensor, metrics: tuple[str]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name in metrics:
        metrics[metric_name] = get_metric(metric_name)(y_true, y_pred)
    return metrics


def get_metric(metric_name: str):
    name_to_fn = {"rmse": root_mean_squared_error}
    return name_to_fn[metric_name]


def root_mean_squared_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true = y_true.detach().numpy().flatten()
    y_pred = y_pred.detach().numpy().flatten()
    return sk_metrics.root_mean_squared_error(y_true, y_pred)
