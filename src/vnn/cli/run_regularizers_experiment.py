import argparse
import io
import multiprocessing as mp

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from vnn.datasets import get_dataset
from vnn.ensemble import (
    fit,
    get_fit_default_args,
    get_metrics,
    get_weights,
    plot,
    predict,
)

reg_configs = [
    # --- No regularizer ---
    {
        "reg_type": "none",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    # --- L2 ---
    {
        "reg_type": "l2",
        "l1_penalty": 0.0,
        "l2_penalty": 1e-2,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    {
        "reg_type": "l2",
        "l1_penalty": 0.0,
        "l2_penalty": 1e-3,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    {
        "reg_type": "l2",
        "l1_penalty": 0.0,
        "l2_penalty": 1e-4,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    # --- L1 ---
    {
        "reg_type": "l1",
        "l1_penalty": 1e-2,
        "l2_penalty": 0.0,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    {
        "reg_type": "l1",
        "l1_penalty": 1e-3,
        "l2_penalty": 0.0,
        "cauchy_penalty": 0.0,
        "cauchy_scale": 0.0,
    },
    # --- Cauchy (vary penalty) ---
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-1,
        "cauchy_scale": 1,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-2,
        "cauchy_scale": 1,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 1,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-4,
        "cauchy_scale": 1,
    },
    # --- Cauchy (vary scale) ---
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 0.1,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 0.5,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 1,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 5,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 10,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 15,
    },
    {
        "reg_type": "cauchy",
        "l1_penalty": 0.0,
        "l2_penalty": 0.0,
        "cauchy_penalty": 1e-3,
        "cauchy_scale": 20,
    },
]


def create_parser() -> argparse.ArgumentParser:
    default_args = get_fit_default_args()

    parser = argparse.ArgumentParser(
        description="Multiple regularization experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hidden_layer_sizes",
        default=default_args["hidden_layer_sizes"],
        nargs="*",
        type=int,
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default=default_args["activation_fn"],
    )
    parser.add_argument(
        "--calc_input_gradient_at",
        default=default_args["calc_input_gradient_at"],
        nargs="*",
        type=float,
    )
    parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
        help="Number of samples/observations to train with.",
    )
    parser.add_argument(
        "--dataset",
        default="piecewise",
        type=str,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--n_workers",
        default=4,
        type=int,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--test_size",
        default=0.2,
        type=float,
        help="Test size",
    )

    return parser


def make_configs(parsed_args: argparse.Namespace) -> list[dict]:
    seeds = [0, 1, 2, 3, 4]

    base = {
        "activation_fn": parsed_args.activation_fn,
        "calc_input_gradient_at": parsed_args.calc_input_gradient_at,
        "hidden_layer_sizes": parsed_args.hidden_layer_sizes,
        "n_samples": parsed_args.n_samples,
        "dataset": parsed_args.dataset,
        "test_size": parsed_args.test_size,
    }

    all_configs = []
    for reg_cfg in reg_configs:
        for seed in seeds:
            cfg = {**base, **reg_cfg, "seed": seed}
            reg_parts = "__".join(f"{k}={v}" for k, v in reg_cfg.items())
            cfg["run_name"] = f"{reg_parts}_seed={seed}"
            all_configs.append(cfg)

    return all_configs


def train_with_config(config: dict) -> None:
    # Set tracking URI in each process (required for spawn method)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("run_regularizers_experiment_multiprocessing")

    with mlflow.start_run(run_name=config["run_name"]):
        fit_args = get_fit_default_args(
            metrics=("rmse",),
            activation_fn=config["activation_fn"],
            calc_input_gradient_at=config["calc_input_gradient_at"],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            random_state=config["seed"],
            l2_penalty=config["l2_penalty"],
            l1_penalty=config["l1_penalty"],
            cauchy_penalty=config["cauchy_penalty"],
            cauchy_scale=config["cauchy_scale"],
        )

        dataset = get_dataset(config["dataset"])
        x, y = dataset.sample(config["n_samples"])
        X = x.reshape(-1, 1)

        # ---- Train/Test split ----
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["seed"]
        )
        X_tr, X_te, y_tr = map(torch.from_numpy, (X_tr, X_te, y_tr))

        # ---- Fit ----
        ensemble = fit(X=X_tr, y=y_tr, **fit_args)
        metrics = get_metrics(ensemble)
        weights = get_weights(ensemble)

        # ---- Predictions ----
        mean_tr, var_tr = predict(ensemble, X_tr)
        mean_te, var_te = predict(ensemble, X_te)

        # --- To Numpy ---
        X_tr = X_tr.detach().numpy().flatten()
        y_tr = y_tr.detach().numpy()
        X_te = X_te.detach().numpy()

        # ---- Figures ----
        fig, ax = plt.subplots()
        plot(X_tr, y_tr, mean_tr, var_tr, ax=ax)
        dataset.plot(ax)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=True)
        ax.set_axisbelow(True)
        plt.tight_layout()

        # ---- Log scalar params ----
        mlflow.log_params(fit_args)
        mlflow.log_param("seed", config["seed"])

        # ---- Log training metrics ----
        for step, epoch_metrics in enumerate(metrics):
            mlflow.log_metrics(epoch_metrics, step=step)

        # ---- Log test metrics ----
        test_rmse = root_mean_squared_error(y_te, mean_te)
        mlflow.log_metric("test_rmse", test_rmse)

        # ---- Log figure ----
        mlflow.log_figure(fig, f"figures/seed_{config['seed']}.png")
        plt.close(fig)

        # ---- Log weights distribution ----
        reg_type = config["reg_type"]

        for subnet_name, w in weights.items():
            w_abs = np.abs(w)
            max_w = w_abs.max()
            upper = max(2.0, max_w * 1.05)
            thresholds = np.linspace(0.0, upper, 100)
            survival = np.array([(w_abs > t).mean() for t in thresholds])

            df = pd.DataFrame(
                {
                    "threshold": thresholds,
                    "survival_prob": survival,
                    "reg_type": reg_type,
                    "seed": config["seed"],
                    "subnet": subnet_name,
                }
            )

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            artifact_name = f"weight_survival/{subnet_name}/seed_{config['seed']}.csv"
            mlflow.log_text(csv_buffer.getvalue(), artifact_name)


def main():
    parser = create_parser()
    parsed_args = parser.parse_args()
    all_configs = make_configs(parsed_args)

    with mp.Pool(processes=8) as pool:
        results = pool.map(train_with_config, all_configs)

    print(f"Completed {len(results)} experiments")


if __name__ == "__main__":
    main()
