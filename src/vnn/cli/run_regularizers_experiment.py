import argparse
import io

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

from vnn.datasets import get_dataset
from vnn.ensemble import (
    fit_ensemble,
    get_default_args,
    get_ensemble_metrics,
    get_ensemble_weights,
    plot_ensemble,
    predict_ensemble,
)

SEEDS = [0, 1, 2, 3, 4]

configs = [
    # --- No regularizer ---
    {"reg_type": "none"},
    # --- L2 ---
    {"reg_type": "l2", "l2_penalty": 1e-2},
    {"reg_type": "l2", "l2_penalty": 1e-3},
    {"reg_type": "l2", "l2_penalty": 1e-4},
    # --- L1 ---
    {"reg_type": "l1", "l1_penalty": 1e-2},
    {"reg_type": "l1", "l1_penalty": 1e-3},
    # --- Cauchy (vary penalty) ---
    {"reg_type": "cauchy", "cauchy_penalty": 1e-1, "cauchy_scale": 1},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-2, "cauchy_scale": 1},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 1},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-4, "cauchy_scale": 1},
    # --- Cauchy (vary scale) ---
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 0.1},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 0.5},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 1},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 5},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 10},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 15},
    {"reg_type": "cauchy", "cauchy_penalty": 1e-3, "cauchy_scale": 20},
]


def create_parser() -> argparse.ArgumentParser:
    default_args = get_default_args()

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

    return parser


def main():
    parser = create_parser()
    parsed_args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("run_regularizers_experiment")

    for config in configs:
        reg_type = config.pop("reg_type")
        for seed in SEEDS:
            run_name = f"{reg_type}_seed_{seed}"
            with mlflow.start_run(
                experiment_id=experiment.experiment_id, run_name=run_name
            ):
                fit_args = get_default_args(
                    metrics=("rmse",),
                    activation_fn=parsed_args.activation_fn,
                    calc_input_gradient_at=parsed_args.calc_input_gradient_at,
                    hidden_layer_sizes=parsed_args.hidden_layer_sizes,
                    random_state=seed,
                    **config,
                )

                dataset = get_dataset(parsed_args.dataset)
                x_obs, y_obs = dataset.sample(parsed_args.n_samples)
                x_obs, y_obs = map(torch.from_numpy, (x_obs, y_obs))
                x_obs_2d = x_obs.reshape(-1, 1)

                # ---- Fit ----
                ensemble = fit_ensemble(X=x_obs_2d, y=y_obs, **fit_args)
                metrics = get_ensemble_metrics(ensemble)
                weights = get_ensemble_weights(ensemble)

                # ---- Predictions ----
                y_pred = predict_ensemble(ensemble, x_obs_2d)

                # ---- Figures ----
                fig, ax = plt.subplots()
                plot_ensemble(x_obs, y_obs, y_pred, ax=ax)
                dataset.plot(ax)
                ax.set_xlabel("x", fontsize=12)
                ax.set_ylabel("y", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.3)
                ax.legend(frameon=True)
                ax.set_axisbelow(True)
                plt.tight_layout()
                plt.legend()

                # ---- Log scalar params ----
                mlflow.log_params(fit_args)
                mlflow.log_param("seed", seed)

                # ---- Log training metrics ----
                for step, epoch_metrics in enumerate(metrics):
                    mlflow.log_metrics(epoch_metrics, step=step)

                # ---- Log figure ----
                mlflow.log_figure(fig, f"figures/seed_{seed}.png")
                plt.close(fig)

                # =====================================================
                # Weight Survival Curve Logging (CSV Artifact)
                # =====================================================

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
                            "seed": seed,
                            "subnet": subnet_name,
                        }
                    )

                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)

                    artifact_name = (
                        f"weight_survival/{subnet_name}/seed_{seed}.csv"
                    )

                    mlflow.log_text(csv_buffer.getvalue(), artifact_name)


if __name__ == "__main__":
    main()
