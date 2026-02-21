import argparse

import mlflow

from vnn.ensemble import aggregate_ensemble_metrics, get_default_args, run_ensemble

configs = [
    {"l1_penalty": 1e-1},
    {"l1_penalty": 1e-2},
    {"l1_penalty": 1e-3},
    {"l1_penalty": 1e-4},
    {"l2_penalty": 1e-1},
    {"l2_penalty": 1e-2},
    {"l2_penalty": 1e-3},
    {"l2_penalty": 1e-4},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 1},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 5},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 10},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 15},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 20},
    {"cauchy_penalty": 1e-1, "cauchy_scale": 1},
    {"cauchy_penalty": 1e-2, "cauchy_scale": 1},
    {"cauchy_penalty": 1e-3, "cauchy_scale": 1},
    {"cauchy_penalty": 1e-4, "cauchy_scale": 1},
]


def create_parser() -> argparse.ArgumentParser:
    default_args = get_default_args()

    parser = argparse.ArgumentParser(
        description="Testing multiple regularization strategies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--activation_fn",
        type=str,
        default=default_args["activation_fn"],
        help="Hidden activation function.",
    )

    parser.add_argument(
        "--calc_input_gradient_at",
        default=default_args["calc_input_gradient_at"],
        nargs="*",
        type=float,
    )

    return parser


def main():
    parser = create_parser()
    parsed_args = parser.parse_args()

    run_args = get_default_args(
        plot=False,
        metrics=("rmse",),
        activation_fn=parsed_args.activation_fn,
        calc_input_gradient_at=parsed_args.calc_input_gradient_at,
    )

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.set_experiment("mve-regularization")

    for config in configs:
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            run_args.update(config)
            ensemble, predictions, statistics, fig = run_ensemble(**run_args)
            ensemble_metrics = aggregate_ensemble_metrics(ensemble)
            mlflow.log_params(run_args)
            mlflow.log_figure(fig, "figures/datasets.png")
            for step, metrics in enumerate(ensemble_metrics):
                mlflow.log_metrics(metrics, step)


if __name__ == "__main__":
    main()
