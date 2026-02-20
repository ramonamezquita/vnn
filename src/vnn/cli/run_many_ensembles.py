import mlflow

from vnn.ensemble import aggregate_ensemble_metrics, get_default_args, run_ensemble

configs = [
    {"l1_penalty": 0.1},
    {"l1_penalty": 0.01},
    {"l1_penalty": 0.001},
    {"l1_penalty": 0.0001},
    {"l2_penalty": 0.1},
    {"l2_penalty": 0.01},
    {"l2_penalty": 0.001},
    {"l2_penalty": 0.0001},
]


def main():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    experiment = mlflow.get_experiment_by_name("deep-ensemble")
    with mlflow.start_run(
        run_name="ensemble_regularization", experiment_id=experiment.experiment_id
    ) as parent_run:
        for config in configs:
            run_name = "__".join([f"{k}_{v}" for k, v in config.items()])
            with mlflow.start_run(
                nested=True, run_name=run_name, experiment_id=experiment.experiment_id
            ) as child_run:
                args = get_default_args(plot=False, metrics=("rmse",))
                args.update(config)
                ensemble, predictions, statistics, fig = run_ensemble(**args)
                ensemble_metrics = aggregate_ensemble_metrics(ensemble)
                mlflow.log_params(args)
                mlflow.log_figure(fig, "figures/datasets.png")
                for step, metrics in enumerate(ensemble_metrics):
                    mlflow.log_metrics(metrics, step)


if __name__ == "__main__":
    main()
