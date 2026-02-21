import argparse

from vnn.ensemble import aggregate_ensemble_metrics, get_default_args, run_ensemble


def create_parser() -> argparse.ArgumentParser:
    default_args = get_default_args()

    parser = argparse.ArgumentParser(
        description="Train a Deep Ensemble of mean-variance estimation (MVE) networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Random seed.",
        default=default_args["random_state"],
    )

    parser.add_argument(
        "--n_total_epochs",
        type=int,
        help="Number of epochs.",
        default=default_args["n_total_epochs"],
    )

    parser.add_argument(
        "--n_warmup_epochs",
        type=int,
        help="Number of warmup epochs.",
        default=default_args["n_warmup_epochs"],
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of models.",
        default=default_args["n_estimators"],
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs.",
        default=default_args["n_jobs"],
    )

    parser.add_argument(
        "--n_hidden_units",
        default=default_args["n_hidden_units"],
        type=int,
        help="Size of the hidden layer.",
    )
    parser.add_argument(
        "--n_samples",
        default=default_args["n_samples"],
        type=int,
        help="Number of samples/observations to train with.",
    )
    parser.add_argument(
        "--activation_fn",
        type=str,
        default=default_args["activation_fn"],
        help="Hidden activation function.",
    )
    parser.add_argument(
        "--max_samples",
        help="The number of samples to draw from X to train each estimator",
        type=float,
        default=default_args["max_samples"],
    )

    parser.add_argument(
        "--learning_rate",
        default=default_args["learning_rate"],
        type=float,
        help="Learning rate.",
    )
    parser.add_argument(
        "--l2_penalty",
        default=default_args["l2_penalty"],
        type=float,
        help="L2 penalty.",
    )

    parser.add_argument(
        "--l1_penalty",
        default=default_args["l1_penalty"],
        type=float,
        help="L1 penalty.",
    )
    parser.add_argument(
        "--cauchy_penalty",
        default=default_args["cauchy_penalty"],
        type=float,
        help="Cauchy penalty.",
    )
    parser.add_argument(
        "--cauchy_scale",
        default=default_args["cauchy_scale"],
        type=float,
        help="Cauchy scale parameter.",
    )
    parser.add_argument(
        "--calc_input_gradient_at",
        default=default_args["calc_input_gradient_at"],
        nargs="*",
        type=float,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=default_args["verbose"],
        help="Controls the verbosity when fitting and predicting.",
    )

    parser.add_argument(
        "--dataset",
        default=default_args["dataset"],
        type=str,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether samples are drawn with replacement.",
        default=False,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot results.",
        default=False,
    )
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Whether to log to mlflow.",
        default=False,
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called bagging with parameters: {vars(args)}")

    ensemble, predictions, statistics, fig = run_ensemble(
        n_estimators=args.n_estimators,
        n_total_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        n_hidden_units=args.n_hidden_units,
        n_jobs=args.n_jobs,
        n_samples=args.n_samples,
        max_samples=args.max_samples,
        bootstrap=args.bootstrap,
        random_state=args.random_state,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
        l1_penalty=args.l1_penalty,
        cauchy_penalty=args.cauchy_penalty,
        cauchy_scale=args.cauchy_scale,
        activation_fn=args.activation_fn,
        dataset=args.dataset,
        plot=args.plot,
        metrics=("rmse",),
        calc_input_gradient_at=args.calc_input_gradient_at,
    )

    ensemble_metrics = aggregate_ensemble_metrics(ensemble)

    if args.use_mlflow:
        import mlflow

        mlflow.set_experiment("deep-ensemble")
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run():
            mlflow.log_params(vars(args))
            mlflow.log_figure(fig, "figures/datasets.png")
            for step, metrics in enumerate(ensemble_metrics):
                mlflow.log_metrics(metrics, step)


if __name__ == "__main__":
    main()
