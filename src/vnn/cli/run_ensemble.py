from vnn.mve import create_parser, run
from vnn.mlflow_logging import log_to_mlflow


def main():
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_ensemble` with parameters: {vars(args)}")

    result = run(
        dataset=args.dataset,
        n_estimators=args.n_estimators,
        n_total_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        n_jobs=args.n_jobs,
        n_samples=args.n_samples,
        hidden_layer_sizes=args.hidden_layer_sizes,
        test_size=args.test_size,
        random_state=args.random_state,
        verbose=args.verbose,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
        l1_penalty=args.l1_penalty,
        cauchy_scale=args.cauchy_scale,
        activation_fn=args.activation_fn,
    )

    if args.use_mlflow:
        import mlflow

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("run_ensemble")
        log_to_mlflow(result, vars(args))


if __name__ == "__main__":
    main()
