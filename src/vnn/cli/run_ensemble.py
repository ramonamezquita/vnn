from argparse import Namespace

from vnn.ensemble import create_parser, train_ensemble


def main(args: Namespace) -> None:
    train_ensemble(
        dataset=args.dataset,
        n_estimators=args.n_estimators,
        n_total_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        n_jobs=args.n_jobs,
        n_samples=args.n_samples,
        hidden_layer_sizes=args.hidden_layer_sizes,
        test_size=args.test_size,
        random_state=args.random_state,
        learning_rate=args.learning_rate,
        l2_penalty=args.l2_penalty,
        l1_penalty=args.l1_penalty,
        cauchy_scale=args.cauchy_scale,
        activation_fn=args.activation_fn,
    )


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_ensemble` with parameters: {vars(args)}")
    main(args)
