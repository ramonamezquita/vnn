import argparse
from dataclasses import asdict, dataclass

from vnn.datasets import DATASETS


@dataclass(kw_only=True, frozen=True)
class RunParams:
    seed: int = 42
    n_total_epochs: int = 1000
    n_warmup_epochs: int = 500
    n_estimators: int = 4
    n_jobs: int = 4
    n_hidden_units: int = 20
    activation_fn: str = "sigmoid"
    bootstrap: bool = True
    max_samples: int = 1
    learning_rate: float = 1e-3
    verbose: int = 0
    dataset: str = "sinusoidal"
    plot: bool = False


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrypoint for running training scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
        default=42,
    )

    parser.add_argument(
        "--n_total_epochs",
        type=int,
        help="Number of epochs.",
        default=10000,
    )

    parser.add_argument(
        "--n_warmup_epochs",
        type=int,
        help="Number of warmup epochs.",
        default=5000,
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        help="Number of models.",
        default=20,
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs.",
        default=4,
    )

    parser.add_argument(
        "--n_hidden_units",
        default=20,
        type=int,
        help="Size of the hidden layer.",
    )

    parser.add_argument(
        "--activation_fn",
        type=str,
        default="sigmoid",
        help="Hidden activation function.",
    )

    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Whether samples are drawn with replacement.",
    )

    parser.add_argument(
        "--max_samples",
        help="The number of samples to draw from X to train each estimator",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="Learning rate.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=2,
        help="Controls the verbosity when fitting and predicting.",
    )

    parser.add_argument(
        "--dataset",
        default="sinusoidal",
        type=str,
        help="Dataset to use.",
        choices=DATASETS,
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Whether to plot results.",
    )

    return parser


def main():
    """Runs a job."""
    from vnn import bagging

    parser = create_parser()
    args = parser.parse_args()

    run_params = RunParams(
        seed=args.seed,
        n_total_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        n_hidden_units=args.n_hidden_units,
        activation_fn=args.activation_fn,
        bootstrap=args.bootstrap,
        max_samples=args.max_samples,
        learning_rate=args.learning_rate,
        verbose=args.verbose,
        dataset=args.dataset,
        plot=args.plot,
    )

    print(f"Called main.py with parameters: {asdict(run_params)}")

    try:
        bagging.run(run_params)
    except Exception as exc:
        raise exc


if __name__ == "__main__":
    main()
