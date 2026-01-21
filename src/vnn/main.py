import argparse
import importlib
from dataclasses import asdict, dataclass

from vnn.datasets import DATASETS


@dataclass(kw_only=True, frozen=True)
class RunParams:
    seed: int = 42
    n_total_epochs: int = 1000
    n_warmup_epochs: int = 500
    n_models: int = 4
    n_jobs: int = 4
    n_hidden_units: int = 20
    sample_size: int = 1
    learning_rate: float = 1e-3
    dataset: str = "sinusoidal"
    plot: bool = False


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrypoint for running training scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        dest="model",
        help="<Required> Name of the job to run.",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed.",
        default=42,
    )

    parser.add_argument(
        "--n_models",
        type=int,
        help="Number of models.",
        default=10,
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of jobs.",
        default=4,
    )

    parser.add_argument(
        "--n_total_epochs",
        type=int,
        help="Number of epochs.",
        default=1000,
    )

    parser.add_argument(
        "--n_warmup_epochs",
        type=int,
        help="Number of warmup epochs.",
        default=500,
    )

    parser.add_argument(
        "--n_hidden_units",
        default=20,
        type=int,
        help="Size of the hidden layer.",
    )

    parser.add_argument(
        "--sample_size",
        help="Sample size",
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

    parser = create_parser()
    args = parser.parse_args()

    model_module = importlib.import_module(f"vnn.{args.model}")
    run_params = RunParams(
        seed=args.seed,
        n_total_epochs=args.n_total_epochs,
        n_warmup_epochs=args.n_warmup_epochs,
        n_models=args.n_models,
        n_jobs=args.n_jobs,
        n_hidden_units=args.n_hidden_units,
        sample_size=args.sample_size,
        learning_rate=args.learning_rate,
        dataset=args.dataset,
        plot=args.plot,
    )

    print(f"Called model `{args.model}` with parameters: {asdict(run_params)}")

    try:
        model_module.run(run_params)
    except Exception as exc:
        raise exc


if __name__ == "__main__":
    main()
