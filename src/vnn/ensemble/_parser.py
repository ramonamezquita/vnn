import argparse

from ._usage import get_default_args


def create_parser() -> argparse.ArgumentParser:
    default_args = get_default_args()

    parser = argparse.ArgumentParser(
        description="Train a Deep Ensemble of MLP networks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
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
        "--n_samples",
        default=default_args["n_samples"],
        type=int,
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
        help="Hidden activation function.",
    )
    parser.add_argument(
        "--test_size",
        help="Test size.",
        type=float,
        default=default_args["test_size"],
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
        help="Cauchy scale.",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=default_args["verbose"],
        help="Controls the verbosity when fitting and predicting.",
    )

    parser.add_argument(
        "--dataset",
        default="piecewise",
        type=str,
        help="Dataset to use.",
    )
    parser.add_argument(
        "--use_mlflow",
        action="store_true",
        help="Whether to log to mlflow.",
        default=False,
    )
    return parser
