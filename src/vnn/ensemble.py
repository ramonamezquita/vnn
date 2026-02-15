"""
Command-line entrypoint for training and evaluating a Deep Ensemble
of mean-variance estimation (MVE) neural networks.

References
----------
[1] Balaji L., Alexander P. and Charles B., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles". https://arxiv.org/abs/1612.01474
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import BaggingRegressor

from vnn.datasets import get_dataset
from vnn.mve import MVE


def run(
    n_estimators: int = 10,
    n_total_epochs: int = 10000,
    n_warmup_epochs: int = 5000,
    n_hidden_units: int = 20,
    n_jobs: int = 4,
    n_samples: int = 100,
    max_samples: float | int | None = 0.8,
    bootstrap: bool = True,
    random_state: int = 0,
    verbose: int = 2,
    learning_rate: float = 1e-3,
    activation_fn: str = "sigmoid",
    dataset: str = "sinusoidal",
    plot: bool = True,
    ax=None,
) -> None:
    base_regressor = MVE(
        n_total_epochs=n_total_epochs,
        n_hidden_units=n_hidden_units,
        n_warmup_epochs=n_warmup_epochs,
        learning_rate=learning_rate,
        activation_fn=activation_fn,
    )

    regr = BaggingRegressor(
        base_regressor,
        n_estimators,
        max_samples=max_samples,
        bootstrap=bootstrap,
        random_state=random_state,
        verbose=verbose,
        n_jobs=n_jobs,
    )

    ds = get_dataset(dataset)
    X, y = ds.sample(n_samples)
    X, y = map(torch.from_numpy, (X, y))
    X_2d = X.reshape(-1, 1)
    regr.fit(X_2d, y)

    # `predictions` holds the output for all estimators.
    predictions: torch.Tensor = torch.stack(
        [regr.estimators_[i].predict(X) for i in range(n_estimators)], axis=0
    )
    predictions.detach_()
    predictions = predictions.numpy()

    means = predictions[:, :, 0]
    vars_ = predictions[:, :, 1]

    # The estimated regression function is just the average of all mean predictions.
    mean_prediction = means.mean(axis=0)

    # The estimated variance is more tricky.
    var_prediction = (vars_ + np.square(means)).mean(axis=0) - np.square(
        mean_prediction
    )

    lb = mean_prediction - 1.96 * np.sqrt(var_prediction)
    ub = mean_prediction + 1.96 * np.sqrt(var_prediction)

    x = X.flatten()

    if ax is None:
        ax = plt.subplot()

    ax.scatter(x, y, label="Observations", s=2, c="black", alpha=0.1)
    ax.plot(x, mean_prediction, label="Ensmeble output")
    ax.plot(x, lb, ls="--", c="gray", label="95% CI")
    ax.plot(x, ub, ls="--", c="gray")
    ax.set_title(f"Number of estimators = {n_estimators}")

    if plot:
        plt.legend()
        plt.show()

    return mean_prediction, var_prediction, ax


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrypoint for running training scripts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--random_state",
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
    )
    parser.add_argument(
        "--plot", action="store_true", help="Whether to plot results.", default=True
    )

    return parser


def main():
    """Parses args and runs bagging."""

    parser = create_parser()
    args = parser.parse_args()

    print(f"Called bagging with parameters: {vars(args)}")

    try:
        run(
            n_estimators=args.n_estimators,
            n_total_epochs=args.n_total_epochs,
            n_warmup_epochs=args.n_warmup_epochs,
            n_hidden_units=args.n_hidden_units,
            n_jobs=args.n_jobs,
            max_samples=args.max_samples,
            bootstrap=args.bootstrap,
            random_state=args.random_state,
            verbose=args.verbose,
            learning_rate=args.learning_rate,
            activation_fn=args.activation_fn,
            dataset=args.dataset,
            plot=args.plot,
        )
    except Exception as exc:
        raise exc


if __name__ == "__main__":
    main()
