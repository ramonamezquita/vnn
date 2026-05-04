import argparse

import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from numpyro.infer import Predictive

from vnn.bnn import BNN
from vnn.datasets import get_dataset
from vnn.inference import nuts
from vnn.priors import get_prior, supported_priors

_CLI_DESCRIPTION = """
Run NUTS to do inference on a Bayesian neural network.
"""


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=_CLI_DESCRIPTION, allow_abbrev=False)
    parser.add_argument(
        "--dataset",
        default="1d_block",
        type=str,
        help="Training dataset.",
    )
    parser.add_argument(
        "--num_samples",
        default=1000,
        type=int,
        help="Number of samples to generate from the Markov chain.",
    )
    parser.add_argument(
        "--num_warmup",
        default=1000,
        type=int,
        help="Number of warmup steps.",
    )
    parser.add_argument(
        "--num_chains",
        default=1,
        type=int,
        help="Number of parallel chains.",
    )
    parser.add_argument(
        "--num_data",
        default=100,
        type=int,
        help="Number of training data points.",
    )
    parser.add_argument(
        "--hidden_layer_sizes",
        nargs="*",
        default=[10],
        type=int,
        help="Hidden layer sizes.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed.",
    )
    parser.add_argument(
        "--step_size",
        default=1e-3,
        type=float,
        help="Step size for the Markov chain",
    )
    parser.add_argument(
        "--sigma_obs",
        default=1.0,
        type=float,
        help="Standard deviaton for the Gaussian likelihood",
    )
    parser.add_argument(
        "--prior",
        default="normal",
        type=str,
        choices=supported_priors(),
        help="Prior distribution.",
    )
    parser.add_argument(
        "--prior_loc",
        default=0.0,
        type=float,
        help="Prior location.",
    )
    parser.add_argument(
        "--prior_scale",
        default=1.0,
        type=float,
        help="Prior scale.",
    )
    parser.add_argument("--infer_sigma_obs", default=False, action="store_true")
    return parser


def main(args: argparse.ArgumentParser) -> None:

    rng_key, rng_subkey = random.split(random.key(args.seed))

    # Get data.
    ds = get_dataset(args.dataset)
    x, y = ds.sample(args.num_data, args.seed)
    X = jnp.array(x.reshape(-1, 1))
    Y = jnp.array(y.reshape(-1, 1))

    # Run MCMC.
    mcmc_args = {
        "step_size": args.step_size,
        "num_warmup": args.num_warmup,
        "num_samples": args.num_samples,
        "num_chains": args.num_chains,
    }
    prior = get_prior(args.prior, args.prior_loc, args.prior_scale)
    hidden_layer_sizes = tuple(args.hidden_layer_sizes)
    sigma_obs = None if args.infer_sigma_obs else args.sigma_obs
    model = BNN(
        hidden_layer_sizes=hidden_layer_sizes,
        prior=prior,
        sigma_obs=sigma_obs,
    )
    mcmc = nuts(rng_key, model, X, Y, **mcmc_args)
    posterior_samples = mcmc.get_samples()

    # Prediction.
    predictive = Predictive(model, posterior_samples, return_sites=["f"])
    predictions = predictive(rng_subkey, X)["f"]

    mean = jnp.mean(predictions[..., 0], axis=0)  # <= mean of means
    perc = jnp.percentile(predictions[..., 0], jnp.array([5.0, 95.0]), axis=0)

    # Plots.
    _, ax = plt.subplots()
    ax = ds.plot_true(ax)
    ax.fill_between(X[:, 0], perc[0, :], perc[1, :], color="purple", alpha=0.2)
    ax.plot(x, mean, color="purple", label="Mean")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    print(f"Called `run_mcmc` with args: {vars(args)}")
    main(args)
