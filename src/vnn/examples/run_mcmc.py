import argparse


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian neural network.")
    parser.add_argument("--num_samples", default=1000, type=int)
    parser.add_argument("--num_warmup", default=1000, type=int)
    parser.add_argument("--num_data", default=100, type=int)
    parser.add_argument("--hidden_sizes", nargs="*", default=[10], type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--step_size", default=1e-3, type=float)
    return parser


if __name__ == "__main__":
    import numpyro

    assert numpyro.__version__.startswith("0.20.1")

    import jax.numpy as jnp
    import jax.random as random
    import matplotlib.pyplot as plt
    import numpy as np
    import numpyro
    import numpyro.distributions as dist

    from vnn.bnn import BNN, mcmc, vpredict_mean
    from vnn.datasets import get_dataset

    parser = create_parser()
    args = parser.parse_args()

    # get data
    ds = get_dataset("1d_block")
    x, y = ds.sample(args.num_data, args.seed)
    X = jnp.array(x.reshape(-1, 1))
    Y = jnp.array(y.reshape(-1, 1))

    # run inference
    rng_key_mcmc, rng_key_predict = random.split(random.key(args.seed))
    model = BNN(hidden_sizes=tuple(args.hidden_sizes), prior=dist.Normal)
    result = mcmc(
        model,
        X,
        Y,
        rng_key_mcmc,
        step_size=args.step_size,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
    )
    samples = result.get_samples()

    # compute mean and confidence interval
    mean_samples = vpredict_mean(model, X, rng_key_predict, samples)
    mean_avg = jnp.mean(mean_samples[..., 0], axis=0)  # <= mean of means
    mean_ci = np.percentile(mean_samples[..., 0], [5.0, 95.0], axis=0)

    # make plots
    _, ax = plt.subplots()
    ax = ds.plot_true(ax)
    ax.fill_between(
        X[:, 0],
        mean_ci[0, :],
        mean_ci[1, :],
        color="purple",
        alpha=0.2,
        label=r"90\% C.I.",
    )
    ax.plot(x, mean_avg, color="purple", label="Mean")
    ax.legend()
