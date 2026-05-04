from typing import Callable

import jax
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from jax import random, vmap
from numpyro import handlers
from numpyro.infer import MCMC, NUTS

# For this example, we will use the WaffleDivorce dataset from Chapter 05, Statistical Rethinking.
DATASET_URL = "https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv"


def model(marriage: np.ndarray, age: np.ndarray, divorce: np.ndarray | None = None):
    """Probabilistic model."""
    # In addition to regular Python statements, the model code also contains primitives like `sample`.
    # These primitives can be interpreted with various side-effects using effect handlers.
    # For now, just remember that a `sample` statement makes this a stochastic function that samples some
    # latent parameters from a prior distribution. Our goal is to infer the posterior distribution of these
    # parameters conditioned on observed data.

    w0 = numpyro.sample("w0", dist.Normal(0.0, 0.2))
    w1 = numpyro.sample("w1", dist.Normal(0.0, 0.5))
    w2 = numpyro.sample("w2", dist.Normal(0.0, 0.5))

    # we put a prior on the observation noise.
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))

    mu = w0 + w1 * marriage + w2 * age  # <= output of the linear model.
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=divorce)


def standardize(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def predict_one(
    rng_key: jax.Array,
    post_samples: dict[str, jax.Array],
    model: Callable,
    *args,
    **kwargs,
):
    model = handlers.seed(handlers.condition(model, post_samples), rng_key)
    model_trace = handlers.trace(model).get_trace(*args, **kwargs)
    return model_trace["obs"]["value"]


def main():
    # Read
    df = pd.read_csv(DATASET_URL, sep=";")

    # Preprocess
    age_values = standardize(df["Marriage"]).values
    marriage_values = standardize(df["MedianAgeMarriage"]).values
    divorce_values = standardize(df["Divorce"]).values

    # Inference
    # Start from this source of randomness. We will split keys for subsequent operations.
    rng_key = random.key(0)
    rng_key_inference, rng_key_predict = random.split(rng_key)
    kernel = NUTS(model)
    num_samples = 2000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples)
    mcmc.run(
        rng_key_inference,
        marriage=marriage_values,
        age=age_values,
        divorce=divorce_values,
    )
    mcmc.print_summary()
    post_samples = mcmc.get_samples()

    # Predict
    vmap_args = (random.split(rng_key_predict, num_samples), post_samples)
    predict_fn = vmap(
        lambda k, s: predict_one(k, s, model, marriage=marriage_values, age=age_values)
    )
    post_predictions = predict_fn(*vmap_args)


if __name__ == "__main__":
    main()
