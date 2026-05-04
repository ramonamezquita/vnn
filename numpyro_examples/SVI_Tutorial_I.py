import math

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from numpyro import param, plate, sample
from numpyro.distributions import constraints
from numpyro.infer import SVI, Trace_ELBO
from numpyro.optim import Adam


def model(data: jax.Array) -> None:
    # this function defines a probability model for the joint p(x, z) as
    #                   p(x, z) = p(x | z) p(z)
    # where:
    # z ~ Beta(alpha0, beta0)
    # x | z ~ Bernoulli(z)

    assert jnp.ndim(data) == 1, "data must be 1-dimensional."

    # sample latent z from the Beta prior p(z).
    alpha_0, beta_0 = 10.0, 10.0
    z = sample("z", dist.Beta(alpha_0, beta_0))

    with plate("data", len(data)):
        return sample("obs", dist.Bernoulli(z), obs=data)


def guide(data: jax.Array) -> None:
    # register the two variational parameters with Pyro.
    alpha_q = param("alpha_q", jnp.array(15.0), constraint=constraints.positive)
    beta_q = param("beta_q", jnp.array(15.0), constraint=constraints.positive)

    # sample z from the distribution Beta(alpha_q, beta_q)
    sample("z", dist.Beta(alpha_q, beta_q))


if __name__ == "__main__":
    # random keys
    root = random.key(0)
    rng_key_data, rng_key_svi = random.split(root)

    # synthetic data
    true_p = 0.4
    data = jnp.astype(random.bernoulli(rng_key_data, true_p, (100,)), float)

    # set up the optimizer
    optimizer = Adam(step_size=0.0005)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    svi_result = svi.run(rng_key_svi, 1000, data)
    params = svi_result.params

    # collect the learned variational parameters
    alpha_q = params["alpha_q"]
    beta_q = params["beta_q"]

    # use some facts about the Beta distribution
    inferred_mean = alpha_q / (alpha_q + beta_q)
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * math.sqrt(factor)

    print(
        "\nBased on the data and our prior belief, the fairness "
        + "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std)
    )
