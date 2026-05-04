from typing import Any, Callable, OrderedDict

import jax
from numpyro import handlers
from numpyro.infer import MCMC, NUTS, init_to_uniform


def nuts(
    rng_key: jax.Array,
    model: Callable,
    *args: Any,
    step_size: float = 1.0,
    num_warmup: int = 2000,
    num_samples: int = 1000,
    num_chains: int = 1,
) -> MCMC:
    """MCMC inference with NUTS sampler.

    Parameters
    ----------
    rng_key : jax.Array
        Random key.

    model : Callable
        Stochastic python function.

    args : Any
        Arugments provided to `model`.

    step_size : float, default=1.0
        Determines the size of a single step taken by the verlet integrator
        while computing the trajectory using Hamiltonian dynamics.

    num_warmup : int
        Number of warmup steps.

    num_samples : int
        Number of samples to generate from the Markov chain.
    """
    kernel = NUTS(model, step_size=step_size, init_strategy=init_to_uniform(radius=1))
    mcmc = MCMC(
        kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains
    )
    mcmc.run(rng_key, *args)
    return mcmc


def trace_model(
    rng_key: jax.Array,
    model: Callable,
    *args: Any,
    data: dict[str, jax.Array] | None = None,
) -> OrderedDict:
    """Run a single forward pass and return the full NumPyro trace.

    Substitutes all primitive calls in `model` with values from `data` whose key matches
    the site name.

    Parameters
    ----------
    rng_key: jax.Array
        Random key.

    model : Callable
        Stochastic python function.

    args : Any
        Arugments provided to `model`.

    data : dict[str, jax.Array]
        Data keyed by site names.
    """
    model = handlers.substitute(handlers.seed(model, rng_key), data)
    model_trace = handlers.trace(model).get_trace(*args)
    return model_trace
