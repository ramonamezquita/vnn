from typing import OrderedDict, Type

import jax
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from jax import vmap
from numpyro import handlers, plate, sample
from numpyro.infer import MCMC, NUTS


def nonlin(x: jax.Array) -> jax.Array:
    """The non-linearity we use in our neural network."""
    return jnp.tanh(x)


class BNN:
    """Bayesian neural network with Gaussian likelihood distribution.

    Parameters
    ----------
    n_features_in: int, default=1
        Number of input features.

    n_features_out: int, default=1
        Number of output (target) features.

    hidden_sizes: tuple of int, default=(10,)
        Sizes of hidden layers.

    prior : Type of Distribution, default=dist.Normal
        Prior for all network weights.
    """

    def __init__(
        self,
        num_features_in: int = 1,
        num_features_out: int = 1,
        hidden_sizes: tuple[int, ...] = (10,),
        prior: Type[dist.Distribution] = dist.Normal,
    ):
        self.num_features_in = num_features_in
        self.num_features_out = num_features_out
        self.hidden_sizes = hidden_sizes
        self.prior = prior

    def __call__(self, X: jax.Array, Y: jax.Array | None = None) -> None:
        """Forward pass.

        Parameters
        ----------
        X : jax.Array, shape=(N, num_features_in)
            Input features.

        Y : jax.Array, shape=(N, num_features_out), default=None
            Target values. If None, `Y` will be sampled from the likelihood distribution.
        """

        assert jnp.ndim(X) > 1, "`X` ndim must be greater than 1."
        assert X.shape[1] == self.num_features_in, "Wrong `X` dimension."
        assert self.hidden_sizes, "`hidden_sizes` cannot be empty."
        N, D_X = X.shape

        if Y is not None:
            assert jnp.ndim(Y) > 1, "`Y` ndim must be greater than 1."
            assert Y.shape[0] == N, "`Y` and `X` lenghts differ."
            assert Y.shape[1] == self.num_features_out, "Wrong `Y` dimension."

        D_Y = self.num_features_out
        L = len(self.hidden_sizes)  # <= number of hidden layers
        layer_sizes = (D_X,) + tuple(self.hidden_sizes)

        # sample hidden layers weights
        z = X
        for i, (in_, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:]), start=1):
            w = sample(f"w{i}", self.prior(jnp.zeros((in_, out)), jnp.ones((in_, out))))
            b = sample(f"b{i}", self.prior(jnp.zeros(out), jnp.ones(out)))
            z = nonlin(jnp.matmul(z, w) + b)  # <= hidden layer activations
            # at the end of each iteration, z.shape=(N, out).

        # sample final layer weights and neural network output
        w = sample(f"w{L + 1}", self.prior(jnp.zeros((out, D_Y)), jnp.ones((out, D_Y))))
        b = sample(f"b{L + 1}", self.prior(jnp.zeros(D_Y), jnp.ones(D_Y)))
        z = jnp.matmul(z, w)  # <= output of the neural network

        # we put a prior on the observation noise
        prec_obs = sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / jnp.sqrt(prec_obs)

        # observe data
        with plate("data", N):
            # note we use to_event(1) because each observation has shape (1,)
            sample("Y", dist.Normal(z, sigma_obs).to_event(1), obs=Y)


def mcmc(
    model: BNN,
    X: jax.Array,
    Y: jax.Array,
    rng_key: jax.Array,
    *,
    step_size: float = 1.0,
    num_warmup: int = 2000,
    num_samples: int = 1000,
) -> MCMC:
    """Helper function for MCMC inference."""
    sampler = NUTS(model, step_size=step_size)
    mcmc = MCMC(sampler, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, X, Y)
    return mcmc


def _trace_model(
    model: BNN,
    X: jax.Array,
    rng_key: jax.Array,
    samples: dict[str, jax.Array],
    Y: jax.Array | None = None,
) -> OrderedDict:
    """Run a single forward pass and return the full NumPyro trace.

    Parameters
    ----------
    model : BNN
        An instance of :class:`BNN`.

    X : jax.Array, shape=(N, D_X)
        Input features.

    rng_key: jax.Array
        Random key.

    samples : dict[str, jax.Array]
        A single posterior sample.

    Y : jax.Array or None
        If None, observations are sampled from the likelihood.
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model if we pass Y=None here.
    model_trace = handlers.trace(model).get_trace(X=X, Y=Y)
    return model_trace


def _num_samples(samples: dict) -> int:
    return next(iter(samples.values())).shape[0]


def predict_obs(
    model: BNN, X: jax.Array, rng_key: jax.Array, samples: dict[str, jax.Array]
) -> jax.Array:
    """Return a noisy observation draw from the likelihood for a single posterior sample.

    For batched samples use `vpredict_obs`.

    Parameters
    ----------
    model : BNN
        An instance of :class:`BNN`.

    X : jax.Array, shape=(N, D_X)
        Input features.

    rng_key: jax.Array
        Random key.

    samples : dict[str, jax.Array]
        A single posterior sample.

    Returns
    -------
    jax.Array
    """
    return _trace_model(model, X, rng_key, samples)["Y"]["value"]


def predict_mean(
    model: BNN, X: jax.Array, rng_key: jax.Array, samples: dict[str, jax.Array]
) -> jax.Array:
    """Return the noiseless NN output (likelihood mean) for a single posterior sample.

    For batched samples use `vpredict_mean`.

    Parameters
    ----------
    model : BNN
        An instance of :class:`BNN`.

    X : jax.Array, shape=(N, D_X)
        Input features.

    rng_key: jax.Array
        Random key.

    samples : dict[str, jax.Array]
        A single posterior sample.

    Returns
    -------
    jax.Array
    """
    return _trace_model(model, X, rng_key, samples)["Y"]["fn"].mean


def vpredict_obs(
    model: BNN, X: jax.Array, rng_key: jax.Array, samples: dict[str, jax.Array]
) -> jax.Array:
    """Return noisy observation draws from the likelihood for all posterior samples.

    Vectorized over the leading sample dimension via `vmap`.

    Parameters
    ----------
    model : BNN
        An instance of :class:`BNN`.

    X : jax.Array, shape=(N, D_X)
        Input features.

    rng_key: jax.Array
        Random key.

    samples : dict[str, jax.Array]
        Multiple posterior samples.

    Returns
    -------
    jax.Array, shape=(num_samples, N, D_Y)
    """

    vmap_args = (samples, random.split(rng_key, _num_samples(samples)))
    return vmap(lambda s, k: predict_obs(model, X, k, s))(*vmap_args)


def vpredict_mean(
    model: BNN, X: jax.Array, rng_key: jax.Array, samples: dict[str, jax.Array]
) -> jax.Array:
    """Return the noiseless NN output (likelihood mean) for all posterior samples.

    Vectorized over the leading sample dimension via `vmap`.

    Parameters
    ----------
    model : BNN
        An instance of :class:`BNN`.

    X : jax.Array, shape=(N, D_X)
        Input features.

    rng_key: jax.Array
        Random key.

    samples : dict[str, jax.Array]
        Multiple posterior samples.

    Returns
    -------
    jax.Array, shape=(num_samples, N, D_Y)
    """
    vmap_args = (samples, random.split(rng_key, _num_samples(samples)))
    return vmap(lambda s, k: predict_mean(model, X, k, s))(*vmap_args)
