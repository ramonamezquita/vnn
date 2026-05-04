import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import deterministic, plate, sample


def nonlin(x: jax.Array) -> jax.Array:
    """The non-linearity we use in our neural network."""
    return jnp.tanh(x)


def bnn(
    X: jax.Array,
    Y: jax.Array | None = None,
    *,
    dim_input: int = 1,
    dim_output: int = 1,
    hidden_layer_sizes: tuple[int, ...] = (100,),
    prior: dist.Distribution = dist.Normal(0, 1),
    sigma_obs: float | None = 1.0,
) -> None:
    """Bayesian neural network with Gaussian likelihood distribution.

    Parameters
    ----------
    X : jax.Array, shape=(N, dim_input)
        Input features.

    Y : jax.Array, shape=(N, dim_output), default=None
        Target values. If None, `Y` will be sampled from the likelihood distribution.

    dim_input: int, default=1
        Number of input features.

    dim_output: int, default=1
        Number of output (target) features.

    hidden_layer_sizes: tuple of int, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    prior : Distribution, default=dist.Normal(0, 1)
        Prior for all network weights.
    """

    assert jnp.ndim(X) == 2, "`X` must be two-dimensional."
    assert X.shape[1] == dim_input, f"Expected input dim {dim_input}, got {X.shape[1]}."
    if Y is not None:
        assert jnp.ndim(Y) == 2, "`Y` must be two-dimensional."
        assert Y.shape[0] == X.shape[0], "`Y` and `X` batch size differ."
        assert Y.shape[1] == dim_output, (
            f"Expected output dim {dim_output}, got {Y.shape[1]}."
        )

    L = len(hidden_layer_sizes)  # <= number of hidden layers.

    # sample all weights independently from the prior
    # `sample` statements makes this a stochastic function that samples some
    # latent parameters from a prior distribution.
    z = X
    h_in = dim_input
    for j, h_out in enumerate(hidden_layer_sizes, start=1):
        w = sample(f"w{j}", prior.expand((h_in, h_out)))
        b = sample(f"b{j}", prior.expand((h_out,)))
        z = nonlin(jnp.matmul(z, w) + b)  # <= hidden layer activations
        h_in = h_out

    # sample final layer weights and neural network output
    w = sample(f"w{L + 1}", prior.expand((h_in, dim_output)))
    b = sample(f"b{L + 1}", prior.expand((dim_output,)))
    z = jnp.matmul(z, w) + b  # <= output of the neural network
    deterministic("f", z)

    # prior on the observation noise
    if sigma_obs is None:
        # use inverse-gamma conjugate.
        prec_obs = sample("prec_obs", dist.Gamma(3.0, 1.0))
        sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    N = X.shape[0]
    with plate("data", N):
        sample("Y", dist.Normal(z, sigma_obs).to_event(1), obs=Y)


class BNN:
    """Bayesian neural network with Gaussian likelihood distribution.

    This class is a thin wrapper around `bnn` that stores model dimensions
    and hyperparameters, providing a stateful interface for repeated calls.

    Parameters
    ----------
    dim_input: int, default=1
        Number of input features.

    dim_output: int, default=1
        Number of output (target) features.

    hidden_layer_sizes: tuple of int, default=(100,)
        The ith element represents the number of neurons in the ith hidden layer.

    prior : Distribution, default=dist.Normal(0, 1)
        Prior for all network weights.
    """

    def __init__(
        self,
        dim_input: int = 1,
        dim_output: int = 1,
        hidden_layer_sizes: tuple[int, ...] = (100,),
        prior: dist.Distribution = dist.Normal(0, 1),
        sigma_obs: float | None = 1.0,
    ):
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.hidden_layer_sizes = hidden_layer_sizes
        self.prior = prior
        self.sigma_obs = sigma_obs

    def __call__(self, X: jax.Array, Y: jax.Array | None = None) -> None:
        """Stochastic forward pass.

        Parameters
        ----------
        X : jax.Array, shape=(N, dim_input)
            Input features.

        Y : jax.Array, shape=(N, dim_output), default=None
            Target values. If None, `Y` will be sampled from the likelihood distribution.
        """
        return bnn(
            X,
            Y,
            dim_input=self.dim_input,
            dim_output=self.dim_output,
            hidden_layer_sizes=self.hidden_layer_sizes,
            prior=self.prior,
            sigma_obs=self.sigma_obs,
        )
