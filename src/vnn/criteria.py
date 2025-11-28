import jax
import jax.numpy as jnp


def normal_negative_log_likelihood(y_true: jax.Array, y_pred: jax.Array) -> jax.Array:
    mu = y_pred[..., 0]
    sigma2 = y_pred[..., 1]
    return (0.5 * (jnp.log(sigma2) + ((y_true - mu) ** 2 / sigma2))).mean()
