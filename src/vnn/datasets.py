import jax
import jax.numpy as jnp
import numpy as np

N_SAMPLES: int = 5000


def get_dataset(name: str) -> tuple[jax.Array, jax.Array]:
    try:
        return DATASETS[name]()
    except KeyError:
        raise KeyError("Dataset {name} not known.")


def sinusoidal() -> tuple[jax.Array, jax.Array]:
    x = jnp.linspace(0, jnp.pi / 2, N_SAMPLES)
    w_c = 5
    w_m = 4
    m_x = jnp.sin(w_m * x)
    y = m_x * jnp.sin(w_c * x)
    sigma2 = 0.02 + 0.02 * jnp.square(1 - m_x)
    noise = np.random.normal(0, np.sqrt(sigma2))
    y += noise
    return x.reshape(-1, 1), y


def cubic() -> tuple[jax.Array, jax.Array]:
    x = jnp.linspace(-3, 3, N_SAMPLES)
    y = x**3
    sigma2 = 9
    noise = np.random.normal(0, np.sqrt(sigma2), N_SAMPLES)
    y += noise
    return x.reshape(-1, 1), y


DATASETS = {"sinusoidal": sinusoidal, "cubic": cubic}
