from typing import Protocol

import jax
import jax.numpy as jnp


class XYFactory(Protocol):
    def __call__(self) -> tuple[jax.Array, jax.Array]: ...


def xy_factory(name: str) -> XYFactory:
    options: dict[str, XYFactory] = {"polynomial": polynomial}

    try:
        return options[name]
    except KeyError:
        raise KeyError(
            f"Dataset {name} is not a valid one. Available datasets are: {list(options)}"
        )


def polynomial() -> tuple[jax.Array, jax.Array]:
    x = jnp.linspace(start=0, stop=2, num=1000).reshape(-1, 1)
    y = 2 * x**3 - 5 * x**2 + 3 * x + 7
    return x, y
