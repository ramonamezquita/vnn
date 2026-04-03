from typing import Callable

import pyro
import torch
from pyro import distributions as dist
from torch.func import functional_call


def get_pyro_distr(
    name: str, loc: float = 0.0, scale: float = 1.0
) -> dist.Distribution:
    name_to_distr = {
        "normal": dist.Normal,
        "cauchy": dist.Cauchy,
        "laplace": dist.Laplace,
    }
    if name not in name_to_distr:
        raise ValueError(f"Distribution {name} not supported.")

    return name_to_distr[name](loc, scale)


def make_pyro_model(
    module: torch.nn.Module, prior: dist.Distribution, sigma: float = 1.0
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

    def pyro_model(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        # Define priors over torch parameters.
        sampled_params: dict[str, torch.Tensor] = {}
        for name, param in module.named_parameters():
            sampled_params[name] = pyro.sample(
                name, prior.expand(param.size()).to_event(param.dim())
            )

        # Forward call.
        mean = functional_call(module, sampled_params, X)
        assert mean.shape == y.shape

        # Define likelihood using `pyro.sample`.
        # Our model implies that:
        # y | x ~ N(f(x), sigma)
        # where the mean f(x) is the forward call function.
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

    return pyro_model
