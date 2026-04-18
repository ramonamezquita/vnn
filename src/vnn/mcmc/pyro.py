from typing import Callable

import pyro
import torch
from pyro import distributions as dist
from torch import nn
from torch.func import functional_call

_NAME_TO_DIST: dict[str, dist.Distribution] = {
    "normal": dist.Normal,
    "cauchy": dist.Cauchy,
    "laplace": dist.Laplace,
}


def supported_distributions() -> list[str]:
    return list(_NAME_TO_DIST)


def pyro_distr(name: str, loc: float = 0.0, scale: float = 1.0) -> dist.Distribution:
    """Returns pyro distribution."""
    if name not in supported_distributions():
        raise ValueError(f"Distribution {name} not supported.")

    return _NAME_TO_DIST[name](loc, scale)


class PyroProbabilisticModel:
    """A probabilistic wrapper around a PyTorch module using Pyro.

    Parameters
    ----------
    module : torch.nn.Module
        The deterministic neural network representing the forward computation.
        Its parameters are treated as latent variables.

    prior : pyro.distributions.Distribution
        Prior distribution over all model parameters.

    likelihood : Callable[[torch.Tensor], pyro.distributions.Distribution]
        A callable that maps the model output (mean) to a likelihood
        distribution over observations.
    """

    def __init__(
        self,
        module: nn.Module,
        prior: dist.Distribution,
        likelihood: Callable[[torch.Tensor], dist.Distribution],
    ):
        self.module = module
        self.prior = prior
        self.likelihood = likelihood

    def __call__(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Run the probabilistic model with observed data.

        Parameters
        ----------
        X : torch.Tensor
            Input features.

        y : torch.Tensor
            Observed targets.

        Returns
        -------
        torch.Tensor
            A sample from the likelihood distribution conditioned on `y`.
        """

        # Define priors over torch parameters.
        # We apply `expand` followed by `to_event` to reshape the
        # prior into a multivariate distribution over the full tensor.
        new_params: dict[str, torch.Tensor] = {}
        for name, p in self.module.named_parameters():
            prior_reshaped = self.prior.expand(p.size()).to_event(p.dim())
            new_params[name] = pyro.sample(name, prior_reshaped)

        # Forward call.
        mean: torch.Tensor = functional_call(self.module, new_params, X)
        assert mean.shape == y.shape

        # Evaluate likelihood using `pyro.sample`.
        return pyro.sample("obs", self.likelihood(mean), obs=y)
