from typing import Callable

import torch
import torch.distributions as dist
from torch.distributions import Distribution
from torch.func import functional_call

_NAME_TO_DIST: dict[str, dist.Distribution] = {
    "normal": dist.Normal,
    "cauchy": dist.Cauchy,
    "laplace": dist.Laplace,
}


def supported_distributions() -> list[str]:
    return list(_NAME_TO_DIST)


def torch_distr(name: str, loc: float = 0.0, scale: float = 1.0) -> dist.Distribution:
    """Returns torch distribution."""
    if name not in supported_distributions():
        raise ValueError(f"Distribution {name} not supported.")

    return _NAME_TO_DIST[name](loc, scale)


class TorchProbabilisticModel:
    def __init__(
        self,
        module: torch.nn.Module,
        prior: Distribution,
        likelihood: Callable[[torch.Tensor], Distribution],
    ):
        self.module = module
        self.prior = prior
        self.likelihood = likelihood

    def __call__(self, W: dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
        return functional_call(self.module, W, X)

    def get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.module.named_parameters())

    def log_prob(
        self, W: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        # The log density is the log posterior of the `nn` parameters log p(W | y):
        #                   log p(W | y) = log p(W) + log p(y | W) - log p(y)
        # where:
        # > log p(W): log-prior of weights `W`.
        # > log p(y | W): log-likelihood of the observed target values `y`.
        # > (X, y) are the fixed observations.
        log_prior = sum(self.prior.log_prob(w).sum() for w in W.values())
        log_likel = self.likelihood(self(W, X)).log_prob(y).sum()
        return log_prior + log_likel
