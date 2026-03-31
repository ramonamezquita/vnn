from typing import Callable

import torch
from torch.distributions import Distribution


class ProbabilisticModel:
    def __init__(
        self,
        nn: torch.nn.Module,
        prior: Distribution,
        likelihood: Callable[[torch.Tensor], Distribution],
    ):
        self.nn = nn
        self.prior = prior
        self.likelihood = likelihood

    def __call__(self, W: dict[str, torch.Tensor], X: torch.Tensor) -> torch.Tensor:
        self.set_parameters(W)
        return self.nn(X)

    def get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        return dict(self.nn.named_parameters())

    def set_parameters(self, W: dict[str, torch.Tensor]) -> None:
        for name, param in self.get_named_parameters().items():
            param.copy_(W[name])

    def log_prob(
        self, W: dict[str, torch.Tensor], X: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:

        # The log density is the posterior of the `nn`, which is given by
        #                   L(W) = log_prior(W) + log_likel(y | loc=NN(W,X))
        # where:
        # > log_prior is the log prior of weights `W`.
        # > log_likel is log-likelihood of the observed target values `y` centered at NN output.
        # > (X, y) are the fixed observations.
        log_prior = sum(self.prior.log_prob(w).sum() for w in W.values())
        log_likel = self.likelihood(self(W, X)).log_prob(y).sum()
        return log_prior + log_likel
