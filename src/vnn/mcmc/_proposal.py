from typing import Callable

import torch
from torch.distributions import Distribution


class Proposal:
    """Generic proposal distribution wrapper for MCMC algorithms.

    This class defines a conditional proposal distribution q(x | condition_on).

    Parameters
    ----------
    distr_factory : Callable[[torch.Tensor], torch.distributions.Distribution]
        A function that takes a conditioning tensor and returns a PyTorch
        Distribution object representing q( . | condition_on).

    """

    def __init__(self, distr_factory: Callable[[torch.Tensor], Distribution]):
        self._distr_factory = distr_factory

    def sample(self, condition_on: torch.Tensor) -> torch.Tensor:
        return self._distr_factory(condition_on).sample()

    def log_prob(self, x: torch.Tensor, condition_on: torch.Tensor) -> torch.Tensor:
        return self._distr_factory(condition_on).log_prob(x)

    def propose(self, W_old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        W_new: dict[str, torch.Tensor] = {}
        for name, w_old in W_old.items():
            W_new[name] = self.sample(condition_on=w_old)
        return W_new
