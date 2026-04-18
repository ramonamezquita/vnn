from typing import Protocol

import torch
from torch import nn
from torch.distributions import Distribution


class Regularizer(Protocol):
    def __call__(self, m: nn.Module) -> torch.Tensor: ...


def make_map_regularizer(distr: Distribution, penalty: float = 1.0) -> Regularizer:
    """Create a parameter regularizer from a probability distribution.

    This constructs a regularization function that penalizes model parameters
    according to the negative log-probability under a given distribution.

    Parameters
    ----------
    distr : Distribution
        A PyTorch distribution object.

    penalty : float, optional
        Scaling factor for the regularization term. Defaults to 1.0.

    Returns
    -------
    Regularizer
    """

    def map_regularizer(m: nn.Module) -> torch.Tensor:
        return -penalty * sum(
            distr.log_prob(p).sum() for p in m.parameters() if p.requires_grad
        )

    return map_regularizer


def no_op_regularizer(m: nn.Module) -> torch.Tensor:
    return torch.tensor(0.0)
