import torch
from pyro.infer import MCMC, NUTS

from .pyro import PyroProbabilisticModel


def nuts(
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    model: PyroProbabilisticModel,
    step_size: float = 1.0,
    num_samples: int = 1,
    warmup_steps: int | None = None,
):
    nuts = NUTS(model, step_size=step_size)
    mcmc = MCMC(nuts, num_samples, warmup_steps)
    mcmc.run(X, y)
    return mcmc
