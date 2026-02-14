"""
Synthetic datasets for regression experiments.
"""

import math

import torch

N_SAMPLES: int = 5000


def get_dataset(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        return DATASETS[name]()
    except KeyError:
        raise KeyError("Dataset {name} not known.")


def sinusoidal() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(0, torch.pi / 2, N_SAMPLES)
    w_c = 5
    w_m = 4
    m_x = torch.sin(w_m * x)
    y = m_x * torch.sin(w_c * x)
    sigma2 = 0.02 + 0.02 * torch.square(1 - m_x)
    noise = torch.normal(0, torch.sqrt(sigma2))
    y += noise
    return x.reshape(-1, 1), y


def cubic() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-3, 3, N_SAMPLES)
    y = x**3
    sigma2 = 9
    noise = torch.normal(0, math.sqrt(sigma2), size=(N_SAMPLES,))
    y += noise
    return x.reshape(-1, 1), y


DATASETS = {"sinusoidal": sinusoidal, "cubic": cubic}
