from dataclasses import dataclass

import torch
from torch import nn


@dataclass(kw_only=True, frozen=True)
class RegularizerOptions:
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    cauchy_scale: float = 0.0


class L1Regularizer:
    def __init__(self, l1_penalty: float = 1e-3):
        self.l1_penalty = l1_penalty

    def __call__(self, model: nn.Module) -> torch.Tensor:
        total_penalty = torch.tensor(0.0)
        for param in model.parameters():
            total_penalty += param.abs().sum()
        return self.l1_penalty * total_penalty


class L2Regularizer:
    def __init__(self, l2_penalty: float = 1e-3):
        self.l2_penalty = l2_penalty

    def __call__(self, model: nn.Module) -> torch.Tensor:
        total_penalty = torch.tensor(0.0)
        for param in model.parameters():
            total_penalty += param.pow(2).sum()
        return self.l2_penalty * total_penalty


class CauchyRegularizer:
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, model: nn.Module) -> torch.Tensor:
        total_penalty: float = 0.0
        n_params: int = 0
        for param in model.parameters():
            total_penalty += torch.log(1 + torch.square(param / self.scale)).sum()
            n_params += param.numel()

        total_penalty /= n_params
        return total_penalty


class ModelRegularizer:
    def __init__(self):
        self._regularizers = list()

    def add_l1_regularizer(self, l1_penalty: float = 1e-3) -> None:
        self._regularizers.append(L1Regularizer(l1_penalty))

    def add_l2_regularizer(self, l2_penalty: float = 1e-3) -> None:
        self._regularizers.append(L2Regularizer(l2_penalty))

    def add_cauchy_regularizer(self, cauchy_scale: float = 1.0) -> None:
        self._regularizers.append(CauchyRegularizer(cauchy_scale))

    def __call__(self, model: nn.Module) -> torch.Tensor:
        reg_penalty = torch.tensor(0.0)
        for reg in self._regularizers:
            reg_penalty += reg(model)
        return reg_penalty


def create_regularizer(reg_options: RegularizerOptions) -> ModelRegularizer:
    regularizer = ModelRegularizer()
    if reg_options.l1_penalty > 0.0:
        regularizer.add_l1_regularizer(reg_options.l1_penalty)
    if reg_options.l2_penalty > 0.0:
        regularizer.add_l2_regularizer(reg_options.l2_penalty)
    if reg_options.cauchy_scale > 0.0:
        regularizer.add_cauchy_regularizer(reg_options.cauchy_scale)
    return regularizer
