from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


@dataclass(kw_only=True, frozen=True)
class RegularizerOptions:
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    cauchy_penalty: float = 0.0
    cauchy_scale: float = 1.0


def l2_penalty(weights: torch.Tensor) -> torch.Tensor:
    return weights.pow(2).sum()


def l1_penalty(weights: torch.Tensor) -> torch.Tensor:
    return weights.abs().sum()


def cauchy_penalty(weights: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.log(1 + torch.square(weights / scale)).sum()


class Regularizer(Protocol):
    def __call__(self, model: nn.Module) -> torch.Tensor: ...


class WeightsRegularizer:
    def __init__(self, penalty_fn, lambda_: float = 1.0):
        self.penalty_fn = penalty_fn
        self.lambda_ = lambda_

    def __call__(self, model: nn.Module) -> torch.Tensor:
        return self.lambda_ * sum(
            self.penalty_fn(p.flatten()) for p in model.parameters()
        )


class ZeroRegularizer:
    def __call__(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)


class CompositeRegularizer:
    def __init__(self, regularizers: list[WeightsRegularizer]):
        self._regularizers = regularizers

    def __call__(self, model: nn.Module) -> torch.Tensor:
        total = torch.tensor(0.0)
        for reg in self._regularizers:
            total = total + reg(model)
        return total


class RegularizerBuilder:
    def __init__(self):
        self._regularizers: list[WeightsRegularizer] = list()

    def add_l1_regularizer(self, lambda_: float = 1.0) -> None:
        self._regularizers.append(WeightsRegularizer(l1_penalty, lambda_))

    def add_l2_regularizer(self, lambda_: float = 1.0) -> None:
        self._regularizers.append(WeightsRegularizer(l2_penalty, lambda_))

    def add_cauchy_regularizer(self, lambda_: float = 1.0, scale: float = 1.0) -> None:
        self._regularizers.append(
            WeightsRegularizer(lambda w: cauchy_penalty(w, scale), lambda_)
        )

    def build(self) -> Regularizer:
        return CompositeRegularizer(list(self._regularizers))


def build_regularizer(reg_options: RegularizerOptions) -> Regularizer:
    builder = RegularizerBuilder()
    if reg_options.l1_penalty > 0.0:
        builder.add_l1_regularizer(reg_options.l1_penalty)
    if reg_options.l2_penalty > 0.0:
        builder.add_l2_regularizer(reg_options.l2_penalty)
    if reg_options.cauchy_penalty > 0.0:
        builder.add_cauchy_regularizer(
            reg_options.cauchy_penalty, reg_options.cauchy_scale
        )
    return builder.build()
