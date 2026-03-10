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
    return weights.pow(2).mean()


def l1_penalty(weights: torch.Tensor) -> torch.Tensor:
    return weights.abs().mean()


def cauchy_penalty(weights: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    return torch.log(1 + torch.square(weights / scale)).mean()


class Regularizer(Protocol):
    def __call__(self, model: nn.Module) -> torch.Tensor: ...


class WeightsRegularizer:
    def __init__(self, penalty_fn, lambda_: float = 1.0):
        self.penalty_fn = penalty_fn
        self.lambda_ = lambda_

    def __call__(self, model: nn.Module) -> torch.Tensor:
        weights = torch.cat([p.flatten() for p in model.parameters()])
        total_penalty = self.penalty_fn(weights)
        return self.lambda_ * total_penalty


class ZeroRegularizer:
    def __call__(self, model: nn.Module) -> torch.Tensor:
        return torch.tensor(0.0)


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
        regularizers = list(self._regularizers)

        def regularizer(model: nn.Module) -> torch.Tensor:
            return sum((reg(model) for reg in regularizers), start=torch.tensor(0.0))

        return regularizer


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
