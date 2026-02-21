import torch
from torch import nn


class L1Regularizer:
    def __init__(self, lambda_: float = 1e-3):
        self.lambda_ = lambda_

    def __call__(self, model: nn.Module) -> torch.Tensor:
        reg_penalty = torch.tensor(0.0)
        for param in model.parameters():
            reg_penalty += param.abs().sum()
        return reg_penalty * self.lambda_


class L2Regularizer:
    def __init__(self, lambda_: float = 1e-3):
        self.lambda_ = lambda_

    def __call__(self, model: nn.Module) -> torch.Tensor:
        reg_penalty = torch.tensor(0.0)
        for param in model.parameters():
            reg_penalty += param.pow(2).sum()
        return reg_penalty * self.lambda_


class CauchyRegularizer:
    def __init__(self, lambda_: float = 1e-3, gamma: float = 1):
        self.lambda_ = lambda_
        self.gamma = gamma

    def __call__(self, model: nn.Module) -> torch.Tensor:
        reg_penalty = 0.0
        for param in model.parameters():
            reg_penalty += torch.log(param.pow(2) + self.gamma**2).sum()
        return self.lambda_ * reg_penalty


class ModelRegularizer:
    def __init__(self):
        self._regularizers = list()

    def add_l1_regularizer(self, lambda_: float = 1e-3) -> None:
        self._regularizers.append(L1Regularizer(lambda_))

    def add_l2_regularizer(self, lambda_: float = 1e-3) -> None:
        self._regularizers.append(L2Regularizer(lambda_))

    def add_cauchy_regularizer(
        self, lambda_: float = 1e-3, gamma: float = 1e-2
    ) -> None:
        self._regularizers.append(CauchyRegularizer(lambda_, gamma))

    def __call__(self, model: nn.Module) -> torch.Tensor:
        reg_penalty = torch.tensor(0.0)
        for reg in self._regularizers:
            reg_penalty += reg(model)
        return reg_penalty
