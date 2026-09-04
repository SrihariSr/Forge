from __future__ import annotations

import array as _array
import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from forge.tensor import Tensor

class Optimizer:
    """Base class for all optimizers."""

    # lr = learning rate
    def __init__(self, parameters: Iterable["Tensor"], lr: float = 0.01) -> None:
        self.parameters = list(parameters)
        self.lr = lr

    def step(self) -> None:
        raise NotImplementedError

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

class SGD(Optimizer):
    # Stochastic Gradient Descent

    def __init__(self, parameters: Iterable["Tensor"], lr: float = 0.01, momentum: float = 0.0) -> None:
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocities = []

        for param in self.parameters:
            self.velocities.append(_array.array(param.dtype.typecode, [0.0] * len(param._data)))

    def step(self) -> None:
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            for j in range(len(param._data)):
                if self.momentum > 0:
                    self.velocities[i][j] = ((self.momentum * self.velocities[i][j]) + param.grad._data[j])
                    param._data[j] -= self.lr * self.velocities[i][j]
                else:
                    param._data[j] -= self.lr * param.grad._data[j]

class Adam(Optimizer):
    # Adam optimizer

    def __init__(self, parameters: Iterable["Tensor"], lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        self.m = []  # first moment (mean of gradients)
        self.v = []  # second moment (mean of squared gradients)

        for param in self.parameters:
            self.m.append(
                _array.array(param.dtype.typecode, [0.0] * len(param._data))
            )
            self.v.append(
                _array.array(param.dtype.typecode, [0.0] * len(param._data))
            )

    def step(self) -> None:
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            for j in range(len(param._data)):
                g = param.grad._data[j]

                # Update first moment (mean of gradients)
                self.m[i][j] = self.beta1 * self.m[i][j] + (1 - self.beta1) * g

                # Update second moment (mean of squared gradients)
                self.v[i][j] = self.beta2 * self.v[i][j] + (1 - self.beta2) * g * g

                # Bias correction
                m_hat = self.m[i][j] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i][j] / (1 - self.beta2 ** self.t)

                # Update parameter
                param._data[j] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

