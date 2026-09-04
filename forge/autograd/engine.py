from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.tensor import Tensor


class Function:
    """Base class for all differentiable operations"""
    def __init__(self) -> None:
        self.inputs: list["Tensor"] = []
        self.saved_tensors: list["Tensor"] = []

    def save_for_backward(self, *tensors: "Tensor") -> None:
        self.saved_tensors = list(tensors)

    def forward(self, *args) -> "Tensor":
        raise NotImplementedError

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        raise NotImplementedError