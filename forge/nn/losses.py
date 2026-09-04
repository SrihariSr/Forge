from __future__ import annotations

from typing import TYPE_CHECKING

from forge.nn.module import Module
import math

if TYPE_CHECKING:
    from forge.tensor import Tensor

class MSELoss(Module):
    # Mean Squared Error

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: "Tensor", target: "Tensor") -> "Tensor":
        diff = pred - target
        return (diff * diff).mean()

    def __repr__(self) -> str:
        return "MSELoss()"

class BCELoss(Module):
   # Binary Cross Entropy Loss

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: "Tensor", target: "Tensor") -> "Tensor":
        from forge.tensor import Tensor
        eps = 1e-7
        pred_clamped = pred.clamp(eps, 1 - eps)

        loss = -(target * pred_clamped.log() + (target * (-1) + 1) * (pred_clamped * (-1) + 1).log())
        return loss.mean()

    def __repr__(self) -> str:
        return "BCELoss()"

class CrossEntropyLoss(Module):
    # Cross Entropy Loss for multi-class classification

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: "Tensor", target) -> "Tensor":
        from forge.tensor import Tensor
        from forge.autograd.engine import Function
        import math
        import array as _array

        # Classes are the last dimension; every earlier dimension is a position
        # being classified. This keeps plain 2D (rows, classes) working while
        # also accepting batched 3D logits (batch, seq, classes).
        num_classes = pred.shape[-1]
        batch_size = pred.numel() // num_classes

        # --- Softmax with numerical stability ---
        probs_data = _array.array(pred.dtype.typecode, [0.0] * len(pred._data))

        for i in range(batch_size):
            row_start = i * num_classes

            # Find max for stability
            row_max = pred._data[row_start]
            for j in range(1, num_classes):
                if pred._data[row_start + j] > row_max:
                    row_max = pred._data[row_start + j]

            # Compute exp(x - max) and sum
            exp_sum = 0.0
            for j in range(num_classes):
                val = math.exp(pred._data[row_start + j] - row_max)
                probs_data[row_start + j] = val
                exp_sum += val

            # Normalize
            for j in range(num_classes):
                probs_data[row_start + j] /= exp_sum

        # --- Negative log likelihood ---
        loss_total = 0.0
        for i in range(batch_size):
            class_idx = target[i]
            prob = probs_data[i * num_classes + class_idx]
            loss_total += -math.log(max(prob, 1e-7))

        loss_val = loss_total / batch_size

        # --- Compute gradient for pred ---
        grad_data = _array.array(pred.dtype.typecode, list(probs_data))

        for i in range(batch_size):
            class_idx = target[i]
            grad_data[i * num_classes + class_idx] -= 1.0

        for i in range(len(grad_data)):
            grad_data[i] /= batch_size

        grad_tensor = Tensor.__new__(Tensor)
        grad_tensor._data = grad_data
        grad_tensor.shape = pred.shape
        grad_tensor.dtype = pred.dtype
        grad_tensor.requires_grad = False
        grad_tensor.grad = None
        grad_tensor._grad_fn = None

        # Create a proper grad_fn so backward() can find pred
        class _CEBackward(Function):
            def __init__(self, pred: "Tensor", grad_for_pred: "Tensor") -> None:
                super().__init__()
                self.inputs = [pred]
                self.saved_grad = grad_for_pred

            def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
                # grad_output is 1.0 (scalar), so just return the precomputed gradient
                return (self.saved_grad,)

        # Building the result tensor
        result = Tensor.__new__(Tensor)
        result._data = _array.array(pred.dtype.typecode, [loss_val])
        result.shape = ()
        result.dtype = pred.dtype
        result.requires_grad = True
        result.grad = None
        result._grad_fn = _CEBackward(pred, grad_tensor)

        return result

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"