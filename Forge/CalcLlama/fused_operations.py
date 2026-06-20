import array as _array
import math
from Forge.CalcLlama.engine import Function


class FusedLinearReLU(Function):
    """Fused matmul + bias add + ReLU in a single pass"""

    def forward(self, x, weight_t, bias):
        self.inputs = [x, weight_t, bias]
        self.save_for_backward(x, weight_t, bias)
        from Forge.tensor import Tensor

        m = x.shape[0]
        n = x.shape[1]
        p = weight_t.shape[1]

        new_data = _array.array(x.dtype.typecode, [])

        for i in range(m):
            for j in range(p):
                # Matmul
                total = 0.0
                for k in range(n):
                    total += x._data[i * n + k] * weight_t._data[k * p + j]

                # Add bias
                total += bias._data[j]

                # ReLU
                if total < 0:
                    total = 0.0

                new_data.append(total)

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (m, p)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        # Save output for backward (need to know where ReLU killed values)
        self._output = result
        return result

    def backward(self, grad_output):
        x, weight_t, bias = self.saved_tensors
        output = self._output
        from Forge.tensor import Tensor, _broadcast_shape, _broadcast_data
        from Forge.CalcLlama.operations import _unbroadcast

        m = x.shape[0]
        n = x.shape[1]
        p = weight_t.shape[1]

        # Apply ReLU mask to grad_output
        relu_grad_data = _array.array(
            grad_output.dtype.typecode,
            [g if o > 0 else 0.0 for g, o in zip(grad_output._data, output._data)]
        )
        relu_grad = Tensor.__new__(Tensor)
        relu_grad._data = relu_grad_data
        relu_grad.shape = grad_output.shape
        relu_grad.dtype = grad_output.dtype
        relu_grad.requires_grad = False
        relu_grad.grad = None
        relu_grad._grad_fn = None

        # Grad for x: relu_grad @ weight_t.T
        # weight_t is (n, p), weight_t.T is (p, n)
        grad_x_data = _array.array(x.dtype.typecode, [0.0] * (m * n))
        for i in range(m):
            for j in range(n):
                total = 0.0
                for k in range(p):
                    total += relu_grad._data[i * p + k] * weight_t._data[j * p + k]
                grad_x_data[i * n + j] = total

        grad_x = Tensor.__new__(Tensor)
        grad_x._data = grad_x_data
        grad_x.shape = x.shape
        grad_x.dtype = x.dtype
        grad_x.requires_grad = False
        grad_x.grad = None
        grad_x._grad_fn = None

        # Grad for weight_t: x.T @ relu_grad
        grad_wt_data = _array.array(x.dtype.typecode, [0.0] * (n * p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for k in range(m):
                    total += x._data[k * n + i] * relu_grad._data[k * p + j]
                grad_wt_data[i * p + j] = total

        grad_wt = Tensor.__new__(Tensor)
        grad_wt._data = grad_wt_data
        grad_wt.shape = weight_t.shape
        grad_wt.dtype = weight_t.dtype
        grad_wt.requires_grad = False
        grad_wt.grad = None
        grad_wt._grad_fn = None

        # Grad for bias: sum relu_grad along rows
        grad_bias_data = _array.array(bias.dtype.typecode, [0.0] * p)
        for i in range(m):
            for j in range(p):
                grad_bias_data[j] += relu_grad._data[i * p + j]

        grad_bias = Tensor.__new__(Tensor)
        grad_bias._data = grad_bias_data
        grad_bias.shape = bias.shape
        grad_bias.dtype = bias.dtype
        grad_bias.requires_grad = False
        grad_bias.grad = None
        grad_bias._grad_fn = None

        return grad_x, grad_wt, grad_bias