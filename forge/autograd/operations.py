from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forge.tensor import Tensor

from forge.autograd.engine import Function
import array as _array
import math as _math

def _unbroadcast(grad: "Tensor", original_shape: tuple[int, ...]) -> "Tensor":
    """Sum grad along dimensions that were broadcast to match original_shape."""
    from forge.tensor import Tensor
    import array as _arr

    grad_shape = grad.shape
    if grad_shape == original_shape:
        return grad

    # Pad original shape with 1s on the left
    pad = len(grad_shape) - len(original_shape)
    padded_original = (1,) * pad + original_shape

    # Sum along broadcast dimensions
    data = list(grad._data)
    current_shape = list(grad_shape)

    for dim in range(len(current_shape)):
        if padded_original[dim] == 1 and current_shape[dim] > 1:
            # Sum along this dimension
            new_size = 1
            for s in current_shape:
                new_size *= s
            new_size //= current_shape[dim]

            new_data = [0.0] * new_size

            # Compute strides
            stride_before = 1
            for d in range(dim + 1, len(current_shape)):
                stride_before *= current_shape[d]
            dim_size = current_shape[dim]

            for i in range(len(new_data)):
                # Figure out position in the reduced array
                outer = i // stride_before
                inner = i % stride_before
                for k in range(dim_size):
                    old_idx = outer * dim_size * stride_before + k * stride_before + inner
                    new_data[i] += data[old_idx]

            data = new_data
            current_shape[dim] = 1

    # Remove leading 1s to match original shape
    final_shape = original_shape
    total = 1
    for s in final_shape:
        total *= s

    result = Tensor.__new__(Tensor)
    result._data = _arr.array(grad.dtype.typecode, data[:total])
    result.shape = final_shape
    result.dtype = grad.dtype
    result.requires_grad = False
    result.grad = None
    result._grad_fn = None
    return result

class Add(Function):
    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from forge.tensor import _broadcast_shape, _broadcast_data, Tensor

        result_shape = _broadcast_shape(a.shape, b.shape)
        data_a = _broadcast_data(a._data, a.shape, result_shape, a.dtype.typecode)
        data_b = _broadcast_data(b._data, b.shape, result_shape, b.dtype.typecode)

        new_data = _array.array(a.dtype.typecode, [x + y for x, y in zip(data_a, data_b)])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = result_shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(grad_output, b.shape)
        return grad_a, grad_b

class Sub(Function):
    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from forge.tensor import _broadcast_shape, _broadcast_data, Tensor

        result_shape = _broadcast_shape(a.shape, b.shape)
        data_a = _broadcast_data(a._data, a.shape, result_shape, a.dtype.typecode)
        data_b = _broadcast_data(b._data, b.shape, result_shape, b.dtype.typecode)

        new_data = _array.array(a.dtype.typecode, [x - y for x, y in zip(data_a, data_b)])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = result_shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result


    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(-grad_output, b.shape)
        return grad_a, grad_b

class Mul(Function):
    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from forge.tensor import _broadcast_shape, _broadcast_data, Tensor

        result_shape = _broadcast_shape(a.shape, b.shape)
        data_a = _broadcast_data(a._data, a.shape, result_shape, a.dtype.typecode)
        data_b = _broadcast_data(b._data, b.shape, result_shape, b.dtype.typecode)

        new_data = _array.array(a.dtype.typecode, [x * y for x, y in zip(data_a, data_b)])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = result_shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output * b, a.shape)
        grad_b = _unbroadcast(grad_output * a, b.shape)
        return grad_a, grad_b

class Pow(Function):
    def forward(self, a: "Tensor", exp: int | float) -> "Tensor":
        self.inputs = [a]
        self.exp = exp
        self.save_for_backward(a)
        from forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [x ** exp for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        return (grad_output * self.exp * (a ** (self.exp - 1)),)

class Neg(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [-x for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        return (-grad_output,)

class Sum(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        self.save_for_backward(a)
        from forge.tensor import Tensor

        total = 0.0
        for x in a._data:
            total += x

        result = Tensor(total)
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor

        # grad flows equally to every element
        numel = 1
        for s in a.shape:
            numel *= s
        grad_data = [grad_output._data[0]] * numel

        import array as _array
        result = Tensor.__new__(Tensor)
        result._data = _array.array(a.dtype.typecode, grad_data)
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Mean(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        self.save_for_backward(a)
        from forge.tensor import Tensor

        total = 0.0
        for x in a._data:
            total += x

        numel = 1
        for s in a.shape:
            numel *= s

        result = Tensor(total / numel)
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor

        numel = 1
        for s in a.shape:
            numel *= s
        grad_val = grad_output._data[0] / numel
        grad_data = [grad_val] * numel

        import array as _array
        result = Tensor.__new__(Tensor)
        result._data = _array.array(a.dtype.typecode, grad_data)
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Relu(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor
        self.save_for_backward(a)
        new_data = _array.array(a.dtype.typecode, [x if x > 0 else 0.0 for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor
        mask = _array.array(a.dtype.typecode, [1.0 if x > 0 else 0.0 for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = _array.array(a.dtype.typecode,
                                    [g * m for g, m in zip(grad_output._data, mask)])
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Transpose(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor
        if len(a.shape) != 2:
            raise ValueError("Transpose only supports 2D tensors")
        rows, cols = a.shape
        new_data: _array.array = _array.array(a.dtype.typecode, [])
        for j in range(cols):
            for i in range(rows):
                new_data.append(a._data[i * cols + j])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (cols, rows)
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        return (grad_output.transpose(),)

class Matmul(Function):
    def forward(self, left: "Tensor", right: "Tensor") -> "Tensor":
            # Save the operands so backward() can compute their gradients later.
            self.inputs = [left, right]
            self.save_for_backward(left, right)

            from forge.tensor import Tensor
            from forge.dtype import float32
            from forge.autograd.mps_backend import MPS_AVAILABLE, mps_matmul, GPU_MIN_WORK
            from forge.autograd.accelerate_backend import ACCELERATE_AVAILABLE, accelerate_matmul

            # Matmul only handles 2D matrices.
            if len(left.shape) != 2 or len(right.shape) != 2:
                raise ValueError("matmul only supports 2D tensors")

            # For A @ B to be valid, A's column count must equal B's row count.
            if left.shape[1] != right.shape[0]:
                raise ValueError(
                    f"matmul shape mismatch: ({left.shape[0]}, {left.shape[1]}) "
                    f"@ ({right.shape[0]}, {right.shape[1]})"
                )

            # Name the three dimensions of the multiplication:
            #   the result is (left_rows x right_cols)
            #   shared_dim is the inner dimension summed over (left cols == right rows)
            left_rows  = left.shape[0]
            shared_dim = left.shape[1]
            right_cols = right.shape[1]

            # Total multiply-add operations. Used to decide if the GPU is worthwhile.
            total_work = left_rows * shared_dim * right_cols

            # Both accelerated backends are single-precision, so they need float32.
            both_float32 = left.dtype == float32 and right.dtype == float32

            if both_float32 and MPS_AVAILABLE and total_work >= GPU_MIN_WORK:
                # Large matrix: the GPU's throughput outweighs its per-call overhead.
                result_data = mps_matmul(
                    left._data, right._data, left_rows, shared_dim, right_cols
                )

            elif both_float32 and ACCELERATE_AVAILABLE:
                # Smaller matrix: the CPU BLAS routine wins (almost no overhead).
                result_data = accelerate_matmul(
                    left._data, right._data, left_rows, shared_dim, right_cols
                )

            else:
                # Fallback: no accelerated backend, or a non-float32 dtype.
                # Compute every output element directly with a triple loop.
                result_data = _array.array(left.dtype.typecode, [])
                for row in range(left_rows):
                    for col in range(right_cols):
                        dot = 0.0
                        # Sum left[row, s] * right[s, col] over the shared dimension.
                        for s in range(shared_dim):
                            dot += left._data[row * shared_dim + s] * right._data[s * right_cols + col]
                        result_data.append(dot)

            # Wrap the flat result back into a Tensor of shape (left_rows x right_cols).
            result = Tensor.__new__(Tensor)
            result._data = result_data
            result.shape = (left_rows, right_cols)
            result.dtype = left.dtype
            result.requires_grad = False
            result.grad = None
            result._grad_fn = None
            return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a, b = self.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b

# ReLU(x) = max(0, x)
# d/dx(ReLU(x)) = 1 if x > 0 else 0
class ReLU(Function): 
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        self.save_for_backward(a)
        from forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [max(0.0, x) for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor

        mask = _array.array(a.dtype.typecode, [1.0 if x > 0 else 0.0 for x in a._data])
        grad_data = _array.array(a.dtype.typecode, [g * m for g, m in zip(grad_output._data, mask)])

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

# Sigmoid(x) = 1 / (1 + e^(-x))
# d/dx(Sigmoid(x)) = Sigmoid(x) * (1 - Sigmoid(x))
class Sigmoid(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor
        import math

        sig_data = _array.array(a.dtype.typecode, [1.0 / (1.0 + math.exp(-x)) for x in a._data])

        result = Tensor.__new__(Tensor)
        result._data = sig_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        self.save_for_backward(result)
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        sig = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(sig.dtype.typecode, [g * s * (1.0 - s) for g, s in zip(grad_output._data, sig._data)])

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = sig.shape
        result.dtype = sig.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

# Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
# d/dx(Tanh(x)) = 1 - Tanh(x)^2
class Tanh(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor
        import math

        tanh_data = _array.array(a.dtype.typecode, [math.tanh(x) for x in a._data])

        result = Tensor.__new__(Tensor)
        result._data = tanh_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        self.save_for_backward(result)
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        tanh_out = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(tanh_out.dtype.typecode, [g * (1.0 - (t * t)) for g, t in zip(grad_output._data, tanh_out._data)])

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = tanh_out.shape
        result.dtype = tanh_out.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

# Log(x) = ln(x)
# d/dx(Log(x)) = 1/x
class Log(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        self.save_for_backward(a)
        from forge.tensor import Tensor
        import math

        new_data = _array.array(a.dtype.typecode, [math.log(x) for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(a.dtype.typecode, [g / x for g, x in zip(grad_output._data, a._data)])

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Clamp(Function):
    def forward(self, a: "Tensor", min_val: float, max_val: float) -> "Tensor":
        self.inputs = [a]
        self.min_val = min_val
        self.max_val = max_val
        self.save_for_backward(a)
        from forge.tensor import Tensor

        new_data = _array.array(
            a.dtype.typecode,
            [max(min_val, min(max_val, x)) for x in a._data]
        )
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        a = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(a.dtype.typecode, [g if self.min_val < x < self.max_val else 0.0 for g, x in zip(grad_output._data, a._data)])

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Softmax(Function):
    def forward(self, a: "Tensor") -> "Tensor":
        self.inputs = [a]
        from forge.tensor import Tensor
        import math

        if len(a.shape) != 2:
            raise ValueError("Softmax currently only supports 2D tensors")

        rows = a.shape[0]
        cols = a.shape[1]

        new_data = _array.array(a.dtype.typecode, [0.0] * len(a._data))

        for i in range(rows):
            row_start = i * cols

            # Finding highest value in this row
            row_max = a._data[row_start]
            for j in range(1, cols):
                if a._data[row_start + j] > row_max:
                    row_max = a._data[row_start + j]

            # Computing exp(x - max) and sum
            # x - max avoids using large values of x which might overflow
            exp_sum = 0.0
            for j in range(cols):
                val = math.exp(a._data[row_start + j] - row_max)
                new_data[row_start + j] = val
                exp_sum += val

            # Normalize
            for j in range(cols):
                new_data[row_start + j] /= exp_sum

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        self.save_for_backward(result)
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        softmax_out = self.saved_tensors[0]
        from forge.tensor import Tensor

        rows = softmax_out.shape[0]
        cols = softmax_out.shape[1]

        grad_data = _array.array(softmax_out.dtype.typecode, [0.0] * len(softmax_out._data))

        for i in range(rows):
            row_start = i * cols

            # Compute dot product of grad and softmax for this row
            dot = 0.0
            for j in range(cols):
                dot += grad_output._data[row_start + j] * softmax_out._data[row_start + j]

            # grad = softmax * (grad_output - dot)
            for j in range(cols):
                grad_data[row_start + j] = softmax_out._data[row_start + j] * (
                    grad_output._data[row_start + j] - dot
                )

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = softmax_out.shape
        result.dtype = softmax_out.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class SelectBatch(Function):
    def forward(self, x: "Tensor", b: int) -> "Tensor":
        self.inputs = [x]
        self.b = b
        self.x_shape = x.shape
        self.x_dtype = x.dtype
        from forge.tensor import Tensor

        batch, seq, dim = x.shape
        start = b * seq * dim
        end = (b + 1) * seq * dim
        new_data = _array.array(x.dtype.typecode, x._data[start:end])

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (seq, dim)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor

        batch, seq, dim = self.x_shape
        total = batch * seq * dim
        grad_data = _array.array(self.x_dtype.typecode, [0.0] * total)

        start = self.b * seq * dim
        for i in range(seq * dim):
            grad_data[start + i] = grad_output._data[i]
        
        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = self.x_shape
        result.dtype = self.x_dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return (result,)

class RowMean(Function):
    """
    Averages each row seperately.
    """

    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        self.save_for_backward(x)
        
        from forge.tensor import Tensor
        rows, cols = x.shape

        means: _array.array = _array.array(x.dtype.typecode, [])
        for r in range(rows):
            total = 0.0
            for c in range(cols):
                total += x._data[r * cols + c]
            means.append(total / cols)
        
        result = Tensor.__new__(Tensor)
        result._data = means
        result.shape = (rows, 1)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return result
    
    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        x = self.saved_tensors[0]
        
        from forge.tensor import Tensor
        rows, cols = x.shape

        grad_data = _array.array(x.dtype.typecode, [])
        for r in range(rows):
            share_of_grad = grad_output._data[r] / cols
            for c in range(cols):
                grad_data.append(share_of_grad)
        
        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return (result,)

# Required to get standard deviation from variance
class Sqrt(Function):
    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        from forge.tensor import Tensor
        import math
        
        new_data = _array.array(x.dtype.typecode, [math.sqrt(d) for d in x._data])

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        self.save_for_backward(result)
        
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        sqrt_x = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(
            sqrt_x.dtype.typecode,
            [g / (2.0 * s) for g, s in zip(grad_output._data, sqrt_x._data)]
        )

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = sqrt_x.shape
        result.dtype = sqrt_x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return (result,)

class CausalMask(Function):
    def forward(self, scores: "Tensor") -> "Tensor":
        self.inputs = [scores]
        from forge.tensor import Tensor

        rows, cols = scores.shape
        new_data = _array.array(scores.dtype.typecode, scores._data)

        NEGINF = -1e50

        for i in range(rows):
            for j in range(cols):
                if j > i:
                    new_data[i * cols + j] = NEGINF

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = scores.shape
        result.dtype = scores.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor

        rows, cols = grad_output.shape
        grad_data = _array.array(grad_output.dtype.typecode, [0.0] * (rows * cols))

        for i in range(rows):
            for j in range(cols):
                if j <= i:
                    grad_data[i * cols + j] = grad_output._data[i * cols + j]

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = grad_output.shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return (result,)

class ConcatColumns(Function):
    """
    Joins 2D tensors side by side: (rows, a) and (rows, b) -> (rows, a + b).
    """

    def forward(self, x: "Tensor", *others: "Tensor") -> "Tensor":
        from forge.tensor import Tensor

        tensors = [x, *others]
        self.inputs = list(tensors)

        rows = x.shape[0]
        for t in tensors:
            if len(t.shape) != 2:
                raise ValueError("concat_columns expects 2D tensors")
            if t.shape[0] != rows:
                raise ValueError(
                    f"concat_columns needs matching row counts, got {rows} and {t.shape[0]}"
                )

        self.widths = [t.shape[1] for t in tensors]
        total_cols = sum(self.widths)

        new_data = _array.array(x.dtype.typecode, [0.0] * (rows * total_cols))

        offset = 0
        for t, width in zip(tensors, self.widths):
            for i in range(rows):
                for j in range(width):
                    new_data[i * total_cols + offset + j] = t._data[i * width + j]
            offset += width

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (rows, total_cols)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor

        rows, total_cols = grad_output.shape

        grads = []
        offset = 0
        for width in self.widths:
            grad_data = _array.array(grad_output.dtype.typecode, [0.0] * (rows * width))
            for i in range(rows):
                for j in range(width):
                    grad_data[i * width + j] = grad_output._data[i * total_cols + offset + j]
            offset += width

            result = Tensor.__new__(Tensor)
            result._data = grad_data
            result.shape = (rows, width)
            result.dtype = grad_output.dtype
            result.requires_grad = False
            result.grad = None
            result._grad_fn = None
            grads.append(result)

        return tuple(grads)
    
class Gelu(Function):
    """
    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        self.save_for_backward(x)
        from forge.tensor import Tensor
        import math
        
        c = math.sqrt(2.0 / math.pi)
        new_data: _array.array = _array.array(x.dtype.typecode, [])
        for d in x._data:
            inner = c * (d + 0.044715 * d**3)
            new_data.append(0.5 * d * (1.0 + math.tanh(inner)))

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        x = self.saved_tensors[0]
        from forge.tensor import Tensor
        import math

        c = math.sqrt(2.0 / math.pi)
        grad_data = _array.array(x.dtype.typecode, [])
        for g, v in zip(grad_output._data, x._data):
            inner = c * (v + 0.044715 * v ** 3)
            t = math.tanh(inner)
            d_inner = c * (1.0 + 3.0 * 0.044715 * v * v)
            local = 0.5 * (1.0 + t) + 0.5 * v * (1.0 - t * t) * d_inner
            grad_data.append(g * local)

        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class UnsqueezeBatch(Function):
    """
    Turns a (rows, cols) tensor into (1, rows, cols) tensor.
    """
    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        self.x_shape = x.shape
        from forge.tensor import Tensor

        result = Tensor.__new__(Tensor)
        result._data = _array.array(x.dtype.typecode, x._data)
        result.shape = (1, x.shape[0], x.shape[1])
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor
        result = Tensor.__new__(Tensor)
        result._data = _array.array(grad_output.dtype.typecode, grad_output._data)
        result.shape = self.x_shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)

class Exp(Function):
    """
    Element-wise e^x.
    """
    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        from forge.tensor import Tensor
        import math

        new_data = _array.array(x.dtype.typecode, [math.exp(d) for d in x._data])

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        self.save_for_backward(result)

        return result
    
    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        exp_x = self.saved_tensors[0]
        from forge.tensor import Tensor

        grad_data = _array.array(
            exp_x.dtype.typecode,
            [g * e for g, e in zip(grad_output._data, exp_x._data)]
        )
        
        result = Tensor.__new__(Tensor)
        result._data = grad_data
        result.shape = exp_x.shape
        result.dtype = exp_x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return (result,)

class FlattenBatch(Function):
    """
    Flattens a 3D batched tensor into a 2D tensor by stacking every sequence in the batch on top of the next.

    (batch, seq, dim)  ->  (batch * seq, dim)
    """
    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        self._x_shape = x.shape
        from forge.tensor import Tensor

        batch, seq, dim = x.shape

        new_data = _array.array(x.dtype.typecode, x._data)

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (batch * seq, dim)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor

        result = Tensor.__new__(Tensor)
        result._data = _array.array(grad_output.dtype.typecode, grad_output._data)
        result.shape = self._x_shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return (result,)


class UnflattenBatch(Function):
    """
    Takes a 2D tensor and splits it into a batch of sequences.
    This is the exact opposite of `FlattenBatch`.

    (batch * seq, dim) -> (batch, seq, dim)
    """
    def forward(self, x: "Tensor", batch: int) -> "Tensor":
        self.inputs = [x]
        from forge.tensor import Tensor
        
        self._x_shape = x.shape

        rows, dim = x.shape
        seq = rows // batch

        new_data = _array.array(x.dtype.typecode, x._data)

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (batch, seq, dim)
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return result
    
    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor

        result = Tensor.__new__(Tensor)
        result._data = _array.array(grad_output.dtype.typecode, grad_output._data)
        result.shape = self._x_shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return (result,)

class StackBatch(Function):
    """
    Stacks several 2D tensors into one 3D tensor.
    
        n tensors of (seq, dim) -> (n, seq, dim)
    """
    def forward(self, *tensors: "Tensor") -> "Tensor":
        self.inputs = list(tensors)
        from forge.tensor import Tensor

        first = tensors[0]
        seq, dim = first.shape
        self.piece_size = seq * dim
        self.piece_shape = (seq, dim)
        self.count = len(tensors)

        # Every tensor needs to have the same shape
        for t in tensors:
            if t.shape != first.shape:
                raise ValueError(f"StackBatch needs matching shapes!")
        
        # Lay the pieces end-to-end
        combined: _array.array = _array.array(first.dtype.typecode)
        for t in tensors:
            combined.extend(t._data)

        result = Tensor.__new__(Tensor)
        result._data = combined
        result.shape = (self.count, seq, dim)
        result.dtype = first.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None

        return result
    
    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor
        
        grads = []
        for i in range(self.count):
            start = i * self.piece_size
            end = start + self.piece_size

            piece = Tensor.__new__(Tensor)
            piece._data = _array.array(grad_output.dtype.typecode, grad_output._data[start:end])
            piece.shape = self.piece_shape
            piece.dtype = grad_output.dtype
            piece.requires_grad = False
            piece.grad = None
            piece._grad_fn = None
            grads.append(piece)
        
        return tuple(grads)

class BatchedMatmul(Function):
    """
    A stack of independant matrix multiplications done in one operation.
    
    Every call into the autograd system builds a Function object, 
    saves tensors for the backward pass, allocates a result
    Tensor and registers it in the graph. 
    """
    def forward(self, left: "Tensor", right: "Tensor") -> "Tensor":
        self.inputs = [left, right]
        self.save_for_backward(left, right)

        # pyrefly: ignore [missing-import]
        from forge.tensor import Tensor
        # pyrefly: ignore [missing-import]
        from forge.autograd.accelerate_backend import ACCELERATE_AVAILABLE, accelerate_matmul
        from forge.dtype import float32

        use_blas = ACCELERATE_AVAILABLE and left.dtype is float32 and right.dtype is float32
        
        if len(left.shape) != 3 or len(right.shape) != 3:
            raise ValueError("BatchedMatmul needs two 3D tensors.")
        
        batch, m, k = left.shape
        rbatch, rk, n = right.shape

        if batch != rbatch:
            raise ValueError(f"BatchedMatmul batch sizes differ: {batch} and {rbatch}.")
        if k != rk:
            raise ValueError(f"BatchedMatmul shape mismatch: ({m}, {k}) @ ({rk}, {n}).")
        
        self.dims = (batch, m, k, n)

        out = _array.array(left.dtype.typecode, [0.0] * (batch * m * n))

        left_size, right_size, out_size = m * k, k * n, m * n

        for b in range(batch):
            lo = b * left_size
            ro = b * right_size
            oo = b * out_size

            if use_blas:
                left_slice = _array.array('f', left._data[lo:lo + left_size])
                right_slice = _array.array('f', right._data[ro:ro + right_size])
                product = accelerate_matmul(left_slice, right_slice, m, k, n)
                out[oo:oo + out_size] = product
            else:
                for i in range(m):
                   for j in range(n):
                       total = 0.0
                       for s in range(k):
                           total += left._data[lo + i * k + s] * right._data[ro + s * n + j]
                       out[oo + i * n + j] = total

        result = Tensor.__new__(Tensor)
        result._data = out
        result.shape = (batch, m, n)
        result.dtype = left.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        
        return result

    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        from forge.tensor import Tensor
        from forge.autograd.accelerate_backend import ACCELERATE_AVAILABLE, accelerate_matmul
        from forge.dtype import float32

        left, right = self.saved_tensors
        batch, m, k, n = self.dims

        use_blas = ACCELERATE_AVAILABLE and left.dtype is float32 and right.dtype is float32

        grad_left = _array.array(left.dtype.typecode, [0.0] * (batch * m * k))
        grad_right = _array.array(right.dtype.typecode, [0.0] * (batch * k * n))

        left_size, right_size, out_size = m * k, k * n, m * n

        for b in range(batch):
            lo, ro, oo = b * left_size, b * right_size, b * out_size
            if use_blas:
                g_slice = _array.array('f', grad_output._data[oo:oo + out_size])
                l_slice = _array.array('f', left._data[lo:lo + left_size])
                r_slice = _array.array('f', right._data[ro:ro + right_size])

                # grad_left = grad_out @ right.T, so transpose right first.
                # (k, n) becomes (n, k).
                r_t = _array.array('f', [0.0] * (k * n))
                for s in range(k):
                    for j in range(n):
                        r_t[j * k + s] = r_slice[s * n + j]
                grad_left[lo:lo + left_size] = accelerate_matmul(g_slice, r_t, m, n, k)

                # grad_right = left.T @ grad_out, so transpose left first.
                # (m, k) becomes (k, m).
                l_t = _array.array('f', [0.0] * (m * k))
                for i in range(m):
                    for s in range(k):
                        l_t[s * m + i] = l_slice[i * k + s]
                grad_right[ro:ro + right_size] = accelerate_matmul(l_t, g_slice, k, m, n)
            else:
                for i in range(m):
                    for s in range(k):
                        total = 0.0
                        for j in range(n):
                            total += grad_output._data[oo + i * n + j] * right._data[ro + s * n + j]
                        grad_left[lo + i * k + s] = total

                for s in range(k):
                    for j in range(n):
                        total = 0.0
                        for i in range(m):
                            total += left._data[lo + i * k + s] * grad_output._data[oo + i * n + j]
                        grad_right[ro + s * n + j] = total

        gl = Tensor.__new__(Tensor)
        gl._data = grad_left
        gl.shape = (batch, m, k)
        gl.dtype = left.dtype
        gl.requires_grad = False
        gl.grad = None
        gl._grad_fn = None

        gr = Tensor.__new__(Tensor)
        gr._data = grad_right
        gr.shape = (batch, k, n)
        gr.dtype = right.dtype
        gr.requires_grad = False
        gr.grad = None
        gr._grad_fn = None

        return (gl, gr)

class BatchedCausalMask(Function):
    """
    Applies the causal mask to each 3D slice seperately.
    """
    MASK_VALUE = -1e30

    def forward(self, x: "Tensor") -> "Tensor":
        self.inputs = [x]
        from forge.tensor import Tensor

        batch, rows, cols = x.shape
        self.dims = (batch, rows, cols)

        out = _array.array(x.dtype.typecode, x._data)
        size = rows * cols

        for b in range(batch):
            off = b * size
            for i in range(rows):
                for j in range(i + 1, cols):
                    out[off + i * cols + j] = self.MASK_VALUE

        result = Tensor.__new__(Tensor)
        result._data = out
        result.shape = x.shape
        result.dtype = x.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result
    
    def backward(self, grad_output: "Tensor") -> tuple["Tensor", ...]:
        """
        Masked entries were replaced by a constant, so they no longer depend on
        the input at all and their gradient is zero. Everything else passed
        through untouched and keeps its gradient.
        """
        from forge.tensor import Tensor

        batch, rows, cols = self.dims
        grad = _array.array(grad_output.dtype.typecode, grad_output._data)
        size = rows * cols

        for b in range(batch):
            off = b * size
            for i in range(rows):
                for j in range(i + 1, cols):
                    grad[off + i * cols + j] = 0.0

        result = Tensor.__new__(Tensor)
        result._data = grad
        result.shape = self.dims
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)
