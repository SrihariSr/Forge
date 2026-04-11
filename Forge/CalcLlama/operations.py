from Forge.CalcLlama.engine import Function
import array as _array
import math as _math

def _unbroadcast(grad, original_shape):
    # Sum grad along dimensions that were broadcast to match original_shape.
    from Forge.tensor import Tensor
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
    def forward(self, a, b):
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from Forge.tensor import _broadcast_shape, _broadcast_data, Tensor

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

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(grad_output, b.shape)
        return grad_a, grad_b

class Sub(Function):
    def forward(self, a, b):
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from Forge.tensor import _broadcast_shape, _broadcast_data, Tensor

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


    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(-grad_output, b.shape)
        return grad_a, grad_b

class Mul(Function):
    def forward(self, a, b):
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from Forge.tensor import _broadcast_shape, _broadcast_data, Tensor

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

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output * b, a.shape)
        grad_b = _unbroadcast(grad_output * a, b.shape)
        return grad_a, grad_b

class Pow(Function):
    def forward(self, a, exp):
        self.inputs = [a]
        self.exp = exp
        self.save_for_backward(a)
        from Forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [x ** exp for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        return (grad_output * self.exp * (a ** (self.exp - 1)),)

class Neg(Function):
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [-x for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output):
        return (-grad_output,)

class Sum(Function):
    def forward(self, a):
        self.inputs = [a]
        self.save_for_backward(a)
        from Forge.tensor import Tensor

        total = 0.0
        for x in a._data:
            total += x

        result = Tensor(total)
        return result

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        self.save_for_backward(a)
        from Forge.tensor import Tensor

        total = 0.0
        for x in a._data:
            total += x

        numel = 1
        for s in a.shape:
            numel *= s

        result = Tensor(total / numel)
        return result

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
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

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor
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



class Sigmoid(Function):
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
        new_data = _array.array(a.dtype.typecode,
                                [1.0 / (1.0 + _math.exp(-x)) for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        self._out_data = new_data
        return result

    def backward(self, grad_output):
        from Forge.tensor import Tensor
        result = Tensor.__new__(Tensor)
        tc = grad_output.dtype.typecode
        result._data = _array.array(tc,
                                    [g * s * (1.0 - s)
                                     for g, s in zip(grad_output._data, self._out_data)])
        result.shape = grad_output.shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)


class Tanh(Function):
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
        new_data = _array.array(a.dtype.typecode, [_math.tanh(x) for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        self._out_data = new_data
        return result

    def backward(self, grad_output):
        from Forge.tensor import Tensor
        result = Tensor.__new__(Tensor)
        tc = grad_output.dtype.typecode
        result._data = _array.array(tc,
                                    [g * (1.0 - t * t)
                                     for g, t in zip(grad_output._data, self._out_data)])
        result.shape = grad_output.shape
        result.dtype = grad_output.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return (result,)


class Transpose(Function):
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
        if len(a.shape) != 2:
            raise ValueError("Transpose only supports 2D tensors")
        rows, cols = a.shape
        new_data = _array.array(a.dtype.typecode, [])
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

    def backward(self, grad_output):
        return (grad_output.transpose(),)

class Matmul(Function):
    def forward(self, a, b):
        self.inputs = [a, b]
        self.save_for_backward(a, b)
        from Forge.tensor import Tensor

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ValueError("matmul only supports 2D tensors")

        if a.shape[1] != b.shape[0]:
            raise ValueError(
                f"matmul shape mismatch: ({a.shape[0]}, {a.shape[1]}) "
                f"@ ({b.shape[0]}, {b.shape[1]})"
            )

        m = a.shape[0]
        n = a.shape[1]
        p = b.shape[1]

        new_data = _array.array(a.dtype.typecode, [])
        for i in range(m):
            for j in range(p):
                total = 0.0
                for k in range(n):
                    total += a._data[i * n + k] * b._data[k * p + j]
                new_data.append(total)

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (m, p)
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output @ b.T
        grad_b = a.T @ grad_output
        return grad_a, grad_b

# ReLU(x) = max(0, x)
# d/dx(ReLU(x)) = 1 if x > 0 else 0
class ReLU(Function): 
    def forward(self, a):
        self.inputs = [a]
        self.save_for_backward(a)
        from Forge.tensor import Tensor

        new_data = _array.array(a.dtype.typecode, [max(0.0, x) for x in a._data])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = a.shape
        result.dtype = a.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
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

    def backward(self, grad_output):
        sig = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
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

    def backward(self, grad_output):
        tanh_out = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        self.save_for_backward(a)
        from Forge.tensor import Tensor
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

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a, min_val, max_val):
        self.inputs = [a]
        self.min_val = min_val
        self.max_val = max_val
        self.save_for_backward(a)
        from Forge.tensor import Tensor

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

    def backward(self, grad_output):
        a = self.saved_tensors[0]
        from Forge.tensor import Tensor

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
    def forward(self, a):
        self.inputs = [a]
        from Forge.tensor import Tensor
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

    def backward(self, grad_output):
        softmax_out = self.saved_tensors[0]
        from Forge.tensor import Tensor

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

