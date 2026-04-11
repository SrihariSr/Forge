import array as _array
from Forge.dtype import Dtype, float32, float64, int32, int64, DEFAULT_DTYPE
from Forge.CalcLlama.operations import Add, Mul, Sub, Pow, Neg, Sum

def _flatten(data):
    if isinstance(data, (int, float)):
        return [data]
    result = []
    for element in data:
        result.extend(_flatten(element))
    return result


def _infer_shape(data):
    if isinstance(data, (int, float)) or not isinstance(data, (list, tuple)):
        return ()
    shape = (len(data),)
    if len(data) > 0 and isinstance(data[0], (int, float, list, tuple)):
        shape += _infer_shape(data[0])
    return shape

def _broadcast_shape(shape_a, shape_b):
    max_length = max(len(shape_a), len(shape_b))

    padded_a = (1,) * (max_length - len(shape_a)) + shape_a
    padded_b = (1,) * (max_length - len(shape_b)) + shape_b

    result = []
    for a, b in zip(padded_a, padded_b):
        if a == b:
            result.append(a)
        elif a == 1:
            result.append(b)
        elif b == 1:
            result.append(a)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape_a} and {shape_b}!")

    return tuple(result)

def _broadcast_data(data, shape, new_shape, dtype_typecode):
    if shape == new_shape:
        return _array.array(dtype_typecode, data)

    # Pad shape with 1s on the left to match length
    pad = len(new_shape) - len(shape)
    padded_shape = (1,) * pad + shape

    # Compute strides for the padded shape
    # A stride tells us how many elements to skip in flat data for each step along a dimension
    strides = []
    stride = 1
    for i in range(len(padded_shape) - 1, -1, -1):
        if padded_shape[i] == 1:
            strides.append(0)
        else:
            strides.append(stride)
        stride *= padded_shape[i]
    strides.reverse()

    # Build the expanded flat data
    total = 1
    for s in new_shape:
        total *= s

    new_data = _array.array(dtype_typecode, [0.0] * total)
    for flat_idx in range(total):
        # Convert flat index to multidimensional index in new_shape
        remaining = flat_idx
        source_flat = 0
        for dim in range(len(new_shape)):
            idx = remaining // (total // new_shape[dim]) if new_shape[dim] > 0 else 0
            size_after = 1
            for d in range(dim + 1, len(new_shape)):
                size_after *= new_shape[d]
            idx = remaining // size_after
            remaining = remaining % size_after
            source_flat += idx * strides[dim]
        new_data[flat_idx] = data[source_flat]

    return new_data

class Tensor:
    def __init__(self, data, dtype=None, requires_grad=False):
        if dtype is None:
            dtype = DEFAULT_DTYPE

        if not isinstance(dtype, Dtype):
            raise TypeError(f"dtype must be a Dtype, got {type(dtype)}")

        self.dtype = dtype

        if isinstance(data, (int, float)):
            self._data = _array.array(dtype.typecode, [data])
            self.shape = ()
        elif isinstance(data, list):
            self.shape = _infer_shape(data)
            flat = _flatten(data)
            self._data = _array.array(dtype.typecode, flat)
        else:
            raise TypeError(f"{type(data)} is not a supported data type for a tensor!")

        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None

    def _rebuild_nested(self):
        if self.shape == ():
            return self._data[0]

        def _build(flat, shape, currentPosition):
            if len(shape) == 1:
                return list(flat[currentPosition:currentPosition + shape[0]]), currentPosition + shape[0]
            result = []
            for i in range(shape[0]):
                sub, currentPosition = _build(flat, shape[1:], currentPosition)
                result.append(sub)
            return result, currentPosition

        nested, _ = _build(self._data, self.shape, 0)
        return nested

    def __repr__(self):
        data_str = self._rebuild_nested()
        if self.dtype == DEFAULT_DTYPE:
            return f"tensor({data_str})"
        return f"tensor({data_str}, dtype={self.dtype})"

    def size(self, dim=None):
        if dim is not None:
            return self.shape[dim]
        return self.shape

    def ndim(self):
        return len(self.shape)

    def numel(self):
        number = 1
        for x in self.shape:
            number *= x
        return number

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) == len(self.shape):
            flat_index = 0
            multiplier = 1
            for i in range(len(self.shape) - 1, -1, -1):
                flat_index += index[i] * multiplier
                multiplier *= self.shape[i]
            return self._data[flat_index]

        if len(index) < len(self.shape):
            flat_index = 0
            multiplier = 1
            for i in range(len(self.shape) - 1, -1, -1):
                if i < len(index):
                    flat_index += index[i] * multiplier
                multiplier *= self.shape[i]

            new_shape = self.shape[len(index):]
            size = 1
            for x in new_shape:
                size *= x

            new_data = _array.array(self.dtype.typecode, self._data[flat_index:flat_index + size])

            result = Tensor.__new__(Tensor)
            result._data = new_data
            result.shape = new_shape
            result.dtype = self.dtype
            result.requires_grad = False
            result.grad = None
            result._grad_fn = None
            return result

        raise IndexError("Too many indices for tensor!")

    def __setitem__(self, index, value):
        if not isinstance(index, tuple):
            index = (index,)

        if len(index) != len(self.shape):
            raise IndexError("Must provide an index for every dimension when setting a value!")

        flat_index = 0
        multiplier = 1
        for i in range(len(self.shape) - 1, -1, -1):
            flat_index += index[i] * multiplier
            multiplier *= self.shape[i]

        self._data[flat_index] = value

    @staticmethod
    def zeros(*shape, dtype=None):
        if dtype is None:
            dtype = DEFAULT_DTYPE
        total = 1
        for s in shape:
            total *= s
        result = Tensor.__new__(Tensor)
        result._data = _array.array(dtype.typecode, [0.0] * total)
        result.shape = shape
        result.dtype = dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    @staticmethod
    def ones(*shape, dtype=None):
        if dtype is None:
            dtype = DEFAULT_DTYPE
        total = 1
        for x in shape:
            total *= x
        result = Tensor.__new__(Tensor)
        result._data = _array.array(dtype.typecode, [1.0] * total)
        result.shape = shape
        result.dtype = dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    @staticmethod
    def full(shape, fill_value, dtype=None):
        if dtype is None:
            dtype = DEFAULT_DTYPE
        total = 1
        for x in shape:
            total *= x
        result = Tensor.__new__(Tensor)
        result._data = _array.array(dtype.typecode, [float(fill_value)] * total)
        result.shape = shape
        result.dtype = dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def _elementwise_op(self, other, op):
        if isinstance(other, (int, float)):
            new_data = _array.array(self.dtype.typecode, [op(x, other) for x in self._data])
            result = Tensor.__new__(Tensor)
            result._data = new_data
            result.shape = self.shape
            result.dtype = self.dtype
            result.requires_grad = False
            result.grad = None
            result._grad_fn = None
            return result

        if not isinstance(other, Tensor):
            raise TypeError(f"Unsupported type: {type(other)}")

        # Broadcasting
        result_shape = _broadcast_shape(self.shape, other.shape)
        data_a = _broadcast_data(self._data, self.shape, result_shape, self.dtype.typecode)
        data_b = _broadcast_data(other._data, other.shape, result_shape, self.dtype.typecode)

        new_data = _array.array(self.dtype.typecode, [op(a, b) for a, b in zip(data_a, data_b)])
        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = result_shape
        result.dtype = self.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self._apply_op(Add, other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self._apply_op(Sub, other)

    def __mul__(self, other):
        if (isinstance(other, (int, float))):
            other = Tensor(other)
        return self._apply_op(Mul, other)

    def __truediv__(self, other):
        return self._elementwise_op(other, lambda a, b: a / b)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self.__add__(other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self._apply_op(Sub, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return self._apply_op(Mul, other)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            other = Tensor(other)
        return other._apply_op(Mul, self ** -1)

    def __neg__(self):
        return self._apply_op(Neg)

    def __pow__(self, exp):
        return self._apply_op(Pow, exp)

    def sum(self):
        from Forge.CalcLlama.operations import Sum
        return self._apply_op(Sum)

    def mean(self):
        from Forge.CalcLlama.operations import Mean
        return self._apply_op(Mean)

    def matmul(self, other):
        from Forge.CalcLlama.operations import Matmul
        return self._apply_op(Matmul, other)

    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, *new_shape):
        new_total = 1
        for x in new_shape:
            new_total *= x

        old_total = self.numel()

        if new_total != old_total:
            raise ValueError(
                f"Cannot reshape ({self.shape}) with {old_total} elements "
                f"into ({new_shape}) with {new_total} elements"
            )

        result = Tensor.__new__(Tensor)
        result._data = _array.array(self.dtype.typecode, self._data)
        result.shape = new_shape
        result.dtype = self.dtype
        result.requires_grad = False
        result.grad = None
        result._grad_fn = None
        return result

    def transpose(self):
        if len(self.shape) != 2:
            raise ValueError("Transpose only supports 2D tensors")
        from Forge.CalcLlama.operations import Transpose
        return self._apply_op(Transpose)

    @property # X.transpose() would be the same as X.T
    def T(self):
        return self.transpose()

    def _apply_op(self, op_class, *args):
        """Apply an operation, recording it in the graph if needed."""
        func = op_class()
        result = func.forward(self, *args)

        requires_grad = self.requires_grad
        for arg in args:
            if isinstance(arg, Tensor) and arg.requires_grad:
                requires_grad = True

        if requires_grad:
            result.requires_grad = True
            result._grad_fn = func

        return result

    def backward(self):
        if self.shape != ():
            raise RuntimeError("backward() can only be called on scalar tensors")

        topo = []
        visited = set()

        def build_topo(tensor):
            if id(tensor) not in visited:
                visited.add(id(tensor))
                if tensor._grad_fn is not None:
                    for inp in tensor._grad_fn.inputs:
                        if isinstance(inp, Tensor):
                            build_topo(inp)
                topo.append(tensor)

        build_topo(self)

        self.grad = Tensor(1.0)

        for tensor in reversed(topo):
            # Handle custom backward (used by CrossEntropyLoss)
            if hasattr(tensor, '_backward_fn') and tensor._backward_fn is not None:
                tensor._backward_fn()

            if tensor._grad_fn is not None:
                grads = tensor._grad_fn.backward(tensor.grad)
                for inp, grad in zip(tensor._grad_fn.inputs, grads):
                    if isinstance(inp, Tensor) and inp.requires_grad:
                        if inp.grad is None:
                            inp.grad = grad
                        else:
                            inp.grad = inp.grad + grad

    def relu(self):
        from Forge.CalcLlama.operations import Relu
        return self._apply_op(Relu)

    def sigmoid(self):
        from Forge.CalcLlama.operations import Sigmoid
        return self._apply_op(Sigmoid)

    def tanh(self):
        from Forge.CalcLlama.operations import Tanh
        return self._apply_op(Tanh)

    def log(self):
        from Forge.CalcLlama.operations import Log
        return self._apply_op(Log)

    def clamp(self, min_val, max_val):
        from Forge.CalcLlama.operations import Clamp
        return self._apply_op(Clamp, min_val, max_val)
    
    def softmax(self):
        from Forge.CalcLlama.operations import Softmax
        return self._apply_op(Softmax)