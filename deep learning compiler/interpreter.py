import ctypes
import array
from graph import topological_order

# Loading compiled kernels and declaring their signatures
_lib = ctypes.CDLL("./kernels.so")
_FP = ctypes.POINTER(ctypes.c_float)

for _name in ("add", "sub", "mul"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [_FP, _FP, _FP, ctypes.c_int]
    _fn.restype = None
_lib.relu.argtypes = [_FP, _FP, ctypes.c_int]
_lib.relu.restype = None
_lib.matmul.argtypes = [_FP, _FP, _FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_lib.matmul.restype = None
_lib.exp_kernel.argtypes = [_FP, _FP, ctypes.c_int]
_lib.exp_kernel.restype = None

def _addr(arr):
    """
    Memory address of an array's data as a float pointer in C.
    """
    return ctypes.cast(arr.buffer_info()[0], _FP)

class Tensor:
    """
    The numbers and their shape, stored at runtime. The interpreter generates
    `Tensor` instances at runtime.
    """
    def __init__(self, data, shape):
        self.data : array.array = data
        self.shape : tuple = shape
    
    @property
    def size(self) -> int:
        total = 1
        for d in self.shape:
            total *= d
        return total
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, data={list(self.data)})"
    
def _empty(shape) -> Tensor:
    """
    Zero-filled tensor of shape `shape`.
    """
    total = 1
    for d in shape:
        total *= d
    return Tensor(array.array('f', bytes(4 * total)), shape)

def run(root, feeds) -> Tensor:
    """
    Execute the graph and return the final Tensor.
    """
    values = {}
    for node in topological_order(root):

        if node.op == "placeholder":
            values[node] = feeds[node.name]

        elif node.op in ("add", "sub", "mul"):
            a = values[node.inputs[0]]
            b = values[node.inputs[1]]
            out = _empty(a.shape)
            kernel = getattr(_lib, node.op)
            kernel(_addr(a.data), _addr(b.data), _addr(out.data), a.size)
            values[node] = out

        elif node.op == "relu":
            x = values[node.inputs[0]]
            out = _empty(x.shape)
            _lib.relu(_addr(x.data), _addr(out.data), x.size)
            values[node] = out

        elif node.op == "matmul":
            a = values[node.inputs[0]]
            b = values[node.inputs[1]]
            m, k = a.shape
            _, n = b.shape
            out = _empty((m, n))
            _lib.matmul(_addr(a.data), _addr(b.data), _addr(out.data), m, k, n)
            values[node] = out

        elif node.op == "exp":
            x = values[node.inputs[0]]
            out = _empty(x.shape)
            _lib.exp_kernel(_addr(x.data), _addr(out.data), x.size)
            values[node] = out

        else:
            raise NotImplementedError(f"Interpreter does not handle operation '{node.op}' yet.")

    return values[root]
