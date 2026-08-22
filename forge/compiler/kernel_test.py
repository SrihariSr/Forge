import ctypes, array

lib = ctypes.CDLL("./kernels.so")

FP = ctypes.POINTER(ctypes.c_float)
lib.fused.argtypes = [FP, FP, FP, FP, ctypes.c_int]
lib.fused.restype  = None

# Test data as flat float32 arrays.
a   = array.array('f', [ 1.0, -2.0,  3.0, -4.0])
b   = array.array('f', [ 0.5,  1.0, -1.0,  2.0])
c   = array.array('f', [ 2.0,  2.0,  2.0,  2.0])
out = array.array('f', [0.0] * 4)

def addr(arr):
    return ctypes.cast(arr.buffer_info()[0], FP)

lib.fused(addr(a), addr(b), addr(c), addr(out), 4)

# Verify against the same maths in plain Python.
expected = [max(0.0, a[i] + b[i]) * c[i] for i in range(4)]
print("C result :", list(out))
print("expected :", expected)
print("match    :", list(out) == expected)