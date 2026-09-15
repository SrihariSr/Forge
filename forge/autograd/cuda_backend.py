from __future__ import annotations
from sys import deactivate_stack_trampoline
import array
import ctypes

# Names vary by platform hence several options are tried
_CUDART_NAMES = ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0", "cudart64_12.dll", "cudart64_110.dll")
_CUBLAS_NAMES = ("libcublas.so", "libcublas.so.12", "libcublas.so.11", "cublas64_12.dll", "cublas64_11.dll")

def _load_first(names: tuple[str, ...]) -> ctypes.CDLL | OSError:
    """
    Try each candidate name and return the first one that works.
    """
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            raise OSError(f"None of the {names} could be loaded.")

try:
    # Vibed to APT by Rosé and Bruno Mars while writing this (lowkey a banger) 
    _cudart = _load_first(_CUDART_NAMES)
    _cublas = _load_first(_CUBLAS_NAMES)

    float_pointer = ctypes.POINTER(ctypes.c_float)
    void_pointer = ctypes.c_void_p

    _cudart.cudaMalloc.argtypes = [ctypes.POINTER(void_pointer), ctypes.c_size_t]
    _cudart.cudaMalloc.restype = ctypes.c_int

    _cudart.cudaFree.argtypes = [void_pointer]
    _cudart.cudaFree.restype = ctypes.c_int

    _cudart.cudaMemcpy.argtypes = [void_pointer, void_pointer, ctypes.c_size_t, ctypes.c_size_t]
    _cudart.cudaMemcpy.restype = ctypes.c_int

    _cudart.cudaDeviceSynchronize.argtypes = []
    _cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    # Holds the library's internal state, created once and reused every call
    _cublas.cublasCreate_v2.argtypes = [ctypes.POINTER(void_pointer)]
    _cublas.cublasCreate_v2.restype = ctypes.c_int

    _cublas.cublasSgemm_v2.argtypes = [
        void_pointer,  # the handle
        ctypes.c_int,  # transpose left matrix?
        ctypes.c_int,  # transpose right matrix?
        ctypes.c_int,  # rows of the result
        ctypes.c_int,  # columns of the result
        ctypes.c_int,  # shared inner dimension
        float_pointer, # alpha parameter
        void_pointer,  # left matrix in GPU memory
        ctypes.c_int,  # column stride in the left matrix
        void_pointer,  # right matrix in GPU memory
        ctypes.c_int,  # column stride in right matrix
        float_pointer, # beta parameter
        void_pointer,  # output matrix in GPU memory
        ctypes.c_int   # column stride in the output matrix
    ]
    _cublas.cublasSgemm_v2.restype = ctypes.c_int

    _handle = void_pointer()
    if _cublas.cublasCreate_v2(ctypes.byref(_handle)) != 0:
        raise OSError("cublasCreate failed :(")

    CUDA_AVAILABLE = True

# For a machine without CUDA
except Exception:
    CUDA_AVAILABLE = False
    _handle = None

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUBLAS_OP_N = 0
BYTES_PER_FLOAT = 4

def _check(status: int, name: str) -> None | RuntimeError:
    """
    Checks the CUDA/cuBLAS status code to diagnose any silently failing backend.
    """
    if status != 0:
        raise RuntimeError(f"{name} failed with status {status}.")


def cuda_matmul(
    left_data: array.array[float],
    right_data: array.array[float],
    left_rows: int,
    shared_dim: int,
    right_cols: int
    ) -> array.array[float] | RuntimeError:
    """
    Multiplies two matrices on an Nvidia GPU using cuBLAS.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this machine :(")
    
    result_data = array.array('f', bytes(BYTES_PER_FLOAT * left_rows * right_cols))

    left_bytes = BYTES_PER_FLOAT * left_rows * shared_dim
    right_bytes = BYTES_PER_FLOAT * shared_dim * right_cols
    result_bytes = BYTES_PER_FLOAT * left_rows * right_cols

    device_left = ctypes.c_void_p()
    device_right = ctypes.c_void_p()
    device_result = ctypes.c_void_p()

    try:
        _check(_cudart.cudaMalloc(ctypes.byref(device_left), left_bytes), "cudaMalloc")
        _check(_cudart.cudaMalloc(ctypes.byref(device_right), right_bytes), "cudaMalloc")
        _check(_cudart.cudaMalloc(ctypes.byref(device_result), result_bytes), "cudaMalloc")

        left_address = ctypes.c_void_p(left_data.buffer_info()[0])
        right_address = ctypes.c_void_p(right_data.buffer_info()[0])
        result_address = ctypes.c_void_p(result_data.buffer_info()[0])

        _check(_cudart.cudaMemcpy(device_left, left_address, left_bytes, CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy to device")
        _check(_cudart.cudaMemcpy(device_right, right_address, right_bytes, CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy to device")

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)

        # cuBLAS reads matrices in the column-major order, Forge uses row-major
        _check(_cublas.cublasSgemm_v2(
            _handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            right_cols, left_rows, shared_dim,
            ctypes.byref(alpha),
            device_right, right_cols,
            device_left, shared_dim,
            ctypes.byref(beta),
            device_result, right_cols,
        ),
        "cublasSgemm"
        )

        _check(_cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

        _check(_cudart.cudaMemcpy(result_address, device_result, result_bytes, CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy to host")

    finally:
        # Freeing memory at the end to ensure no failure can leak GPU memory.
        # A long run with a leak can exhaust memory.
        for pointer in (device_left, device_right, device_result):
            if pointer:
                _cudart.cudaFree(pointer)
    
    return result_data