from __future__ import annotations
import array
import ctypes

_HIP_NAMES = ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5")
_ROCBLAS_NAMES = ("librocblas.so", "/opt/rocm/lib/librocblas.so", "librocblas.so.4", "librocblas.so.3")

def _load_first(names: tuple[str, ...]) -> ctypes.CDLL | OSError:
    """
    Try each name and return the first that loads.
    """
    last = None
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            last = error
    raise OSError(f"None of {names} could be loaded: {last}")
    
try:
    _hip = _load_first(_HIP_NAMES)
    _rocblas = _load_first(_ROCBLAS_NAMES)

    float_pointer = ctypes.POINTER(ctypes.c_float)
    void_pointer = ctypes.c_void_p

    _hip.hipMalloc.argtypes = [ctypes.POINTER(void_pointer), ctypes.c_size_t]
    _hip.hipMalloc.restype = ctypes.c_int

    _hip.hipFree.argtypes = [void_pointer]
    _hip.hipFree.restype = ctypes.c_int

    _hip.hipMemcpy.argtypes = [void_pointer, void_pointer, ctypes.c_size_t, ctypes.c_int]
    _hip.hipMemcpy.restype = ctypes.c_int


    _hip.hipDeviceSynchronize.argtypes = []
    _hip.hipDeviceSynchronize.restype = ctypes.c_int

    _rocblas.rocblas_sgemm.argtypes = [
        void_pointer,  # the handle
        ctypes.c_int,  # transpose the left matrix?
        ctypes.c_int,  # transpose the right matrix?
        ctypes.c_int,  # rows of the result
        ctypes.c_int,  # columns of the result
        ctypes.c_int,  # the shared inner dimension being summed over
        float_pointer, # alpha factor
        void_pointer,  # left matrix, in GPU memory
        ctypes.c_int,  # column stride of the left matrix
        void_pointer,  # right matrix, in GPU memory
        ctypes.c_int,  # column stride of the right matrix
        float_pointer, # beta factor
        void_pointer,  # output matrix, in GPU memory
        ctypes.c_int,  # column stride of the output matrix
    ]

    _rocblas.rocblas_sgemm.restype = ctypes.c_int

    _rocblas.rocblas_create_handle.argtypes = [ctypes.POINTER(void_pointer)]
    _rocblas.rocblas_create_handle.restype = ctypes.c_int

    _handle = void_pointer()
    if _rocblas.rocblas_create_handle(ctypes.byref(_handle)) != 0:
        raise OSError("rocblas_create_handle failed.")
    
    ROCM_AVAILABLE = True

except Exception:
    ROCM_AVAILABLE = False
    _handle = None


HIP_MEMCPY_HOST_TO_DEVICE = 1
HIP_MEMCPY_DEVICE_TO_HOST = 2

ROCBLAS_OPERATION_NONE = 111

BYTES_PER_FLOAT = 4



def _check(status: int, what: str) -> None:
    """
    Checks the HIP/rocBLAS status code to diagnose any silently failing backend.
    """
    if status != 0:
        raise RuntimeError(f"{what} failed with status {status}")


def rocm_matmul(left_data: array.array[float], right_data: array.array[float],
                left_rows: int, shared_dim: int,
                right_cols: int) -> array.array[float]:
    """
    Multiplies two matrices on an AMD GPU using rocBLAS.
    """
    if not ROCM_AVAILABLE:
        raise RuntimeError("ROCm is not available on this machine")

    result_data = array.array('f', bytes(BYTES_PER_FLOAT * left_rows * right_cols))

    left_bytes = BYTES_PER_FLOAT * left_rows * shared_dim
    right_bytes = BYTES_PER_FLOAT * shared_dim * right_cols
    result_bytes = BYTES_PER_FLOAT * left_rows * right_cols

    device_left = ctypes.c_void_p()
    device_right = ctypes.c_void_p()
    device_result = ctypes.c_void_p()

    try:
        _check(_hip.hipMalloc(ctypes.byref(device_left), left_bytes), "hipMalloc")
        _check(_hip.hipMalloc(ctypes.byref(device_right), right_bytes), "hipMalloc")
        _check(_hip.hipMalloc(ctypes.byref(device_result), result_bytes), "hipMalloc")

        left_address = ctypes.c_void_p(left_data.buffer_info()[0])
        right_address = ctypes.c_void_p(right_data.buffer_info()[0])
        result_address = ctypes.c_void_p(result_data.buffer_info()[0])

        _check(_hip.hipMemcpy(device_left, left_address, left_bytes,
                              HIP_MEMCPY_HOST_TO_DEVICE), "hipMemcpy to device")
        _check(_hip.hipMemcpy(device_right, right_address, right_bytes,
                              HIP_MEMCPY_HOST_TO_DEVICE), "hipMemcpy to device")

        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)

        # rocBLAS reads matrices in the column-major order, Forge uses row-major
        _check(_rocblas.rocblas_sgemm(
            _handle,
            ROCBLAS_OPERATION_NONE, ROCBLAS_OPERATION_NONE,
            right_cols, left_rows, shared_dim,
            ctypes.byref(alpha),
            device_right, right_cols,
            device_left, shared_dim,
            ctypes.byref(beta),
            device_result, right_cols,
        ), "rocblas_sgemm")

        # rocBLAS calls are asynchronous: the call returns before the GPU has
        # finished. Copying back without waiting would read a partial result.
        _check(_hip.hipDeviceSynchronize(), "hipDeviceSynchronize")

        _check(_hip.hipMemcpy(result_address, device_result, result_bytes,
                              HIP_MEMCPY_DEVICE_TO_HOST), "hipMemcpy to host")

    finally:
        # Freed even if something above raised, so a failure does not leak GPU
        # memory. A long training run would otherwise exhaust it.
        for pointer in (device_left, device_right, device_result):
            if pointer:
                _hip.hipFree(pointer)

    return result_data