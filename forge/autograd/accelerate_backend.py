import array
import ctypes

try:
    _lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    
    cblas_sgemm = _lib.cblas_sgemm

    float_pointer = ctypes.POINTER(ctypes.c_float)

    cblas_sgemm.argtypes = [
        ctypes.c_int,   # storage order (row-major vs column-major)
        ctypes.c_int,   # whether to transpose the left matrix
        ctypes.c_int,   # whether to transpose the right matrix
        ctypes.c_int,   # number of rows in the result
        ctypes.c_int,   # number of columns in the result
        ctypes.c_int,   # the shared inner dimension being summed over
        ctypes.c_float, # alpha: the product is scaled by this
        float_pointer,  # pointer to the left matrix's data
        ctypes.c_int,   # row stride of the left matrix
        float_pointer,  # pointer to the right matrix's data
        ctypes.c_int,   # row stride of the right matrix
        ctypes.c_float, # beta: the existing result is scaled by this
        float_pointer,  # pointer to the output matrix's data
        ctypes.c_int,   # row stride of the output matrix 
    ]

    cblas_sgemm.restype = None

    ACCELERATE_AVAILABLE = True

# Only occurs on non-macOS systems
except Exception:
    ACCELERATE_AVAILABLE = False

CBLAS_ROW_MAJOR = 101
CBLAS_NO_TRANSPOSE = 111

BYTES_PER_FLOAT = 4

def accelerate_matmul(left_data, right_data, left_rows, shared_dim, right_cols):
    """
    Multiplies two matrices on the CPU using Accelerate.

    Returns a flat float32 array of the result.
    """
    result_data = array.array('f', bytes(BYTES_PER_FLOAT * left_rows * right_cols))

    left_pointer = ctypes.cast(left_data.buffer_info()[0], float_pointer)
    right_pointer = ctypes.cast(right_data.buffer_info()[0], float_pointer)
    result_pointer = ctypes.cast(result_data.buffer_info()[0], float_pointer)

    cblas_sgemm(
        CBLAS_ROW_MAJOR,      # data is laid out row by row
        CBLAS_NO_TRANSPOSE,   # do not transpose the left matrix
        CBLAS_NO_TRANSPOSE,   # do not transpose the right matrix
        left_rows,            # rows of the result
        right_cols,           # columns of the result
        shared_dim,           # the inner dimension summed over
        1.0,                  # alpha: scale the product by 1.0
        left_pointer,         # the left matrix
        shared_dim,           # left's row stride
        right_pointer,        # the right matrix
        right_cols,           # right's row stride
        0.0,                  # beta: ignore the zeroed existing result
        result_pointer,       # where to write the answer
        right_cols,           # result's row stride
    )

    return result_data
