import array

try:
    import Metal
    import MetalPerformanceShaders as MPS

    _DEVICE = Metal.MTLCreateSystemDefaultDevice()
    _QUEUE = _DEVICE.newCommandQueue() if _DEVICE is not None else None

    MPS_AVAILABLE = _DEVICE is not None and _QUEUE is not None
    if MPS_AVAILABLE:
        _SHARED = Metal.MTLResourceStorageModeShared
except Exception:
    _DEVICE = None
    _QUEUE = None
    MPS_AVAILABLE = False

_FLOAT_BYTES = 4
GPU_MIN_WORK = 200000

def _descriptor(rows, cols):
    return MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        rows, cols, cols * _FLOAT_BYTES, MPS.MPSDataTypeFloat32
    )

def mps_matmul(a_data, b_data, m, k, n):
    buf_a = _DEVICE.newBufferWithBytes_length_options_(
        a_data.tobytes(), m * k * _FLOAT_BYTES, _SHARED
    )
    buf_b = _DEVICE.newBufferWithBytes_length_options_(
        b_data.tobytes(), k * n * _FLOAT_BYTES, _SHARED
    )
    buf_c = _DEVICE.newBufferWithLength_options_(m * n * _FLOAT_BYTES, _SHARED)

    mat_a = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buf_a, _descriptor(m, k))
    mat_b = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buf_b, _descriptor(k, n))
    mat_c = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(buf_c, _descriptor(m, n))

    kernel = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
        _DEVICE, False, False, m, n, k, 1.0, 0.0
    )

    cmd = _QUEUE.commandBuffer()
    kernel.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(
        cmd, mat_a, mat_b, mat_c
    )
    cmd.commit()
    cmd.waitUntilCompleted()

    out = array.array('f')
    out.frombytes(buf_c.contents().as_buffer(m * n * _FLOAT_BYTES))
    return out
