import array
import Metal
import MetalPerformanceShaders as MPS

# --- matrices, row-major, float32 ---
M, K, N = 2, 3, 2
A = array.array('f', [1, 2, 3, 4, 5, 6])      # 2x3
B = array.array('f', [7, 8, 9, 10, 11, 12])   # 3x2
# expected A @ B = [[58, 64], [139, 154]]
F = 4  # bytes per float32

# --- 1. device + command queue ---
device = Metal.MTLCreateSystemDefaultDevice()
print("GPU:", device.name())
queue = device.newCommandQueue()

# --- 2. GPU buffers holding the matrix bytes ---
opts = Metal.MTLResourceStorageModeShared
bufA = device.newBufferWithBytes_length_options_(A.tobytes(), len(A) * F, opts)
bufB = device.newBufferWithBytes_length_options_(B.tobytes(), len(B) * F, opts)
bufC = device.newBufferWithLength_options_(M * N * F, opts)

# --- 3. wrap each buffer as an MPSMatrix (buffer + shape description) ---
def desc(rows, cols):
    return MPS.MPSMatrixDescriptor.matrixDescriptorWithRows_columns_rowBytes_dataType_(
        rows, cols, cols * F, MPS.MPSDataTypeFloat32
    )

matA = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(bufA, desc(M, K))
matB = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(bufB, desc(K, N))
matC = MPS.MPSMatrix.alloc().initWithBuffer_descriptor_(bufC, desc(M, N))

# --- 4. the kernel: C = 1.0 * A @ B + 0.0 * C ---
matmul = MPS.MPSMatrixMultiplication.alloc().initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta_(
    device, False, False, M, N, K, 1.0, 0.0
)

# --- 5. encode, run on GPU, wait ---
cmd = queue.commandBuffer()
matmul.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix_(cmd, matA, matB, matC)
cmd.commit()
cmd.waitUntilCompleted()

# --- 6. read result back from the GPU buffer ---
out = array.array('f')
out.frombytes(bufC.contents().as_buffer(M * N * F))
print("GPU result:", list(out))
print("expected:  ", [58.0, 64.0, 139.0, 154.0])