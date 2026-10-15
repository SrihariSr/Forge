# Changelog

## 0.4.0

- Added `SME/sme_matmul.c`: a hand-written Arm SME FP32 matmul for Apple M4.
  Output is bit-identical to Accelerate's `cblas_sgemm`. On an M4 Max
  (macOS 27.0.1), timed alternately against Accelerate: 1.06x to 1.18x
  faster from 1024 to 2048, 0.83x at 1000, 0.67x at 512. Build and verify:
  `clang -O2 -mcpu=apple-m4 sme_matmul.c -o sme_matmul -framework Accelerate && ./sme_matmul verify`

## 0.3.0

CUDA backend. Matrix multiplication dispatches to cuBLAS on machines with an
NVIDIA GPU, alongside the existing Metal and Accelerate backends. Verified on
an NVIDIA Tesla T4.

ROCm backend. The same through rocBLAS on AMD hardware. Untested: written and
logic-checked, but never run on an AMD GPU.

NVIDIA/matmul.cu and AMD/matmul.hip: three hand-written matmul kernels each
(naive, shared-memory tiled, register blocked). The CUDA register-blocked
kernel reaches 2718 GFLOPS on a T4, within 1.9x of cuBLAS. The HIP versions
are untested.

## 0.2.1

Correctness fixes found in an audit of the library.

CrossEntropyLoss ignored the incoming gradient, breaking the chain rule for
any composite loss. Sum and Mean silently downcast float64 to float32, which
meant gradient checking on matmul had been failing. Reflected operators and
the backward pass seed used the default dtype rather than the tensor's.
Indexing past the end of a tensor returned a wrong value instead of raising.
BatchedMatmul compared dtypes by identity rather than equality, silently
skipping the BLAS path. CausalMask used a constant that overflows float32 to
negative infinity. UnflattenBatch did not check divisibility.
MultiHeadAttention did not validate that embed_dim divides by num_heads.
Sigmoid overflowed for large negative inputs. The compiler package could not
be imported before its kernels were built, making the build command
impossible to run.

Documentation corrected: the quick-start example did not run, the repository
layout listed a path that does not exist, and an unsupported claim about
put-call parity was removed.

## 0.2.0

Batching. The GPT, attention, Linear, LayerNorm and CrossEntropyLoss now
accept 3D input of shape (batch, seq, features). Previously GPT.forward
silently discarded every sequence but the first.

New operations: FlattenBatch, UnflattenBatch, StackBatch, BatchedMatmul,
BatchedCausalMask.

Type annotations throughout the library.

Fixed: a requires_grad typo in StackBatch, duplicate Sigmoid, Tanh and Add
definitions where the first of each was dead code, SimpleAttention returning
a Python list for batches larger than one, and a wrong return annotation in
the compiler's code generator.

## 0.1.1

Package rename from `forge-ml` to `forge-dl`.

## 0.1.0

First release.



