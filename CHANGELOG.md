# Changelog

## 0.1.0

First release.

## 0.1.1

Package rename from `forge-ml` to `forge-dl`.

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