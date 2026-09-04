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
BatchedTranspose, BatchedCausalMask, SliceLastDim, ConcatLastDim.

Type annotations throughout the library.

Fixed: a requires_grad typo in StackBatch, duplicate Sigmoid, Tanh and Add
definitions where the first of each was dead code, SimpleAttention returning
a Python list for batches larger than one, and a wrong return annotation in
the compiler's code generator.
