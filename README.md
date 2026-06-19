# Forge 🧠

A fully differentiable machine learning library built from scratch using only Python. Every operation is implemented using first principles.

## Why This Exists

Forge isn't a wrapper around existing libraries. It's a ground-up implementation of the core ideas behind modern Machine Learning frameworks, built to deeply understand how tools like PyTorch, JAX, and MLX work under the hood.

**Notable Achievement:** Trained a character-level transformer language model entirely on this framework.

## Features

### Core Tensor Engine
- Multi-dimensional tensor class with typed memory backend
- Memory efficient storage using Python's `array` module instead of Python lists leading to 6x memory usage reduction
- Full broadcasting support following NumPy semantics
- Reshape, transpose, and element-wise operations

### Automatic Differentiation
- Automatic gradient computation tracks operations as they happen and computes gradients by walking backwards through the computation, just like PyTorch
- Support for arbitrary compositions of differentiable operations
- Numerical gradient verification using central finite differences
- Custom `Function` API for user-defined differentiable operations

### Neural Network Module System
- `Module` base class with automatic parameter registration using `__setattr__`
- Recursive parameter discovery across nested modules
- Train and eval mode switching
- Built-in layers: `Linear`, `Embedding`, `ReLU`, `Sigmoid` and `Tanh`

### Transformer Architecture
- Each position in a sequence learns which other positions are important to it, using the standard Query/Key/Value attention formula
- Transformer blocks with residual connections and feed-forward networks
- Character-level language model
- Successfully trained to generate coherent text

### Optimizers
- SGD with momentum
- Adam with bias correction and adaptive learning rates

### Loss Functions
- MSELoss (Mean Squared Error)
- BCELoss (Binary Cross Entropy)
- CrossEntropyLoss with numerically stable softmax

### Operator Fusion
- Graph-level pattern detection that identifies `Linear` to `ReLU` sequences
- Automatic replacement with fused `FusedLinearReLU` operations
- Inspired by XLA and TorchInductor optimization passes

### Serialization
- Save/load model weights to a `.json` file
- State dict (a dictionary of all model weights) matching PyTorch's API convention
- Dot-separated parameter naming for nested module hierarchies


## Quick Start

### Create Tensors
```python
from Forge import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

print(a + b)        # Element-wise addition
print(a @ b)        # Matrix multiplication
print(a.T)          # Transpose
print(a.shape)      # (2, 2)
```

### Automatic Differentiation
```python
x = Tensor([2.0, 3.0], requires_grad=True)
y = ((x * x) + x).sum()
y.backward()
print(x.grad)       # dy/dx = 2x + 1
```

### Train a Neural Network
```python
from NeuralNetwork.layers import Linear, MSELoss
from Optim.optimizer import SGD

model = Linear(2, 1)
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    pred = model(x)
    loss = criterion(pred, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Operator Fusion
```python
from NeuralNetwork.layers import Linear, ReLULayer, Sequential

model = Sequential(
    Linear(2, 8),    # These two get fused into
    ReLULayer(),     # a single FusedLinearReLU
    Linear(8, 1),
)
print(model)         # Shows 3 layers
model.optimize()
print(model)         # Shows 2 layers: Linear+ReLU fused
```

### Train a Transformer
```python
from NeuralNetwork.layers import CharTransformer
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

model = CharTransformer(vocab_size=8, embed_dim=16, ff_dim=32, seq_len=8)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

```

---

## Benchmarks

Benchmarked on Apple M4 Max. All times are median of 10 runs.

| Operation | Forge | PyTorch | Ratio |
|---|---|---|---|
| Matmul 64×64 | 20.710 ms | 0.002 ms | 11292.4x |
| Matmul 128×128 | 167.914 ms | 0.006 ms | 30076.0x |
| Add (100k elements) | 4.435 ms | 0.044 ms | 100.4x |
| ReLU (100k elements) | 0.371 ms | 0.002 ms | 228.1x |
| Training (100 epochs) | 13.201 ms | 6.669 ms | 2.0x |

> Run the `runBenchmarks.py` file if you wish to reproduce these results (results may vary with hardware specifications).

### Performance Analysis

This library is written in pure Python with no C extensions. The performance gap with PyTorch is hence expected. This can be explained due to the following reasons:

1. **Python loop overhead**: Forge's matmul is implemented using triple nested Python loops. PyTorch uses optimized BLAS libraries written in C/C++.
2. **No SIMD/vectorization**: PyTorch uses SSE/AVX instructions that process multiple numbers in a single CPU cycle. Python processes one number at a time.
3. **No hardware acceleration**: PyTorch on Apple Silicon leverages the AMX coprocessor and Accelerate framework. Pure Python cannot utilize these hardware features.
4. **Memory allocation**: Forge creates new typed arrays for every operation. PyTorch uses memory pools and in-place operations.

## Transformer Training Results

Trained a character-level transformer using 2,152 parameters on the pattern "hello world":

```
Epoch   0, Loss: 2.0553
Epoch 100, Loss: 0.4421
Epoch 300, Loss: 0.4032
Epoch 490, Loss: 0.3521

Generated: "hello world world held hello"
```

### Key Observations

**Learning Rate Sensitivity:**
Training at `lr=0.01` caused instability with loss spikes at later epochs. Reducing to `lr=0.001` eliminated the instability and produced significantly better generation quality. This demonstrates the importance of hyperparameter tuning.

**Loss vs Generation Quality:**
Seed search across 20 random initializations revealed that lower training loss doesn't always correlate with better text generation. Models with the lowest loss converged to repetitive local minima (outputting "wo wo wo wo"), while models with slightly higher loss learned richer, more accurate patterns ("hello world world held hello").

The model was run on 20 different seed values to identify the best seed value to use. However, the seed value with the least loss did not produce the best text generation as initially expected. Models with slightly higher loss learned better, more accurate patterns. This demonstrates that the loss function is not always a perfect indicator of the model's performance.

**Positional Encoding:**
The model occasionally confuses similar subsequences (generating "held" instead of "hello") due to the lack of explicit positional encoding. This demonstrates why positional information is critical in transformer architectures. Without it, the model cannot distinguish between identical characters at different sequence positions.

## Gradient Verification

All analytical gradients are verified against numerical gradients using central finite differences:

```python
from Forge.CalcLlama import grad_check
from Forge.dtype import float64

def mse_func(pred):
    target = Tensor([1.0, 2.0, 3.0], dtype=float64)
    return ((pred - target) ** 2).mean()

pred = Tensor([1.5, 2.5, 3.5], dtype=float64, requires_grad=True)
assert grad_check(mse_func, [pred])
```

Gradient checking uses `float64` to avoid false failures from `float32` precision limitations. The double-sided finite difference formula `f(x) = lim(h->0) (f(x+h) - f(x-h)) / 2h` provides an accuracy proportional to h² compared to h for one-sided differences which is better for small values of h.

## Technical Decisions

### Why typed arrays instead of Python lists?
A Python float object uses 24 bytes (object header + reference count + value). A `float32` in a typed array uses 4 bytes which is a 6x memory reduction. For models with millions of parameters, this difference becomes significant.

### Why dynamic graphs over static graphs?
Dynamic graphs (define-by-run) allow standard Python control flow in the forward pass: if statements, for loops, and variable-length sequences work naturally. 

### Why operator fusion at the graph level?
Real ML compilers such as XLA, TorchInductor, and Core ML optimize by detecting operation patterns and replacing them with specialized fused kernels. The fusion pass in this library demonstrates this principle by detecting `Linear` to `ReLU` patterns and replacing them with a single `FusedLinearReLU` that performs matmul + bias + activation in one pass through the data.

## What I Learned

Building this framework from scratch taught me:

- **How automatic differentiation works** at the implementation level; not just the math, but the graph construction, topological sorting, and gradient accumulation.
- **Why numerical stability matters**: softmax overflow prevention, log(0) clamping, and why gradient checking requires float64.
- **Optimizer dynamics**: how SGD momentum accelerates convergence, why Adam adapts per-parameter, and how learning rate affects training stability.
- **Transformer internals**: how self-attention computes and propagates gradients through Query/Key/Value projections, softmax, and residual connections.
- **Compiler-level optimization**: how pattern matching on computation graphs enables operator fusion, and why this matters for performance.
- **Training dynamics**: sensitivity to initialization, the disconnect between loss and generation quality, and the importance of early stopping.

## Installation

```bash
git clone https://github.com/SrihariSr/Forge.git
cd Forge
pip install -e .
```

No external dependencies required. Python 3.10+ only.