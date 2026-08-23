# Forge

A machine learning library and deep learning compiler, written from scratch in Python.

No PyTorch, no TensorFlow, no NumPy in the core. Every operation, from the autograd engine to the attention mechanism, is implemented from first principles. On top of it sits a compiler that analyses computation graphs, fuses operations, and emits optimised C at run time.

Full documentation, derivations and benchmarks: **https://github.com/SrihariSr/Forge**

---

## Install

```bash
pip install forge-dl
```

Python 3.10 or later. The core library has no dependencies.

Optional extras:

```bash
pip install "forge-dl[plots]"    # matplotlib, for the option pricing charts
pip install "forge-dl[bench]"    # numpy, for the compiler's BLAS comparison
pip install "forge-dl[metal]"    # pyobjc, for the Apple GPU backend
```

---

## Quick start

Tensors and automatic differentiation:

```python
from forge import Tensor

a = Tensor([[1.0, 2.0], [3.0, 4.0]])
b = Tensor([[5.0, 6.0], [7.0, 8.0]])

print(a + b)      # element-wise addition
print(a @ b)      # matrix multiplication
print(a.T)        # transpose

x = Tensor([2.0, 3.0], requires_grad=True)
y = ((x * x) + x).sum()
y.backward()
print(x.grad)     # dy/dx = 2x + 1
```

A transformer, built on that autograd engine:

```python
from forge.nn import GPT, CrossEntropyLoss
from forge.optim import Adam

model = GPT(vocab_size=65, embed_dim=128, num_heads=4,
            ff_dim=256, num_layers=4, seq_len=32)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.002)

logits = model([[1, 2, 3, 4]])
loss = criterion(logits, [2, 3, 4, 5])

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## What is in it

**A differentiable tensor library.** Reverse-mode automatic differentiation on a define-by-run graph, a typed memory backend, and broadcasting that follows NumPy semantics without depending on NumPy. Every operation is checked against a numerical gradient computed by central finite differences.

**A neural network framework.** A `Module` system with automatic parameter registration, layers, optimisers and loss functions.

**A decoder-only transformer.** Multi-head causal self-attention, LayerNorm, GELU, learned positional encoding and pre-norm residual blocks, all built on the autograd engine above and gradient-checked individually.

**Hardware backends.** Matrix multiplication dispatches to the Apple Metal GPU, to Apple Accelerate BLAS on the CPU, or to a pure Python fallback, chosen by problem size.

**A deep learning compiler.** Builds a graph, plans which operations can share one pass over memory, generates C for each group, compiles it with gcc, and loads it back through ctypes while the program runs.

---

## The compiler

The C kernels are not built during installation, because they need a compiler and are architecture-specific. Build them once:

```bash
python -m forge.compiler.build
```

Importing `forge.compiler` before running that raises an error telling you to.

```python
from forge.compiler import placeholder, relu, compile_graph, run_compiled

a, b, c = placeholder("a"), placeholder("b"), placeholder("c")
graph = relu(a + b) * c

steps = compile_graph(graph)      # analyse, fuse, generate C, compile
result = run_compiled(graph, steps, feeds)
```

For `relu(a + b) * c` the compiler writes this, and nothing else in the project wrote it:

```c
void fused_kernel(const float* in0, const float* in1, const float* in2,
                  float* out, int n){
    for (int i = 0; i < n; i++) {
        float t3 = (in0[i] + in1[i]);
        float t4 = (t3 > 0.0f ? t3 : 0.0f);
        out[i] = (t4 * in2[i]);
    }
}
```

Three kernels become one. Three passes over memory become one. The intermediates live in registers and never reach main memory.

---

## Results

Measured on an Apple M4 Max. Every optimised path is verified to produce output identical to the unoptimised path before it is timed.

| Optimisation | Workload | Before | After | Speedup |
|---|---|---|---|---|
| Operator fusion | 16M elements | 8.0ms | 3.4ms | 2.35x |
| Blocked, threaded matmul | 768 x 768 | 348.0ms | 3.8ms | 91.5x |
| Both, 4-layer MLP | 256 x 256 | 43.0ms | 1.3ms | 33.6x |

How the matmul got there, with the same arithmetic throughout:

| Stage | Time | GFLOPS | This step | Cumulative |
|---|---|---|---|---|
| Naive triple loop | 350.6ms | 3 | | 1.0x |
| Loop reordering | 28.4ms | 32 | 12.3x | 12.3x |
| Cache blocking | 32.0ms | 28 | 0.89x | 11.0x |
| restrict | 32.0ms | 28 | 1.00x | 11.0x |
| Threading and tuning | 3.8ms | 238 | 8.4x | 92.0x |

Against OpenBLAS, called through NumPy on the same machine, at 768 x 768: 234 GFLOPS against 351, a gap of 1.5x.

A 550,977-parameter GPT trained on the tinyshakespeare corpus with data-parallel training across 8 CPU cores brought character-level cross-entropy from 3.66 to 1.44 over 33,113 steps.

---

## Limitations

The core library is pure Python, hence it is slower than a production framework.

The compiler handles the forward pass over six operations, with no autograd. Only matmul is threaded; the element-wise kernels are single-threaded, which is fine because they are memory-bound rather than compute-bound.

Metal and Accelerate backends require macOS and PyObjC. If not available, it defaults to pure Python.

---

Built by Srihari Srinivasan. MIT licensed.

The full write-up, including the mathematics behind each component and the bugs worth knowing about can be found at **https://github.com/SrihariSr/Forge**