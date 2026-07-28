import array
import ctypes
import os
import random
import subprocess
import tempfile
import time

try:
    import numpy
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

from graph import placeholder, relu, matmul, topological_order
from interpreter import run, Tensor, _lib, _addr, _empty
from compiled_run import compile_graph, run_compiled

# Unoptimised matmul, compiled at startup so the report has a real "before".
NAIVE_SRC = r"""
void matmul_naive(const float* a, const float* b, float* out, int m, int n, int p){
    for (int i = 0; i < m; i++){
        for (int k = 0; k < p; k++){
            float sum = 0.0f;
            for (int j = 0; j < n; j++){
                sum += a[i*n + j] * b[j*p + k];
            }
            out[i*p + k] = sum;
        }
    }
}
"""

def bench(fn, reps = 3) -> float:
    """
    Mean milliseconds per call, discarding the first run so one-off costs such
    as page mapping do not skew the result.
    """
    fn()
    start = time.time()
    for _ in range(reps):
        fn()
    return (time.time() - start) / reps * 1000.0

def rand_tensor(shape) -> Tensor:
    """
    A `Tensor` of the given shape filled with random float32 values.
    """
    total = 1
    for d in shape:
        total *= d
    return Tensor(array.array('f', [random.uniform(-2, 2) for _ in range(total)]), shape)

def max_diff(first, second) -> float:
    """
    Largest absolute difference between two tensors, used to prove an
    optimisation did not change the answer.
    """
    return max(abs(x - y) for x, y in zip(first.data, second.data))

def load_naive() -> ctypes.CDLL:
    """
    Compile and load the unoptimised matmul used as the comparison baseline.
    """
    folder = tempfile.mkdtemp(prefix = "bench_")
    source = os.path.join(folder, "naive.c")
    library = os.path.join(folder, "naive.so")

    with open(source, "w") as f:
        f.write(NAIVE_SRC)

    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", source, "-o", library],
                   check = True, capture_output = True)

    lib = ctypes.CDLL(library)
    FP = ctypes.POINTER(ctypes.c_float)
    lib.matmul_naive.argtypes = [FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.matmul_naive.restype = None
    return lib

def run_unoptimised(root, feeds, naive) -> Tensor:
    """
    Execute a graph with no optimisations: one kernel per node, a fresh buffer
    every time, and the naive matmul.
    """
    values = {}

    for node in topological_order(root):
        if node.op == "placeholder":
            values[node] = feeds[node.name]

        elif node.op == "matmul":
            a = values[node.inputs[0]]
            b = values[node.inputs[1]]
            m, n = a.shape
            _, p = b.shape
            out = _empty((m, p))
            naive.matmul_naive(_addr(a.data), _addr(b.data), _addr(out.data), m, n, p)
            values[node] = out

        elif node.op in ("add", "sub", "mul"):
            a = values[node.inputs[0]]
            b = values[node.inputs[1]]
            out = _empty(a.shape)
            getattr(_lib, node.op)(_addr(a.data), _addr(b.data), _addr(out.data), a.size)
            values[node] = out

        elif node.op == "relu":
            x = values[node.inputs[0]]
            out = _empty(x.shape)
            _lib.relu(_addr(x.data), _addr(out.data), x.size)
            values[node] = out

    return values[root]

def measure_fusion() -> list:
    """
    Time element-wise chains with and without fusion, across several sizes.
    Returns rows of (label, size, before, after).
    """
    random.seed(1)
    a = placeholder("a")
    b = placeholder("b")
    c = placeholder("c")
    graph = relu(a + b) * c
    steps = compile_graph(graph)

    rows = []
    for count in (100_000, 4_000_000, 16_000_000):
        feeds = {name: rand_tensor((count,)) for name in ("a", "b", "c")}
        assert max_diff(run(graph, feeds), run_compiled(graph, steps, feeds)) == 0.0

        reps = 10 if count <= 1_000_000 else 3
        before = bench(lambda: run(graph, feeds), reps)
        after = bench(lambda: run_compiled(graph, steps, feeds), reps)
        rows.append(("fusion", f"{count:,} elements", before, after))

    return rows

def measure_matmul(naive) -> list:
    """
    Time the blocked matmul against the naive triple loop, across several sizes.
    """
    random.seed(2)
    rows = []

    for size in (256, 512, 768):
        a = rand_tensor((size, size))
        b = rand_tensor((size, size))
        slow_out = _empty((size, size))
        fast_out = _empty((size, size))

        naive.matmul_naive(_addr(a.data), _addr(b.data), _addr(slow_out.data), size, size, size)
        _lib.matmul(_addr(a.data), _addr(b.data), _addr(fast_out.data), size, size, size)
        assert max_diff(slow_out, fast_out) < 1e-2

        before = bench(lambda: naive.matmul_naive(_addr(a.data), _addr(b.data),
                                                  _addr(slow_out.data), size, size, size))
        after = bench(lambda: _lib.matmul(_addr(a.data), _addr(b.data),
                                          _addr(fast_out.data), size, size, size))
        rows.append(("blocked matmul", f"{size} x {size}", before, after))

    return rows

def measure_against_numpy(naive) -> list:
    """
    Time the naive matmul, the compiled blocked matmul, and NumPy on the same
    data. NumPy calls BLAS, decades of hand-tuned assembly, so it is the
    yardstick that says how far the compiler actually got.
    Returns rows of (size, naive, blocked, numpy).
    """
    random.seed(4)
    rows = []

    for size in (256, 512, 768):
        a = rand_tensor((size, size))
        b = rand_tensor((size, size))
        slow_out = _empty((size, size))
        fast_out = _empty((size, size))

        left = numpy.frombuffer(a.data, dtype = numpy.float32).reshape(size, size)
        right = numpy.frombuffer(b.data, dtype = numpy.float32).reshape(size, size)
        expected = left @ right

        naive.matmul_naive(_addr(a.data), _addr(b.data), _addr(slow_out.data), size, size, size)
        _lib.matmul(_addr(a.data), _addr(b.data), _addr(fast_out.data), size, size, size)
        ours = numpy.frombuffer(fast_out.data, dtype = numpy.float32).reshape(size, size)
        assert float(numpy.abs(expected - ours).max()) < 1e-2

        slow = bench(lambda: naive.matmul_naive(_addr(a.data), _addr(b.data),
                                                _addr(slow_out.data), size, size, size))
        fast = bench(lambda: _lib.matmul(_addr(a.data), _addr(b.data),
                                         _addr(fast_out.data), size, size, size))
        blas = bench(lambda: left @ right, 10)
        rows.append((size, slow, fast, blas))

    return rows

def measure_pipeline(naive) -> list:
    """
    Time a 4-layer MLP forward pass at three levels of optimisation.
    Returns rows of (label, milliseconds).
    """
    random.seed(3)
    dim = 256

    node = placeholder("in")
    feeds = {"in": rand_tensor((dim, dim))}
    for i in range(4):
        feeds[f"W{i}"] = rand_tensor((dim, dim))
        feeds[f"b{i}"] = rand_tensor((dim, dim))
        node = relu(matmul(node, placeholder(f"W{i}")) + placeholder(f"b{i}"))

    steps = compile_graph(node)
    assert max_diff(run_unoptimised(node, feeds, naive),
                    run_compiled(node, steps, feeds)) < 1e-1

    return [
        ("unoptimised", bench(lambda: run_unoptimised(node, feeds, naive))),
        ("blocked matmul", bench(lambda: run(node, feeds))),
        ("blocked matmul + fusion", bench(lambda: run_compiled(node, steps, feeds))),
    ]

def print_numpy_table(rows):
    """
    Show the naive matmul, the compiled one, and BLAS side by side. NumPy calls
    BLAS, which is decades of hand-tuned assembly with SIMD and multithreading,
    so the slowdown factor is a straight measure of how much ground the compiler
    made up against production tooling.
    """
    print()
    print("MATRIX MULTIPLICATION, MEASURED AGAINST BLAS (using NumPy)")
    print("  " + "\u2500" * 70)
    print(f"{'size':<12}{'naive':>11}{'compiled':>11}{'BLAS':>10}"
          f"{'slowdown against BLAS':>26}")
    print(f"{'':<12}{'':>11}{'':>11}{'':>10}{'naive':>14}{'compiled':>12}")
    print("  " + "\u2500" * 70)

    for size, slow, fast, blas in rows:
        label = f"{size} x {size}"
        print(f"{label:<12}{slow:>9.1f}ms{fast:>9.1f}ms{blas:>8.2f}ms"
              f"{slow / blas:>13.0f}x{fast / blas:>11.0f}x")

    gains = [slow / fast for _, slow, fast, _ in rows]
    print("  " + "\u2500" * 70)
    print(f"The compiler closed the gap to BLAS by {sum(gains) / len(gains):.1f}x on average.")
    print()

def main():
    naive = load_naive()

    breakdown = measure_fusion() + measure_matmul(naive)
    pipeline = measure_pipeline(naive)
    numpy_rows = measure_against_numpy(naive) if HAVE_NUMPY else None

    print()
    print("FORGE'S DEEP LEARNING COMPILER")
    print()

    print(f"{'optimisation':<16}{'workload':>20}{'before':>12}{'after':>12}{'speedup':>10}")
    print("  " + "\u2500" * 70)
    for label, workload, before, after in breakdown:
        print(f"{label:<16}{workload:>20}{before:>10.1f}ms{after:>10.1f}ms"
              f"{before / after:>9.2f}x")

    if numpy_rows is not None:
        print_numpy_table(numpy_rows)
    else:
        print()
        print("NumPy not installed, skipping the BLAS comparison.")

    print("4-layer MLP forward pass, 256 x 256")
    print("  " + "\u2500" * 70)
    slowest = max(t for _, t in pipeline)
    baseline = pipeline[0][1]
    for label, taken in pipeline:
        blocks = "\u2588" * max(1, round(taken / slowest * 26))
        speedup = "" if taken == baseline else f"{baseline / taken:.2f}x"
        print(f"{label:<26}{taken:>8.1f}ms  {blocks:<28}{speedup}")

if __name__ == "__main__":
    main()