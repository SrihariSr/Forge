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

# Each stage of the matmul as it was actually developed, so the report can show
# where every factor of speedup came from. These are compiled at startup and
# only ever used for measurement; the real kernel lives in kernels.c.
STAGES_SRC = r"""
#include <string.h>
#include <pthread.h>

/* 1. Naive. For each output element, walk a row of a and DOWN a column of b.
 *    Going down a column jumps a whole row's width in memory every step, so
 *    most of each cache line fetched is thrown away. */
void mm_naive(const float* a, const float* b, float* out, int m, int n, int p){
    for (int i = 0; i < m; i++)
        for (int k = 0; k < p; k++){
            float sum = 0.0f;
            for (int j = 0; j < n; j++) sum += a[i*n + j] * b[j*p + k];
            out[i*p + k] = sum;
        }
}

/* 2. Loop order swapped so the innermost loop runs over k. Now b and out both
 *    step forward one float at a time, which is what caches are built for.
 *    Identical arithmetic, only the order of memory access changed. */
void mm_reordered(const float* a, const float* b, float* out, int m, int n, int p){
    memset(out, 0, (size_t)m*p*sizeof(float));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++){
            float a_ij = a[i*n + j];
            for (int k = 0; k < p; k++) out[i*p + k] += a_ij * b[j*p + k];
        }
}

/* 3. Cache blocking. A large matrix does not fit in cache, so data is evicted
 *    before it can be reused. Working in tiles finishes everything that touches
 *    a tile while it is still resident. */
#define B3 64
void mm_blocked(const float* a, const float* b, float* out, int m, int n, int p){
    memset(out, 0, (size_t)m*p*sizeof(float));
    for (int i0=0;i0<m;i0+=B3) for (int j0=0;j0<n;j0+=B3) for (int k0=0;k0<p;k0+=B3){
        int ie=(i0+B3<m)?i0+B3:m, je=(j0+B3<n)?j0+B3:n, ke=(k0+B3<p)?k0+B3:p;
        for (int i=i0;i<ie;i++) for (int j=j0;j<je;j++){
            float a_ij=a[i*n+j];
            for (int k=k0;k<ke;k++) out[i*p+k]+=a_ij*b[j*p+k];
        }
    }
}

/* 4. restrict, promising the compiler the three arrays do not overlap, so it
 *    is free to reorder loads and stores. */
void mm_restrict(const float* restrict a, const float* restrict b,
                 float* restrict out, int m, int n, int p){
    memset(out, 0, (size_t)m*p*sizeof(float));
    for (int i0=0;i0<m;i0+=B3) for (int j0=0;j0<n;j0+=B3) for (int k0=0;k0<p;k0+=B3){
        int ie=(i0+B3<m)?i0+B3:m, je=(j0+B3<n)?j0+B3:n, ke=(k0+B3<p)?k0+B3:p;
        for (int i=i0;i<ie;i++) for (int j=j0;j<je;j++){
            float a_ij=a[i*n+j];
            for (int k=k0;k<ke;k++) out[i*p+k]+=a_ij*b[j*p+k];
        }
    }
}
"""


def load_stages() -> ctypes.CDLL:
    """
    Compile every intermediate version of the matmul, so the report can show
    what each optimisation actually contributed rather than only the total.
    """
    folder = tempfile.mkdtemp(prefix = "stages_")
    source = os.path.join(folder, "stages.c")
    library = os.path.join(folder, "stages.so")
    with open(source, "w") as f:
        f.write(STAGES_SRC)
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", source, "-o", library,
                    "-lpthread"], check = True, capture_output = True)

    lib = ctypes.CDLL(library)
    FP = ctypes.POINTER(ctypes.c_float)
    for name in ("mm_naive", "mm_reordered", "mm_blocked", "mm_restrict"):
        fn = getattr(lib, name)
        fn.argtypes = [FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        fn.restype = None
    return lib


def measure_journey(stages) -> list:
    """
    Time every version of the matmul on the same problem, in the order they
    were built, and check each against the naive one before timing it.
    Returns rows of (label, milliseconds).
    """
    random.seed(9)
    size = 768
    a = rand_tensor((size, size))
    b = rand_tensor((size, size))
    ref = _empty((size, size))
    out = _empty((size, size))

    stages.mm_naive(_addr(a.data), _addr(b.data), _addr(ref.data),
                    size, size, size)

    rows = []
    for label, fn in (("naive",                    stages.mm_naive),
                      ("+ loop reordering",        stages.mm_reordered),
                      ("+ cache blocking",         stages.mm_blocked),
                      ("+ restrict",               stages.mm_restrict)):
        fn(_addr(a.data), _addr(b.data), _addr(out.data), size, size, size)
        assert max_diff(ref, out) < 1e-1, f"{label} changed the answer"
        rows.append((label, bench(lambda f = fn: f(_addr(a.data), _addr(b.data),
                                                   _addr(out.data),
                                                   size, size, size), 7)))

    # the kernel actually shipped in kernels.c: adaptive block size, 8 threads
    _lib.matmul(_addr(a.data), _addr(b.data), _addr(out.data), size, size, size)
    assert max_diff(ref, out) < 1e-1, "the shipped kernel changed the answer"
    rows.append(("+ threading and tuning",
                 bench(lambda: _lib.matmul(_addr(a.data), _addr(b.data),
                                           _addr(out.data), size, size, size), 15)))
    return rows


def print_journey(rows):
    """
    Show what each optimisation contributed, in the order they were applied.
    """
    size = 768
    flops = 2.0 * size ** 3
    first = rows[0][1]

    print()
    print("HOW THE MATMUL GOT FAST, 768 x 768")
    print("  " + "\u2500" * 70)
    print(f"{'stage':<26}{'time':>10}{'GFLOPS':>10}{'step':>10}{'total':>10}")
    print("  " + "\u2500" * 70)

    previous = first
    for label, ms in rows:
        gflops = flops / (ms / 1000.0) / 1e9
        step = previous / ms
        total = first / ms
        step_text = "" if ms == first else f"{step:.2f}x"
        print(f"{label:<26}{ms:>8.1f}ms{gflops:>10.0f}{step_text:>10}{total:>9.1f}x")
        previous = ms
    print("  " + "\u2500" * 70)
    print("Same arithmetic throughout. Every gain came from moving memory")
    print("differently or from using more cores.")
    print()


def bench(fn, reps = 15, warmups = 3) -> float:
    """
    Median milliseconds per call.

    The median is used rather than the mean because a single unlucky run, where
    the operating system schedules something else on the core, drags a mean
    noticeably but barely moves a median.

    Several warmup calls are discarded first. One is not always enough: BLAS in
    particular spins up a thread pool and picks a kernel on its first call, and
    that cost lands entirely on whichever measurement runs first.
    """
    for _ in range(warmups):
        fn()
    times = []
    for _ in range(reps):
        start = time.time()
        fn()
        times.append((time.time() - start) * 1000.0)
    times.sort()
    return times[len(times) // 2]

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

def warm_blas():
    """
    Burn in the BLAS thread pool and kernel dispatch before anything is timed.

    Without this the first BLAS call absorbs the whole start-up cost, which
    produced a nonsense reading where a 256x256 multiply appeared to take
    longer than a 512x512 one on eight times the work.
    """
    warm = numpy.random.rand(512, 512).astype(numpy.float32)
    for _ in range(20):
        _ = warm @ warm

def measure_against_numpy(naive) -> list:
    """
    Time the naive matmul, the compiled blocked matmul, and BLAS on the same
    data. NumPy calls BLAS, which is decades of hand-tuned assembly using every
    core, so it is the yardstick for how far the compiler actually got.
    Returns rows of (size, naive, blocked, blas).
    """
    random.seed(4)
    warm_blas()
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
        mine = numpy.frombuffer(fast_out.data, dtype = numpy.float32).reshape(size, size)
        assert float(numpy.abs(expected - mine).max()) < 1e-2

        slow = bench(lambda: naive.matmul_naive(_addr(a.data), _addr(b.data),
                                                _addr(slow_out.data), size, size, size))
        fast = bench(lambda: _lib.matmul(_addr(a.data), _addr(b.data),
                                         _addr(fast_out.data), size, size, size))
        # BLAS finishes in under a millisecond, so it needs many more
        # repetitions than our kernels to give a stable reading.
        blas = bench(lambda: left @ right, 200, warmups = 20)
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
    Show the naive matmul, the compiled one, and BLAS side by side.

    GFLOPS is reported alongside the times because it is comparable across
    sizes in a way milliseconds are not: a 768 multiply does 27 times the work
    of a 256 one, so its longer runtime says nothing on its own.
    """
    print()
    print("MATRIX MULTIPLICATION, MEASURED AGAINST BLAS (using NumPy)")
    print("  " + "\u2500" * 70)
    print(f"{'size':<12}{'naive':>10}{'compiled':>10}{'BLAS':>9}"
          f"{'GFLOPS':>18}{'BLAS is':>11}")
    print(f"{'':<12}{'':>10}{'':>10}{'':>9}{'mine':>11}{'BLAS':>7}{'faster by':>11}")
    print("  " + "\u2500" * 70)

    for size, slow, fast, blas in rows:
        label = f"{size} x {size}"
        # a size x size multiply is 2 * size^3 floating point operations
        flops = 2.0 * size ** 3
        ours_gf = flops / (fast / 1000.0) / 1e9
        blas_gf = flops / (blas / 1000.0) / 1e9
        print(f"{label:<12}{slow:>8.1f}ms{fast:>8.1f}ms{blas:>7.2f}ms"
              f"{ours_gf:>11.0f}{blas_gf:>7.0f}{blas_gf / ours_gf:>10.1f}x")

    gains = [slow / fast for _, slow, fast, _ in rows]
    print("  " + "\u2500" * 70)
    print(f"The compiler closed the gap to BLAS by {sum(gains) / len(gains):.1f}x on average.")
    print()

def main():
    naive = load_naive()
    stages = load_stages()

    journey = measure_journey(stages)
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

    print_journey(journey)

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