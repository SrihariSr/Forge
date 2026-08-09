"""
Check the threaded matmul against a reference.

A data race does not crash and does not warn. It produces answers that are
mostly right, with a few elements wrong, and different ones each run. So the
bar here is bit-identical output, repeated, not "close enough".

Run from inside the compiler folder, after rebuilding kernels.so.
"""

import ctypes
import array
import random

lib = ctypes.CDLL("./kernels.so")
FP = ctypes.POINTER(ctypes.c_float)
lib.matmul.argtypes = [FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.matmul.restype = None

def addr(a):
    return ctypes.cast(a.buffer_info()[0], FP)

def reference(a, b, m, n, p):
    # Deliberately slow and obviously correct, in double precision.
    out = [0.0] * (m * p)
    for i in range(m):
        for k in range(p):
            s = 0.0
            for j in range(n):
                s += a[i*n + j] * b[j*p + k]
            out[i*p + k] = s
    return out

def run(m, n, p, seed):
    random.seed(seed)
    a = array.array('f', [random.uniform(-2, 2) for _ in range(m*n)])
    b = array.array('f', [random.uniform(-2, 2) for _ in range(n*p)])
    o = array.array('f', [0.0] * (m*p))
    lib.matmul(addr(a), addr(b), addr(o), m, n, p)
    return a, b, o

# Shapes chosen so several exercise more than one row block, since a single
# block means only one thread runs and nothing is being tested.
# BLOCK is 64, so m must exceed 64 for threading to engage at all.
shapes = [
    (1, 1, 1),          # degenerate
    (5, 7, 9),          # smaller than one block
    (64, 64, 66),       # exactly one block
    (65, 65, 65),       # just over one block, so two threads
    (100, 70, 90),      # two blocks
    (128, 128, 128),    # two blocks, exact
    (200, 64, 150),     # four blocks
    (300, 80, 120),     # five blocks
    (513, 64, 64),      # nine blocks, awkward remainder
]

print("threaded matmul verification")
print(f"{'shape':<20}{'row blocks':>11}{'max error':>13}")
print("-" * 46)

failures = 0
for (m, n, p) in shapes:
    a, b, o = run(m, n, p, seed=1)
    want = reference(a, b, m, n, p)
    err = max(abs(x - y) for x, y in zip(o, want))
    blocks = (m + 63) // 64
    ok = err < 1e-3
    if not ok:
        failures += 1
    print(f"{m}x{n} @ {n}x{p}".ljust(20) + f"{blocks:>11}{err:>13.2e}"
          + ("" if ok else "   FAIL"))

# Races are intermittent, so repeat one threaded shape many times. A single
# clean run proves very little.
print()
print("repeating a multi-block shape 40 times, since races are intermittent")
m, n, p = 300, 80, 120
first = None
mismatched_runs = 0
for trial in range(40):
    _, _, o = run(m, n, p, seed=7)      # same seed, so every run must agree
    if first is None:
        first = list(o)
    elif list(o) != first:
        mismatched_runs += 1

print(f"  runs differing from the first: {mismatched_runs}   (must be 0)")
if mismatched_runs:
    failures += 1

print()
print("FAILURES:", failures)