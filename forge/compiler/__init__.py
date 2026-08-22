"""
A deep learning compiler.

Builds a computation graph, plans which operations can share a single pass over
memory, generates C for each group, compiles it with gcc, and loads it back
through ctypes while the program is running.

The C kernels are not built during installation. Run the build script once
before using this module:

    python -m forge.compiler.build
"""

from forge.compiler.graph import placeholder, relu, matmul, topological_order
from forge.compiler.fusion import fuse
from forge.compiler.codegen import generate_c, compile_and_load
from forge.compiler.compiled_run import compile_graph, run_compiled

__all__ = [
    "placeholder", "relu", "matmul", "topological_order",
    "fuse", "generate_c", "compile_and_load",
    "compile_graph", "run_compiled",
]
