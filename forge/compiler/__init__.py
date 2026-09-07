"""
A deep learning compiler.

Builds a computation graph, plans which operations can share a single pass over
memory, generates C for each group, compiles it with gcc, and loads it back
through ctypes while the program is running.

The C kernels are not built during installation. Run the build script once
before using this module:

    python -m forge.compiler.build
"""

__all__ = [
    "placeholder", "relu", "matmul", "topological_order",
    "fuse", "generate_c", "compile_and_load",
    "compile_graph", "run_compiled",
]


def __getattr__(name):
    # Imported on first use rather than eagerly, so that
    # `python -m forge.compiler.build` can run before the kernels exist.
    # interpreter.py raises on import when kernels.so is missing, which would
    # otherwise make the build command impossible to invoke.
    if name in ("placeholder", "relu", "matmul", "topological_order"):
        from forge.compiler import graph
        return getattr(graph, name)
    if name == "fuse":
        from forge.compiler.fusion import fuse
        return fuse
    if name in ("generate_c", "compile_and_load"):
        from forge.compiler import codegen
        return getattr(codegen, name)
    if name in ("compile_graph", "run_compiled"):
        from forge.compiler import compiled_run
        return getattr(compiled_run, name)
    raise AttributeError(f"module 'forge.compiler' has no attribute '{name}'")
