"""
Build the C kernels.

    python -m forge.compiler.build

The kernels are deliberately not compiled during `pip install`. They need a C
compiler and they are architecture-specific, so building them is a separate,
explicit step rather than something that happens silently and fails obscurely.
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build():
    source = os.path.join(HERE, "kernels.c")
    library = os.path.join(HERE, "kernels.so")

    command = ["gcc", "-O2", "-shared", "-fPIC", source, "-o", library,
               "-lm", "-lpthread"]

    print(" ".join(command))
    result = subprocess.run(command, capture_output = True, text = True)

    if result.returncode != 0:
        print(result.stderr, file = sys.stderr)
        raise SystemExit("kernel build failed")

    print(f"built {library}")


if __name__ == "__main__":
    build()
