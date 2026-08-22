"""
Forge: a machine learning library and deep learning compiler written from
scratch in Python.

The core has no dependencies. Every operation, from the autograd engine to the
attention mechanism, is implemented from first principles.
"""

from forge.tensor import Tensor
from forge.dtype import float32, float64, int32, int64
from forge.serialization import save_model, load_model

__version__ = "0.1.0"

__all__ = [
    "Tensor",
    "float32", "float64", "int32", "int64",
    "save_model", "load_model",
]
