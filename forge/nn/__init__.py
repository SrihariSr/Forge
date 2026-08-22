"""
Neural network building blocks: the Module system, layers, and losses.

Every layer here is built from the differentiable operations in forge.autograd,
so none of them needs a hand-written backward pass.
"""

from forge.nn.module import Module
from forge.nn.parameter import Parameter
from forge.nn.layers import (
    Linear, ReLULayer, FusedLinearReLULayer, Sequential, Embedding,
    SimpleAttention, TransformerBlock, CharTransformer,
    LayerNorm, MultiHeadAttention, GPTBlock, GPT,
)
from forge.nn.losses import MSELoss, BCELoss, CrossEntropyLoss

__all__ = [
    "Module", "Parameter",
    "Linear", "ReLULayer", "FusedLinearReLULayer", "Sequential", "Embedding",
    "SimpleAttention", "TransformerBlock", "CharTransformer",
    "LayerNorm", "MultiHeadAttention", "GPTBlock", "GPT",
    "MSELoss", "BCELoss", "CrossEntropyLoss",
]
