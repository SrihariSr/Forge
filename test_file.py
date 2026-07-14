import sys; sys.path.insert(0, ".")
import random; random.seed(0)
from Forge import Tensor
from NeuralNetwork.layers import MultiHeadAttention

mha = MultiHeadAttention(16, 4)
x = Tensor([[[float(i + j) for j in range(16)] for i in range(5)]])   # (1, 5, 16)
out = mha(x)
print(mha)
print("output shape:", out.shape, "(expect (5, 16))")

out.sum().backward()
print("all heads get gradients:",
      all(getattr(mha, f"query_{h}").weight.grad is not None for h in range(4)))