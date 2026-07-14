import sys; sys.path.insert(0, ".")
from Forge import Tensor
from Forge.dtype import float64
from Forge.CalcLlama import grad_check
from NeuralNetwork.layers import LayerNorm

print("division grad_check:", grad_check(
    lambda t: (t / Tensor([2., 4.], dtype=float64)).sum(),
    [Tensor([4., 9.], dtype=float64, requires_grad=True)]))

ln = LayerNorm(4)
out = ln(Tensor([[1., 2., 3., 4.], [10., 12., 14., 16.]]))
row = [out._data[c] for c in range(4)]
mean = sum(row) / 4
std = (sum((v - mean) ** 2 for v in row) / 4) ** 0.5
print(f"row 0 -> mean {mean:+.6f} (expect ~0), std {std:.6f} (expect ~1)")
print("layernorm params:", len(list(ln.parameters())), "(expect 2)")