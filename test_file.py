import sys; sys.path.insert(0, ".")
from Forge import Tensor
from Forge.dtype import float64
from Forge.CalcLlama import grad_check
x = Tensor([[[1.,2.],[3.,4.]],[[5.,6.],[7.,8.]]], dtype=float64, requires_grad=True)
print("SelectBatch grad_check:", grad_check(lambda t: t.select_batch(1).sum(), [x]))