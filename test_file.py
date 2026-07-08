import sys; sys.path.insert(0, ".")
import random
from Forge import Tensor
from Forge.dtype import float32, float64
from Forge.CalcLlama.accelerate_backend import ACCELERATE_AVAILABLE
from Forge.CalcLlama import grad_check

print("ACCELERATE_AVAILABLE:", ACCELERATE_AVAILABLE)   # expect True on your Mac

# correctness: float32 (BLAS) vs float64 (loop)
random.seed(0); M,K,N = 16,20,12
a=[random.uniform(-1,1) for _ in range(M*K)]; b=[random.uniform(-1,1) for _ in range(K*N)]
f32=(Tensor([a[i*K:(i+1)*K] for i in range(M)],dtype=float32)@Tensor([b[i*N:(i+1)*N] for i in range(K)],dtype=float32))._data
f64=(Tensor([a[i*K:(i+1)*K] for i in range(M)],dtype=float64)@Tensor([b[i*N:(i+1)*N] for i in range(K)],dtype=float64))._data
print("max rel err:", f"{max(abs(x-y)/(abs(y)+1e-9) for x,y in zip(f32,f64)):.2e}")
print("grad_check:", grad_check(lambda t:(t@Tensor([[1.,0.],[0.,1.]],dtype=float64)).sum(),
                                [Tensor([[1.,2.],[3.,4.]],dtype=float64,requires_grad=True)]))