import sys; sys.path.insert(0, ".")
import random; random.seed(0)
from NeuralNetwork.layers import GPT

m = GPT(12, 16, 4, 32, 2, 6)
print("repr:", repr(m))
print("params:", len(list(m.parameters())), "(expect 74)")
print("has block_0:", hasattr(m, "block_0"))