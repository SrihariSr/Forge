import array as _array

class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def step(self):
        for p in self.parameters:
            if p.grad is not None:
                p._data = _array.array(p.dtype.typecode, [p._data[i] - self.lr * p.grad._data[i] for i in range(len(p._data))])
