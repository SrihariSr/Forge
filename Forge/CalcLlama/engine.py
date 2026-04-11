class Function:
    # Base class for all differentiable operations.
    def __init__(self):
        self.inputs = []
        self.saved_tensors = []

    def save_for_backward(self, *tensors):
        self.saved_tensors = list(tensors)

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError