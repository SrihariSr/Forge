from Forge.tensor import Tensor

class Parameter(Tensor):
    # A tensor that is automatically marked as requiring gradients
    
    def __init__(self, data):
        if isinstance(data, Tensor):
            super().__init__(data._rebuild_nested(), dtype=data.dtype, requires_grad=True)
        else:
            super().__init__(data, requires_grad=True)

    def __repr__(self):
        return f"Parameter({self._rebuild_nested()})"