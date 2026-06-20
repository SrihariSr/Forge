from NeuralNetwork.parameter import Parameter


class Module:
    """Base class for all neural network modules."""

    def __init__(self):
        self._Parameters = {}
        self._modules = {}
        self.training = True

    def forward(self, *args):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if '_Parameters' not in self.__dict__:
                self.__dict__['_Parameters'] = {}
            self._Parameters[name] = value
        elif isinstance(value, Module):
            if '_modules' not in self.__dict__:
                self.__dict__['_modules'] = {}
            self._modules[name] = value
        super().__setattr__(name, value)

    def Parameters(self):
        # Return all Parameters, including from child modules.
        params = list(self._Parameters.values())
        for module in self._modules.values():
            params.extend(module.Parameters())

        return params

    def parameters(self):
        return self.Parameters()

    def zero_grad(self):
        # Reset all Parameter gradients to None.
        for param in self.Parameters():
            param.grad = None

    def train(self):
        # Set module to training mode.
        self.training = True
        for module in self._modules.values():
            module.train()

    def eval(self):
        # Set module to evaluation mode.
        self.training = False
        for module in self._modules.values():
            module.eval()

    
    def state_dict(self):
        # Return a dictionary of all parameters with their names
        state = {}

        # Own parameters
        for name, param in self._Parameters.items():
            state[name] = param

        # Child module parameters (with prefix)
        for mod_name, module in self._modules.items():
            child_state = module.state_dict()
            for key, value in child_state.items():
                state[f"{mod_name}.{key}"] = value

        return state

    def load_state_dict(self, state):
        # Load parameters from a state dictionary
        own_state = self.state_dict()

        for name, param in own_state.items():
            if name not in state:
                raise KeyError(f"Missing key in state dict: {name}")

            source = state[name]

            if param.shape != source.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"expected {param.shape}, got {source.shape}"
                )

            # Copy data
            for i in range(len(param._data)):
                param._data[i] = source._data[i]