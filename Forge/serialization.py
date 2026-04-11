import json
import array as _array

def save_model(model, filepath):
    # Save a model's parameters to a file
    state = model.state_dict()

    save_data = {}
    for name, param in state.items():
        save_data[name] = {
            'shape': list(param.shape),
            'dtype': param.dtype.typecode,
            'data': list(param._data),
        }

    with open(filepath, 'w') as f:
        json.dump(save_data, f)

    print(f"Model saved to {filepath}")
    print(f"  Parameters: {len(save_data)}")
    total = sum(len(v['data']) for v in save_data.values())
    print(f"  Total values: {total}")


def load_model(model, filepath):
    # Load parameters from a file into a model
    from Forge.tensor import Tensor
    from Forge.dtype import float32, float64, int32, int64

    typecode_to_dtype = {
        'f': float32,
        'd': float64,
        'i': int32,
        'l': int64,
    }

    with open(filepath, 'r') as f:
        save_data = json.load(f)

    # Convert saved data back to tensors
    state = {}
    for name, info in save_data.items():
        shape = tuple(info['shape'])
        typecode = info['dtype']
        data = info['data']
        dtype = typecode_to_dtype[typecode]

        tensor = Tensor.__new__(Tensor)
        tensor._data = _array.array(typecode, data)
        tensor.shape = shape
        tensor.dtype = dtype
        tensor.requires_grad = True
        tensor.grad = None
        tensor._grad_fn = None
        state[name] = tensor

    model.load_state_dict(state)

    print(f"Model loaded from {filepath}")
    print(f"Parameters: {len(state)}")