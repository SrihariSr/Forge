from NeuralNetwork.layers import Linear, ReLULayer, FusedLinearReLULayer


def optimize_model(layers):
    """
    Takes a list of layers and returns an optimized list
    with fused operations where possible.
    """
    optimized = []
    i = 0

    while i < len(layers):
        # Pattern: Linear followed by ReLU
        if (i + 1 < len(layers) and isinstance(layers[i], Linear) and isinstance(layers[i + 1], ReLULayer)):
            linear = layers[i]
            in_f = linear.weight.shape[1]
            out_f = linear.weight.shape[0]

            fused = FusedLinearReLULayer(in_f, out_f)

            # Copy weights from the original linear layer
            fused.weight._data = linear.weight._data.__class__(
                linear.weight.dtype.typecode, linear.weight._data
            )

            # Copy bias — need to flatten from (1, out_f) to (out_f,)
            if linear.bias is not None:
                import array as _array
                fused.bias._data = _array.array(
                    linear.bias.dtype.typecode, list(linear.bias._data)
                )

            optimized.append(fused)
            i += 2  # Skip both the Linear and ReLU

        else:
            optimized.append(layers[i])
            i += 1

    return optimized