def grad_check(func, inputs, eps=1e-3, tol=1e-2):
    
    # Using first principles for differentiation: df/dx = lim(h->0) [f(x + h) - f(x - h)] / 2h
    # We use a small value of h (eps) to approximate the limit.

    for inp in inputs:
        inp.grad = None

    output = func(*inputs)
    output.backward()

    analytical_grads = [inp.grad for inp in inputs]

    # Check each input
    for idx, inp in enumerate(inputs):
        analytical_grad = analytical_grads[idx]

        # Compute numerical gradient for each element
        for i in range(len(inp._data)):
            original = inp._data[i]

            # f(x + h)
            inp._data[i] = original + eps
            # Clear grads and recompute
            for inp2 in inputs:
                inp2.grad = None
                inp2._grad_fn = None
            out_plus = func(*inputs)

            # f(x - h)
            inp._data[i] = original - eps
            for inp2 in inputs:
                inp2.grad = None
                inp2._grad_fn = None
            out_minus = func(*inputs)

            # Restore
            inp._data[i] = original

            # Numerical gradient
            numerical = (out_plus._data[0] - out_minus._data[0]) / (2 * eps)
            analytical = analytical_grad._data[i]

            diff = abs(numerical - analytical)
            if diff > tol:
                raise AssertionError(
                    f"Gradient check failed for input {idx}, element {i}: "
                    f"numerical={numerical:.6f}, analytical={analytical:.6f}, "
                    f"diff={diff:.6f}"
                )

    return True