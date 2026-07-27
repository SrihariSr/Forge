import ctypes
from graph import topological_order
from fusion import fuse
from codegen import generate_c, compile_and_load, group_external_inputs
from interpreter import Tensor, _empty, _addr, _lib

def compile_graph(root):
    """
    Turns a graph into an executable plan by:
     - Group the nodes.
     - Walk the graph in execution order.
     - For each node, either record a plain kernel step, or, if it belongs to
       a fusion group we have not handled yet, generate and compile a custom
       kernel for that whole group.
    """

    groups = fuse(root)
    
    # Lookup of each node to it's group (if any)
    node_to_group = {}
    for g in groups:
        for n in g:
            node_to_group[n] = g
    
    steps = []
    # Which groups already have a step, so that the same group isn't compiled more than once
    emitted = set()

    for node in topological_order(root):
        if node.op in ("placeholder", "constant"):
            continue

        group = node_to_group.get(node)

        # This node isn't in any group, which means it is a matmul
        if group is None:
            steps.append(("kernel", node))
        
        if id(group) in emitted:
            continue
        emitted.add(id(group))

        # Values that need to read from memory
        inputs = group_external_inputs(group)

        # The kernel is given a unique name based on the group's output node id
        name = f"fused_{group[-1].id}"
        
        # Generating C code for this chain of operations
        c_src = generate_c(group, inputs, name)
        fn = compile_and_load(c_src, name, len(inputs))

        steps.append(("fused", group, inputs, fn))

def run_compiled(root, steps, feeds):
    """
    Execute a compiled plan on real data.

    The answers are identical to the baseline runner's, but we get there with
    fewer passes over memory, fewer temporary buffers, and fewer crossings from
    Python into C.
    """

    # `values` remembers the Tensor each node produced, so later steps can look up the results of earlier ones.
    values = {}

    # Placeholders are not computed, their data comes from the caller.
    for node in topological_order(root):
        if node.op == "placeholder":
            values[node] = feeds[node.name]

    for step in steps:

        if step[0] == "kernel":
            # An unfused operation, run with a fixed library kernel
            node = step[1]

            if node.op == "matmul":
                # Matrix multiplication.
                a = values[node.inputs[0]] # left matrix,  shape (m x k)
                b = values[node.inputs[1]] # right matrix, shape (k x n)
                m, k = a.shape
                _, n = b.shape
                out = _empty((m, n)) # result is (m x n)
                _lib.matmul(_addr(a.data), _addr(b.data), _addr(out.data), m, k, n)
                values[node] = out

            elif node.op in ("add", "sub", "mul"):
                # An element-wise binary op that did not end up in a group
                a = values[node.inputs[0]]
                b = values[node.inputs[1]]
                out = _empty(a.shape)

                getattr(_lib, node.op)(_addr(a.data), _addr(b.data), _addr(out.data), a.size)
                values[node] = out

            elif node.op == "relu":
                x = values[node.inputs[0]]
                out = _empty(x.shape)
                _lib.relu(_addr(x.data), _addr(out.data), x.size)
                values[node] = out

        else:
            # A fused group, run with the kernel generated for it.
            _, group, inputs, fn = step

            # Collect the input Tensors in exactly the order the generated function expects its parameters
            in_tensors = [values[n] for n in inputs]

            # Allocate the output buffer
            out = _empty(in_tensors[0].shape)

            # Build the argument list for the C call:
            #  the address of each input array,
            #  then the address of the output array,
            #  then how many elements to process.
            args = ([_addr(t.data) for t in in_tensors] + [_addr(out.data), in_tensors[0].size])

            # The call into C computes the entire chain of operations.
            fn(*args)

            values[group[-1]] = out

    return values[root]