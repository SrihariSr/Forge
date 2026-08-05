_counter = 0 # global ID counter for readable names

def _next_id() -> int:
    global _counter
    i = _counter
    _counter += 1
    return i

def _wrap(x):
    if isinstance(x, Node):
        return x
    return Node("constant", [], name=f"constant({x})", value=x)

class Node:
    """
    One node in a graph. It is exactly one of the following:
        - A placeholder
        - A constant
        - An operation
    
    Fields:
        `op` : A string naming the operation.
        `inputs` : The dependant nodes consumed.
        `name` : Readable label
        `value` : The value of a constant (only used by constant nodes)
    """
    def __init__(self, op, inputs=None, name=None, value=None):
        self.op = op
        self.inputs = inputs if inputs is not None else []
        self.id = _next_id()
        self.name = name if name is not None else f"n{self.id}"
        self.value = value
    
    def __add__(self, other):
        return Node("add", [self, _wrap(other)])
    
    def __sub__(self, other):
        return Node("sub", [self, _wrap(other)])
    
    def __mul__(self, other):
        return Node("mul", [self, _wrap(other)])

    def exp(self):
        """
        Element-wise e to the power of the node's value 
        """
        return Node("exp", [self])

    def __repr__(self):
        ins = ", ".join(inp.name for inp in self.inputs)
        return f"{self.name} = {self.op}({ins})"
    
def placeholder(name) -> Node:
    """
    A slot for data provided at runtime.
    """
    return Node("placeholder", [], name=name)

def constant(value) -> Node:
    """
    A fixed value provided to the node.
    """
    return Node("constant", [], name=f"constant({value})", value=value)

def relu(x) -> Node:
    """
    relu(x) = max(0, x) applied elementwise.
    """
    return Node("relu", [_wrap(x)])

def matmul(a, b) -> Node:
    """
    Matrix multiplication of `a` and `b` (`a` @ `b`).
    """
    return Node("matmul", [_wrap(a), _wrap(b)])

def topological_order(root):
    """
    Recursive Depth-first search of the topologically sorted graph.
    """
    order = []
    visited = set()

    def visit(node):
        if node in visited:
            return
        visited.add(node)
        for inp in node.inputs:
            visit(inp)
        order.append(node)
    
    visit(root)
    return order

def print_graph(root):
    """
    Pretty-print the whole graph in execution order.
    """
    print("computation graph (in execution order):")
    for node in topological_order(root):
        if node.op == "placeholder":
            print(f"  {node.name:<14} = placeholder")
        elif node.op == "constant":
            print(f"  {node.name:<14} = constant {node.value}")
        else:
            ins = ", ".join(inp.name for inp in node.inputs)
            print(f"{node.name:<14} = {node.op}({ins})")
    print(f"output is {root.name}")