import ctypes
import os
import subprocess
import tempfile

C_EXP = {
    "add": "({} + {})",
    "sub": "({} - {})",
    "mul": "({} * {})",
    "relu": "{0} > 0.0f ? {0} : 0.0f"
}

def generate_c(group, group_inputs, func_name="fused_kernel") -> list:
    """
    Build the C source for one fused group.
      `group`        : nodes in the group, in execution order.
      `group_inputs` : nodes feeding the group from OUTSIDE (become parameters).
    
    Every intermediate variable is a local variable so lives only in the register, not main memory
    which leads to a speedup.
    """
    param_of = {node: f"in {i}" for i, node in enumerate(group_inputs)}

    params = ", ".join(f"const float* {param_of[n]}" for n in group_inputs)

    # Building the C function
    lines = []
    lines.append(f"void {func_name}({params}, float* out, int n){{")
    lines.append("for (int i = 0; i < n; i++) {")

    value_of = {node: f"{param_of[node]}[i]" for node in group_inputs}

    last = group[-1]

    for node in group:
        arg_names = [value_of[inp] for inp in node.inputs]
        
        if node.op == "relu":
            expr = C_EXP["relu"].format(arg_names[0])
        else:
            expr = C_EXP[node.op].format(*arg_names)
        
        if node is last:
            lines.append(f"out[i] = {expr};")
        else:
            var = f"t{node.id}"
            value_of[node] = var
            lines.append(f"float {var} = {expr};")
    
    lines.append("}")
    lines.append("}")

    return "\n".join(lines)

def compile_and_load(c_source, func_name, num_inputs):
    """
    Takes a generated C source code, compiles it into a shared library,
    load that library, and return a Python-callable handle to the function inside it. 
    """
    tmpdir = tempfile.mkdtemp(prefix="dlc_")
    c_path = os.path.join(tmpdir, f"{func_name}.c")
    so_path = os.path.join(tmpdir, f"{func_name}.so")

    with open(c_path, "w") as f:
        f.write(c_source)
    
    # Runs the command as it would b run in the terminal. 
    subprocess.run(["gcc", "-O2", "-shared", "-fPIC", c_path, "-o", so_path], check=True, capture_output=True)

    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, func_name)

    FP = ctypes.POINTER(ctypes.c_float)

    fn.argtypes = [FP]*num_inputs + [FP, ctypes.c_int]

    fn.restype = None

    return fn

def group_external_inputs(group):
    """
    Works out which values a fused group needs to read from memory.
    """
    members = set(group)

    inputs = []
    seen = set()

    for node in group:
        for inp in node.inputs:
            if inp not in members and inp not in seen:
                seen.add(inp)
                input.append(inp)
    
    return inputs
    

