from graph import topological_order

ELEMENTWISE_OPS = {"add", "sub", "mul", "relu", "exp"}

def count_consumers(root) -> dict:
    """
    Counts how many nodes use each node as an input.
    """
    order = topological_order(root)
    consumers = {n: 0 for n in order}
    for n in order:
        for inp in n.inputs:
            consumers[inp] += 1
    consumers[root] += 1

    return consumers

def fuse(root) -> list:
    """
    Partition element-wise operations into fusion groups. Each group is a list of nodes
    in execution order that becomes one fused kernel.
    """
    order = topological_order(root)
    pos = {n: i for i, n in enumerate(order)} # node: topological index
    consumers = count_consumers(root)
    assigned = set()
    groups = []

    for node in reversed(order):
        if node.op not in ELEMENTWISE_OPS or node in assigned:
            continue

        members = {node}
        frontier = [node]
        while frontier:
            current = frontier.pop()
            for inp in current.inputs:
                if (inp.op in ELEMENTWISE_OPS and inp not in members and consumers[inp] == 1):
                    members.add(inp)
                    frontier.append(inp)
        
        group = sorted(members, key=lambda n: pos[n])
        assigned |= members
        groups.append(group)
    
    groups.sort(key=lambda g: pos[g[-1]])
    return groups

def print_groups(root, groups):
    """
    Shows the fusion plan: each group (a future kernel) plus any unfused ops.
    """
    in_group = set()
    for g in groups:
        in_group.update(g)
    print("fusion plan (each group becomes one kernel):")
    for i, g in enumerate(groups):
        body = "- ".join(n.op for n in g)
        print(f"group {i}: [{body}] output = {g[-1].name}")
    unfused = [n for n in topological_order(root) if n not in in_group and n.op not in ("placeholder", "const")]
    if unfused:
        print("seperate kernel each:", ", ".join(f"{n.op}({n.name})" for n in unfused))
