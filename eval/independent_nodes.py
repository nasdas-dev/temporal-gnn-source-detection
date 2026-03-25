import random

def independent_nodes(H, depth=2):
    """Find a set of independent nodes in the graph H such that no two nodes are within 'depth' hops of each other."""
    n = H.number_of_nodes()
    nodes = set(range(n))
    result = []

    adj = [set(H.neighbors(i)) for i in range(n)]
    while nodes:
        node = random.choice(list(nodes))
        result.append(node)

        disable = {node}
        frontier = {node}
        for _ in range(depth):
            next_frontier = set()
            for u in frontier:
                next_frontier |= adj[u]
            frontier = next_frontier
            disable |= frontier

        nodes -= disable

    return result