import networkx as nx
import matplotlib.pyplot as plt

def make_de_bruijn_graph(H_array, start_t, end_t, time_reverse=False, directed=False):
    # TODO: not quite correct, no need to use t as part of the node, see paper
    B = nx.DiGraph()
    nodes = set(H_array[:, 0]).union(H_array[:, 1])
    nodes = [int(n) for n in nodes]
    previous_contacts = {n: [] for n in nodes}

    for n in nodes:
        B.add_node((n, n, end_t))
    for n in nodes:
        B.add_node((n, n, start_t))
    for n in nodes:
        B.add_edge((n, n, start_t), (n, n, end_t), directed=True, diff=end_t - start_t)

    for u, v, t in H_array:
        u = int(u); v = int(v); t = int(t)
        for x, y, t1 in previous_contacts[u]:
            B.add_edge((x, y, t1), (u, v, t), directed=True, diff=t-t1)
        if not directed:
            for x, y, t1 in previous_contacts[v]:
                B.add_edge((x, y, t1), (v, u, t), directed=True, diff=t-t1)
        previous_contacts[v].append((u, v, t))
        B.add_edge((u, u, start_t), (u, v, t), directed=True, diff=t-start_t)
        B.add_edge((u, v, t), (v, v, end_t), directed=True, diff=end_t-t)
        if not directed:
            previous_contacts[u].append((v, u, t))
            B.add_edge((v, v, start_t), (v, u, t), directed=True, diff=t-start_t)
            B.add_edge((v, u, t), (u, u, end_t), directed=True, diff=end_t-t)

    if time_reverse:
        B = nx.reverse(B)
    return B


def plot_de_bruijn(G, path, plot_labels):
    base_pos = nx.spring_layout(G)
    pos = {n: (n[2], base_pos[n][1]) for n in G.nodes()}
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=plot_labels, node_size=500, arrowsize=20)
    plt.savefig(path, dpi=200)
    plt.close()