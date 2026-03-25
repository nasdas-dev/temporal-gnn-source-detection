import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def make_rozenshtein_graph(H_array, start_t, end_t, time_reverse=False, directed=False):
    G = nx.DiGraph()
    nodes = set(H_array[:, 0]).union(H_array[:, 1])
    times = defaultdict(set) # to keep track of times for each node

    # add start and end nodes for each node
    for n in nodes:
        G.add_node((int(n), end_t))
        times[n].add(int(end_t))
    for n in nodes:
        G.add_node((int(n), start_t))
        times[n].add(int(start_t))

    # add contact nodes and edges
    for u, v, t in H_array:
        G.add_node((int(u), int(t)))
        G.add_node((int(v), int(t)))
        G.add_edge((int(u), int(t)), (int(v), int(t)))
        if not directed:
            G.add_edge((int(v), int(t)), (int(u), int(t)))
        times[u].add(int(t))
        times[v].add(int(t))

    # connect each node down the timeline
    for n, ts in times.items():
        ts = sorted(ts)
        for i in range(len(ts) - 1):
            if time_reverse:
                G.add_edge((int(n), ts[i + 1]), (int(n), ts[i]))
            else:
                G.add_edge((int(n), ts[i]), (int(n), ts[i + 1]))

    return G


def plot_rozenshtein(G, path, plot_labels):
    pos = {(n, t): (t, -n) for n, t in G.nodes()}  # x=time, y=node
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=plot_labels, node_size=500, arrowsize=20)
    plt.savefig(path, dpi=200)
    plt.close()