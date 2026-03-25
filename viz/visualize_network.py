import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import modularity


def degree_histogram(H, bins, save_to):
    deg = [d for _, d in H.degree()]
    dia = nx.diameter(H)
    nodes = H.number_of_nodes()
    edges = H.number_of_edges()

    plt.hist(deg, bins=bins)
    plt.title(f"Diameter={dia}, nodes={nodes}, edges={edges}")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_to)
    plt.close()


def visualize_communities(H, seed, save_to):
    coms = nx.community.louvain_communities(H)
    Q = modularity(H, coms)
    pos = nx.spring_layout(H, seed=seed)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(coms):
        nx.draw_networkx_nodes(H, pos, node_size=40, nodelist=list(c), node_color=[cmap(i)])

    nx.draw_networkx_edges(H, pos, alpha=0.3)
    plt.title(f"Modularity Q={Q:.3f}, nodes={H.number_of_nodes()}, edges={H.number_of_edges()}")
    plt.savefig(save_to)
    plt.close()
    return coms


def visualize_network_timeslice(H, t, coms, seed, save_to):
    pos = nx.spring_layout(H, seed=seed)
    x_vals, y_vals = zip(*pos.values())
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)
    xlim = (min(x_vals) - 0.03 * x_range, max(x_vals) + 0.03 * x_range)
    ylim = (min(y_vals) - 0.03 * y_range, max(y_vals) + 0.03 * y_range)

    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(coms):
        nx.draw_networkx_nodes(H, pos, node_size=40, nodelist=list(c), node_color=[cmap(i)])

    for off, transp in zip([-2, -1, 0, 1, 2], [0.05, 0.1, 0.2, 0.1, 0.05]):
        edges_t = [(u,v) for u,v,d in H.edges(data=True) if (t+off) in d.get("times",[])]
        Gt = nx.Graph()
        Gt.add_nodes_from(sorted(H.nodes()))
        Gt.add_edges_from(edges_t)
        nx.draw_networkx_edges(Gt, pos, alpha=transp)

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(save_to)
    plt.close()


def visualize_small_network(H, seed, save_to, old_node_label=None, edge_label=None):
    """Visualize small network with node id's, old node id's, and potentially edge time stamps."""
    if old_node_label is None:
        labels = {n: str(n) for n in sorted(H.nodes())}
    else: # add old node ids as labels in exponent
        labels = {n: f"{n}$^{{{H.nodes[n][old_node_label]}}}$" for n in sorted(H.nodes())}

    edge_labels = {}
    if edge_label is not None:
        edge_labels = {(u, v): 't=' + ','.join(map(str, H.edges[u, v].get(edge_label, [])))
                       for u, v in H.edges()}

    pos = nx.spring_layout(H, seed=seed)
    plt.figure(figsize=(8, 5))
    nx.draw(H, pos, nodelist=sorted(H.nodes()), labels=labels, with_labels=True, node_color='skyblue', arrows=True)
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels)
    plt.savefig(save_to, dpi=200)
    plt.close()

