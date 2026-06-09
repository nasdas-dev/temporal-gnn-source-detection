import time

import networkx as nx
import numpy as np

from setup.reduction import apply_network_reduction, largest_component


def load_network(cfg, label_attribute="old_id"):
    """Load a temporal network and apply optional scientific reductions."""
    if cfg.nwk.t_max < cfg.sir.end_t:
        raise ValueError("t_max of network is smaller than end_t of SIR, which would lead to wrong results")

    if cfg.nwk.type == "empirical":
        time_steps = getattr(cfg.nwk, "time_steps", None)
        if time_steps is not None and cfg.nwk.t_max > time_steps:
            raise ValueError("t_max of network is larger than time_steps, the actual maximum time of the network")
        H = read_networkx(
            "nwk/" + cfg.nwk.name + ".csv",
            t_max=cfg.nwk.t_max,
            directed=cfg.nwk.directed,
            label_attribute=label_attribute,
        )
    elif cfg.nwk.type == "synthetic":
        H = generate_synthetic_graph(cfg)
    else:
        raise ValueError(f"Unknown network type: {cfg.nwk.type}")

    H, reduction_report = apply_network_reduction(H, cfg.nwk)
    if reduction_report:
        reduced_t_max = int(reduction_report["reduced"]["t_max"])
        cfg.nwk.t_max = reduced_t_max
        cfg.sir.start_t = 0
        cfg.sir.end_t = min(int(cfg.sir.end_t), reduced_t_max)
        if reduction_report.get("time", {}).get("reindex_to_zero", False):
            cfg.sir.end_t = reduced_t_max
        print(
            " --- Applied reduction "
            f"{reduction_report['reduction_id']}: "
            f"{reduction_report['original']['nodes']} nodes/"
            f"{reduction_report['original']['edges']} edges/"
            f"{reduction_report['original']['contacts']} contacts -> "
            f"{reduction_report['reduced']['nodes']} nodes/"
            f"{reduction_report['reduced']['edges']} edges/"
            f"{reduction_report['reduced']['contacts']} contacts"
        )

    print(f" --- The reduced network has {len(list(H.nodes()))} nodes and {len(list(H.edges()))} edges")
    return H


def generate_synthetic_graph(cfg):
    if cfg.nwk.name == "erdos_renyi":
        G = nx.erdos_renyi_graph(cfg.nwk.n, cfg.nwk.p, seed=cfg.nwk.seed, directed=cfg.nwk.directed)
    elif cfg.nwk.name == "barabasi_albert":
        if cfg.nwk.directed == True:
            raise ValueError("Barabasi-Albert model does not support directed graphs.")
        G = nx.barabasi_albert_graph(cfg.nwk.n, cfg.nwk.m, seed=cfg.nwk.seed)
    else:
        raise ValueError(f"Unknown synthetic network model: {cfg.nwk.name}")

    for u, v in G.edges():
        G[u][v]["times"] = list(range(cfg.nwk.t_max + 1))
    return G


def read_networkx(fname, t_max, directed=False, label_attribute=None):
    """Read a temporal network CSV into a NetworkX graph.

    Each edge receives a ``times`` list with contacts at or before ``t_max``.
    Self-contacts and duplicate edge-time contacts are ignored.  The largest
    connected component is kept and node labels are converted to integers.
    """
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    start = time.time()
    print("Reading network from", fname, "but stop after", t_max, "time steps...", end=" ")

    nr_self_contacts = 0
    nr_multiple_events = 0
    try:
        with open(fname) as f:
            for line in f:
                fields = line.strip().split()
                if len(fields) == 3 and int(fields[2]) <= t_max:
                    u = int(fields[0])
                    v = int(fields[1])
                    if u != v:
                        t = int(fields[2])
                        if G.has_edge(u, v):
                            if t not in G.edges[u, v]["times"]:
                                G.edges[u, v]["times"].append(t)
                            else:
                                nr_multiple_events += 1
                        else:
                            G.add_edge(u, v)
                            G.edges[u, v]["times"] = [t]
                    else:
                        nr_self_contacts += 1
    except Exception:
        print("Error in reading network", fname)
        raise

    print(f"Done in {time.time() - start:.2f} seconds")
    if nr_self_contacts > 0 or nr_multiple_events > 0:
        print(" --- Ignored", nr_self_contacts, "self-contacts and", nr_multiple_events, "multiple events.")

    G, removed = largest_component(G)
    if removed:
        print(f" --- Removed {removed} disconnected nodes.")

    return nx.convert_node_labels_to_integers(G, label_attribute=label_attribute, ordering="sorted")


def make_array_from_networkx(H):
    """From a NetworkX temporal graph, return sorted rows ``(u, v, t)``."""
    rows = []
    for u, v, data in H.edges(data=True):
        rows.extend([[u, v, t] for t in data["times"]])

    arr = np.array(rows)
    if len(arr) == 0:
        return np.zeros((0, 3), dtype=np.int64)
    arr = arr[arr[:, 2].argsort()]
    return arr

