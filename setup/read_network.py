import networkx as nx
import numpy as np
import time


def _cfg_get(cfg, key, default=None):
    """Read a key from either a dict-like object or Config object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _largest_component(G):
    if G.number_of_nodes() == 0:
        return G.copy(), 0
    components = nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G)
    largest = max(components, key=len)
    removed = G.number_of_nodes() - len(largest)
    return G.subgraph(largest).copy(), removed


def _incident_activity(H):
    activity = {n: 0 for n in H.nodes()}
    for u, v, data in H.edges(data=True):
        count = len(data.get("times", []))
        activity[u] += count
        activity[v] += count
    return activity


def _temporal_neighbors(H, node):
    if H.is_directed():
        return set(H.successors(node)) | set(H.predecessors(node))
    return set(H.neighbors(node))


def _contact_count_between(H, u, v):
    count = 0
    if H.has_edge(u, v):
        count += len(H.edges[u, v].get("times", []))
    if H.is_directed() and H.has_edge(v, u):
        count += len(H.edges[v, u].get("times", []))
    return count


def _static_edges_to_selected(H, node, selected):
    count = 0
    for other in selected:
        if H.is_directed():
            if H.has_edge(node, other):
                count += 1
            if H.has_edge(other, node):
                count += 1
        elif H.has_edge(node, other):
            count += 1
    return count


def _activity_snowball_sample(H, sample_cfg):
    """Connected, activity-biased node sampling under a simple cost budget.

    The sampler starts from the most temporally active node and grows through
    the current frontier. At each step it picks the feasible frontier node with
    the largest contact activity into the selected set, while respecting an
    optional ``max_node_edge_cost = n_selected * m_selected`` budget.
    """
    n_original = H.number_of_nodes()
    if n_original == 0:
        return H.copy()

    max_cost = _cfg_get(sample_cfg, "max_node_edge_cost", None)
    target_nodes = _cfg_get(sample_cfg, "target_nodes", None)
    min_nodes = int(_cfg_get(sample_cfg, "min_nodes", 5))
    seed = int(_cfg_get(sample_cfg, "seed", 42))
    rng = np.random.default_rng(seed)

    activity = _incident_activity(H)
    tie_break = {n: float(rng.random()) for n in H.nodes()}
    start = max(H.nodes(), key=lambda n: (activity[n], H.degree(n), tie_break[n]))

    selected = {start}
    frontier = _temporal_neighbors(H, start) - selected
    selected_edge_count = 0

    def reached_target():
        if target_nodes is not None and len(selected) >= int(target_nodes):
            return True
        if target_nodes is None and max_cost is None:
            return True
        return False

    while not reached_target():
        pool = frontier
        if not pool:
            # The input should already be connected, but fall back gracefully.
            pool = set(H.nodes()) - selected
        if not pool:
            break

        candidates = []
        for node in pool:
            add_edges = _static_edges_to_selected(H, node, selected)
            if add_edges == 0 and selected:
                continue
            new_n = len(selected) + 1
            new_m = selected_edge_count + add_edges
            new_cost = new_n * max(new_m, 1)
            within_cost = max_cost is None or new_cost <= float(max_cost)
            within_target = target_nodes is None or new_n <= int(target_nodes)
            forced_min = new_n <= min_nodes
            if (within_cost and within_target) or forced_min:
                connection_activity = sum(_contact_count_between(H, node, s) for s in selected)
                candidates.append((
                    connection_activity,
                    add_edges,
                    activity[node],
                    H.degree(node),
                    tie_break[node],
                    node,
                ))

        if not candidates:
            break

        *_, picked = max(candidates)
        added_edges = _static_edges_to_selected(H, picked, selected)
        selected.add(picked)
        selected_edge_count += added_edges
        frontier.discard(picked)
        frontier |= _temporal_neighbors(H, picked) - selected

    return H.subgraph(selected).copy()


def sample_temporal_network(H, sample_cfg):
    method = _cfg_get(sample_cfg, "method", None)
    if method in (None, "none", "None", False):
        return H
    if method != "activity_snowball":
        raise ValueError(f"Unknown network sampling method: {method}")

    original_nodes = H.number_of_nodes()
    original_edges = H.number_of_edges()
    original_contacts = sum(len(d.get("times", [])) for _, _, d in H.edges(data=True))

    sampled = _activity_snowball_sample(H, sample_cfg)
    sampled, removed = _largest_component(sampled)
    if removed:
        print(f" --- Sampling removed {removed} disconnected sampled nodes.")

    if sampled.number_of_nodes() < original_nodes:
        sampled = nx.convert_node_labels_to_integers(
            sampled,
            label_attribute="sample_id",
            ordering="sorted",
        )

    sampled_nodes = sampled.number_of_nodes()
    sampled_edges = sampled.number_of_edges()
    sampled_contacts = sum(len(d.get("times", [])) for _, _, d in sampled.edges(data=True))
    original_cost = original_nodes * max(original_edges, 1)
    sampled_cost = sampled_nodes * max(sampled_edges, 1)
    reduction = original_cost / max(sampled_cost, 1)

    sampled.graph["sample"] = {
        "method": method,
        "seed": int(_cfg_get(sample_cfg, "seed", 42)),
        "original_nodes": int(original_nodes),
        "original_edges": int(original_edges),
        "original_contacts": int(original_contacts),
        "sampled_nodes": int(sampled_nodes),
        "sampled_edges": int(sampled_edges),
        "sampled_contacts": int(sampled_contacts),
        "node_edge_cost_reduction": float(reduction),
        "max_node_edge_cost": _cfg_get(sample_cfg, "max_node_edge_cost", None),
    }

    print(
        " --- Sampled network with activity_snowball: "
        f"{original_nodes} nodes/{original_edges} edges -> "
        f"{sampled_nodes} nodes/{sampled_edges} edges "
        f"(node*edge cost {reduction:.2f}x smaller)"
    )
    return sampled


def load_network(cfg, label_attribute="old_id"):
    if cfg.nwk.t_max < cfg.sir.end_t:
        raise ValueError("t_max of network is smaller than end_t of SIR, which would lead to wrong results")

    if cfg.nwk.type == "empirical":
        time_steps = _cfg_get(cfg.nwk, "time_steps", None)
        if time_steps is not None and cfg.nwk.t_max > time_steps:
            raise ValueError("t_max of network is larger than time_steps, the actual maximum time of the network")
        H = read_networkx('nwk/' + cfg.nwk.name + '.csv', t_max=cfg.nwk.t_max, directed=cfg.nwk.directed,
                          label_attribute = label_attribute)
    elif cfg.nwk.type == "synthetic":
        H = generate_synthetic_graph(cfg)
    else:
        raise ValueError(f"Unknown network type: {cfg.nwk.type}")

    H = sample_temporal_network(H, _cfg_get(cfg.nwk, "sample", None))
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

def read_networkx(fname, t_max, directed = False, label_attribute = None):
    """Read a temporal network from a csv file into a networkx graph. Each edge gets and attribute 'times' which is a
    list of time steps when the contact happened. Self-contacts and multiple contacts are ignored.
    Node labels are converted to integers in a sorted manner starting from 0. Contacts after t_max are ignored.
    The parameter label_attribute can be used to store the original node labels as a node attribute."""
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    start = time.time()
    print('Reading network from', fname, 'but stop after', t_max, 'time steps...', end=' ')

    nr_self_contacts = 0
    nr_multiple_events = 0
    try:
        with open(fname) as f:
            for l in f:
                a = l.strip().split()
                if len(a) == 3 and (int(a[2]) <= t_max):
                    u = int(a[0])
                    v = int(a[1])
                    if u != v:
                        b = int(a[2])
                        if G.has_edge(u,v):
                            if b not in G.edges[u,v]['times']:
                                G.edges[u,v]['times'].append(b)
                            else:
                                nr_multiple_events += 1
                        else:
                            G.add_edge(u,v)
                            G.edges[u,v]['times'] = [b]
                    else:
                        nr_self_contacts += 1
    except:
        print('Error in reading network', fname)
        exit(1)

    print(f"Done in {time.time() - start:.2f} seconds")
    if nr_self_contacts > 0 or nr_multiple_events > 0:
        print(' --- Ignored', nr_self_contacts, 'self-contacts and', nr_multiple_events, 'multiple events.')

    # take largest connected component
    G, removed = _largest_component(G)
    if removed:
        print(f" --- Removed {removed} disconnected nodes.")

    # relabel nodes to integers starting from 0
    H = nx.convert_node_labels_to_integers(G, label_attribute = label_attribute, ordering='sorted')
    return H


def make_array_from_networkx(H):
    """From the networkx graph, make a numpy array with rows (u, v, t) and sorted by t."""
    rows = []
    for u, v, data in H.edges(data=True):
        rows.extend([[u, v, t] for t in data['times']])

    arr = np.array(rows)
    arr = arr[arr[:, 2].argsort()] # sort by time
    return arr
