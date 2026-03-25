import networkx as nx
import numpy as np
import time

def load_network(cfg, label_attribute="old_id"):
    if cfg.nwk.t_max < cfg.sir.end_t:
        raise ValueError("t_max of network is smaller than end_t of SIR, which would lead to wrong results")

    if cfg.nwk.type == "empirical":
        if cfg.nwk.time_steps is not None and cfg.nwk.t_max > cfg.nwk.time_steps:
            raise ValueError("t_max of network is larger than time_steps, the actual maximum time of the network")
        H = read_networkx('nwk/' + cfg.nwk.name + '.csv', t_max=cfg.nwk.t_max, directed=cfg.nwk.directed,
                          label_attribute = label_attribute)
    elif cfg.nwk.type == "synthetic":
        H = generate_synthetic_graph(cfg)
    else:
        raise ValueError(f"Unknown network type: {cfg.nwk.type}")

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
    largest = max(nx.connected_components(G), key=len)
    if len(largest) < G.number_of_nodes():
        print(f" --- Removed {G.number_of_nodes() - len(largest)} disconnected nodes.")
    G = G.subgraph(largest).copy()

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