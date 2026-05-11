"""
Second-order De Bruijn graph from a temporal contact network.

Following Qarkaxhija, Perri, Scholtes.
"De Bruijn goes Neural: Causality-Aware GNNs for Time Series Data on Dynamic Graphs."
arXiv:2209.08311v1, 2022.

Nodes V^(2): unique directed node-pairs (u, v) observed in any contact event.
Edges E^(2): ((u,v), (v,w)) exists if there are times t1 < t2 with t2-t1 <= delta
             such that (u,v;t1) and (v,w;t2) are contacts.
Edge weight w((u,v),(v,w)): count of such causal walk completions.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict

import networkx as nx


def make_de_bruijn_graph(
    H_array,
    delta: int | None = None,
    directed: bool = False,
) -> nx.DiGraph:
    """Build G^(2): the second-order De Bruijn graph of causal walks.

    Parameters
    ----------
    H_array : array-like of shape (n_contacts, 3)
        Each row (u, v, t). Sorting is done internally.
    delta : int or None
        Maximum allowed time gap between consecutive contacts in a causal walk.
        ``None`` means no constraint (default).
    directed : bool
        If ``False``, each undirected contact (u,v;t) also implies (v,u;t).

    Returns
    -------
    nx.DiGraph
        Nodes are ``(u, v)`` ordered pairs. Edges carry ``'weight'`` attribute
        = number of causal walk completions through that edge.
    """
    B = nx.DiGraph()

    # Sort contacts by time
    contacts: list[tuple[int, int, int]] = sorted(
        (int(u), int(v), int(t)) for u, v, t in H_array
    )
    if not directed:
        extra = [(v, u, t) for u, v, t in contacts if u != v]
        contacts = sorted(contacts + extra, key=lambda x: x[2])

    # Register all unique (u, v) pairs as De Bruijn nodes
    for u, v, _ in contacts:
        B.add_node((u, v))

    # Index: for each arriving node v, list of (t1, db_src_node (u, v)) contacts.
    # Lists remain sorted because ``contacts`` is sorted by time.
    arrivals: dict[int, list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    arrival_times: dict[int, list[int]] = defaultdict(list)
    for u, v, t in contacts:
        arrivals[v].append((t, (u, v)))
        arrival_times[v].append(t)

    # Build causal edges: for each (v, w, t2), link from all (u, v, t1) with t1 < t2
    edge_weights: dict[tuple[tuple[int, int], tuple[int, int]], int] = defaultdict(int)
    for v, w, t2 in contacts:
        db_dst = (v, w)
        times = arrival_times[v]
        hi = bisect_left(times, t2)  # only strict t1 < t2
        lo = 0 if delta is None else bisect_left(times, t2 - delta)
        for _, db_src in arrivals[v][lo:hi]:
            edge_weights[(db_src, db_dst)] += 1

    for (db_src, db_dst), weight in edge_weights.items():
        B.add_edge(db_src, db_dst, weight=weight)

    return B


def plot_de_bruijn(G: nx.DiGraph, path: str, plot_labels: bool = True) -> None:
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=plot_labels, node_size=500, arrowsize=20)
    plt.savefig(path, dpi=200)
    plt.close()
