"""
k-th order De Bruijn graph from a temporal contact network.

Following Qarkaxhija, Perri, Scholtes.
"De Bruijn goes Neural: Causality-Aware GNNs for Time Series Data on Dynamic Graphs."
arXiv:2209.08311v1, 2022.

Nodes V^(k): unique directed node sequences (v0, ..., v{k-1}) that occur as
causal walks of length k - 1. Edges E^(k) connect overlapping sequences when
their concatenation is a causal walk of length k. Edge weights count temporal
realizations of those causal walk completions.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable

import networkx as nx


def _expanded_contacts(
    H_array: Iterable,
    directed: bool,
) -> list[tuple[int, int, int]]:
    contacts: list[tuple[int, int, int]] = sorted(
        ((int(u), int(v), int(t)) for u, v, t in H_array),
        key=lambda x: x[2],
    )
    if not directed:
        contacts.extend([(v, u, t) for u, v, t in contacts if u != v])
        contacts.sort(key=lambda x: x[2])
    return contacts


def _extend_causal_walks(
    walks: list[tuple[tuple[int, ...], int]],
    outgoing: dict[int, list[tuple[int, int]]],
    outgoing_times: dict[int, list[int]],
    delta: int | None,
) -> list[tuple[tuple[int, ...], int]]:
    """Extend temporal realizations of causal walks by one contact."""
    extended: list[tuple[tuple[int, ...], int]] = []
    for seq, last_t in walks:
        current = seq[-1]
        times = outgoing_times.get(current)
        if not times:
            continue

        lo = bisect_right(times, last_t)  # strict chronological order
        hi = len(times) if delta is None else bisect_right(times, last_t + delta)
        for t_next, next_node in outgoing[current][lo:hi]:
            extended.append((seq + (next_node,), t_next))
    return extended


def make_de_bruijn_graph(
    H_array,
    delta: int | None = None,
    directed: bool = False,
    order: int = 2,
) -> nx.DiGraph:
    """Build G^(k): the k-th order De Bruijn graph of causal walks.

    Parameters
    ----------
    H_array : array-like of shape (n_contacts, 3)
        Each row (u, v, t). Sorting is done internally.
    delta : int or None
        Maximum allowed time gap between consecutive contacts in a causal walk.
        ``None`` means no constraint (default).
    directed : bool
        If ``False``, each undirected contact (u,v;t) also implies (v,u;t).
    order : int
        De Bruijn order k. Must be at least 2. Nodes are causal walks with
        ``order`` original nodes, i.e. ``order - 1`` temporal contacts.

    Returns
    -------
    nx.DiGraph
        Nodes are ordered tuples of length ``order``. Edges carry ``'weight'``
        attribute = number of causal walk completions through that edge.
    """
    if order < 2:
        raise ValueError(f"De Bruijn order must be >= 2, got {order}")

    B = nx.DiGraph()

    contacts = _expanded_contacts(H_array, directed=directed)
    if not contacts:
        return B

    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing_times: dict[int, list[int]] = defaultdict(list)
    for u, v, t in contacts:
        outgoing[u].append((t, v))
        outgoing_times[u].append(t)

    # Temporal realizations of length-one causal walks (single contacts).
    walks: list[tuple[tuple[int, ...], int]] = [
        ((u, v), t) for u, v, t in contacts
    ]

    # Nodes of G^(k) are causal walks with k - 1 contacts.
    for _ in range(2, order):
        walks = _extend_causal_walks(walks, outgoing, outgoing_times, delta)
        if not walks:
            return B

    for seq, _ in walks:
        B.add_node(seq)

    # Edges of G^(k) are causal walks with k contacts, projected to overlapping
    # source/destination k-node sequences.
    edge_walks = _extend_causal_walks(walks, outgoing, outgoing_times, delta)
    edge_weights: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = defaultdict(int)
    for seq, _ in edge_walks:
        db_src = seq[:-1]
        db_dst = seq[1:]
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
