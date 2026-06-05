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


def _check_limit(name: str, value: int, limit: int | None, *, order: int) -> None:
    if limit is not None and value > limit:
        raise RuntimeError(
            f"De Bruijn order k={order} exceeded {name} limit: "
            f"{value:,} > {limit:,}. Reduce dbgnn.delta, increase "
            "dbgnn.time_bin_size, sample the network, or raise the limit."
        )


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
    max_temporal_states: int | None = None,
    max_db_nodes: int | None = None,
    max_db_edges: int | None = None,
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
    max_temporal_states, max_db_nodes, max_db_edges : int or None
        Optional safety limits. They do not change the graph; they fail early
        with an actionable error before an impractically large higher-order
        graph consumes all RAM.

    Returns
    -------
    nx.DiGraph
        Nodes are ordered tuples of length ``order``. Edges carry ``'weight'``
        attribute = number of causal walk completions through that edge.
    """
    node_list, edge_triples, stats = make_de_bruijn_graph_compact(
        H_array,
        delta=delta,
        directed=directed,
        order=order,
        max_temporal_states=max_temporal_states,
        max_db_nodes=max_db_nodes,
        max_db_edges=max_db_edges,
    )
    B = nx.DiGraph()
    B.add_nodes_from(node_list)
    for src, dst, weight in edge_triples:
        B.add_edge(node_list[src], node_list[dst], weight=weight)
    B.graph["stats"] = stats
    return B


def make_de_bruijn_graph_compact(
    H_array,
    delta: int | None = None,
    directed: bool = False,
    order: int = 2,
    max_temporal_states: int | None = None,
    max_db_nodes: int | None = None,
    max_db_edges: int | None = None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, int, int]], dict[str, int | bool | None]]:
    """Build compact arrays for the k-th order De Bruijn graph.

    This is equivalent to :func:`make_de_bruijn_graph`, but avoids storing one
    Python object per temporal realization. Identical ``(walk, last_time)``
    states are grouped with a count; extending a grouped state multiplies the
    downstream edge weight by that count.
    """
    if order < 2:
        raise ValueError(f"De Bruijn order must be >= 2, got {order}")

    contacts = _expanded_contacts(H_array, directed=directed)
    stats: dict[str, int | bool | None] = {
        "order": order,
        "delta": delta,
        "directed": directed,
        "expanded_contacts": len(contacts),
        "temporal_state_count": 0,
        "n_db_nodes": 0,
        "db_edge_count": 0,
    }
    if not contacts:
        return [], [], stats

    outgoing: dict[int, list[tuple[int, int]]] = defaultdict(list)
    outgoing_times: dict[int, list[int]] = defaultdict(list)
    for u, v, t in contacts:
        outgoing[u].append((t, v))
        outgoing_times[u].append(t)

    states: dict[tuple[tuple[int, ...], int], int] = defaultdict(int)
    for u, v, t in contacts:
        states[((u, v), t)] += 1
    _check_limit("temporal-state", len(states), max_temporal_states, order=order)

    # Nodes of G^(k) are temporal realizations of k-node causal walks, grouped
    # by both their node sequence and final contact time.
    for _ in range(2, order):
        next_states: dict[tuple[tuple[int, ...], int], int] = defaultdict(int)
        for (seq, last_t), count in states.items():
            current = seq[-1]
            times = outgoing_times.get(current)
            if not times:
                continue

            lo = bisect_right(times, last_t)
            hi = len(times) if delta is None else bisect_right(times, last_t + delta)
            for t_next, next_node in outgoing[current][lo:hi]:
                next_states[(seq + (next_node,), t_next)] += count

        states = next_states
        _check_limit("temporal-state", len(states), max_temporal_states, order=order)
        if not states:
            return [], [], stats

    node_list: list[tuple[int, ...]] = []
    node_to_idx: dict[tuple[int, ...], int] = {}
    for seq, _ in states:
        if seq not in node_to_idx:
            node_to_idx[seq] = len(node_list)
            node_list.append(seq)
            _check_limit("DB-node", len(node_list), max_db_nodes, order=order)

    edge_weights: dict[tuple[int, int], int] = defaultdict(int)
    for (seq, last_t), count in states.items():
        current = seq[-1]
        times = outgoing_times.get(current)
        if not times:
            continue

        lo = bisect_right(times, last_t)
        hi = len(times) if delta is None else bisect_right(times, last_t + delta)
        src_idx = node_to_idx[seq]
        for t_next, next_node in outgoing[current][lo:hi]:
            dst_seq = seq[1:] + (next_node,)
            dst_idx = node_to_idx.get(dst_seq)
            if dst_idx is None:
                dst_idx = len(node_list)
                node_to_idx[dst_seq] = dst_idx
                node_list.append(dst_seq)
                _check_limit("DB-node", len(node_list), max_db_nodes, order=order)
            edge_weights[(src_idx, dst_idx)] += count
            _check_limit("DB-edge", len(edge_weights), max_db_edges, order=order)

    edge_triples = [(src, dst, weight) for (src, dst), weight in edge_weights.items()]
    stats.update({
        "temporal_state_count": len(states),
        "n_db_nodes": len(node_list),
        "db_edge_count": len(edge_triples),
    })
    return node_list, edge_triples, stats


def plot_de_bruijn(G: nx.DiGraph, path: str, plot_labels: bool = True) -> None:
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 5))
    nx.draw(G, pos, with_labels=plot_labels, node_size=500, arrowsize=20)
    plt.savefig(path, dpi=200)
    plt.close()
