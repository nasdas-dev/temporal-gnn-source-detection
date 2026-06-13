"""
Graph builder functions — one per model type.

Each function takes a NetworkX temporal graph ``H`` (nodes 0…N-1, edges carry
a ``'times'`` list attribute) and returns a ``dict`` of pre-computed tensors
plus ``"n_nodes": int``.  The dict is passed unchanged to the model's forward
function during training and inference.

Supported builders
------------------
build_static_graph          → StaticGNN
build_temporal_activation   → BacktrackingNetwork (Ru et al.)
build_temporal_snapshots    → TemporalGNN (time-slice SAGEConv)
build_de_bruijn_graph       → DBGNN (Qarkaxhija et al.)
build_dag_event_graph       → DAGGNN (Rey et al.)
"""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np
import torch


def normalize_gcn_edges(
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    n_nodes: int,
    add_self_loops: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GCN-style symmetric edge weights for message passing.

    The returned graph uses source-to-destination edges and weights
    ``w(u, v) / sqrt(S(u) S(v))``, where ``S(x)`` is the incoming weighted
    strength after optional self-loops have been added.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")

    device = edge_index.device
    edge_index = edge_index.to(dtype=torch.long)
    edge_weight = edge_weight.to(dtype=torch.float32, device=device)

    if add_self_loops and n_nodes > 0:
        loops = torch.arange(n_nodes, dtype=torch.long, device=device)
        loop_index = torch.stack([loops, loops], dim=0)
        loop_weight = torch.ones(n_nodes, dtype=torch.float32, device=device)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    if edge_index.numel() == 0:
        return edge_index, edge_weight

    src, dst = edge_index[0], edge_index[1]
    strength = torch.zeros(n_nodes, dtype=torch.float32, device=device)
    strength.scatter_add_(0, dst, edge_weight)
    denom = torch.sqrt(strength[src] * strength[dst]).clamp(min=1e-12)
    return edge_index, edge_weight / denom


# ---------------------------------------------------------------------------
# Unified temporal coarse-graining  (shared across all temporal models)
# ---------------------------------------------------------------------------

def coarsen_temporal_network(H: nx.Graph, delta_t: int) -> tuple[nx.Graph, dict]:
    """Coarse-grain a temporal contact network in time.

    Bins every edge's contact times into non-overlapping windows of width
    ``delta_t`` (``binned = (t - t_min) // delta_t``) and returns a *new*
    temporal graph carrying the binned, de-duplicated ``'times'`` lists.

    This is applied **once, before any model-specific graph construction**, so
    that the TemporalGNN snapshots, the BacktrackingNetwork edge textures, and
    the DBGNN De Bruijn graph are all derived from the *same* coarse-grained
    network.  ``delta_t <= 1`` returns the input graph unchanged.

    Parameters
    ----------
    H:
        Temporal NetworkX graph. Edges carry a ``'times'`` list attribute.
    delta_t:
        Temporal bin width in original time steps (``>= 1``).

    Returns
    -------
    (H_coarse, stats)
        ``H_coarse`` is a graph of the same directed/undirected type with binned
        ``'times'``.  ``stats`` records before/after sizes:
        ``delta_t``, ``t_min``, ``t_max_before``, ``t_max_after``,
        ``contacts_before``, ``contacts_after``, ``n_edges_before``,
        ``n_edges_after``.
    """
    if delta_t < 1:
        raise ValueError(f"delta_t must be >= 1, got {delta_t}")

    all_times = [int(t) for _, _, data in H.edges(data=True) for t in data.get("times", [])]
    contacts_before = len(all_times)
    n_edges_before = H.number_of_edges()
    t_min = min(all_times) if all_times else 0
    t_max_before = max(all_times) if all_times else 0

    if delta_t <= 1:
        return H, {
            "delta_t": 1,
            "t_min": int(t_min),
            "t_max_before": int(t_max_before),
            "t_max_after": int(t_max_before),
            "contacts_before": int(contacts_before),
            "contacts_after": int(contacts_before),
            "n_edges_before": int(n_edges_before),
            "n_edges_after": int(n_edges_before),
        }

    H_coarse = H.__class__()  # preserve directed/undirected type
    H_coarse.add_nodes_from(H.nodes())
    contacts_after = 0
    for u, v, data in H.edges(data=True):
        binned = sorted({(int(t) - t_min) // delta_t for t in data.get("times", [])})
        if not binned:
            continue
        H_coarse.add_edge(u, v, times=binned)
        contacts_after += len(binned)

    t_max_after = (t_max_before - t_min) // delta_t
    stats = {
        "delta_t": int(delta_t),
        "t_min": int(t_min),
        "t_max_before": int(t_max_before),
        "t_max_after": int(t_max_after),
        "contacts_before": int(contacts_before),
        "contacts_after": int(contacts_after),
        "n_edges_before": int(n_edges_before),
        "n_edges_after": int(H_coarse.number_of_edges()),
    }
    return H_coarse, stats


# ---------------------------------------------------------------------------
# Static projection  (StaticGNN)
# ---------------------------------------------------------------------------

def build_static_graph(
    H: nx.Graph,
    use_edge_weights: bool = False,
) -> dict:
    """Collapse temporal edges into a static graph.

    Every node pair that had *any* contact becomes one undirected edge.
    PyG convention is used: both (u, v) and (v, u) are included.

    Parameters
    ----------
    H:
        Temporal NetworkX graph. Edges carry a ``'times'`` list attribute.
    use_edge_weights:
        If ``True``, edge weight = number of temporal contacts between the two
        nodes, normalised to [0, 1].  If ``False``, ``edge_weight`` is ``None``.

    Returns
    -------
    dict with keys:
        ``n_nodes``     int
        ``edge_index``  LongTensor [2, 2*|E|]
        ``edge_weight`` FloatTensor [2*|E|] or ``None``
    """
    n_nodes = H.number_of_nodes()
    src_list: list[int] = []
    dst_list: list[int] = []
    w_list:   list[float] = []

    for u, v, data in H.edges(data=True):
        contact_count = float(len(data.get("times", [1])))
        # forward direction
        src_list.append(u); dst_list.append(v); w_list.append(contact_count)
        # reverse direction (undirected)
        src_list.append(v); dst_list.append(u); w_list.append(contact_count)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    edge_weight: torch.Tensor | None = None
    if use_edge_weights:
        w = torch.tensor(w_list, dtype=torch.float32)
        edge_weight = w / w.max()  # normalise to [0, 1]

    return {
        "n_nodes":     n_nodes,
        "edge_index":  edge_index,
        "edge_weight": edge_weight,
    }


# ---------------------------------------------------------------------------
# Temporal activation pattern  (BacktrackingNetwork)
# ---------------------------------------------------------------------------

def build_temporal_activation(
    H: nx.Graph,
    directed: bool | None = None,
    dense_edge_attr: bool = False,
) -> dict:
    """Build the aggregated graph + temporal activation patterns.

    For each edge (u, v) in the static projection, ``edge_attr[e, t] = 1`` iff
    the edge was active at time step *t*.  Undirected contacts are represented
    in both directions. Directed contacts are kept in their observed direction.

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.
    directed:
        If ``True``, keep observed edge directions. If ``False``, add reverse
        copies for every edge. If ``None``, infer this from ``H``.
    dense_edge_attr:
        If ``True``, also return dense binary ``edge_attr`` for compatibility.
        The default returns a sparse exact representation of the same vectors.

    Returns
    -------
    dict with keys:
        ``n_nodes``              int
        ``T``                    int   — number of time steps (t_max + 1)
        ``edge_index``           LongTensor [2, E]
        ``edge_time_index``      LongTensor [nnz] — active time per nonzero
        ``edge_time_edge_index`` LongTensor [nnz] — owning edge per nonzero

    When ``dense_edge_attr=True``, the returned dict also includes:
        ``edge_attr`` FloatTensor [E, T] — binary activation pattern
    """
    n_nodes  = H.number_of_nodes()
    all_times = [t for _, _, data in H.edges(data=True) for t in data.get("times", [])]
    if not all_times:
        raise ValueError("Graph has no temporal edge data ('times' attribute missing).")
    t_max = max(all_times)
    T = t_max + 1
    is_directed = H.is_directed() if directed is None else bool(directed)

    src_list: list[int] = []
    dst_list: list[int] = []
    edge_time_index: list[int] = []
    edge_time_edge_index: list[int] = []
    attr_list: list[torch.Tensor] | None = [] if dense_edge_attr else None

    def append_edge(src: int, dst: int, times: list[int]) -> None:
        edge_id = len(src_list)
        src_list.append(src)
        dst_list.append(dst)
        edge_time_index.extend(times)
        edge_time_edge_index.extend([edge_id] * len(times))
        if attr_list is not None:
            act = torch.zeros(T, dtype=torch.float32)
            if times:
                act[torch.tensor(times, dtype=torch.long)] = 1.0
            attr_list.append(act)

    for u, v, data in H.edges(data=True):
        times = sorted(set(int(t) for t in data.get("times", [])))
        append_edge(int(u), int(v), times)
        if not is_directed:
            append_edge(int(v), int(u), times)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    graph_data = {
        "n_nodes":              n_nodes,
        "T":                    T,
        "edge_index":           edge_index,
        "edge_time_index":      torch.tensor(edge_time_index, dtype=torch.long),
        "edge_time_edge_index": torch.tensor(edge_time_edge_index, dtype=torch.long),
        "n_edges":              edge_index.size(1),
        "directed":             is_directed,
    }
    if attr_list is not None:
        graph_data["edge_attr"] = torch.stack(attr_list, dim=0)

    return graph_data


# ---------------------------------------------------------------------------
# Time-sliced snapshots  (TemporalGNN)
# ---------------------------------------------------------------------------

def build_temporal_snapshots(
    H: nx.Graph,
    group_by_time: int = 1,
    directed: bool = False,
) -> dict:
    """Build a dict of edge_index tensors, one per time-slice.

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.
    group_by_time:
        Aggregate contacts within windows of this width into one snapshot.
        Default 1 means one snapshot per unique time step.
    directed:
        If ``False`` (default) both (u,v) and (v,u) are included per snapshot.

    Returns
    -------
    dict with keys:
        ``n_nodes``       int
        ``num_snapshots`` int
        ``edge_indeces``  dict[int, LongTensor [2, E_t]]  — keyed by slice index
    """
    n_nodes = H.number_of_nodes()

    # Collect all (u, v, t) triples
    rows: list[tuple[int, int, int]] = []
    for u, v, data in H.edges(data=True):
        for t in data.get("times", []):
            rows.append((u, v, t))
    if not rows:
        raise ValueError("Graph has no temporal edge data ('times' attribute missing).")

    arr = np.array(rows)
    t_min = int(arr[:, 2].min())
    arr[:, 2] = (arr[:, 2] - t_min) // group_by_time  # re-index slice

    # Group by slice index
    slices: dict[int, list[tuple[int, int]]] = {}
    for u, v, s in arr.tolist():
        s = int(s)
        if s not in slices:
            slices[s] = []
        slices[s].append((int(u), int(v)))
        if not directed:
            slices[s].append((int(v), int(u)))

    time_order = sorted(slices)
    edge_indeces: dict[int, torch.Tensor] = {}
    for s in time_order:
        edges = slices[s]
        edge_indeces[s] = torch.tensor(edges, dtype=torch.long).T  # [2, E_s]

    return {
        "n_nodes":       n_nodes,
        "num_snapshots": len(edge_indeces),
        "edge_indeces":  edge_indeces,
        "time_order":    time_order,
        "group_by_time": group_by_time,
    }


# ---------------------------------------------------------------------------
# De Bruijn graph  (DBGNN)
# ---------------------------------------------------------------------------

def _coarsen_contact_times(
    H_array: np.ndarray,
    time_bin_size: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Aggregate contacts into wider time bins for higher-order graphs."""
    if time_bin_size < 1:
        raise ValueError(f"time_bin_size must be >= 1, got {time_bin_size}")
    if len(H_array) == 0:
        return H_array, {
            "time_bin_size": time_bin_size,
            "time_bin_t_min": 0,
            "contacts_before_time_bin": 0,
            "contacts_after_time_bin": 0,
        }

    arr = np.asarray(H_array, dtype=np.int64).copy()
    t_min = int(arr[:, 2].min())
    if time_bin_size > 1:
        arr[:, 2] = (arr[:, 2] - t_min) // time_bin_size
        arr = np.unique(arr, axis=0)
    return arr, {
        "time_bin_size": time_bin_size,
        "time_bin_t_min": t_min,
        "contacts_before_time_bin": int(len(H_array)),
        "contacts_after_time_bin": int(len(arr)),
    }


def build_de_bruijn_graph(
    H: nx.Graph,
    directed: bool | None = None,
    delta: int | None = None,
    order: int = 2,
    time_bin_size: int = 1,
    max_temporal_states: int | None = None,
    max_db_nodes: int | None = None,
    max_db_edges: int | None = None,
) -> dict:
    """Build the k-th order De Bruijn graph for DBGNN.

    Following Qarkaxhija et al. (arXiv:2209.08311): nodes are unique directed
    causal walks ``(v0, ..., v{k-1})``; an edge connects two overlapping
    k-node walks when their concatenation is a causal walk of length ``k``.
    Edge weights count temporal realizations of those causal completions.

    Also builds the static time-aggregated graph for the first-order GCN branch.

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.
    directed:
        If ``True``, treat contacts as directed. If ``None``, infer from ``H``.
    delta:
        Maximum allowed time gap for causal walks. ``None`` = no constraint.
    order:
        De Bruijn order k. Must be at least 2.
    time_bin_size:
        Aggregate contacts into bins of this width before constructing the
        higher-order graph. ``1`` keeps the original time scale. Values above
        one collapse duplicate ``(u, v, bin)`` contacts, mirroring the
        interaction aggregation used in the DBGNN paper for dense proximity
        data.
    max_temporal_states, max_db_nodes, max_db_edges:
        Optional fail-fast limits for higher-order graph construction.

    Returns
    -------
    dict with keys:
        ``n_nodes``              int
        ``n_db_nodes``           int
        ``db_edge_index``        LongTensor [2, E_db]
        ``db_edge_weight``       FloatTensor [E_db]  GCN-normalised, incl. self-loops
        ``db_node_to_original``  LongTensor [n_db, order]
        ``db_node_last``         LongTensor [n_db]
        ``static_edge_index``    LongTensor [2, E_st]  — time-aggregated graph
        ``static_edge_weight``   FloatTensor [E_st]    — GCN-normalised, incl. self-loops
    """
    from utils.make_de_bruijn_graph import make_de_bruijn_graph_compact as _make_db_compact
    from setup.read_network import make_array_from_networkx

    if order < 2:
        raise ValueError(f"DBGNN requires order >= 2, got {order}")

    n_nodes = H.number_of_nodes()
    is_directed = H.is_directed() if directed is None else bool(directed)
    H_array_raw = make_array_from_networkx(H)
    H_array, bin_stats = _coarsen_contact_times(H_array_raw, time_bin_size)

    empty_static_ei, empty_static_ew = normalize_gcn_edges(
        torch.zeros(2, 0, dtype=torch.long),
        torch.zeros(0, dtype=torch.float32),
        n_nodes,
        add_self_loops=True,
    )
    _empty = {
        "n_nodes":             n_nodes,
        "order":               order,
        "directed":            is_directed,
        "n_db_nodes":          0,
        "db_edge_count":       0,
        "time_bin_size":       time_bin_size,
        "db_stats":            {
            "order": order,
            "delta": delta,
            "directed": is_directed,
            "expanded_contacts": 0,
            "temporal_state_count": 0,
            "n_db_nodes": 0,
            "db_edge_count": 0,
            **bin_stats,
        },
        "db_edge_index":       torch.zeros(2, 0, dtype=torch.long),
        "db_edge_weight":      torch.zeros(0, dtype=torch.float32),
        "db_node_to_original": torch.zeros(0, order, dtype=torch.long),
        "db_node_last":        torch.zeros(0, dtype=torch.long),
        "static_edge_index":   empty_static_ei,
        "static_edge_weight":  empty_static_ew,
    }
    if len(H_array) == 0:
        return _empty

    # ----------------------------------------------------------------
    # De Bruijn graph G^(k)
    # ----------------------------------------------------------------
    node_list, edge_triples, db_stats = _make_db_compact(
        H_array,
        delta=delta,
        directed=is_directed,
        order=order,
        max_temporal_states=max_temporal_states,
        max_db_nodes=max_db_nodes,
        max_db_edges=max_db_edges,
    )
    db_stats = {**db_stats, **bin_stats}
    raw_n_db_nodes = len(node_list)

    # A De Bruijn graph with no causal walk completions (no edges) carries no
    # higher-order information beyond the self-loops that would be added below.
    # This is exactly the fully time-collapsed Δt in the H2 sweep: a single time
    # bin admits no strictly-increasing causal walk. There, order k>=3 already
    # yields zero nodes, while k=2 yields the single-contact nodes with no edges.
    # We drop the higher-order branch in BOTH cases so the model falls back to the
    # first-order static branch IDENTICALLY for every order — the collapse
    # endpoint reduces to the same pure static GCN regardless of k, instead of
    # letting k=2 keep self-loop-only nodes blended into the readout. This makes
    # the H2 k2-vs-k3 collapse points order-invariant by construction.
    if raw_n_db_nodes > 0 and edge_triples:
        # DB node -> original-node mapping
        db_node_to_original = torch.tensor(
            [[int(v) for v in node] for node in node_list], dtype=torch.long
        )  # [n_db, order]
        db_node_last = db_node_to_original[:, -1]

        # DB edges with raw count weights
        db_src = torch.tensor([e[0] for e in edge_triples], dtype=torch.long)
        db_dst = torch.tensor([e[1] for e in edge_triples], dtype=torch.long)
        raw_w  = torch.tensor([e[2] for e in edge_triples], dtype=torch.float32)
        db_edge_index_raw = torch.stack([db_src, db_dst], dim=0)  # [2, E_db]
        db_edge_index, db_edge_weight = normalize_gcn_edges(
            db_edge_index_raw, raw_w, raw_n_db_nodes, add_self_loops=True
        )
        n_db_nodes = raw_n_db_nodes
        db_edge_count = len(edge_triples)
    else:
        # No causal structure -> empty De Bruijn graph for every order.
        n_db_nodes = 0
        db_edge_count = 0
        db_node_to_original = torch.zeros(0, order, dtype=torch.long)
        db_node_last = torch.zeros(0, dtype=torch.long)
        db_edge_index = torch.zeros(2, 0, dtype=torch.long)
        db_edge_weight = torch.zeros(0, dtype=torch.float32)
        if raw_n_db_nodes > 0:
            # Record the dropped higher-order nodes for transparency in the cost
            # diagnostics (e.g. k=2 builds contact nodes that carry no causal info).
            db_stats = {**db_stats, "dropped_no_causal_edges": True,
                        "raw_n_db_nodes": int(raw_n_db_nodes)}

    # ----------------------------------------------------------------
    # Static time-aggregated graph G^(1)  (for first-order GCN branch)
    # ----------------------------------------------------------------
    st_weight_by_edge: dict[tuple[int, int], float] = defaultdict(float)
    for u, v, _ in H_array.tolist():
        st_weight_by_edge[(int(u), int(v))] += 1.0
        if not is_directed and u != v:
            st_weight_by_edge[(int(v), int(u))] += 1.0

    if st_weight_by_edge:
        static_edges = list(st_weight_by_edge.items())
        static_edge_index_raw = torch.tensor(
            [[u for (u, _), _ in static_edges], [v for (_, v), _ in static_edges]],
            dtype=torch.long,
        )
        static_w_raw = torch.tensor([w for _, w in static_edges], dtype=torch.float32)
    else:
        static_edge_index_raw = torch.zeros(2, 0, dtype=torch.long)
        static_w_raw = torch.zeros(0, dtype=torch.float32)
    static_edge_index, static_edge_weight = normalize_gcn_edges(
        static_edge_index_raw, static_w_raw, n_nodes, add_self_loops=True
    )

    return {
        "n_nodes":             n_nodes,
        "order":               order,
        "directed":            is_directed,
        "n_db_nodes":          n_db_nodes,
        "db_edge_count":       db_edge_count,
        "time_bin_size":       time_bin_size,
        "db_stats":            db_stats,
        "db_edge_index":       db_edge_index,
        "db_edge_weight":      db_edge_weight,
        "db_node_to_original": db_node_to_original,
        "db_node_last":        db_node_last,
        "static_edge_index":   static_edge_index,
        "static_edge_weight":  static_edge_weight,
    }


# ---------------------------------------------------------------------------
# Temporal event graph / DAG  (DAGGNN)
# ---------------------------------------------------------------------------

def build_dag_event_graph(
    H: nx.Graph,
    delta_t: int | None = None,
) -> dict:
    """Build the temporal event graph (TEG) as a DAG for DAGGNN.

    Each contact event (u, v, t) becomes a node in the TEG. A directed edge
    (e1 → e2) is added when event e1=(u,v,t1) causally enables e2=(v,w,t2):
    they share node v and t2 > t1.  The result is a DAG.

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.
    delta_t:
        Maximum time gap for a causal link.  ``None`` means no limit.

    Returns
    -------
    dict with keys:
        ``n_nodes``         int
        ``n_events``        int  — number of contact events (TEG nodes)
        ``dag_edge_index``  LongTensor [2, E_dag]  — forward causal edges
        ``event_to_node``   LongTensor [n_events]  — arriving node per event
        ``event_src_node``  LongTensor [n_events]  — departing node per event
        ``event_times``     LongTensor [n_events]  — time of each event
    """
    n_nodes = H.number_of_nodes()

    # Collect all directed contact events (both directions for undirected edges)
    events: list[tuple[int, int, int]] = []
    for u, v, data in H.edges(data=True):
        for t in data.get("times", []):
            events.append((int(u), int(v), int(t)))
            if u != v:
                events.append((int(v), int(u), int(t)))

    if not events:
        return {
            "n_nodes":        n_nodes,
            "n_events":       0,
            "dag_edge_index": torch.zeros(2, 0, dtype=torch.long),
            "event_to_node":  torch.zeros(0, dtype=torch.long),
            "event_src_node": torch.zeros(0, dtype=torch.long),
            "event_times":    torch.zeros(0, dtype=torch.long),
        }

    # Sort by time
    events.sort(key=lambda e: e[2])
    n_events = len(events)

    event_src_arr = np.array([e[0] for e in events], dtype=np.int64)
    event_dst_arr = np.array([e[1] for e in events], dtype=np.int64)
    event_t_arr   = np.array([e[2] for e in events], dtype=np.int64)

    # Build causal edges efficiently:
    # For each event i=(u,v,t1), find all events j=(v,w,t2) where t2>t1
    # (and t2-t1 <= delta_t if specified).
    # Group events by their SOURCE node for fast lookup.
    from collections import defaultdict
    node_to_events: dict[int, list[int]] = defaultdict(list)
    for i, (u, v, t) in enumerate(events):
        node_to_events[u].append(i)  # events that START at node u

    causal_src: list[int] = []
    causal_dst: list[int] = []

    for i, (u, v, t1) in enumerate(events):
        # Find events starting at node v (arriving node of event i)
        for j in node_to_events[v]:
            t2 = events[j][2]
            if t2 <= t1:
                continue
            if delta_t is not None and (t2 - t1) > delta_t:
                continue
            causal_src.append(i)
            causal_dst.append(j)

    if causal_src:
        dag_edge_index = torch.tensor(
            [causal_src, causal_dst], dtype=torch.long
        )  # [2, E_dag]
    else:
        dag_edge_index = torch.zeros(2, 0, dtype=torch.long)

    return {
        "n_nodes":        n_nodes,
        "n_events":       n_events,
        "dag_edge_index": dag_edge_index,
        "event_to_node":  torch.from_numpy(event_dst_arr),
        "event_src_node": torch.from_numpy(event_src_arr),
        "event_times":    torch.from_numpy(event_t_arr),
    }
