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

import networkx as nx
import numpy as np
import torch


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
) -> dict:
    """Build the aggregated static graph + binary temporal activation patterns.

    For each edge (u, v) in the static projection, ``edge_attr[e, t] = 1`` iff
    the edge was active at time step *t*.  Both (u, v) and (v, u) are included
    with the same activation pattern (undirected contacts).

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.

    Returns
    -------
    dict with keys:
        ``n_nodes``     int
        ``T``           int   — number of time steps (t_max + 1)
        ``edge_index``  LongTensor [2, 2*|E|]
        ``edge_attr``   FloatTensor [2*|E|, T]  — binary activation pattern
    """
    n_nodes  = H.number_of_nodes()
    all_times = [t for _, _, data in H.edges(data=True) for t in data.get("times", [])]
    if not all_times:
        raise ValueError("Graph has no temporal edge data ('times' attribute missing).")
    t_max = max(all_times)
    T = t_max + 1

    src_list:  list[int] = []
    dst_list:  list[int] = []
    attr_list: list[torch.Tensor] = []

    for u, v, data in H.edges(data=True):
        act = torch.zeros(T, dtype=torch.float32)
        for t in data.get("times", []):
            act[t] = 1.0

        src_list.append(u); dst_list.append(v); attr_list.append(act)
        src_list.append(v); dst_list.append(u); attr_list.append(act.clone())

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.stack(attr_list, dim=0)  # [2|E|, T]

    return {
        "n_nodes":    n_nodes,
        "T":          T,
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
    }


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

def build_de_bruijn_graph(
    H: nx.Graph,
    directed: bool = False,
    delta: int | None = None,
) -> dict:
    """Build the second-order De Bruijn graph for DBGNN.

    Following Qarkaxhija et al. (arXiv:2209.08311): nodes are unique directed
    node-pairs (u, v) observed in contacts; an edge ((u,v),(v,w)) exists if
    there are times t1 < t2 (with t2-t1 ≤ delta) such that (u,v;t1) and
    (v,w;t2) are contacts.  Edge weights count causal walk completions.

    Also builds the static time-aggregated graph for the first-order GCN branch.

    Parameters
    ----------
    H:
        Temporal NetworkX graph with ``'times'`` edge attribute.
    directed:
        If ``True``, treat contacts as directed. Default ``False``.
    delta:
        Maximum allowed time gap for causal walks. ``None`` = no constraint.

    Returns
    -------
    dict with keys:
        ``n_nodes``              int
        ``n_db_nodes``           int
        ``db_edge_index``        LongTensor [2, E_db]
        ``db_edge_weight``       FloatTensor [E_db]  normalised per destination
        ``db_node_to_original``  LongTensor [n_db, 2]  — (u, v) per DB node
        ``static_edge_index``    LongTensor [2, E_st]  — time-aggregated graph
        ``static_edge_weight``   FloatTensor [E_st]    — normalised per destination
    """
    from utils.make_de_bruijn_graph import make_de_bruijn_graph as _make_db
    from setup.read_network import make_array_from_networkx

    n_nodes = H.number_of_nodes()
    H_array = make_array_from_networkx(H)

    _empty = {
        "n_nodes":             n_nodes,
        "n_db_nodes":          0,
        "db_edge_index":       torch.zeros(2, 0, dtype=torch.long),
        "db_edge_weight":      torch.zeros(0, dtype=torch.float32),
        "db_node_to_original": torch.zeros(0, 2, dtype=torch.long),
        "static_edge_index":   torch.zeros(2, 0, dtype=torch.long),
        "static_edge_weight":  torch.zeros(0, dtype=torch.float32),
    }
    if len(H_array) == 0:
        return _empty

    # ----------------------------------------------------------------
    # De Bruijn graph G^(2)
    # ----------------------------------------------------------------
    B = _make_db(H_array, delta=delta, directed=directed)

    node_list   = list(B.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    n_db_nodes  = len(node_list)

    if n_db_nodes == 0:
        return _empty

    # DB node → original-node mapping
    db_node_to_original = torch.tensor(
        [[int(u), int(v)] for u, v in node_list], dtype=torch.long
    )  # [n_db, 2]

    # DB edges with raw weights
    edge_triples = [(node_to_idx[u], node_to_idx[v], d.get("weight", 1))
                    for u, v, d in B.edges(data=True)]
    if edge_triples:
        db_src = torch.tensor([e[0] for e in edge_triples], dtype=torch.long)
        db_dst = torch.tensor([e[1] for e in edge_triples], dtype=torch.long)
        raw_w  = torch.tensor([e[2] for e in edge_triples], dtype=torch.float32)
        db_edge_index = torch.stack([db_src, db_dst], dim=0)  # [2, E_db]
        # Normalise weights so they sum to 1 per destination (w(u,v)/S(v))
        dst_sum = torch.zeros(n_db_nodes, dtype=torch.float32)
        dst_sum.scatter_add_(0, db_dst, raw_w)
        db_edge_weight = raw_w / dst_sum[db_dst].clamp(min=1e-8)
    else:
        db_edge_index  = torch.zeros(2, 0, dtype=torch.long)
        db_edge_weight = torch.zeros(0, dtype=torch.float32)

    # ----------------------------------------------------------------
    # Static time-aggregated graph G^(1)  (for first-order GCN branch)
    # ----------------------------------------------------------------
    st_src, st_dst, st_w_raw = [], [], []
    for u, v, data in H.edges(data=True):
        cnt = float(len(data.get("times", [1])))
        st_src.append(u); st_dst.append(v); st_w_raw.append(cnt)
        st_src.append(v); st_dst.append(u); st_w_raw.append(cnt)

    if st_src:
        st_dst_t = torch.tensor(st_dst, dtype=torch.long)
        st_w_t   = torch.tensor(st_w_raw, dtype=torch.float32)
        deg_sum  = torch.zeros(n_nodes, dtype=torch.float32)
        deg_sum.scatter_add_(0, st_dst_t, st_w_t)
        static_edge_index  = torch.tensor([st_src, st_dst], dtype=torch.long)
        static_edge_weight = st_w_t / deg_sum[st_dst_t].clamp(min=1e-8)
    else:
        static_edge_index  = torch.zeros(2, 0, dtype=torch.long)
        static_edge_weight = torch.zeros(0, dtype=torch.float32)

    return {
        "n_nodes":             n_nodes,
        "n_db_nodes":          n_db_nodes,
        "db_edge_index":       db_edge_index,
        "db_edge_weight":      db_edge_weight,
        "db_node_to_original": db_node_to_original,
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
