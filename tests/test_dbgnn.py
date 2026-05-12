from __future__ import annotations

import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from utils.make_de_bruijn_graph import make_de_bruijn_graph


_ROOT = Path(__file__).resolve().parents[1]

_GRAPH_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "graph_builder_direct", _ROOT / "gnn" / "graph_builder.py"
)
_GRAPH_BUILDER = importlib.util.module_from_spec(_GRAPH_BUILDER_SPEC)
assert _GRAPH_BUILDER_SPEC.loader is not None
_GRAPH_BUILDER_SPEC.loader.exec_module(_GRAPH_BUILDER)
build_de_bruijn_graph = _GRAPH_BUILDER.build_de_bruijn_graph
normalize_gcn_edges = _GRAPH_BUILDER.normalize_gcn_edges

_DBGNN_SPEC = importlib.util.spec_from_file_location(
    "dbgnn_direct", _ROOT / "gnn" / "dbgnn.py"
)
_DBGNN = importlib.util.module_from_spec(_DBGNN_SPEC)
assert _DBGNN_SPEC.loader is not None
_DBGNN_SPEC.loader.exec_module(_DBGNN)
DBGNN = _DBGNN.DBGNN


def _edge_weights(G: nx.DiGraph):
    return {(u, v): d["weight"] for u, v, d in G.edges(data=True)}


def test_de_bruijn_order3_counts_duplicate_causal_realizations_and_delta():
    events = np.array([
        [1, 4, 0],   # reversed relative to 0 -> 1, so it cannot extend that walk
        [0, 1, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [2, 3, 10],  # excluded by delta=5 after 1 -> 2 at t=3
    ])

    graph = make_de_bruijn_graph(events, directed=True, delta=5, order=3)

    assert (0, 1, 2) in graph.nodes
    assert (1, 2, 3) in graph.nodes
    assert graph[(0, 1, 2)][(1, 2, 3)]["weight"] == 2
    assert (0, 1, 4) not in graph.nodes


def test_de_bruijn_directed_and_undirected_behaviour_differ():
    events = np.array([
        [0, 1, 1],
        [2, 1, 2],
    ])

    directed = make_de_bruijn_graph(events, directed=True, delta=None, order=2)
    undirected = make_de_bruijn_graph(events, directed=False, delta=None, order=2)

    assert ((0, 1), (1, 2)) not in _edge_weights(directed)
    assert _edge_weights(undirected)[((0, 1), (1, 2))] == 1


def test_normalize_gcn_edges_adds_self_loops_and_symmetric_weights():
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_weight = torch.tensor([2.0, 3.0], dtype=torch.float32)

    norm_index, norm_weight = normalize_gcn_edges(edge_index, edge_weight, n_nodes=3)

    assert torch.equal(
        norm_index,
        torch.tensor([[0, 1, 0, 1, 2], [1, 2, 0, 1, 2]], dtype=torch.long),
    )
    expected = torch.tensor([
        2.0 / (1.0 * 3.0) ** 0.5,
        3.0 / (3.0 * 4.0) ** 0.5,
        1.0,
        1.0 / 3.0,
        1.0 / 4.0,
    ])
    assert torch.allclose(norm_weight, expected)


def test_bipartite_sum_and_mean_aggregation_match_eq2_inputs():
    values = torch.tensor([[
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]])
    dst = torch.tensor([1, 1, 2], dtype=torch.long)

    sum_model = DBGNN(hidden_channels=2, num_conv_layers=0, order=2, bipartite_agg="sum")
    sum_out, sum_has = sum_model._aggregate_bipartite(values, dst, n_nodes=4)

    mean_model = DBGNN(hidden_channels=2, num_conv_layers=0, order=2, bipartite_agg="mean")
    mean_out, mean_has = mean_model._aggregate_bipartite(values, dst, n_nodes=4)

    assert torch.allclose(sum_out[0, 1], torch.tensor([4.0, 6.0]))
    assert torch.allclose(sum_out[0, 2], torch.tensor([5.0, 6.0]))
    assert torch.allclose(mean_out[0, 1], torch.tensor([2.0, 3.0]))
    assert torch.equal(sum_has, torch.tensor([False, True, True, False]))
    assert torch.equal(mean_has, sum_has)


def test_dbgnn_forward_smoke_for_order2_and_order3():
    H = nx.DiGraph()
    H.add_nodes_from(range(4))
    H.add_edge(0, 1, times=[1, 2])
    H.add_edge(1, 2, times=[3])
    H.add_edge(2, 3, times=[4])

    x = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ]])

    for order in (2, 3):
        graph_data = build_de_bruijn_graph(H, order=order, delta=5)
        model = DBGNN(
            hidden_channels=4,
            num_conv_layers=1,
            order=order,
            bipartite_agg="sum",
            dropout_rate=0.0,
        )
        model.eval()
        out = model(
            x,
            graph_data["db_edge_index"],
            graph_data["db_edge_weight"],
            graph_data["db_node_to_original"],
            graph_data["db_node_last"],
            graph_data["static_edge_index"],
            graph_data["static_edge_weight"],
        )

        assert out.shape == (1, 4)
        assert torch.isneginf(out[0, 0])
        assert torch.isfinite(out[0, 1:]).all()
