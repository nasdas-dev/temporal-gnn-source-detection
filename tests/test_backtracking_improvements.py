import importlib.util
from pathlib import Path

import networkx as nx
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_attr(module_name: str, path: Path, attr: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, attr)


BacktrackingNetwork = _load_attr(
    "backtracking_network_under_test",
    ROOT / "gnn" / "backtracking_network.py",
    "BacktrackingNetwork",
)
build_temporal_activation = _load_attr(
    "graph_builder_under_test",
    ROOT / "gnn" / "graph_builder.py",
    "build_temporal_activation",
)


def test_sparse_edge_textures_match_dense_projection() -> None:
    H = nx.Graph()
    H.add_edge(0, 1, times=[0, 2, 4])
    H.add_edge(1, 2, times=[1, 3])

    dense = build_temporal_activation(H, dense_edge_attr=True)
    sparse = build_temporal_activation(H)

    torch.manual_seed(7)
    model = BacktrackingNetwork(
        node_feat_dim=3,
        edge_feat_dim=dense["T"],
        hidden_dim=8,
        num_layers=2,
    )
    x = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )

    dense_out = model(x, dense["edge_index"], edge_attr=dense["edge_attr"])
    sparse_out = model(
        x,
        sparse["edge_index"],
        edge_time_index=sparse["edge_time_index"],
        edge_time_edge_index=sparse["edge_time_edge_index"],
        n_edges=sparse["n_edges"],
    )

    assert torch.allclose(sparse_out, dense_out, atol=1e-6)


def test_temporal_activation_preserves_directed_edges() -> None:
    H = nx.DiGraph()
    H.add_edge(0, 1, times=[0, 2])
    H.add_edge(2, 1, times=[1])

    graph = build_temporal_activation(H)

    assert graph["directed"] is True
    assert graph["edge_index"].tolist() == [[0, 2], [1, 1]]
    assert graph["edge_time_edge_index"].tolist() == [0, 0, 1]
    assert graph["edge_time_index"].tolist() == [0, 2, 1]


def test_temporal_activation_duplicates_undirected_edges() -> None:
    H = nx.Graph()
    H.add_edge(0, 1, times=[0, 2])

    graph = build_temporal_activation(H)

    assert graph["directed"] is False
    assert graph["edge_index"].tolist() == [[0, 1], [1, 0]]
    assert graph["edge_time_edge_index"].tolist() == [0, 0, 1, 1]
    assert graph["edge_time_index"].tolist() == [0, 2, 0, 2]
