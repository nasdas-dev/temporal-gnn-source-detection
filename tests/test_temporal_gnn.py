from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch

torch.set_num_threads(1)

from gnn.temporal_gnn import TemporalGNN
from training.trainer import temporal_gnn_forward


def test_temporal_gnn_addnorm_matches_vectorized_forward() -> None:
    torch.manual_seed(7)
    model = TemporalGNN(
        in_channels=3,
        hidden_channels=8,
        out_channels=1,
        num_snapshots=2,
        residual=True,
        layer_norm=True,
        dropout_rate=0.0,
    )
    model.eval()

    edge_indeces = {
        0: torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        1: torch.tensor([[3, 2, 1], [2, 1, 0]], dtype=torch.long),
    }
    graph_data = {
        "n_nodes": 4,
        "edge_indeces": edge_indeces,
        "time_order": [0, 1],
    }
    x_batch = torch.randn(3, 4, 3)

    vectorized = temporal_gnn_forward(model, x_batch, graph_data, torch.device("cpu"))
    single = torch.stack(
        [model(x, edge_indeces, graph_data["time_order"]) for x in x_batch],
        dim=0,
    )

    assert vectorized.shape == (3, 4)
    assert torch.allclose(vectorized, single, atol=1e-6)
    assert torch.allclose(vectorized.exp().sum(dim=1), torch.ones(3), atol=1e-6)


def test_temporal_gnn_addnorm_can_be_disabled() -> None:
    model = TemporalGNN(
        in_channels=3,
        hidden_channels=4,
        out_channels=1,
        num_snapshots=3,
        residual=False,
        layer_norm=False,
    )

    assert model.residual is False
    assert model.layer_norm is False
    assert len(model.norms) == 3


def test_temporal_gnn_learns_toy_source_signal() -> None:
    torch.manual_seed(11)
    model = TemporalGNN(
        in_channels=3,
        hidden_channels=12,
        out_channels=1,
        num_snapshots=2,
        residual=True,
        layer_norm=True,
        dropout_rate=0.0,
        readout="jumping_mean",
    )
    graph_data = {
        "n_nodes": 5,
        "edge_indeces": {
            0: torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            1: torch.tensor([[4, 3, 2, 1], [3, 2, 1, 0]], dtype=torch.long),
        },
        "time_order": [0, 1],
    }
    labels = torch.arange(5).repeat_interleave(8)
    x_batch = torch.zeros(labels.numel(), 5, 3)
    x_batch[:, :, 0] = 1.0
    x_batch[torch.arange(labels.numel()), labels, 0] = 0.0
    x_batch[torch.arange(labels.numel()), labels, 1] = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=0.03, weight_decay=0.0)
    initial = torch.nn.functional.nll_loss(
        temporal_gnn_forward(model, x_batch, graph_data, torch.device("cpu")),
        labels,
    ).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        out = temporal_gnn_forward(model, x_batch, graph_data, torch.device("cpu"))
        loss = torch.nn.functional.nll_loss(out, labels)
        loss.backward()
        optimizer.step()

    final = torch.nn.functional.nll_loss(
        temporal_gnn_forward(model, x_batch, graph_data, torch.device("cpu")),
        labels,
    ).item()

    assert final < initial * 0.35
    assert final < 0.25
