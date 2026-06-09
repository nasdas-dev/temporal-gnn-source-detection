from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn.conv import SAGEConv


class TemporalGNN(torch.nn.Module):
    """Simple temporal GNN: one SAGEConv layer per time-slice, applied in reverse.

    Designed for 2D input x of shape [N, in_channels] (single sample).
    Batching is handled externally by temporal_gnn_forward in training/trainer.py,
    which replicates each snapshot graph into a disconnected batched graph.

    Residual AddNorm blocks and a lightweight Jumping-Knowledge-style readout
    keep this intentionally modest baseline trainable when the temporal graph
    is represented by several contact snapshots.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_snapshots: int,
        *,
        residual: bool = True,
        layer_norm: bool = True,
        dropout_rate: float = 0.0,
        readout: str = "jumping_mean",
    ) -> None:
        super().__init__()
        if readout not in {"last", "jumping_mean"}:
            raise ValueError("TemporalGNN readout must be 'last' or 'jumping_mean'.")
        self.residual = residual
        self.layer_norm = layer_norm
        self.readout = readout
        self.lin_pre = torch.nn.Linear(in_channels, hidden_channels)
        self.input_norm = torch.nn.LayerNorm(hidden_channels) if layer_norm else torch.nn.Identity()
        self.convs = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_snapshots)]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(hidden_channels) if layer_norm else torch.nn.Identity()
                for _ in range(num_snapshots)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        readout_dim = hidden_channels * 2 if readout == "jumping_mean" else hidden_channels
        self.lin_post = torch.nn.Linear(readout_dim, out_channels)

    def encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Encode final S/I/R node states before temporal propagation."""
        return self.input_norm(F.relu(self.lin_pre(x)))

    def apply_temporal_layer(
        self,
        layer_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one snapshot layer with optional residual AddNorm."""
        h = F.relu(self.convs[layer_idx](x, edge_index))
        h = self.dropout(h)
        if self.residual:
            h = h + x
        return self.norms[layer_idx](h)

    def score_nodes(self, x: torch.Tensor, history: list[torch.Tensor]) -> torch.Tensor:
        """Return unnormalised source scores from temporal hidden states."""
        if self.readout == "jumping_mean":
            temporal_mean = torch.stack(history, dim=0).mean(dim=0)
            x = torch.cat([x, temporal_mean], dim=-1)
        return self.lin_post(x)

    def forward(self, x: torch.Tensor, edge_indeces: dict, time_order: list[int] | None = None) -> torch.Tensor:
        """Forward pass for a single sample.

        Parameters
        ----------
        x:
            Node feature matrix, shape [N, in_channels].
        edge_indeces:
            Dict mapping time-slice index → edge_index LongTensor [2, E_t].

        Returns
        -------
        log_probs : Tensor [N]
            Log-softmax over nodes (source probability distribution).
        """
        x = self.encode_input(x)                             # [N, hidden]
        history = [x]
        order = sorted(edge_indeces) if time_order is None else list(time_order)
        for count, t in enumerate(reversed(order)):
            x = self.apply_temporal_layer(count, x, edge_indeces[t])
            history.append(x)
        x = self.score_nodes(x, history).squeeze(-1)         # [N]
        return F.log_softmax(x, dim=-1)                      # [N]
