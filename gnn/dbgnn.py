"""
De Bruijn Graph Neural Network (DBGNN).

Faithful source-detection adaptation of Qarkaxhija, Perri, and Scholtes,
"De Bruijn goes Neural: Causality-Aware GNNs for Time Series Data on
Dynamic Graphs" (arXiv:2209.08311v1).

The implementation keeps the paper's architectural core:

1. Message passing on a k-th order De Bruijn graph G^(k), where nodes are
   causal walks of length k - 1 and edges are causal walk completions.
2. Parallel first-order GCN message passing on the weighted static projection.
3. Bipartite Eq. 2 readout from higher-order nodes back to original nodes.

It adapts the input/output to this thesis' source-detection pipeline: node
features are final SIR states and the model returns a masked log-softmax over
candidate source nodes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList


class WeightedGCNLayer(torch.nn.Module):
    """Paper-style weighted GCN propagation with pre-normalized edge weights."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = Linear(in_dim, out_dim, bias=True)
        self.out_dim = out_dim

    def forward(
        self,
        h: torch.Tensor,                         # [B, N, in_dim]
        edge_index: torch.Tensor,                 # [2, E] src -> dst
        edge_weight: torch.Tensor,                # [E] normalized GCN weights
    ) -> torch.Tensor:                            # [B, N, out_dim]
        B, N, in_dim = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        if E == 0:
            return F.elu(self.linear(h))

        messages = h[:, src, :] * edge_weight.to(h.device).view(1, E, 1)
        dst_idx = dst.view(1, E, 1).expand(B, E, in_dim)
        agg = h.new_zeros(B, N, in_dim)
        agg.scatter_add_(1, dst_idx, messages)
        return F.elu(self.linear(agg))


class DBGNN(torch.nn.Module):
    """DBGNN for epidemic source detection."""

    def __init__(
        self,
        hidden_channels: int,
        num_conv_layers: int,
        order: int = 2,
        bipartite_agg: str = "sum",
        dropout_rate: float = 0.2,
        conv_type: str = "gcn",
    ) -> None:
        super().__init__()
        if order < 2:
            raise ValueError(f"DBGNN requires order >= 2, got {order}")
        if bipartite_agg not in {"sum", "mean", "max", "min"}:
            raise ValueError(
                "bipartite_agg must be one of {'sum', 'mean', 'max', 'min'}, "
                f"got {bipartite_agg!r}"
            )

        self.hidden_channels = hidden_channels
        self.order = order
        self.bipartite_agg = bipartite_agg
        self.dropout_rate = dropout_rate
        self.conv_type = conv_type

        self.proj_ho = Linear(3 * order, hidden_channels)
        self.proj_fo = Linear(3, hidden_channels)

        self.ho_convs = ModuleList(
            [WeightedGCNLayer(hidden_channels, hidden_channels) for _ in range(num_conv_layers)]
        )
        self.fo_convs = ModuleList(
            [WeightedGCNLayer(hidden_channels, hidden_channels) for _ in range(num_conv_layers)]
        )

        self.bipartite_proj = Linear(hidden_channels, hidden_channels, bias=True)
        self.out = Linear(hidden_channels, 1)

    def _aggregate_bipartite(
        self,
        values: torch.Tensor,       # [B, n_db, D]
        dst: torch.Tensor,          # [n_db]
        n_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, n_db, D = values.shape
        device = values.device
        counts = torch.zeros(n_nodes, dtype=values.dtype, device=device)
        counts.scatter_add_(0, dst, torch.ones(n_db, dtype=values.dtype, device=device))
        has_db = counts > 0

        if self.bipartite_agg in {"sum", "mean"}:
            out = values.new_zeros(B, n_nodes, D)
            idx = dst.view(1, n_db, 1).expand(B, n_db, D)
            out.scatter_add_(1, idx, values)
            if self.bipartite_agg == "mean":
                out = out / counts.clamp(min=1).view(1, n_nodes, 1)
            return out, has_db

        reduce_name = "amax" if self.bipartite_agg == "max" else "amin"
        fill = -float("inf") if self.bipartite_agg == "max" else float("inf")
        out = values.new_full((B, n_nodes, D), fill)
        idx = dst.view(1, n_db, 1).expand(B, n_db, D)
        out.scatter_reduce_(1, idx, values, reduce=reduce_name, include_self=True)
        out = torch.where(has_db.view(1, n_nodes, 1), out, torch.zeros_like(out))
        return out, has_db

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, 3] original SIR states
        db_edge_index: torch.Tensor,         # [2, E_db] De Bruijn edges incl. self-loops
        db_edge_weight: torch.Tensor,        # [E_db] GCN-normalized weights
        db_node_to_original: torch.Tensor,   # [n_db, order]
        db_node_last: torch.Tensor,          # [n_db]
        static_edge_index: torch.Tensor,     # [2, E_st] static edges incl. self-loops
        static_edge_weight: torch.Tensor,    # [E_st] GCN-normalized weights
    ) -> torch.Tensor:                       # [B, N] log-probabilities
        B, N, _ = x.shape
        n_db = db_node_to_original.shape[0]
        D = self.hidden_channels
        device = x.device

        # Higher-order branch on G^(k).
        if n_db > 0:
            walk_idx = db_node_to_original.to(device)                 # [n_db, order]
            x_walk = x[:, walk_idx, :].reshape(B, n_db, 3 * self.order)
            h_db = F.elu(self.proj_ho(x_walk))

            ei_db = db_edge_index.to(device)
            ew_db = db_edge_weight.to(device)
            for conv in self.ho_convs:
                h_db = conv(h_db, ei_db, ew_db)
                if self.dropout_rate > 0 and self.training:
                    h_db = F.dropout(h_db, p=self.dropout_rate)
        else:
            h_db = torch.zeros(B, 0, D, device=device)

        # First-order branch on the weighted static projection.
        h_fo = F.elu(self.proj_fo(x))
        ei_st = static_edge_index.to(device)
        ew_st = static_edge_weight.to(device)
        for conv in self.fo_convs:
            h_fo = conv(h_fo, ei_st, ew_st)
            if self.dropout_rate > 0 and self.training:
                h_fo = F.dropout(h_fo, p=self.dropout_rate)

        # Bipartite Eq. 2: aggregate h_u^(k,l) + h_v^(1,g) for DB nodes ending at v.
        if n_db > 0:
            dst = db_node_last.to(device)
            augmented = h_db + h_fo[:, dst, :]
            bipartite_agg, _ = self._aggregate_bipartite(augmented, dst, N)
        else:
            bipartite_agg = torch.zeros(B, N, D, device=device)

        h_out = F.elu(self.bipartite_proj(bipartite_agg))
        scores = self.out(h_out).squeeze(-1)

        susceptible_mask = x[..., 0].bool()
        scores = scores.masked_fill(susceptible_mask, float("-inf"))
        return F.log_softmax(scores, dim=-1)
