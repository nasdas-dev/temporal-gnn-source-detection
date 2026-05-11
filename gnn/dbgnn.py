"""
De Bruijn Graph Neural Network (DBGNN)
Adapted from: Qarkaxhija et al., 'De Bruijn goes Neural: Causality-Aware GNNs
for Time Series Data on Dynamic Graphs.' arXiv:2209.08311v1, 2022.

Architecture (paper Section 3)
-------------------------------
1. Higher-order branch: message passing on G^(2) (De Bruijn graph)
   - Nodes: unique directed node-pairs (u, v) with input features cat(x_u, x_v)
   - l layers of frequency-weighted mean aggregation (Eq. 1)
2. First-order branch: g layers of GCN on the static time-aggregated graph G^(1)
   - Same number of layers as higher-order branch
3. Bipartite projection G^b (Eq. 2): for each original node v, sum the
   higher-order embeddings of all (u, v) De Bruijn nodes, apply W^b, add
   the first-order embedding h_v^{1,g}.
4. Final linear layer → mask susceptible nodes → log-softmax.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList


# ---------------------------------------------------------------------------
# Shared message-passing layer
# ---------------------------------------------------------------------------

class MPLayer(torch.nn.Module):
    """Single mean-aggregation message passing layer (ELU activation).

    Supports pre-normalised edge weights (sum-to-1 per destination).
    Update: h'_v = ELU( W_self * h_v + W_agg * sum_{u→v} w(u,v) * h_u )
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.W_self  = Linear(in_dim, out_dim, bias=False)
        self.W_agg   = Linear(in_dim, out_dim, bias=True)
        self.out_dim = out_dim

    def forward(
        self,
        h: torch.Tensor,                          # [B, N, in_dim]
        edge_index: torch.Tensor,                  # [2, E]  src → dst
        edge_weight: torch.Tensor | None = None,   # [E] normalised (sum-to-1 per dst)
    ) -> torch.Tensor:                             # [B, N, out_dim]
        B, N, _ = h.shape
        D = self.out_dim
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        h_self = self.W_self(h)  # [B, N, D]

        if E == 0:
            return F.elu(h_self)

        h_src_proj = self.W_agg(h[:, src, :])  # [B, E, D]

        if edge_weight is not None:
            h_src_proj = h_src_proj * edge_weight.to(h.device).view(1, -1, 1)

        dst_idx = dst.view(1, -1, 1).expand(B, E, D)
        agg = h.new_zeros(B, N, D)
        agg.scatter_add_(1, dst_idx, h_src_proj)

        if edge_weight is None:
            deg = h.new_zeros(N)
            deg.scatter_add_(0, dst, torch.ones(E, device=h.device))
            agg = agg / deg.clamp(min=1).view(1, N, 1)

        return F.elu(h_self + agg)


# ---------------------------------------------------------------------------
# Full DBGNN model
# ---------------------------------------------------------------------------

class DBGNN(torch.nn.Module):
    """
    De Bruijn GNN for epidemic source detection.

    Parameters
    ----------
    hidden_channels:
        Hidden embedding dimension D for both branches.
    num_conv_layers:
        Number of message-passing layers in both branches (g = l).
    conv_type:
        Unused — kept for API compatibility with YAML configs.
    dropout_rate:
        Dropout probability applied after each convolution.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_conv_layers: int,
        conv_type: str = "sage",
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate    = dropout_rate

        # Input projections
        self.proj_ho  = Linear(6, hidden_channels)   # De Bruijn node: cat(x_u, x_v) — 6-dim
        self.proj_fo  = Linear(3, hidden_channels)   # first-order node: x_v — 3-dim

        # Higher-order De Bruijn convolutions (branch 1)
        self.ho_convs = ModuleList(
            [MPLayer(hidden_channels, hidden_channels) for _ in range(num_conv_layers)]
        )

        # First-order GCN convolutions (branch 2)
        self.fo_convs = ModuleList(
            [MPLayer(hidden_channels, hidden_channels) for _ in range(num_conv_layers)]
        )

        # Bipartite projection W^b (Eq. 2 in paper)
        self.bipartite_proj = Linear(hidden_channels, hidden_channels, bias=True)

        # Final readout: per-node embedding → scalar score
        self.out = Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, 3]  original SIR states
        db_edge_index: torch.Tensor,         # [2, E_db]  De Bruijn graph edges
        db_edge_weight: torch.Tensor,        # [E_db]     normalised edge weights
        db_node_to_original: torch.Tensor,   # [n_db, 2]  (u, v) per DB node
        static_edge_index: torch.Tensor,     # [2, E_st]  static graph edges
        static_edge_weight: torch.Tensor,    # [E_st]     normalised edge weights
    ) -> torch.Tensor:                       # [B, N]     log-probabilities
        B, N, _ = x.shape
        n_db = db_node_to_original.shape[0]
        D    = self.hidden_channels
        device = x.device

        # --------------------------------------------------------
        # 1. Higher-order branch: message passing on G^(2)
        # --------------------------------------------------------
        if n_db > 0:
            u_idx = db_node_to_original[:, 0].to(device)  # [n_db]
            v_idx = db_node_to_original[:, 1].to(device)  # [n_db]
            x_u = x[:, u_idx, :]                          # [B, n_db, 3]
            x_v = x[:, v_idx, :]                          # [B, n_db, 3]
            h_db = F.elu(self.proj_ho(torch.cat([x_u, x_v], dim=-1)))  # [B, n_db, D]

            ei_db = db_edge_index.to(device)
            ew_db = db_edge_weight.to(device)
            for conv in self.ho_convs:
                h_db = conv(h_db, ei_db, ew_db)
                if self.dropout_rate > 0 and self.training:
                    h_db = F.dropout(h_db, p=self.dropout_rate)
        else:
            v_idx = torch.zeros(0, dtype=torch.long, device=device)
            h_db  = torch.zeros(B, 0, D, device=device)

        # --------------------------------------------------------
        # 2. First-order branch: GCN on static graph G^(1)
        # --------------------------------------------------------
        h_fo = F.elu(self.proj_fo(x))   # [B, N, D]
        ei_st = static_edge_index.to(device)
        ew_st = static_edge_weight.to(device)
        for conv in self.fo_convs:
            h_fo = conv(h_fo, ei_st, ew_st)
            if self.dropout_rate > 0 and self.training:
                h_fo = F.dropout(h_fo, p=self.dropout_rate)

        # --------------------------------------------------------
        # 3. Bipartite projection (Eq. 2): G^(2) → original nodes
        #    For each original node v, aggregate embeddings of (u,v) DB nodes.
        # --------------------------------------------------------
        bipartite_agg = torch.zeros(B, N, D, device=device)
        if n_db > 0:
            v_exp = v_idx.view(1, -1, 1).expand(B, n_db, D)
            bipartite_agg.scatter_add_(1, v_exp, h_db)

            # Degree: how many DB nodes map to each original node
            deg = torch.zeros(N, device=device)
            deg.scatter_add_(0, v_idx, torch.ones(n_db, device=device))
            bipartite_agg = bipartite_agg / deg.clamp(min=1).view(1, N, 1)

        # Apply W^b only where DB nodes exist; add first-order embeddings
        has_db = (deg > 0).float().view(1, N, 1) if n_db > 0 else torch.zeros(1, N, 1, device=device)
        h_out = F.elu(self.bipartite_proj(bipartite_agg)) * has_db + h_fo  # [B, N, D]

        # --------------------------------------------------------
        # 4. Score → mask susceptible nodes → log-softmax
        # --------------------------------------------------------
        scores = self.out(h_out).squeeze(-1)          # [B, N]
        susceptible_mask = x[..., 0].bool()           # [B, N]  — S channel
        scores = scores.masked_fill(susceptible_mask, float("-inf"))
        return F.log_softmax(scores, dim=-1)           # [B, N]
