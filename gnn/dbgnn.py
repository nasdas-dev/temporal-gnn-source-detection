"""
De Bruijn Graph Neural Network (DBGNN)
Adapted from: Qarkaxhija et al., 'De Bruijn goes Neural: Causality-Aware GNNs
for Time Series Data on Dynamic Graphs', arXiv:2209.08311

Architecture overview
---------------------
A temporal contact network G(V, E^t) is transformed into a De Bruijn graph B
where nodes represent *causal walks* (sequences of temporally ordered contacts)
and directed edges connect walks that are causally compatible (suffix of one =
prefix of another).

For source detection we:
1. Map original SIR node features to De Bruijn node features.
2. Run message passing on the De Bruijn graph.
3. Read out the embedding of each original node via its *end-time sentinel*
   node in B (the (n, n, t_end) node which acts as a summary for node n).
4. Project to scores → log-softmax.

Paper notation → code mapping
------------------------------
- B             → nx.DiGraph from make_de_bruijn_graph()
- (x, y, t)     → De Bruijn node for contact (x, y) at time t
- (n, n, t_end) → sentinel node for original node n (readout anchor)
- h_i^(l)       → node embeddings at layer l  [B_batch, db_N, D]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList


# ---------------------------------------------------------------------------
# Single DBGNN layer  (SAGEConv-style, manually batched)
# ---------------------------------------------------------------------------

class DBGNNLayer(torch.nn.Module):
    """One layer of message passing on the De Bruijn graph.

    Implements a mean-aggregator SAGE-style update without PyG batching
    overhead, directly operating on [B, db_N, D] tensors.

    Update rule:
        h^(l+1)_dst = ReLU( W_self * h^(l)_dst + W_agg * mean_{src→dst} h^(l)_src )
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_self = Linear(hidden_dim, hidden_dim, bias=False)
        self.W_agg  = Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        h: torch.Tensor,           # [B, db_N, D]
        edge_index: torch.Tensor,  # [2, E_db]  src → dst
    ) -> torch.Tensor:             # [B, db_N, D]
        B, db_N, D = h.shape
        src, dst = edge_index[0], edge_index[1]
        E = src.shape[0]

        if E == 0:
            return F.relu(self.W_self(h))

        # Gather source features
        h_src = h[:, src, :]                              # [B, E, D]

        # Scatter-add into aggregation buffer
        dst_idx = dst.view(1, -1, 1).expand(B, E, D)
        agg_sum = h.new_zeros(B, db_N, D)
        agg_sum.scatter_add_(1, dst_idx, h_src)

        # Compute per-node incoming degree for mean normalisation
        deg = h.new_zeros(db_N)
        deg.scatter_add_(0, dst, torch.ones(E, device=h.device))
        deg = deg.clamp(min=1).view(1, db_N, 1)

        agg_mean = agg_sum / deg                          # [B, db_N, D]
        h_new = F.relu(self.W_self(h) + self.W_agg(agg_mean))
        return h_new


# ---------------------------------------------------------------------------
# Full DBGNN model
# ---------------------------------------------------------------------------

class DBGNN(torch.nn.Module):
    """
    De Bruijn GNN for epidemic source detection.

    Args
    ----
    hidden_channels:
        Hidden embedding dimension D.
    num_conv_layers:
        Number of De Bruijn graph convolution layers.
    dropout_rate:
        Dropout probability applied after each convolution.
    """

    def __init__(
        self,
        hidden_channels: int,
        num_conv_layers: int,
        conv_type: str = "sage",    # currently only "sage" is supported
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate = dropout_rate

        # Separate input projections: sentinels have 3-dim input (one-hot SIR),
        # event nodes have 6-dim (concatenation of src+dst SIR features).
        self.proj_sentinel = Linear(3, hidden_channels)
        self.proj_event    = Linear(6, hidden_channels)

        # De Bruijn graph convolution layers
        self.convs = ModuleList(
            [DBGNNLayer(hidden_channels) for _ in range(num_conv_layers)]
        )

        # Final readout: node embedding → scalar score
        self.out = Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,                    # [B, N, 3]  original SIR states
        edge_index: torch.Tensor,            # [2, E_db]  De Bruijn graph edges
        db_node_to_original: torch.Tensor,   # [db_N, 2]  (u_orig, v_orig) per DB node
        sentinel_end_indices: torch.Tensor,  # [N]        DB index of (n,n,t_end) per orig node
        is_sentinel: torch.Tensor,           # [db_N]     True if DB node is a sentinel
    ) -> torch.Tensor:                       # [B, N]     log-probabilities
        B, N, _ = x.shape
        db_N = db_node_to_original.shape[0]
        D    = self.hidden_channels

        # ----------------------------------------------------------------
        # 1. Construct De Bruijn node features from original SIR states
        # ----------------------------------------------------------------
        u_orig = db_node_to_original[:, 0]  # [db_N]
        v_orig = db_node_to_original[:, 1]  # [db_N]

        x_u = x[:, u_orig, :]               # [B, db_N, 3]
        x_v = x[:, v_orig, :]               # [B, db_N, 3]

        # Sentinel projection: only u (= v for sentinels) matters
        h_sent  = F.relu(self.proj_sentinel(x_u))            # [B, db_N, D]
        # Event projection: concatenate both endpoint features
        h_event = F.relu(self.proj_event(torch.cat([x_u, x_v], dim=-1)))  # [B, db_N, D]

        # Blend: select projection based on node type
        mask = is_sentinel.to(x.device).view(1, db_N, 1).expand(B, db_N, D)
        h = torch.where(mask, h_sent, h_event)               # [B, db_N, D]

        # ----------------------------------------------------------------
        # 2. De Bruijn graph convolution
        # ----------------------------------------------------------------
        ei = edge_index.to(x.device)
        for conv in self.convs:
            h = conv(h, ei)
            if self.dropout_rate > 0 and self.training:
                h = F.dropout(h, p=self.dropout_rate)

        # ----------------------------------------------------------------
        # 3. Readout: extract embedding of end-time sentinel for each node
        # ----------------------------------------------------------------
        sent_idx = sentinel_end_indices.to(x.device)         # [N]
        node_emb = h[:, sent_idx, :]                         # [B, N, D]

        # ----------------------------------------------------------------
        # 4. Score → mask susceptible nodes → log-softmax
        # ----------------------------------------------------------------
        scores = self.out(node_emb).squeeze(-1)              # [B, N]
        susceptible_mask = x[..., 0].bool()                  # [B, N]
        scores = scores.masked_fill(susceptible_mask, float("-inf"))
        return F.log_softmax(scores, dim=-1)                 # [B, N]
