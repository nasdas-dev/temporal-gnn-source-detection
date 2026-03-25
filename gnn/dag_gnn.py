"""
DAG-GNN — Directed Acyclic Graph Convolutional Network for Source Detection
Inspired by: Rey et al., 'Directed Acyclic Graph Convolutional Networks',
arXiv:2506.12218, and the temporal event graph formalism of
Saramäki et al., arXiv:1912.03904.

Architecture overview
---------------------
A temporal contact network G(V, E^t) is transformed into a *temporal event
graph* (TEG):
  - Nodes   = individual contact events (u, v, t)
  - Directed edges = causal links: (u,v,t1) → (v,w,t2) when t2 > t1
    (event at (u,v) can causally enable the next event at (v,w))

This TEG is a DAG (no cycles because t2 > t1 strictly).

For source detection we propagate information *backward* through the causal
chain (reverse DAG edges), so information flows from late observations toward
early causal predecessors — naturally favouring source nodes.

Paper notation → code mapping
------------------------------
- E^t         → contact events list, sorted by time
- v_e         → event node for contact (u, v, t)
- h_e^(l)     → event embedding at layer l  [B, n_events, D]
- agg_v       → scatter-mean of event embeddings to original nodes
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList


# ---------------------------------------------------------------------------
# Single DAG-GNN layer  (manually batched, no PyG dependency)
# ---------------------------------------------------------------------------

class DAGConvLayer(torch.nn.Module):
    """One layer of message passing on the (reversed) event DAG.

    Implements a mean-aggregator SAGE-style update directly on
    [B, n_events, D] tensors.

    Update rule:
        h^(l+1)_e = ReLU( W_self * h^(l)_e + W_agg * mean_{e'→e} h^(l)_e' )

    Messages flow along reversed DAG edges (backward in time).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.W_self = Linear(hidden_dim, hidden_dim, bias=False)
        self.W_agg  = Linear(hidden_dim, hidden_dim, bias=True)

    def forward(
        self,
        h: torch.Tensor,                # [B, n_events, D]
        rev_edge_index: torch.Tensor,   # [2, E_dag]  reversed causal edges
    ) -> torch.Tensor:                  # [B, n_events, D]
        B, n_events, D = h.shape
        E = rev_edge_index.shape[1]

        if E == 0:
            return F.relu(self.W_self(h))

        src, dst = rev_edge_index[0], rev_edge_index[1]

        # Gather source features
        h_src = h[:, src, :]                                  # [B, E, D]

        # Scatter-add into aggregation buffer
        dst_idx = dst.view(1, -1, 1).expand(B, E, D)
        agg_sum = h.new_zeros(B, n_events, D)
        agg_sum.scatter_add_(1, dst_idx, h_src)

        # Mean normalisation
        deg = h.new_zeros(n_events)
        deg.scatter_add_(0, dst, torch.ones(E, device=h.device))
        deg = deg.clamp(min=1).view(1, n_events, 1)

        agg_mean = agg_sum / deg                              # [B, n_events, D]
        h_new = F.relu(self.W_self(h) + self.W_agg(agg_mean))
        return h_new


# ---------------------------------------------------------------------------
# Full DAG-GNN model
# ---------------------------------------------------------------------------

class DAGGNN(torch.nn.Module):
    """
    Temporal Event Graph GNN for epidemic source detection.

    Each contact event (u, v, t) is represented as a node in the temporal
    event graph.  Causal edges connect events that are compatible in time.
    Message passing proceeds backward along the causal chain, aggregating
    evidence from later events toward the probable source.

    Args
    ----
    hidden_channels:
        Hidden embedding dimension D.
    num_conv_layers:
        Number of DAG convolution layers.
    dropout_rate:
        Dropout probability after each convolution.
    agg:
        Aggregation strategy for mapping event embeddings back to original
        nodes: ``"mean"`` or ``"sum"``.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_conv_layers: int = 3,
        dropout_rate: float = 0.2,
        agg: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout_rate    = dropout_rate
        self.agg             = agg

        # Event feature projection: cat(x_src, x_dst) [6] → [D]
        self.proj_event = Linear(6, hidden_channels)

        # DAG convolution layers (operate on reversed DAG edges)
        self.convs = ModuleList(
            [DAGConvLayer(hidden_channels) for _ in range(num_conv_layers)]
        )

        # Final readout: aggregated node embedding → scalar score
        self.out = Linear(hidden_channels, 1)

    def forward(
        self,
        x: torch.Tensor,               # [B, N, 3]  original SIR states
        dag_edge_index: torch.Tensor,  # [2, E_dag] forward causal edges (will be reversed)
        event_to_node: torch.Tensor,   # [n_events] arriving node index per event
        event_src_node: torch.Tensor,  # [n_events] departing node index per event
    ) -> torch.Tensor:                 # [B, N]     log-probabilities
        B, N, _ = x.shape
        n_events = event_src_node.shape[0]
        D = self.hidden_channels

        # ----------------------------------------------------------------
        # Edge case: no events in the temporal graph
        # ----------------------------------------------------------------
        if n_events == 0:
            scores = torch.zeros(B, N, device=x.device)
            susceptible_mask = x[..., 0].bool()
            scores = scores.masked_fill(susceptible_mask, float("-inf"))
            return F.log_softmax(scores, dim=-1)

        # ----------------------------------------------------------------
        # 1. Build event features from original node SIR states
        # ----------------------------------------------------------------
        ev_src  = event_src_node.to(x.device)   # [n_events]
        ev_dst  = event_to_node.to(x.device)     # [n_events]

        x_src = x[:, ev_src, :]                  # [B, n_events, 3]
        x_dst = x[:, ev_dst, :]                  # [B, n_events, 3]
        h = F.relu(self.proj_event(torch.cat([x_src, x_dst], dim=-1)))  # [B, n_events, D]

        # ----------------------------------------------------------------
        # 2. Reverse the DAG for backward propagation
        #    Forward edge: e1 → e2 (e1 causally enables e2, t(e1) < t(e2))
        #    Reversed edge: e2 → e1 (information flows from later to earlier)
        # ----------------------------------------------------------------
        if dag_edge_index.numel() > 0:
            rev_ei = dag_edge_index.flip(0).to(x.device)   # [2, E_dag]
        else:
            rev_ei = dag_edge_index.to(x.device)

        # ----------------------------------------------------------------
        # 3. DAG convolution layers (backward propagation)
        # ----------------------------------------------------------------
        for conv in self.convs:
            h = conv(h, rev_ei)
            if self.dropout_rate > 0 and self.training:
                h = F.dropout(h, p=self.dropout_rate)

        # ----------------------------------------------------------------
        # 4. Aggregate event embeddings → original node embeddings
        #    For each node v: aggregate all events where event_to_node == v
        # ----------------------------------------------------------------
        node_emb = torch.zeros(B, N, D, device=x.device)
        dst_idx  = ev_dst.view(1, -1, 1).expand(B, n_events, D)
        node_emb.scatter_add_(1, dst_idx, h)                 # [B, N, D]

        if self.agg == "mean":
            count = torch.zeros(N, device=x.device)
            count.scatter_add_(0, ev_dst, torch.ones(n_events, device=x.device))
            count = count.clamp(min=1).view(1, N, 1)
            node_emb = node_emb / count

        # ----------------------------------------------------------------
        # 5. Score → mask susceptible nodes → log-softmax
        # ----------------------------------------------------------------
        scores = self.out(node_emb).squeeze(-1)              # [B, N]
        susceptible_mask = x[..., 0].bool()
        scores = scores.masked_fill(susceptible_mask, float("-inf"))
        return F.log_softmax(scores, dim=-1)                 # [B, N]
