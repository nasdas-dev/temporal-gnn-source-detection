"""
Backtracking Network (BN) — Ru et al., AAAI 2023
'Inferring Patient Zero on Temporal Networks via Graph Neural Networks'

Implements the architecture depicted in Figure 2 (page 3) of the paper.

The BN learns an inverse mapping from the final SIR snapshot S_T and the
aggregated temporal network G̃ back to the initial source S_0.

Key equations (from the paper):
  Initialisation:
    h^0_i         = p^v(C_i)           node feature projection     (p^v = Linear + ReLU)
    g^0_{(i,j)}   = p^e(X_{(i,j)})    edge feature projection     (p^e = Linear + ReLU)

  Layer l update  (kernel-based convolutional operator, Eqs. 3–4):
    g^(l+1)_{(i,j)} = f^e( cat( g^l_{(i,j)},  h^l_i ) )          edge update
    h^(l+1)_i       = ReLU( f^v(h^l_i) + Σ_{j→i} g^(l+1)_{(j,i)} )  node update

    where f^e, f^v are FC layers with ReLU activations.

  Expert knowledge (Eq. 6):
    h_s ← −∞   for every susceptible node s
    (forces detection probability of susceptible nodes to zero after softmax)

  Objective (Eq. 5):
    L = −Σ_i  y_i · log softmax(h_i)   (cross-entropy)
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ModuleList


# ---------------------------------------------------------------------------
# Single BN convolution layer
# ---------------------------------------------------------------------------

class BNConvLayer(torch.nn.Module):
    """
    One layer of the Backtracking Network convolution (Eqs. 3 & 4).

    Edge update:
        g^(l+1)_{src→dst} = f^e( cat( g^l_{src→dst},  h^l_src ) )

    Node update (aggregate incoming edges at each destination):
        h^(l+1)_dst = ReLU( f^v(h^l_dst) + Σ_{src→dst} g^(l+1)_{src→dst} )

    Both f^e and f^v are single fully-connected layers with ReLU (as described
    in the paper: "f^e and f^v are represented by two independent
    fully-connected layers with activation function ReLU").
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        # f^e: edge update — input = cat(edge hidden, source node hidden)
        self.f_e = Sequential(Linear(2 * hidden_dim, hidden_dim), ReLU())
        # f^v: node self-transform
        self.f_v = Sequential(Linear(hidden_dim, hidden_dim), ReLU())

    def forward(
        self,
        h: torch.Tensor,          # [B, N, D]  node hiddens
        g: torch.Tensor,          # [B, E, D]  edge hiddens
        edge_index: torch.Tensor, # [2, E]
    ):
        """
        Returns updated (h_new, g_new) with the same shapes as the inputs.
        """
        src, dst = edge_index[0], edge_index[1]  # each [E]
        B, N, D = h.shape

        # ------------------------------------------------------------------
        # Edge update (Eq. 3)
        # g^(l+1)_{(src,dst)} = f^e( cat( g^l_{(src,dst)},  h^l_src ) )
        # ------------------------------------------------------------------
        h_src = h[:, src, :]                           # [B, E, D]
        g_new = self.f_e(torch.cat([g, h_src], dim=-1))  # [B, E, D]

        # ------------------------------------------------------------------
        # Node update (Eq. 4)
        # Aggregate incoming edge messages at each destination node via
        # scatter-add, then add the self-transformed node hidden.
        # ------------------------------------------------------------------
        dst_idx = dst.view(1, -1, 1).expand(B, -1, D)  # [B, E, D]
        agg = h.new_zeros(B, N, D)                      # [B, N, D]
        agg.scatter_add_(1, dst_idx, g_new)             # accumulate at dst

        h_new = F.relu(self.f_v(h) + agg)              # [B, N, D]

        return h_new, g_new


# ---------------------------------------------------------------------------
# Full Backtracking Network
# ---------------------------------------------------------------------------

class BacktrackingNetwork(torch.nn.Module):
    """
    Backtracking Network (BN) — Ru et al., AAAI 2023, Figure 2.

    Args:
        node_feat_dim:  Dimension of node features (3 for one-hot SIR states).
        edge_feat_dim:  Dimension of edge features = number of time slices T
                        (binary activation pattern per edge).
        hidden_dim:     Hidden embedding dimension D.
        num_layers:     Number of BN convolutional layers L.
    """

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        # Initial projections (p^v and p^e from the paper)
        self.p_v = Sequential(Linear(node_feat_dim, hidden_dim), ReLU())
        self.p_e = Sequential(Linear(edge_feat_dim, hidden_dim), ReLU())

        # L BN convolution layers
        self.convs = ModuleList([BNConvLayer(hidden_dim) for _ in range(num_layers)])

        # Final linear projection h^L_i → scalar score  (the "another projection
        # layer" h_i ∈ ℝ mentioned just before Eq. 5)
        self.final = Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, 3]  one-hot node states at final time
        edge_index: torch.Tensor,  # [2, E]     aggregated graph G̃_a
        edge_attr: torch.Tensor,   # [E, T]     binary activation patterns X_e
    ) -> torch.Tensor:
        """
        Args:
            x:          Node feature matrix. Each row is a one-hot SIR state:
                        susceptible=[1,0,0], infectious=[0,1,0], recovered=[0,0,1].
                        Shape: [B, N, 3]  (batched) or [N, 3] (single sample).
            edge_index: COO edge index of the aggregated (static) network G̃_a.
                        Shape: [2, E].
            edge_attr:  Binary vector per edge recording at which time slices the
                        edge was active (X_e from the paper).
                        Shape: [E, T].

        Returns:
            Log-probabilities over nodes. Shape: [B, N] or [N] (single sample).
            The node with the highest value is the inferred patient zero.
        """
        # Support both batched [B, N, 3] and unbatched [N, 3] input
        unbatched = x.dim() == 2
        if unbatched:
            x = x.unsqueeze(0)          # [1, N, 3]
        B, N, _ = x.shape

        # ------------------------------------------------------------------
        # 1. Initial projections
        # ------------------------------------------------------------------
        h = self.p_v(x)                                         # [B, N, D]
        g = self.p_e(edge_attr)                                 # [E,  D]
        # Edge hiddens start shared across batch items; they diverge at layer 1
        # because f^e uses per-sample node hiddens h^l_src (see Eq. 3).
        g = g.unsqueeze(0).expand(B, -1, -1).contiguous()      # [B, E, D]

        # ------------------------------------------------------------------
        # 2. L layers of BN convolution
        # ------------------------------------------------------------------
        for conv in self.convs:
            h, g = conv(h, g, edge_index)

        # ------------------------------------------------------------------
        # 3. Final projection to scalar score per node
        # ------------------------------------------------------------------
        scores = self.final(h).squeeze(-1)                      # [B, N]

        # ------------------------------------------------------------------
        # 4. Expert knowledge (Eq. 6)
        # Susceptible nodes (state index 0 in the one-hot encoding) cannot be
        # the source: their score is set to −∞ so softmax maps them to 0.
        # ------------------------------------------------------------------
        susceptible_mask = x[..., 0].bool()                     # [B, N]
        scores = scores.masked_fill(susceptible_mask, float('-inf'))

        # ------------------------------------------------------------------
        # 5. Log-softmax → log detection probabilities (Eq. 5)
        # ------------------------------------------------------------------
        log_probs = F.log_softmax(scores, dim=-1)               # [B, N]

        if unbatched:
            log_probs = log_probs.squeeze(0)                    # [N]

        return log_probs
