"""Static MLP baseline for source detection.

This baseline intentionally ignores graph *structure* (no message passing, no
edges).  Unlike a naive flatten-the-snapshot MLP, it is **permutation
equivariant**: every node is scored by the *same* shared function of its own
SIR state plus a permutation-invariant global summary of the snapshot.

Why equivariance matters here
-----------------------------
A flattened ``[B, N*3] -> Linear(N)`` MLP ties output unit ``i`` to node index
``i``.  On a *single fixed network*, that lets the model memorise per-node
source signatures (node identity implicitly encodes degree / centrality /
position).  That is a transductive shortcut the structural GNNs deliberately do
not have, so it makes the graph-free baseline look unfairly strong.

This model removes that shortcut.  It contains no per-node-index parameters:
permuting the input nodes permutes the output logits identically and changes
nothing else.  It is therefore a fair graph-free reference — it can use the SIR
snapshot (its own state and the global S/I/R composition) but not *which* node
it is.  It is a DeepSets-style set function (Zaheer et al., 2017).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear


class StaticMLP(torch.nn.Module):
    """Permutation-equivariant, graph-free baseline over SIR snapshots.

    Each node is encoded by a shared per-node MLP, combined with a
    permutation-invariant global context (mean over nodes), and scored by a
    shared head.  No parameter is indexed by node identity.

    Parameters
    ----------
    num_preprocess_layers:
        Number of shared per-node fully connected preprocessing layers before
        the main hidden block.
    embed_dim_preprocess:
        Width of preprocessing layers.
    num_postprocess_layers:
        Number of shared per-node fully connected layers after the global
        context fusion.
    num_hidden_layers:
        Number of shared per-node hidden fully connected layers.
    num_node_features:
        Number of SIR channels per node. This is normally 3.
    n_nodes:
        Number of nodes in the snapshot. Kept for interface compatibility; the
        model is agnostic to it (no node-indexed parameters), so it also works
        when ``N`` differs between graphs.
    hidden_channels:
        Width of hidden and postprocessing layers.
    dropout_rate:
        Dropout probability after hidden activations.
    batch_norm:
        Whether to use batch normalisation after linear layers.
    skip:
        Whether to concatenate the raw per-node SIR features before the final
        layer (a per-node skip connection — still identity-free).
    """

    def __init__(
        self,
        num_preprocess_layers: int,
        embed_dim_preprocess: int,
        num_postprocess_layers: int,
        num_hidden_layers: int,
        num_node_features: int,
        n_nodes: int,
        hidden_channels: int,
        dropout_rate: float,
        batch_norm: bool,
        skip: bool,
    ) -> None:
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("StaticMLP requires at least one hidden layer")

        self.num_preprocess_layers = num_preprocess_layers
        self.num_postprocess_layers = num_postprocess_layers
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.skip = skip
        self.n_nodes = n_nodes
        self.num_node_features = num_node_features
        # Per-node feature dimension (NOT n_nodes * features): the model is
        # applied independently and identically to every node.
        self.input_dim = num_node_features

        self.preprocess = torch.nn.Sequential()
        self.hidden = torch.nn.Sequential()
        self.fuse = torch.nn.Sequential()
        self.postprocess = torch.nn.Sequential()
        self.final = torch.nn.Sequential()

        # ----------------------------------------------------------------
        # Shared per-node preprocessing
        # ----------------------------------------------------------------
        for layer_idx in range(num_preprocess_layers):
            in_features = self.input_dim if layer_idx == 0 else embed_dim_preprocess
            self.preprocess.append(Linear(in_features, embed_dim_preprocess))
            if batch_norm:
                self.preprocess.append(torch.nn.BatchNorm1d(embed_dim_preprocess))
            self.preprocess.append(torch.nn.PReLU())
            self.preprocess.append(torch.nn.Dropout(p=dropout_rate))

        # ----------------------------------------------------------------
        # Shared per-node hidden block
        # ----------------------------------------------------------------
        for layer_idx in range(num_hidden_layers):
            if layer_idx == 0:
                in_features = (
                    self.input_dim
                    if num_preprocess_layers == 0
                    else embed_dim_preprocess
                )
            else:
                in_features = hidden_channels
            self.hidden.append(Linear(in_features, hidden_channels))
            if batch_norm:
                self.hidden.append(torch.nn.BatchNorm1d(hidden_channels))
            self.hidden.append(torch.nn.PReLU())
            self.hidden.append(torch.nn.Dropout(p=dropout_rate))

        # ----------------------------------------------------------------
        # Global-context fusion (permutation invariant).
        # Each node hidden is concatenated with the mean of all node hiddens,
        # then projected back to hidden_channels.  This is the only channel
        # through which a node "sees" the rest of the snapshot, and it is
        # symmetric over nodes, so it carries no identity information.
        # ----------------------------------------------------------------
        self.fuse.append(Linear(2 * hidden_channels, hidden_channels))
        if batch_norm:
            self.fuse.append(torch.nn.BatchNorm1d(hidden_channels))
        self.fuse.append(torch.nn.PReLU())
        self.fuse.append(torch.nn.Dropout(p=dropout_rate))

        # ----------------------------------------------------------------
        # Shared per-node postprocessing
        # ----------------------------------------------------------------
        for _ in range(num_postprocess_layers):
            self.postprocess.append(Linear(hidden_channels, hidden_channels))
            if batch_norm:
                self.postprocess.append(torch.nn.BatchNorm1d(hidden_channels))
            self.postprocess.append(torch.nn.PReLU())
            self.postprocess.append(torch.nn.Dropout(p=dropout_rate))

        # ----------------------------------------------------------------
        # Shared per-node scoring head → one scalar logit per node.
        # ----------------------------------------------------------------
        final_in = hidden_channels + self.input_dim if skip else hidden_channels
        self.final.append(Linear(final_in, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities over source nodes.

        Parameters
        ----------
        x:
            Tensor of shape ``[batch_size, n_nodes, num_node_features]``.

        Returns
        -------
        torch.Tensor
            Log-softmax over nodes, shape ``[batch_size, n_nodes]``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"StaticMLP expects [B, N, F] input, got shape {tuple(x.shape)}"
            )
        B, N, F_in = x.shape

        # Apply the shared per-node stack by folding nodes into the batch dim,
        # so BatchNorm1d normalises over (batch * nodes) feature vectors.
        node_x = x.reshape(B * N, F_in)
        h = self.preprocess(node_x)
        h = self.hidden(h)                       # [B*N, hidden]
        h = h.reshape(B, N, -1)                  # [B, N, hidden]

        # Permutation-invariant global context: mean over nodes, broadcast back.
        global_ctx = h.mean(dim=1, keepdim=True).expand(-1, N, -1)  # [B, N, hidden]
        h = torch.cat([h, global_ctx], dim=-1)   # [B, N, 2*hidden]

        h = self.fuse(h.reshape(B * N, -1))      # [B*N, hidden]
        h = self.postprocess(h)                  # [B*N, hidden]

        if self.skip:
            h = torch.cat([node_x, h], dim=-1)   # per-node raw skip (identity-free)

        scores = self.final(h).reshape(B, N)     # [B, N] one logit per node
        return F.log_softmax(scores, dim=1)
