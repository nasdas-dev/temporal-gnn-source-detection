"""Static MLP baseline for source detection.

The model intentionally ignores graph structure.  Each SIR observation is the
full network snapshot flattened into one vector, and the network predicts a
probability distribution over candidate source nodes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Linear


class StaticMLP(torch.nn.Module):
    """Graph-free baseline over flattened SIR snapshots.

    Parameters
    ----------
    num_preprocess_layers:
        Number of fully connected preprocessing layers before the main hidden
        block.
    embed_dim_preprocess:
        Width of preprocessing layers.
    num_postprocess_layers:
        Number of fully connected layers after the main hidden block.
    num_hidden_layers:
        Number of hidden fully connected layers.
    num_node_features:
        Number of SIR channels per node. This is normally 3.
    n_nodes:
        Number of source classes and nodes in the snapshot.
    hidden_channels:
        Width of hidden and postprocessing layers.
    dropout_rate:
        Dropout probability after hidden activations.
    batch_norm:
        Whether to use batch normalisation after linear layers.
    skip:
        Whether to concatenate the raw flattened input before the final layer.
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
        self.input_dim = n_nodes * num_node_features

        self.preprocess = torch.nn.Sequential()
        self.hidden = torch.nn.Sequential()
        self.postprocess = torch.nn.Sequential()
        self.final = torch.nn.Sequential()

        for layer_idx in range(num_preprocess_layers):
            in_features = self.input_dim if layer_idx == 0 else embed_dim_preprocess
            self.preprocess.append(Linear(in_features, embed_dim_preprocess))
            if batch_norm:
                self.preprocess.append(torch.nn.BatchNorm1d(embed_dim_preprocess))
            self.preprocess.append(torch.nn.PReLU())
            self.preprocess.append(torch.nn.Dropout(p=dropout_rate))

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

        for _ in range(num_postprocess_layers):
            self.postprocess.append(Linear(hidden_channels, hidden_channels))
            if batch_norm:
                self.postprocess.append(torch.nn.BatchNorm1d(hidden_channels))
            self.postprocess.append(torch.nn.PReLU())
            self.postprocess.append(torch.nn.Dropout(p=dropout_rate))

        final_in = hidden_channels + self.input_dim if skip else hidden_channels
        self.final.append(Linear(final_in, n_nodes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return log-probabilities over source nodes.

        Parameters
        ----------
        x:
            Tensor of shape ``[batch_size, n_nodes, num_node_features]``.
        """
        if x.dim() != 3:
            raise ValueError(
                f"StaticMLP expects [B, N, F] input, got shape {tuple(x.shape)}"
            )
        x_flat = x.reshape(x.size(0), self.input_dim)
        h = self.preprocess(x_flat)
        h = self.hidden(h)
        h = self.postprocess(h)
        if self.skip:
            h = torch.cat([x_flat, h], dim=-1)
        return F.log_softmax(self.final(h), dim=1)
