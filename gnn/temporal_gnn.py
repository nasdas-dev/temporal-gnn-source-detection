import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class TemporalGNN(torch.nn.Module):
    """Primitive temporal GNN: one SAGEConv layer per time-slice, applied in reverse.

    Designed for 2D input x of shape [N, in_channels] (single sample).
    Batching is handled externally by temporal_gnn_forward in training/trainer.py,
    which loops over batch items to keep each SAGEConv call strictly 2D.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_snapshots: int) -> None:
        super().__init__()
        self.lin_pre  = torch.nn.Linear(in_channels, hidden_channels)
        self.convs     = torch.nn.ModuleList(
            [SAGEConv(hidden_channels, hidden_channels) for _ in range(num_snapshots)]
        )
        self.lin_post  = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_indeces: dict) -> torch.Tensor:
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
        x = F.relu(self.lin_pre(x))                          # [N, hidden]
        for count, t in enumerate(reversed(edge_indeces.keys())):
            x = F.relu(self.convs[count](x, edge_indeces[t]))  # [N, hidden]
        x = self.lin_post(x).squeeze(-1)                     # [N]
        return F.log_softmax(x, dim=-1)                      # [N]


