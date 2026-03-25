"""
Dataset utilities for SIR-snapshot training data.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SIRDataset(Dataset):
    """Index-based dataset over SIR simulation snapshots.

    Stores all samples in memory as float32 tensors of shape ``[N, 3]`` (one
    per sample).  Labels are the source-node indices.

    Parameters
    ----------
    mc_S, mc_I, mc_R:
        Arrays of shape ``[n_nodes, mc_runs, n_nodes]`` (int8).
        ``mc_S[s, r, v] = 1`` means node *v* is susceptible in run *r* started
        from source *s*.
    """

    def __init__(
        self,
        mc_S: np.ndarray,
        mc_I: np.ndarray,
        mc_R: np.ndarray,
    ) -> None:
        n_nodes, mc_runs, _ = mc_S.shape
        # Flatten [n_nodes, mc_runs, n_nodes] → [n_nodes*mc_runs, n_nodes]
        S = mc_S.reshape(n_nodes * mc_runs, n_nodes)
        I = mc_I.reshape(n_nodes * mc_runs, n_nodes)
        R = mc_R.reshape(n_nodes * mc_runs, n_nodes)

        # Stack to one-hot [n_nodes*mc_runs, n_nodes, 3]
        self.X: torch.Tensor = torch.tensor(
            np.stack([S, I, R], axis=-1), dtype=torch.float32
        )
        # Labels: repeat each source index mc_runs times
        self.y: torch.Tensor = torch.tensor(
            np.repeat(np.arange(n_nodes), mc_runs), dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
