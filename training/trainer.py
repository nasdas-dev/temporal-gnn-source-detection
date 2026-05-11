"""
Unified training and inference loop for all source-detection models.

The Trainer is model-agnostic: it accepts a ``forward_fn`` (from the model
registry) that knows how to call the specific model correctly, regardless of
whether it uses PyG-style batching (StaticGNN) or internal batching
(BacktrackingNetwork, TemporalGNN).

Usage
-----
::

    from gnn import MODEL_REGISTRY
    from training import Trainer, SIRDataset

    spec       = MODEL_REGISTRY["backtracking"]
    graph_data = spec.builder_fn(H)
    model      = spec.cls(node_feat_dim=3, edge_feat_dim=graph_data["T"],
                          hidden_dim=32, num_layers=6)

    trainer = Trainer(model, spec.forward_fn, graph_data, device)
    train_losses, val_losses = trainer.fit(dataset, ...)
    probs = trainer.predict(X_truth)
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .data import SIRDataset


class LossGuardAbort(RuntimeError):
    """Raised when training loss becomes clearly unusable for this run."""

    def __init__(self, reason: str, epoch: int, train_loss: float, val_loss: float) -> None:
        self.reason = reason
        self.epoch = epoch
        self.train_loss = train_loss
        self.val_loss = val_loss
        super().__init__(
            f"{reason} at epoch {epoch}: train={train_loss:.6g}, val={val_loss:.6g}"
        )


@dataclass(frozen=True)
class LossGuardConfig:
    """Configuration for nonsensical-loss early aborts."""

    enabled: bool = True
    warmup_epochs: int = 20
    divergence_factor: float = 1.5
    uniform_tolerance: float = 0.02
    uniform_window: int = 80
    min_improvement: float = 0.01


def make_loss_guard_config(raw: dict | None) -> LossGuardConfig | None:
    """Return a typed loss-guard config, or ``None`` when disabled/missing."""
    if raw is None:
        return None
    allowed = LossGuardConfig.__dataclass_fields__.keys()
    cfg = LossGuardConfig(**{k: v for k, v in raw.items() if k in allowed})
    return cfg if cfg.enabled else None


def check_loss_guard(
    train_losses: list[float],
    val_losses: list[float],
    epoch: int,
    n_nodes: int,
    cfg: LossGuardConfig | None,
) -> str | None:
    """Return an abort reason if the current loss history is nonsensical."""
    if cfg is None:
        return None

    train_loss = train_losses[-1]
    val_loss = val_losses[-1]
    if not math.isfinite(train_loss) or not math.isfinite(val_loss):
        return "non_finite_loss"

    if epoch < cfg.warmup_epochs:
        return None

    uniform_loss = math.log(max(n_nodes, 2))
    if val_loss > cfg.divergence_factor * uniform_loss:
        return "divergent_validation_loss"

    if len(val_losses) >= cfg.uniform_window:
        recent = val_losses[-cfg.uniform_window:]
        mean_recent = float(np.mean(recent))
        near_uniform = abs(mean_recent - uniform_loss) <= cfg.uniform_tolerance * uniform_loss
        improvement = recent[0] - min(recent)
        if near_uniform and improvement < cfg.min_improvement:
            return "uniform_stall"

    return None


# ---------------------------------------------------------------------------
# Forward dispatch functions
# One per model family.  Each receives (model, x_batch [B,N,F], graph_data, device)
# and returns [B, N] log-probabilities.
# ---------------------------------------------------------------------------

def static_gnn_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, F]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    """PyG-style batching: replicate graph B times, flatten all nodes."""
    B, N, F = x_batch.shape
    edge_index  = graph_data["edge_index"].to(device)
    edge_weight = graph_data.get("edge_weight")
    E = edge_index.size(1)

    x = x_batch.reshape(B * N, F).to(device)

    offsets = torch.arange(B, device=device) * N
    offsets = offsets.repeat_interleave(E)
    batched_ei = edge_index.repeat(1, B) + offsets.unsqueeze(0)

    batched_ew: torch.Tensor | None = None
    if edge_weight is not None:
        batched_ew = edge_weight.to(device).repeat(B)

    batch_vec = torch.arange(B, device=device).repeat_interleave(N)

    return model(x, batched_ei, batched_ew, batch_vec)  # [B, N]


def backtracking_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    x          = x_batch.to(device)
    edge_index = graph_data["edge_index"].to(device)
    edge_attr  = graph_data["edge_attr"].to(device)
    return model(x, edge_index, edge_attr)  # [B, N]


def temporal_gnn_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    # SAGEConv (PyG) requires strictly 2D input [N, F].
    # We loop over the batch dimension so each call receives [N, F].
    # This is correct with gradients and works for any batch size.
    edge_indeces = {
        t: ei.to(device) for t, ei in graph_data["edge_indeces"].items()
    }
    time_order = graph_data.get("time_order")
    outputs = [
        model(x_batch[b].to(device), edge_indeces, time_order)   # [N]
        for b in range(x_batch.size(0))
    ]
    return torch.stack(outputs, dim=0)               # [B, N]


def dbgnn_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    x                   = x_batch.to(device)
    db_edge_index       = graph_data["db_edge_index"].to(device)
    db_edge_weight      = graph_data["db_edge_weight"].to(device)
    db_node_to_original = graph_data["db_node_to_original"].to(device)
    static_edge_index   = graph_data["static_edge_index"].to(device)
    static_edge_weight  = graph_data["static_edge_weight"].to(device)
    return model(x, db_edge_index, db_edge_weight, db_node_to_original,
                 static_edge_index, static_edge_weight)


def dag_gnn_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    x              = x_batch.to(device)
    dag_edge_index = graph_data["dag_edge_index"].to(device)
    event_to_node  = graph_data["event_to_node"].to(device)
    event_src_node = graph_data["event_src_node"].to(device)
    return model(x, dag_edge_index, event_to_node, event_src_node)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Train and evaluate any registered source-detection model.

    Parameters
    ----------
    model:
        Instantiated (but not yet trained) ``torch.nn.Module``.
    forward_fn:
        One of ``static_gnn_forward``, ``backtracking_forward``, etc.
        Retrieved via ``MODEL_REGISTRY[name].forward_fn``.
    graph_data:
        Dict returned by the model's builder function.  Tensors stay on CPU
        here; forward functions move them to ``device`` as needed.
    device:
        Target device for training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        forward_fn,
        graph_data: dict,
        device: torch.device,
    ) -> None:
        self.model      = model.to(device)
        self.forward_fn = forward_fn
        self.graph_data = graph_data
        self.device     = device

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward(self, x_batch: torch.Tensor) -> torch.Tensor:
        """x_batch: [B, N, F] on CPU → [B, N] log-probs."""
        return self.forward_fn(self.model, x_batch, self.graph_data, self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: SIRDataset,
        batch_size: int     = 128,
        epochs:     int     = 500,
        patience:   int     = 5,
        lr:         float   = 1e-3,
        weight_decay: float = 5e-4,
        test_size:  float   = 0.30,
        seed:       int     = 42,
        wandb_run=None,
        rep: int = 0,
        loss_guard: dict | None = None,
    ) -> tuple[list[float], list[float]]:
        """Train with early stopping on validation NLL.

        Parameters
        ----------
        dataset:
            A ``SIRDataset`` containing all MC training samples.
        wandb_run:
            Optional W&B run object.  If provided, per-epoch losses are logged
            as ``train/loss_rep{rep}`` and ``val/loss_rep{rep}``.
        rep:
            Repetition index (for W&B key naming).

        Returns
        -------
        train_losses, val_losses:
            Per-epoch average NLL (one value per epoch trained).
        """
        n_total = len(dataset)
        indices = np.arange(n_total)
        labels  = dataset.y.numpy()

        tr_idx, va_idx = train_test_split(
            indices,
            test_size    = test_size,
            stratify     = labels,
            random_state = seed,
        )

        train_loader = DataLoader(
            Subset(dataset, tr_idx),
            batch_size = batch_size,
            shuffle    = True,
            num_workers = 0,
        )
        val_loader = DataLoader(
            Subset(dataset, va_idx),
            batch_size = batch_size,
            shuffle    = False,
            num_workers = 0,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val      = float("inf")
        best_state    = None
        patience_ctr  = 0
        guard_cfg     = make_loss_guard_config(loss_guard)
        n_nodes       = int(self.graph_data.get("n_nodes", 1))

        for epoch in range(1, epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float()
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                out  = self._forward(x_batch)
                loss = F.nll_loss(out, y_batch, reduction="sum")
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.float()
                    y_batch = y_batch.to(self.device)
                    out      = self._forward(x_batch)
                    val_loss += F.nll_loss(out, y_batch, reduction="sum").item()

            tl = train_loss / len(tr_idx)
            vl = val_loss   / len(va_idx)
            train_losses.append(tl)
            val_losses.append(vl)

            if (epoch % 20 == 0) or epoch == 1:
                print(f"  [{epoch:>4}/{epochs}]  train={tl:.4f}  val={vl:.4f}")

            if wandb_run is not None:
                wandb_run.log({
                    f"train/loss_rep{rep}": tl,
                    f"val/loss_rep{rep}":   vl,
                    "epoch": epoch,
                })

            guard_reason = check_loss_guard(
                train_losses, val_losses, epoch, n_nodes, guard_cfg
            )
            if guard_reason is not None:
                raise LossGuardAbort(guard_reason, epoch, tl, vl)

            # Early stopping
            if vl < best_val:
                best_val   = vl
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val={best_val:.4f})")
                    break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return train_losses, val_losses

    def predict(
        self,
        X_truth: np.ndarray,    # [n_nodes, n_runs, n_nodes] int8
        batch_size: int = 256,
    ) -> np.ndarray:
        """Run inference on ground-truth snapshots.

        Parameters
        ----------
        X_truth:
            SIR state array of shape ``[n_nodes, n_runs, n_nodes]`` (int8).
            ``X_truth[s, r, v]`` is the one-hot SIR index for node *v* in run
            *r* started from source *s*.  Typically built from stacking
            ``truth_S``, ``truth_I``, ``truth_R``.

        Returns
        -------
        probs : ndarray of shape ``[n_nodes * n_runs, n_nodes]``
            Predicted **probabilities** (not log-probs) over candidate source
            nodes.  Row order: all runs of source 0, then source 1, etc.
        """
        n_nodes, n_runs, _ = X_truth.shape
        n_total = n_nodes * n_runs

        # Build one-hot tensor [n_total, n_nodes, 3]
        X_flat = X_truth.reshape(n_total, n_nodes)  # int8 state index
        # X_truth entries are already 0/1 per-state channel — stack S/I/R
        # (this function receives the stacked one-hot directly as a [n,r,v] array
        # where each value is already the channel indicator — see calling convention)
        # We expect X_truth to be passed as stacked [n_total, n_nodes, 3] float.

        raise RuntimeError(
            "predict() must receive X_truth as [n_total, n_nodes, 3] float tensor. "
            "Use predict_from_tensor() instead."
        )

    def predict_from_tensor(
        self,
        truth_S: np.ndarray,    # [n_nodes, n_runs, n_nodes] int8
        truth_I: np.ndarray,
        truth_R: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Run inference on all (source, run) ground-truth combinations.

        Parameters
        ----------
        truth_S, truth_I, truth_R:
            Arrays of shape ``[n_nodes, n_runs, n_nodes]`` (int8).

        Returns
        -------
        probs : ndarray of shape ``[n_nodes * n_runs, n_nodes]``
            Predicted probabilities (softmax, not log) over source nodes.
        """
        n_nodes, n_runs, _ = truth_S.shape
        n_total = n_nodes * n_runs

        # Stack to [n_total, n_nodes, 3]
        S = truth_S.reshape(n_total, n_nodes)
        I = truth_I.reshape(n_total, n_nodes)
        R = truth_R.reshape(n_total, n_nodes)
        X = torch.tensor(
            np.stack([S, I, R], axis=-1), dtype=torch.float32
        )  # [n_total, n_nodes, 3]

        probs = np.zeros((n_total, n_nodes), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for start in range(0, n_total, batch_size):
                end     = min(start + batch_size, n_total)
                x_batch = X[start:end]                # [B, n_nodes, 3]
                log_p   = self._forward(x_batch)       # [B, n_nodes]
                probs[start:end] = log_p.exp().cpu().numpy()

        return probs
