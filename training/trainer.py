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
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from .checkpointing import (
    array_fingerprint,
    assert_compatible,
    atomic_json_dump,
    atomic_torch_save,
    capture_rng_state,
    checkpoint_timestamp,
    compatibility_hash,
    restore_rng_state,
    torch_load,
)
from .data import SIRDataset


def _move_graph_data_to_device(value, device: torch.device):
    """Recursively move static graph tensors to the training device once."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_graph_data_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_graph_data_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_graph_data_to_device(v, device) for v in value)
    return value


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
    # Judge the run by its BEST validation epoch, not the latest one: the trainer
    # restores best-val weights before inference, so a transient spike in a noisy
    # loss is not a failed run. Models like BacktrackingNetwork have an unstable /
    # oscillating loss but still hit good minima that get checkpointed; slow
    # starters sit high for many epochs while still descending. Abort only when
    # even the best val stays above the uniform baseline AND is no longer
    # improving — i.e. the model never actually learned.
    best_val = min(val_losses)
    if best_val > cfg.divergence_factor * uniform_loss:
        window = max(2, cfg.warmup_epochs // 2)
        recent = val_losses[-window:]
        if len(recent) < window or (recent[0] - min(recent)) < cfg.min_improvement:
            return "divergent_validation_loss"

    if len(val_losses) >= cfg.uniform_window:
        recent = val_losses[-cfg.uniform_window:]
        mean_recent = float(np.mean(recent))
        near_uniform = abs(mean_recent - uniform_loss) <= cfg.uniform_tolerance * uniform_loss
        improvement = recent[0] - min(recent)
        if near_uniform and improvement < cfg.min_improvement:
            return "uniform_stall"

    return None


def make_train_val_split(
    dataset: SIRDataset,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic stratified split used by Trainer.fit."""
    n_total = len(dataset)
    indices = np.arange(n_total)
    labels = dataset.y.numpy()
    tr_idx, va_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )
    return np.asarray(tr_idx, dtype=np.int64), np.asarray(va_idx, dtype=np.int64)


def fit_compatibility_metadata(
    *,
    dataset: SIRDataset,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    test_size: float,
    seed: int,
    checkpoint_metadata: dict | None,
) -> tuple[dict, np.ndarray, np.ndarray, str]:
    """Return compatibility metadata, split indices, and its hash."""
    tr_idx, va_idx = make_train_val_split(dataset, test_size, seed)
    metadata = {
        **(checkpoint_metadata or {}),
        "trainer": {
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "patience": int(patience),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "test_size": float(test_size),
            "seed": int(seed),
            "n_total": int(len(dataset)),
            "train_indices": array_fingerprint(tr_idx),
            "val_indices": array_fingerprint(va_idx),
        },
    }
    return metadata, tr_idx, va_idx, compatibility_hash(metadata)


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
    B, N, num_features = x_batch.shape
    edge_index  = graph_data["edge_index"].to(device)
    edge_weight = graph_data.get("edge_weight")
    E = edge_index.size(1)

    x = x_batch.reshape(B * N, num_features).to(device)

    offsets = torch.arange(B, device=device) * N
    offsets = offsets.repeat_interleave(E)
    batched_ei = edge_index.repeat(1, B) + offsets.unsqueeze(0)

    batched_ew: torch.Tensor | None = None
    if edge_weight is not None:
        batched_ew = edge_weight.to(device).repeat(B)

    batch_vec = torch.arange(B, device=device).repeat_interleave(N)

    return model(x, batched_ei, batched_ew, batch_vec)  # [B, N]


def static_mlp_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, F]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    """Graph-free batching for the flattened-observation MLP baseline."""
    return model(x_batch.to(device, non_blocking=True))


def backtracking_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    x = x_batch.to(device, non_blocking=True)
    edge_index = graph_data["edge_index"]
    edge_attr = graph_data.get("edge_attr")
    if edge_attr is not None:
        return model(x, edge_index, edge_attr=edge_attr)  # [B, N]
    return model(
        x,
        edge_index,
        edge_time_index=graph_data["edge_time_index"],
        edge_time_edge_index=graph_data["edge_time_edge_index"],
        n_edges=graph_data["n_edges"],
    )  # [B, N]


def temporal_gnn_forward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,   # [B, N, 3]
    graph_data: dict,
    device: torch.device,
) -> torch.Tensor:           # [B, N]
    """Vectorized TemporalGNN batching.

    Each time-slice graph is replicated B times as one disconnected PyG graph,
    so each temporal layer is evaluated once per batch instead of once per
    sample. This keeps the primitive temporal baseline faithful while avoiding
    pathological Python-loop runtime on real networks.
    """
    B, N, num_features = x_batch.shape
    x = x_batch.reshape(B * N, num_features).to(device)
    edge_indeces = {
        t: ei.to(device) for t, ei in graph_data["edge_indeces"].items()
    }
    time_order = list(graph_data.get("time_order", sorted(edge_indeces)))

    x = model.encode_input(x)
    history = [x]
    offsets = torch.arange(B, device=device) * N
    for count, t in enumerate(reversed(time_order)):
        edge_index = edge_indeces[t]
        E = edge_index.size(1)
        batch_offsets = offsets.repeat_interleave(E)
        batched_ei = edge_index.repeat(1, B) + batch_offsets.unsqueeze(0)
        x = model.apply_temporal_layer(count, x, batched_ei)
        history.append(x)

    scores = model.score_nodes(x, history).view(B, N)
    return F.log_softmax(scores, dim=-1)


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
    db_node_last        = graph_data["db_node_last"].to(device)
    static_edge_index   = graph_data["static_edge_index"].to(device)
    static_edge_weight  = graph_data["static_edge_weight"].to(device)
    return model(x, db_edge_index, db_edge_weight, db_node_to_original, db_node_last,
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
        self.device     = device
        self.graph_data = _move_graph_data_to_device(graph_data, device)
        self.last_fit_info: dict = {}

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
        grad_clip_norm: float | None = None,
        wandb_run=None,
        rep: int = 0,
        loss_guard: dict | None = None,
        optuna_trial=None,
        optuna_report_sign: float = 1.0,
        optuna_step_offset: int = 0,
        checkpoint_dir: str | Path | None = None,
        checkpoint_metadata: dict | None = None,
        checkpoint_enabled: bool = True,
        checkpoint_resume: bool = True,
        checkpoint_fresh: bool = False,
        checkpoint_save_every: int = 1,
        final_model_path: str | Path | None = None,
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
        optuna_trial:
            Optional Optuna trial.  When provided, validation loss is reported
            after each epoch so Optuna pruners can stop unpromising trials.
        optuna_report_sign:
            Multiplier for the reported validation loss.  Use ``-1`` for
            maximize studies so lower validation NLL becomes a larger
            intermediate value.
        optuna_step_offset:
            Offset added to epoch numbers when reporting repeated fits in the
            same trial.

        Returns
        -------
        train_losses, val_losses:
            Per-epoch average NLL (one value per epoch trained).
        """
        fit_metadata, tr_idx, va_idx, fit_hash = fit_compatibility_metadata(
            dataset=dataset,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            lr=lr,
            weight_decay=weight_decay,
            test_size=test_size,
            seed=seed,
            checkpoint_metadata=checkpoint_metadata,
        )

        train_loader = DataLoader(
            Subset(dataset, tr_idx),
            batch_size = batch_size,
            shuffle    = True,
            num_workers = 0,
            pin_memory = self.device.type == "cuda",
        )
        val_loader = DataLoader(
            Subset(dataset, va_idx),
            batch_size = batch_size,
            shuffle    = False,
            num_workers = 0,
            pin_memory = self.device.type == "cuda",
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_losses: list[float] = []
        val_losses:   list[float] = []
        best_val      = float("inf")
        best_epoch    = 0
        best_state    = None
        patience_ctr  = 0
        guard_cfg     = make_loss_guard_config(loss_guard)
        n_nodes       = int(self.graph_data.get("n_nodes", 1))
        start_epoch   = 1
        stopped_early = False
        checkpoint_save_every = max(1, int(checkpoint_save_every))
        ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        latest_path = ckpt_dir / "latest.pt" if ckpt_dir is not None else None
        best_path = ckpt_dir / "best.pt" if ckpt_dir is not None else None
        final_path = Path(final_model_path) if final_model_path is not None else None

        def _base_payload(epoch: int) -> dict:
            return {
                "version": 1,
                "kind": "trainer_epoch",
                "saved_at": checkpoint_timestamp(),
                "compatibility_hash": fit_hash,
                "metadata": fit_metadata,
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_val": float(best_val),
                "patience_ctr": int(patience_ctr),
                "train_losses": list(train_losses),
                "val_losses": list(val_losses),
                "train_indices": tr_idx,
                "val_indices": va_idx,
                "rng_state": capture_rng_state(),
            }

        def _save_manifest(status: str, epoch: int) -> None:
            if ckpt_dir is None:
                return
            atomic_json_dump(
                {
                    "status": status,
                    "compatibility_hash": fit_hash,
                    "updated_at": checkpoint_timestamp(),
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    "best_val": None if best_val == float("inf") else float(best_val),
                    "latest": str(latest_path) if latest_path is not None else "",
                    "best": str(best_path) if best_path is not None else "",
                    "final_model": str(final_path) if final_path is not None else "",
                    "metadata": fit_metadata,
                },
                ckpt_dir / "manifest.json",
            )

        def _save_latest(epoch: int, status: str = "training") -> None:
            if not checkpoint_enabled or latest_path is None:
                return
            payload = {
                **_base_payload(epoch),
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "status": status,
            }
            atomic_torch_save(payload, latest_path)
            _save_manifest(status, epoch)

        def _save_best(epoch: int) -> None:
            if not checkpoint_enabled or best_path is None:
                return
            payload = {
                **_base_payload(epoch),
                "kind": "trainer_best",
                "model_state": self.model.state_dict(),
                "status": "best",
            }
            atomic_torch_save(payload, best_path)

        if (
            checkpoint_enabled
            and checkpoint_resume
            and not checkpoint_fresh
            and latest_path is not None
            and latest_path.exists()
        ):
            payload = torch_load(latest_path, map_location=self.device)
            assert_compatible(payload, fit_hash, latest_path)
            self.model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            restore_rng_state(payload.get("rng_state"))
            train_losses = list(payload.get("train_losses", []))
            val_losses = list(payload.get("val_losses", []))
            best_val = float(payload.get("best_val", best_val))
            best_epoch = int(payload.get("best_epoch", 0))
            patience_ctr = int(payload.get("patience_ctr", 0))
            start_epoch = int(payload.get("epoch", 0)) + 1
            print(
                f"  Resuming from {latest_path} at epoch {start_epoch} "
                f"(best val={best_val:.4f})"
            )
        elif checkpoint_enabled and checkpoint_fresh and ckpt_dir is not None:
            _save_manifest("fresh_start", 0)

        for epoch in range(start_epoch, epochs + 1):
            # --- Train ---
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.float()
                y_batch = y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                out  = self._forward(x_batch)
                loss = F.nll_loss(out, y_batch, reduction="mean")
                loss.backward()
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                optimizer.step()
                train_loss += loss.item() * y_batch.size(0)

            # --- Validate ---
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.float()
                    y_batch = y_batch.to(self.device, non_blocking=True)
                    out      = self._forward(x_batch)
                    val_loss += (
                        F.nll_loss(out, y_batch, reduction="mean").item()
                        * y_batch.size(0)
                    )

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

            if optuna_trial is not None:
                optuna_trial.report(optuna_report_sign * vl, step=optuna_step_offset + epoch)
                if optuna_trial.should_prune():
                    import optuna

                    raise optuna.TrialPruned(
                        f"pruned at epoch {epoch}: val_loss={vl:.6g}"
                    )

            # Early stopping
            if vl < best_val:
                best_val   = vl
                best_epoch = epoch
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_ctr = 0
                _save_best(epoch)
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"  Early stopping at epoch {epoch} (best val={best_val:.4f})")
                    stopped_early = True
                    _save_latest(epoch, status="early_stopped")
                    break

            if epoch % checkpoint_save_every == 0 or epoch == epochs:
                _save_latest(epoch)

        # Restore best weights
        if checkpoint_enabled and best_path is not None and best_path.exists():
            payload = torch_load(best_path, map_location=self.device)
            assert_compatible(payload, fit_hash, best_path)
            self.model.load_state_dict(payload["model_state"])
        elif best_state is not None:
            self.model.load_state_dict(best_state)

        epochs_trained = len(train_losses)
        if checkpoint_enabled and final_path is not None:
            final_payload = {
                "version": 1,
                "kind": "final_model",
                "saved_at": checkpoint_timestamp(),
                "compatibility_hash": fit_hash,
                "metadata": fit_metadata,
                "model_state": self.model.state_dict(),
                "best_epoch": int(best_epoch),
                "best_val": None if best_val == float("inf") else float(best_val),
                "epochs_trained": int(epochs_trained),
                "stopped_early": bool(stopped_early),
                "train_losses": list(train_losses),
                "val_losses": list(val_losses),
                "train_indices": tr_idx,
                "val_indices": va_idx,
            }
            atomic_torch_save(final_payload, final_path)
            _save_manifest("trained", max(start_epoch - 1, epochs_trained))

        self.last_fit_info = {
            "compatibility_hash": fit_hash,
            "metadata": fit_metadata,
            "train_indices": tr_idx,
            "val_indices": va_idx,
            "best_epoch": int(best_epoch),
            "best_val": None if best_val == float("inf") else float(best_val),
            "epochs_trained": int(epochs_trained),
            "checkpoint_dir": str(ckpt_dir) if ckpt_dir is not None else "",
            "latest_checkpoint": str(latest_path) if latest_path is not None else "",
            "best_checkpoint": str(best_path) if best_path is not None else "",
            "final_model": str(final_path) if final_path is not None else "",
            "resumed": bool(start_epoch > 1),
        }
        return train_losses, val_losses

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
