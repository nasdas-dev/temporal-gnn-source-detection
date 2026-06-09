from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from main_optuna import _remaining_trials_for_target
from training import CheckpointError, SIRDataset, Trainer
from training.checkpointing import (
    assert_compatible,
    atomic_torch_save,
    compatibility_hash,
    torch_load,
)

torch.set_num_threads(1)


class TinySourceModel(torch.nn.Module):
    def __init__(self, n_nodes: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.linear = torch.nn.Linear(n_nodes * 3, n_nodes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.linear(x.reshape(x.size(0), -1))
        return torch.nn.functional.log_softmax(scores, dim=-1)


def tiny_forward(model, x_batch, graph_data, device):
    return model(x_batch.to(device))


def tiny_dataset(n_nodes: int = 4, mc_runs: int = 5) -> SIRDataset:
    mc_S = np.ones((n_nodes, mc_runs, n_nodes), dtype=np.int8)
    mc_I = np.zeros_like(mc_S)
    mc_R = np.zeros_like(mc_S)
    for source in range(n_nodes):
        mc_S[source, :, source] = 0
        mc_I[source, :, source] = 1
    return SIRDataset(mc_S, mc_I, mc_R)


def test_checkpoint_round_trip_and_compatibility(tmp_path):
    metadata = {"model": "tiny", "rep": 0}
    compat = compatibility_hash(metadata)
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"compatibility_hash": compat, "value": 3}, path)

    payload = torch_load(path)

    assert payload["value"] == 3
    assert_compatible(payload, compat, path)
    with pytest.raises(CheckpointError):
        assert_compatible(payload, compatibility_hash({"model": "other"}), path)


def test_trainer_fit_resumes_from_latest_checkpoint(tmp_path):
    dataset = tiny_dataset()
    graph_data = {"n_nodes": 4}
    ckpt_dir = tmp_path / "tiny"

    torch.manual_seed(11)
    trainer = Trainer(TinySourceModel(4), tiny_forward, graph_data, torch.device("cpu"))
    train_losses, val_losses = trainer.fit(
        dataset=dataset,
        batch_size=4,
        epochs=5,
        patience=99,
        lr=0.01,
        weight_decay=0.0,
        test_size=0.25,
        seed=7,
        checkpoint_dir=ckpt_dir,
        checkpoint_metadata={"case": "resume"},
        final_model_path=ckpt_dir / "final_model_rep0.pt",
    )

    assert len(train_losses) == 5
    assert len(val_losses) == 5
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    assert (ckpt_dir / "final_model_rep0.pt").exists()

    latest = torch_load(ckpt_dir / "latest.pt")
    latest["epoch"] = 3
    latest["train_losses"] = latest["train_losses"][:3]
    latest["val_losses"] = latest["val_losses"][:3]
    atomic_torch_save(latest, ckpt_dir / "latest.pt")

    resumed = Trainer(TinySourceModel(4), tiny_forward, graph_data, torch.device("cpu"))
    train_losses, val_losses = resumed.fit(
        dataset=dataset,
        batch_size=4,
        epochs=5,
        patience=99,
        lr=0.01,
        weight_decay=0.0,
        test_size=0.25,
        seed=7,
        checkpoint_dir=ckpt_dir,
        checkpoint_metadata={"case": "resume"},
        final_model_path=ckpt_dir / "final_model_rep0.pt",
    )

    assert resumed.last_fit_info["resumed"] is True
    assert len(train_losses) == 5
    assert len(val_losses) == 5
    assert torch_load(ckpt_dir / "latest.pt")["epoch"] == 5

    incompatible = Trainer(TinySourceModel(4), tiny_forward, graph_data, torch.device("cpu"))
    with pytest.raises(CheckpointError):
        incompatible.fit(
            dataset=dataset,
            batch_size=4,
            epochs=5,
            patience=99,
            lr=0.01,
            weight_decay=0.0,
            test_size=0.25,
            seed=7,
            checkpoint_dir=ckpt_dir,
            checkpoint_metadata={"case": "different"},
            final_model_path=ckpt_dir / "final_model_rep0.pt",
        )


def test_remaining_trials_targets_total_finished_count():
    trials = [
        SimpleNamespace(state="COMPLETE"),
        SimpleNamespace(state="PRUNED"),
        SimpleNamespace(state="RUNNING"),
    ]
    terminal = {"COMPLETE", "PRUNED", "FAIL"}

    assert _remaining_trials_for_target(trials, terminal, 5) == 3
    assert _remaining_trials_for_target(trials, terminal, 2) == 0


def test_runner_dry_run_passes_stable_checkpoint_dir(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "run_all_experiments.py",
            "--dry-run",
            "--no-hpo",
            "--skip-viz",
            "--skip-tables",
            "--preset",
            "fast",
            "--networks",
            "france_office",
            "--r0",
            "1.0",
            "--models",
            "static_gnn",
            "--output",
            str(tmp_path),
            "--run-name",
            "dry",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expected = tmp_path / "dry" / "checkpoints" / "france_office" / "r0_10" / "static_gnn"
    assert "--checkpoint-dir" in result.stdout
    assert str(expected) in result.stdout
