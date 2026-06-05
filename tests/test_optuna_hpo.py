from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from hpo.search_space import apply_trial_params, describe_search_space, suggest_hyperparameters
from main_optuna import _truth_indices_for_rep, resolve_truth_budget


class FirstChoiceTrial:
    """Deterministic Optuna-like trial for search-space unit tests."""

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, name, low, high, *, log=False):
        return low

    def suggest_int(self, name, low, high):
        return low


def test_search_spaces_cover_model_specific_parameters():
    expected = {
        "static_gnn": "static_gnn.use_edge_weights",
        "static_mlp": "static_mlp.num_hidden_layers",
        "backtracking": "backtracking.num_layers",
        "temporal_gnn": "temporal_gnn.group_by_time",
        "dag_gnn": "dag_gnn.delta_t",
        "dbgnn": "dbgnn.time_bin_size",
    }
    for model, key in expected.items():
        space = describe_search_space(model)
        assert "train.lr" in space
        assert "train.weight_decay" in space
        assert key in space


def test_suggested_parameters_patch_nested_config():
    cfg = {
        "model": "static_gnn",
        "train": {"lr": 0.001, "batch_size": 128},
        "static_gnn": {"hidden_channels": 64, "use_edge_weights": False},
    }
    params = suggest_hyperparameters(
        FirstChoiceTrial(),
        cfg,
        "static_gnn",
        max_batch_size=64,
    )
    apply_trial_params(cfg, params)

    assert cfg["train"]["batch_size"] <= 64
    assert cfg["train"]["lr"] == pytest.approx(1e-4)
    assert cfg["static_gnn"]["use_edge_weights"] is False
    assert cfg["static_gnn"]["hidden_channels"] == 16


def test_truth_budget_keeps_hpo_and_final_windows_disjoint():
    budget = resolve_truth_budget(
        data_n_runs=1000,
        eval_cfg={"n_truth": 1000},
        train_cfg={"reps": 1},
        hpo_cfg={"reps": 1, "n_truth": None, "truth_start": 0, "final_truth_start": None, "final_n_truth": None},
    )

    assert budget.hpo_truth_start == 0
    assert budget.hpo_n_truth == 500
    assert budget.final_truth_start == 500
    assert budget.final_n_truth == 500


def test_truth_indices_respect_eval_truth_start():
    cfg = {"truth_start": 10}
    assert _truth_indices_for_rep(cfg, rep=0, n_truth=5, n_runs=30, reps=2).tolist() == [10, 11, 12, 13, 14]
    assert _truth_indices_for_rep(cfg, rep=1, n_truth=5, n_runs=30, reps=2).tolist() == [15, 16, 17, 18, 19]
    with pytest.raises(ValueError):
        _truth_indices_for_rep(cfg, rep=0, n_truth=20, n_runs=30, reps=2)


def test_optuna_dry_run_does_not_require_optuna_dependency():
    result = subprocess.run(
        [
            sys.executable,
            "main_optuna.py",
            "--dry-run",
            "--cfg",
            "exp/france_office/static_gnn.yml",
            "--data",
            "france_office:latest",
            "--study-name",
            "dry_static",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Optuna HPO dry run" in result.stdout
    assert "static_gnn.hidden_channels" in result.stdout


def test_runner_with_hpo_dry_run_adds_hpo_stage(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "run_all_experiments.py",
            "--dry-run",
            "--with-hpo",
            "--hpo-trials",
            "2",
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

    with open(tmp_path / "dry" / "status.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["stage"] for row in rows] == [
        "tsir",
        "hpo",
        "train",
        "train",
        "hpo",
        "train",
        "train",
        "eval",
    ]
    assert [row["model"] for row in rows if row["stage"] == "hpo"] == [
        "static_gnn_optuna",
        "static_mlp_optuna",
    ]
    assert [row["model"] for row in rows if row["stage"] == "train"] == [
        "static_gnn",
        "static_gnn_optuna",
        "static_mlp",
        "static_mlp_optuna",
    ]
