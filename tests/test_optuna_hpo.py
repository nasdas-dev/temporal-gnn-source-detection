from __future__ import annotations

import csv
import subprocess
import sys

import pytest

from hpo.search_space import (
    apply_trial_params,
    default_trial_params,
    describe_search_space,
    suggest_hyperparameters,
)
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


def test_default_trial_params_extracts_base_config_values():
    cfg = {
        "model": "static_gnn",
        "train": {"lr": 0.001, "weight_decay": 5e-4, "batch_size": 128, "test_size": 0.3},
        "static_gnn": {
            "num_preprocess_layers": 1,
            "embed_dim_preprocess": 32,
            "num_conv_layers": 4,
            "hidden_channels": 64,
            "dropout_rate": 0.1,
            "batch_norm": True,
            "skip": True,
            "use_edge_weights": False,
        },
    }
    params = default_trial_params(cfg, "static_gnn")
    assert params["train.lr"] == pytest.approx(0.001)
    assert params["train.batch_size"] == 128
    assert params["static_gnn.hidden_channels"] == 64
    assert params["static_gnn.num_conv_layers"] == 4
    assert params["static_gnn.batch_norm"] is True


def test_default_trial_params_snaps_out_of_grid_values_into_the_search_space():
    # patience=5 is not an offered choice ([10, 20, 30]); it must snap to the
    # nearest valid choice so the enqueued trial is accepted by Optuna.
    cfg = {
        "model": "static_gnn",
        "train": {"lr": 9e-3, "weight_decay": 5e-4, "batch_size": 128, "patience": 5},
        "static_gnn": {"num_conv_layers": 99, "hidden_channels": 64},
    }
    params = default_trial_params(cfg, "static_gnn")
    assert params["train.patience"] == 10              # nearest categorical choice
    assert params["train.lr"] == pytest.approx(5e-3)   # clipped into the float range
    assert params["static_gnn.num_conv_layers"] == 5   # clipped into int[2, 5]


def test_default_trial_params_respects_locked_params():
    cfg = {
        "model": "dbgnn",
        "train": {"lr": 0.001},
        "dbgnn": {"order": 3, "hidden_channels": 64},
        "hpo": {"locked_params": ["dbgnn.order"]},
    }
    params = default_trial_params(cfg, "dbgnn")
    assert "dbgnn.order" not in params
    assert params["dbgnn.hidden_channels"] == 64


def test_default_trial_params_are_consumable_by_the_suggester():
    # Every value the default extractor emits must be a value the suggester can
    # legitimately return, so an enqueued default trial reproduces the config.
    cfg = {
        "model": "static_gnn",
        "train": {"lr": 0.001, "weight_decay": 5e-4, "batch_size": 128, "test_size": 0.25, "patience": 20},
        "static_gnn": {
            "num_preprocess_layers": 1,
            "embed_dim_preprocess": 32,
            "num_postprocess_layers": 0,
            "num_conv_layers": 3,
            "aggr": "sum",
            "hidden_channels": 64,
            "dropout_rate": 0.2,
            "batch_norm": True,
            "skip": True,
            "use_edge_weights": False,
        },
    }
    optuna = pytest.importorskip("optuna")
    defaults = default_trial_params(cfg, "static_gnn")
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=0)
    )
    study.enqueue_trial(defaults, skip_if_exists=True)
    consumed: dict = {}

    def objective(trial):
        consumed.update(
            suggest_hyperparameters(
                trial, cfg, "static_gnn", max_batch_size=10**9, tune_n_mc=False, max_n_mc=500
            )
        )
        return 0.0

    study.optimize(objective, n_trials=1)
    # The enqueued default trial must have reproduced every default value.
    for key, value in defaults.items():
        assert consumed[key] == value


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
