from __future__ import annotations

import csv
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import yaml

import run_all_experiments as runner
import run_dbgnn_higher_order_experiment as ho_runner
from training.trainer import LossGuardConfig, check_loss_guard
from utils.make_de_bruijn_graph import make_de_bruijn_graph


_GRAPH_BUILDER_SPEC = importlib.util.spec_from_file_location(
    "graph_builder_direct", Path(__file__).resolve().parents[1] / "gnn" / "graph_builder.py"
)
_GRAPH_BUILDER = importlib.util.module_from_spec(_GRAPH_BUILDER_SPEC)
assert _GRAPH_BUILDER_SPEC.loader is not None
_GRAPH_BUILDER_SPEC.loader.exec_module(_GRAPH_BUILDER)
build_temporal_snapshots = _GRAPH_BUILDER.build_temporal_snapshots


def _slow_de_bruijn(events, delta=None, directed=False):
    contacts = sorted((int(u), int(v), int(t)) for u, v, t in events)
    if not directed:
        contacts = sorted(contacts + [(v, u, t) for u, v, t in contacts], key=lambda x: x[2])
    B = nx.DiGraph()
    for u, v, _ in contacts:
        B.add_node((u, v))
    for u, v, t1 in contacts:
        for v2, w, t2 in contacts:
            if v != v2 or t1 >= t2:
                continue
            if delta is not None and t2 - t1 > delta:
                continue
            src, dst = (u, v), (v2, w)
            if B.has_edge(src, dst):
                B[src][dst]["weight"] += 1
            else:
                B.add_edge(src, dst, weight=1)
    return B


def _edge_weights(G: nx.DiGraph):
    return {(u, v): d["weight"] for u, v, d in G.edges(data=True)}


def test_temporal_snapshots_are_sorted():
    H = nx.Graph()
    H.add_edge(0, 1, times=[10, 1])
    H.add_edge(1, 2, times=[4])

    graph_data = build_temporal_snapshots(H, group_by_time=1)

    assert graph_data["time_order"] == sorted(graph_data["time_order"])
    assert list(graph_data["edge_indeces"].keys()) == graph_data["time_order"]


def test_de_bruijn_builder_matches_slow_reference():
    events = np.array([
        [0, 1, 1],
        [1, 2, 3],
        [2, 1, 5],
        [1, 0, 8],
    ])

    fast = make_de_bruijn_graph(events, delta=4, directed=False)
    slow = _slow_de_bruijn(events, delta=4, directed=False)

    assert set(fast.nodes()) == set(slow.nodes())
    assert _edge_weights(fast) == _edge_weights(slow)


def test_loss_guard_detects_non_finite_divergence_and_uniform_stall():
    cfg = LossGuardConfig(warmup_epochs=2, uniform_window=4, min_improvement=0.01)
    assert check_loss_guard([1.0], [float("nan")], 1, 20, cfg) == "non_finite_loss"
    assert check_loss_guard([1.0, 1.0], [1.0, 10.0], 2, 20, cfg) == "divergent_validation_loss"

    uniform = math.log(20)
    train = [uniform] * 4
    val = [uniform * 1.001] * 4
    assert check_loss_guard(train, val, 4, 20, cfg) == "uniform_stall"


def test_generated_max_quality_configs_are_consistent():
    preset = runner.PRESETS["max_quality"]
    tsir = runner.build_tsir_config("france_office", "r0_20", preset)
    model = runner.build_model_config("france_office", "dbgnn", "r0_20", preset)
    eval_cfg = runner.build_eval_config("france_office", "r0_20", preset)

    assert tsir["sir"]["n_runs"] == 1000
    assert tsir["sir"]["mc_runs"] == 500
    assert tsir["nwk"]["directed"] is False
    assert model["eval"]["min_outbreak"] == 1
    assert model["eval"]["n_truth"] == 1000
    assert model["train"]["reps"] == 1
    assert model["train"]["n_mc"] == 500
    assert model["dbgnn"]["order"] == 2
    assert model["dbgnn"]["delta"] == 24
    assert model["dbgnn"]["bipartite_agg"] == "sum"
    assert model["dbgnn"]["directed"] is False
    assert eval_cfg["eval"]["min_outbreak"] == 1
    assert tsir["sir"]["n_runs"] >= model["train"]["reps"] * model["eval"]["n_truth"]


def test_added_network_parameters_match_final_grid():
    preset = runner.PRESETS["max_quality"]

    students = runner.build_tsir_config("students", "r0_25", preset)
    biasca = runner.build_tsir_config("biasca", "r0_20", preset)
    olten = runner.build_tsir_config("olten", "r0_25", preset)

    assert students["sir"]["beta"] == 0.187
    assert students["sir"]["mu"] == 0.01
    assert biasca["sir"]["beta"] == 0.274
    assert biasca["sir"]["mu"] == 0.001
    assert olten["sir"]["beta"] == 0.343
    assert olten["sir"]["mu"] == 0.001


def test_dry_run_expands_full_thesis_matrix(tmp_path):
    cmd = [
        sys.executable,
        "run_all_experiments.py",
        "--dry-run",
        "--skip-viz",
        "--skip-tables",
        "--output",
        str(tmp_path),
        "--run-name",
        "dry",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "Thesis Final Experiment Runner" in result.stdout

    with open(tmp_path / "dry" / "status.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = len(runner.NETWORKS) * len(runner.R0_LABELS) * (1 + len(runner.MODELS) + 1)
    assert len(rows) == expected
    assert sum(r["stage"] == "train" for r in rows) == len(runner.NETWORKS) * len(runner.R0_LABELS) * len(runner.MODELS)


def test_dbgnn_higher_order_dry_run_expands_orders_and_sampling(tmp_path):
    cmd = [
        sys.executable,
        "run_dbgnn_higher_order_experiment.py",
        "--dry-run",
        "--output",
        str(tmp_path),
        "--run-name",
        "dry",
        "--networks",
        "students",
        "--r0",
        "1.0",
        "--orders",
        "2",
        "4",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "DBGNN Higher-Order Experiment Runner" in result.stdout
    assert "balanced_activity_snowball" in result.stdout

    with open(tmp_path / "dry" / "status.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert sum(r["stage"] == "tsir" for r in rows) == 1
    assert [r["order"] for r in rows if r["stage"] == "train"] == ["2", "4"]

    tsir_cfg = (tmp_path / "dry" / "configs" / "students" / "r0_10" / "tsir.yml").read_text()
    k2_cfg = yaml.safe_load((tmp_path / "dry" / "configs" / "students" / "r0_10" / "dbgnn_k2.yml").read_text())
    k4_cfg = yaml.safe_load((tmp_path / "dry" / "configs" / "students" / "r0_10" / "dbgnn_k4.yml").read_text())
    assert "balanced_activity_snowball" in tsir_cfg
    assert "target_nodes: 300" in tsir_cfg
    assert k2_cfg["dbgnn"]["delta"] == 24
    assert k2_cfg["dbgnn"]["time_bin_size"] == 1
    assert k2_cfg["train"]["batch_size"] == 16
    assert k4_cfg["dbgnn"]["order"] == 4
    assert k4_cfg["dbgnn"]["delta"] == 4
    assert k4_cfg["dbgnn"]["time_bin_size"] == 4
    assert k4_cfg["train"]["batch_size"] == 4
    assert k4_cfg["dbgnn"]["max_temporal_states"] == 2_000_000


def test_dbgnn_higher_order_defaults_match_requested_grid_and_run_order(tmp_path):
    cmd = [
        sys.executable,
        "run_dbgnn_higher_order_experiment.py",
        "--dry-run",
        "--output",
        str(tmp_path),
        "--run-name",
        "dry",
        "--networks",
        "students",
        "lyon_ward",
        "--r0",
        "0.8",
        "1.0",
        "--orders",
        "2",
        "3",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    with open(tmp_path / "dry" / "status.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    train_rows = [r for r in rows if r["stage"] == "train"]
    assert [(r["network"], r["order"], r["r0_label"]) for r in train_rows] == [
        ("lyon_ward", "2", "r0_08"),
        ("lyon_ward", "2", "r0_10"),
        ("lyon_ward", "3", "r0_08"),
        ("lyon_ward", "3", "r0_10"),
        ("students", "2", "r0_08"),
        ("students", "2", "r0_10"),
        ("students", "3", "r0_08"),
        ("students", "3", "r0_10"),
    ]

    manifest = yaml.safe_load((tmp_path / "dry" / "manifest.json").read_text())
    assert manifest["r0_labels"] == ["r0_08", "r0_10"]
    assert manifest["orders"] == [2, 3]
    assert manifest["networks"] == ["lyon_ward", "students"]


def test_dbgnn_sampling_budget_uses_students_factor_72():
    stats = {"students": ho_runner.read_full_network_stats("students")}
    budget = ho_runner.compute_sample_budget(stats, "students", 72)
    assert budget == stats["students"].node_edge_cost // 72


def test_dbgnn_runner_timeout_marks_command_for_skip(tmp_path):
    rc, stdout = ho_runner.run_command(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        tmp_path / "timeout.log",
        dry_run=False,
        timeout_seconds=1,
    )

    assert rc == 124
    assert "TIMEOUT_SKIP" in stdout
    assert "TIMEOUT_SKIP" in (tmp_path / "timeout.log").read_text()


def test_plot_generation_from_synthetic_eval_arrays(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "read_network_meta", lambda network: {"t_max": 10})
    data_dir = tmp_path / "data"
    (data_dir / "modelrun").mkdir(parents=True)
    (data_dir / "evalrun").mkdir(parents=True)

    n_nodes, n_runs = 5, 4
    n = n_nodes * n_runs
    arrays = {
        "ranks": np.tile(np.array([1, 2, 3, 4]), n_nodes).astype(np.int64),
        "outbreak_sizes": np.linspace(1 / n_nodes, 1.0, n).astype(np.float32),
        "sel": np.ones(n, dtype=bool),
        "true_sources": np.repeat(np.arange(n_nodes), n_runs),
    }
    np.savez_compressed(data_dir / "modelrun" / "eval_arrays_rep0.npz", **arrays)
    np.savez_compressed(data_dir / "evalrun" / "eval_arrays_uniform.npz", **arrays)

    status = [
        {"network": "france_office", "r0_label": "r0_08", "stage": "train", "model": "static_gnn", "status": "success", "run_id": "modelrun"},
        {"network": "france_office", "r0_label": "r0_08", "stage": "eval", "model": "", "status": "success", "run_id": "evalrun"},
    ]

    run_dir = tmp_path / "results"
    runner.plot_scenario_outputs(run_dir, status, "france_office", "r0_08")

    fig_dir = run_dir / "france_office" / "r0_08" / "figures"
    assert (fig_dir / "top5_vs_outbreak_compare.pdf").exists()
    assert (fig_dir / "top5_vs_outbreak_compare.png").exists()
    assert (fig_dir / "top5_vs_outbreak_compare.README.md").exists()
