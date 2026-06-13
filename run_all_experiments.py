"""
Thesis-final experiment pipeline.

Runs the complete source-detection sweep for the thesis networks, R0 values,
GNN models, heuristic baselines, result tables, and plots.

Every run maintains a publication bundle under:
    results/thesis_final/<run-name>/result/

The bundle mirrors metrics, figures, tables, and lightweight run assets, and
includes latex_inputs.json for downstream LaTeX/report generation.

Default run:
    python run_all_experiments.py
        # uses the day-scale paired Optuna protocol: paper_24h, 5 HPO trials,
        # network-scope HPO reuse across R0s, short HPO trial budgets,
        # capped final training

Useful controls:
    python run_all_experiments.py --dry-run
    python run_all_experiments.py --preset fast --networks lyon_ward --models static_gnn
    python run_all_experiments.py --hpo-scope scenario --preset max_quality --hpo-trials 30 \
        --max-train-epochs 0 --max-train-patience 0
    python run_all_experiments.py --no-hpo --preset fast
        # disables the default paired <method> and <method>_optuna final evaluations
    python run_all_experiments.py --resume --run-name 20260512_010000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz.rank_vs_outbreak import load_eval_arrays
from viz.style import MODEL_COLORS, MODEL_LABELS, apply_style, finish_fig, model_style
from scripts.publication_bundle import sync_publication_result
from hpo import apply_trial_params
from setup.reduction import NetworkStats, read_full_network_stats


WANDB_PROJECT = "source-detection"

NETWORKS = ["lyon_ward", "malawi", "france_office", "students", "biasca", "olten", "escort", "pig_data"]
MODELS = ["static_gnn", "temporal_gnn", "backtracking", "dbgnn_k2", "dbgnn_k3"]
MODEL_ALIASES = {
    "dbgnn": ["dbgnn_k2", "dbgnn_k3"],
    "all": MODELS,
}
TRAINABLE_BASELINES = ["static_mlp"]
HEURISTIC_BASELINES_FAST = ["uniform", "random", "degree", "closeness", "betweenness", "jordan_center"]
# Sterchi et al.'s full baseline set: Random, Jordan, Betweenness, SME, MCMF.
# soft_margin (SME) now runs on the stored MC pool (artifact SME), so it is
# affordable at n_mc=500 and belongs in the default comparison.
HEURISTIC_BASELINES_PAPER = HEURISTIC_BASELINES_FAST + ["mc_mean_field", "soft_margin"]
HEURISTIC_BASELINES_EXPENSIVE = ["mcs_mean_field"]
HEURISTIC_BASELINES = HEURISTIC_BASELINES_PAPER
# Per-observation baselines: betweenness/jordan_center are computed on the
# *infected subgraph* of every outbreak (Sterchi-faithful, but ~100x slower than
# a static prior) and soft_margin (SME) compares every observation to the stored
# MC pool. On the large reduced networks (≈300 nodes × n_truth observations)
# these can take hours per network. ``--no-expensive-baselines`` drops them while
# keeping the cheap, fully-vectorised baselines (incl. the MCMF probabilistic
# baseline). The cheap subset alone is a complete, fast comparison; add the
# expensive set back for the definitive Sterchi-parity table.
EXPENSIVE_BASELINES = {"betweenness", "jordan_center", "soft_margin"}
HEURISTIC_BASELINES_CHEAP = [b for b in HEURISTIC_BASELINES_PAPER if b not in EXPENSIVE_BASELINES]
BASELINE_ALIASES = {
    "fast": HEURISTIC_BASELINES_FAST,
    "cheap": HEURISTIC_BASELINES_CHEAP,
    "paper": HEURISTIC_BASELINES_PAPER,
    "all": HEURISTIC_BASELINES_PAPER + HEURISTIC_BASELINES_EXPENSIVE,
}
BASELINES = TRAINABLE_BASELINES + HEURISTIC_BASELINES
OPTUNA_SUFFIX = "_optuna"
R0_LABELS = ["r0_08", "r0_10", "r0_11", "r0_15", "r0_20", "r0_25"]
MIN_OUTBREAK = 2

R0_VALUES = {
    "r0_08": 0.8,
    "r0_10": 1.0,
    "r0_11": 1.1,
    "r0_15": 1.5,
    "r0_20": 2.0,
    "r0_25": 2.5,
}

BETAS = {
    "lyon_ward":    {"r0_08": 0.012, "r0_10": 0.015, "r0_11": 0.016, "r0_15": 0.024, "r0_20": 0.038, "r0_25": 0.059},
    "malawi":       {"r0_08": 0.025, "r0_10": 0.041, "r0_11": 0.050, "r0_15": 0.105, "r0_20": 0.244, "r0_25": 0.542},
    "france_office":{"r0_08": 0.058, "r0_10": 0.070, "r0_11": 0.076, "r0_15": 0.107, "r0_20": 0.159, "r0_25": 0.233},
    "students":     {"r0_08": 0.034, "r0_10": 0.045, "r0_11": 0.051, "r0_15": 0.078, "r0_20": 0.124, "r0_25": 0.187},
    "biasca":       {"r0_08": 0.016, "r0_10": 0.031, "r0_11": 0.041, "r0_15": 0.113, "r0_20": 0.274, "r0_25": 0.480},
    "olten":        {"r0_08": 0.025, "r0_10": 0.039, "r0_11": 0.047, "r0_15": 0.096, "r0_20": 0.195, "r0_25": 0.343},
    "escort":       {"r0_08": 0.016, "r0_10": 0.020, "r0_11": 0.022, "r0_15": 0.030, "r0_20": 0.040, "r0_25": 0.050},
    "pig_data":     {"r0_08": 0.032, "r0_10": 0.040, "r0_11": 0.044, "r0_15": 0.060, "r0_20": 0.080, "r0_25": 0.100},
}

# Recovery rate. Standardised across the whole test suite so every network
# shares the same SIR dynamical regime (Sterchi et al. fix mu WLOG; the C
# simulator requires mu < 1, so we use a single small value rather than 1).
# Networks whose mu changes here have their beta re-calibrated automatically at
# run time (the beta cache is mu-aware), and end_t is calibrated to the 40%
# infected-snapshot target — so all three knobs stay self-consistent.
STANDARD_MU = 0.01
MUS = {network: STANDARD_MU for network in NETWORKS}

# Target fraction of nodes infected at the observed snapshot. end_t is
# calibrated per network/R0 to hit this (Sterchi et al. use ≈ 0.40).
TARGET_INFECTED = 0.40

TEMPORAL_GROUP_BY_TIME = {
    "lyon_ward": 6,
    "malawi": 16,
    "france_office": 12,
    "students": 50,
    "biasca": 32,
    "olten": 32,
}


@dataclass(frozen=True)
class Preset:
    n_runs: int
    mc_runs: int
    n_mc: int
    reps: int
    n_truth: int


PRESETS = {
    "paper_24h": Preset(n_runs=300, mc_runs=180, n_mc=160, reps=1, n_truth=150),
    "balanced": Preset(n_runs=500, mc_runs=300, n_mc=300, reps=1, n_truth=300),
    "max_quality": Preset(n_runs=1000, mc_runs=500, n_mc=500, reps=1, n_truth=1000),
    "fast": Preset(n_runs=120, mc_runs=80, n_mc=80, reps=1, n_truth=40),
    # Multi-seed protocol for the in-depth Optuna tuner experiment
    # (run_tuner_experiment.py). reps=3 mirrors Sterchi et al.: the final model
    # is trained and evaluated three times and averaged. n_runs leaves room for
    # a disjoint HPO-validation window + the 3*250 held-out final-test window.
    "tuner": Preset(n_runs=1200, mc_runs=500, n_mc=500, reps=3, n_truth=250),
}

LOSS_GUARD = {
    "enabled": True,
    "warmup_epochs": 20,
    "divergence_factor": 1.5,
    "uniform_tolerance": 0.02,
    "uniform_window": 80,
    "min_improvement": 0.01,
}

# Gradient-clipping norm applied to every trained model. Tames
# BacktrackingNetwork's unstable/oscillating loss while barely touching the
# well-behaved GNNs (their grad norms rarely exceed this).
GRAD_CLIP_NORM = 1.0

# Frozen, Sterchi-aligned training hyperparameters (Table 4): the optimiser
# setup is fixed and only the architecture is tuned. batch_size applies to the
# static models; temporal models keep memory-appropriate batches (see
# build_model_config).
STERCHI_TRAIN = {
    "lr": 0.001,
    "weight_decay": 0.0005,
    "batch_size": 128,
    "test_size": 0.30,
    # Train up to 500 epochs and let early-stopping (patience 20) decide when to
    # stop, exactly as Sterchi et al. — never a hard cap that bites before
    # convergence. Early stopping means real epoch counts stay modest
    # (Sterchi's networks stopped at 23–68 epochs), so this is GPU-affordable.
    "epochs": 500,
    "patience": 20,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper_24h",
                   help="Runtime/quality preset. Use max_quality for the full expensive grid.")
    p.add_argument("--n-runs", type=int, default=None,
                   help="Override the preset's ground-truth simulation count (TSIR n_runs). "
                        "Raise together with --n-truth for tighter metric estimates.")
    p.add_argument("--mc-runs", type=int, default=None,
                   help="Override the preset's Monte Carlo simulation count stored per source.")
    p.add_argument("--n-mc", type=int, default=None,
                   help="Override the per-repetition number of MC simulations used for training.")
    p.add_argument("--reps", type=int, default=None,
                   help="Override the number of independent train/eval repetitions.")
    p.add_argument("--n-truth", type=int, default=None,
                   help="Override the preset's evaluation sample count. Must be <= --n-runs; "
                        "cheap (inference + metrics only, no extra training).")
    p.add_argument("--target-infected", type=float, default=TARGET_INFECTED,
                   help="Target fraction of nodes infected at the observed snapshot. "
                        "end_t is calibrated per network/R0 to hit this (Sterchi ≈ 0.40).")
    p.add_argument("--target-infected-n-probe", type=int, default=64,
                   help="Number of probe outbreaks per end_t bisection step when "
                        "calibrating the infected-snapshot fraction.")
    p.add_argument("--networks", nargs="+", default=NETWORKS)
    p.add_argument("--reverse-networks", action="store_true",
                   help="Run selected networks in reverse order, useful for splitting work across machines.")
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--baselines", nargs="+", default=["paper"],
                   help="Heuristic baseline keys or presets: fast, cheap, paper, all. Default: paper")
    p.add_argument("--no-expensive-baselines", dest="exclude_expensive_baselines",
                   action="store_true",
                   help="Drop the per-observation baselines (subgraph betweenness/jordan_center "
                        "and the SME soft_margin), which can take hours on the large reduced "
                        "networks. Keeps the cheap vectorised baselines incl. MCMF.")
    p.add_argument("--r0", nargs="+", default=["all"], help="R0 labels: r0_08 ... r0_25, numeric values, or all")
    p.add_argument("--output", default="results/thesis_final", help="Root results directory")
    p.add_argument("--experiment-name", default="thesis_final",
                   help="Logical experiment name recorded in the publication bundle and LaTeX inputs.")
    p.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run directory and skip terminal stages")
    p.add_argument("--force", action="store_true", help="Rerun stages even when a terminal status exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    p.add_argument("--save-probs", action="store_true", help="Save probs_rep*.pt tensors from main_train.py")
    p.add_argument("--with-hpo", dest="with_hpo", action="store_true",
                   help="Run paired untuned and Optuna-tuned final evaluations (default)")
    p.add_argument("--no-hpo", dest="with_hpo", action="store_false",
                   help="Disable Optuna and run only the untuned configs")
    p.add_argument("--hpo-trials", type=int, default=12,
                   help="Optuna trials per network/R0/model when --with-hpo is enabled. "
                        "12 suffices now that only architecture is tuned and selection "
                        "uses smoothed validation loss; raise to ~30 for an exhaustive sweep.")
    p.add_argument("--hpo-timeout", type=int, default=None,
                   help="Optional Optuna timeout in seconds per study")
    p.add_argument("--hpo-scope", choices=["network", "scenario"], default="network",
                   help="Tune once per network/model and reuse across R0s, or tune every scenario")
    p.add_argument("--hpo-reference-r0", default="r0_10",
                   help="R0 used for network-scope HPO when present; falls back to the first selected R0")
    p.add_argument("--hpo-n-truth", type=int, default=100,
                   help="Validation truth runs per HPO study, clamped to the preset budget")
    p.add_argument("--hpo-n-mc", type=int, default=80,
                   help="MC samples used inside HPO trials only")
    p.add_argument("--hpo-epochs", type=int, default=120,
                   help="Epoch cap used inside HPO trials only")
    p.add_argument("--hpo-patience", type=int, default=12,
                   help="Early-stopping patience cap used inside HPO trials only")
    p.add_argument("--max-train-epochs", type=int, default=500,
                   help="Cap final train.epochs for generated configs; set 0 to keep template values. "
                        "Default 500 matches Sterchi (rely on early-stopping, not the cap, to converge).")
    p.add_argument("--max-train-patience", type=int, default=20,
                   help="Cap final train.patience for generated configs; set 0 to keep template values")
    p.add_argument("--hpo-sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--hpo-pruner", choices=["hyperband", "median", "none"], default="hyperband")
    p.add_argument("--hpo-enqueue-default", dest="hpo_enqueue_default", action="store_true", default=True,
                   help="Evaluate the base config as a protected first Optuna trial so the tuned "
                        "model can never be selected worse than the default (default on).")
    p.add_argument("--no-hpo-enqueue-default", dest="hpo_enqueue_default", action="store_false",
                   help="Disable enrolling the base config as a protected candidate trial.")
    p.add_argument("--reduction", choices=["safe_1h", "none"], default="safe_1h",
                   help="Network reduction policy for TSIR artifacts. Default: safe_1h")
    p.add_argument("--no-reduction", dest="reduction", action="store_const", const="none",
                   help="Disable default network reduction.")
    p.add_argument("--target-runtime-seconds", type=int, default=7200,
                   help="Timeout (s) for each TSIR/HPO/train/eval subprocess. Default 2h.")
    p.add_argument("--sample-target-nodes", type=int, default=300,
                   help="Target nodes for safe_1h node sampling.")
    p.add_argument("--time-window-steps", default="auto",
                   help="Temporal window length for safe_1h, or auto.")
    p.add_argument("--reduction-seed", type=int, default=42,
                   help="Seed for deterministic representative reduction.")
    p.add_argument("--reduction-reps", type=int, default=1,
                   help="Number of reduction seeds to record for robustness runs.")
    p.add_argument("--use-full-betas", action="store_true",
                   help="Use the static BETAS table instead of per-artifact beta calibration.")
    p.add_argument("--skip-tsir", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--skip-tables", action="store_true")
    p.set_defaults(with_hpo=True)
    return p


def parse_args(
    argv: list[str] | None = None,
    default_overrides: dict[str, Any] | None = None,
) -> argparse.Namespace:
    """Parse CLI args, optionally injecting alternative defaults.

    ``default_overrides`` lets sibling runners (e.g. run_tuner_experiment.py)
    reuse this pipeline with a different default protocol while still allowing
    every value to be overridden on the command line.
    """
    parser = build_parser()
    if default_overrides:
        unknown = set(default_overrides) - {action.dest for action in parser._actions}
        if unknown:
            raise ValueError(f"Unknown default_overrides keys: {sorted(unknown)}")
        parser.set_defaults(**default_overrides)
    return parser.parse_args(argv)


def _positive_cap(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def apply_final_train_caps(cfg: dict[str, Any], args: argparse.Namespace | None) -> None:
    """Bound definitive training runs for day-scale experiment execution."""
    if args is None:
        return
    train_cfg = cfg.setdefault("train", {})
    epoch_cap = _positive_cap(getattr(args, "max_train_epochs", None))
    patience_cap = _positive_cap(getattr(args, "max_train_patience", None))
    if epoch_cap is not None:
        train_cfg["epochs"] = min(int(train_cfg.get("epochs", epoch_cap)), epoch_cap)
    if patience_cap is not None:
        train_cfg["patience"] = min(int(train_cfg.get("patience", patience_cap)), patience_cap)


def effective_hpo_n_truth(args: argparse.Namespace, preset: Preset) -> int:
    """Keep HPO and final truth windows disjoint even on small presets."""
    requested = _positive_cap(getattr(args, "hpo_n_truth", None)) or max(1, preset.n_runs // 3)
    return min(requested, max(1, preset.n_runs // 3))


def effective_hpo_n_mc(args: argparse.Namespace, preset: Preset) -> int | None:
    requested = _positive_cap(getattr(args, "hpo_n_mc", None))
    if requested is None:
        return None
    return min(requested, preset.n_mc, preset.mc_runs)


def attach_hpo_budget(cfg: dict[str, Any], args: argparse.Namespace, preset: Preset) -> None:
    hpo_cfg = cfg.setdefault("hpo", {})
    hpo_cfg["n_truth"] = effective_hpo_n_truth(args, preset)
    hpo_cfg["trial_epochs"] = _positive_cap(getattr(args, "hpo_epochs", None))
    hpo_cfg["trial_patience"] = _positive_cap(getattr(args, "hpo_patience", None))
    hpo_cfg["trial_n_mc"] = effective_hpo_n_mc(args, preset)
    # Select models on the smoothed validation NLL (mean of the last
    # `val_loss_window` validation losses), as in Sterchi et al., instead of the
    # noisy truth-window MRR. The validation split is large and i.i.d. with
    # training, so this signal is stable even at a few trials; the truth-window
    # metrics are still logged for monitoring but no longer drive selection.
    hpo_cfg["metric"] = "eval/val_nll"
    hpo_cfg["direction"] = "minimize"
    hpo_cfg["val_loss_window"] = 5


def final_eval_window(args: argparse.Namespace, preset: Preset) -> tuple[int, int]:
    """Held-out truth window used by the HPO-paired GNN finals.

    Mirrors ``main_optuna.resolve_truth_budget`` for the runner protocol
    (``hpo.truth_start = 0``, ``hpo.reps = 1`` — ``attach_hpo_budget`` never
    overrides these), so the heuristic/probabilistic baselines can be scored on
    the *identical* window as the GNN finals instead of the default ``[0,
    n_truth)``. Sterchi's benchmark requires every method on one data substrate;
    without this the GNN window (shifted past the HPO-validation runs) and the
    baseline window only partially overlap.

    With ``--no-hpo`` there is no shift and the finals use ``[0, n_truth)``.

    Returns ``(truth_start, n_truth)``. Assumes the artifact was generated with
    ``preset.n_runs`` ground-truth runs (true for a fresh TSIR stage), so this
    equals the window ``main_optuna`` wrote into ``best_config.yml``.
    """
    if not bool(getattr(args, "with_hpo", False)):
        return 0, preset.n_truth
    # hpo_truth_start(0) + hpo_reps(1) * hpo_n_truth
    final_truth_start = effective_hpo_n_truth(args, preset)
    final_remaining = preset.n_runs - final_truth_start
    final_n_truth = min(preset.n_truth, max(1, final_remaining // max(1, preset.reps)))
    return final_truth_start, final_n_truth


def normalize_r0_labels(raw: list[str]) -> list[str]:
    if "all" in raw:
        return R0_LABELS
    out = []
    numeric_to_label = {f"{v:.1f}": k for k, v in R0_VALUES.items()}
    for item in raw:
        key = item.strip()
        if key in R0_VALUES:
            out.append(key)
            continue
        try:
            label = numeric_to_label[f"{float(key):.1f}"]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Unknown R0 selector '{item}'. Use one of {R0_LABELS}.") from exc
        out.append(label)
    return out


def resolve_hpo_reference_r0(raw: str, r0_labels: list[str]) -> str:
    if not r0_labels:
        raise ValueError("At least one R0 label is required.")
    if raw in {"auto", "middle"}:
        return "r0_10" if "r0_10" in r0_labels else r0_labels[len(r0_labels) // 2]
    try:
        label = normalize_r0_labels([raw])[0]
    except ValueError:
        label = raw
    return label if label in r0_labels else r0_labels[0]


def normalize_model_keys(raw: list[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        expanded = MODEL_ALIASES.get(item, [item])
        for model in expanded:
            if model not in out:
                out.append(model)
    return out


def normalize_baseline_keys(raw: list[str], exclude_expensive: bool = False) -> list[str]:
    out: list[str] = []
    for item in raw:
        expanded = BASELINE_ALIASES.get(item, [item])
        for baseline in expanded:
            if baseline not in out:
                out.append(baseline)
    known = set(HEURISTIC_BASELINES_PAPER + HEURISTIC_BASELINES_EXPENSIVE)
    unknown = [baseline for baseline in out if baseline not in known]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s): {unknown}. Known presets/keys: "
            f"{sorted(known | set(BASELINE_ALIASES))}"
        )
    if exclude_expensive:
        dropped = [b for b in out if b in EXPENSIVE_BASELINES]
        out = [b for b in out if b not in EXPENSIVE_BASELINES]
        if dropped:
            print(f"  --no-expensive-baselines: dropping {dropped} (subgraph centrality + SME)")
    return out


def optuna_variant_key(method: str) -> str:
    return f"{method}{OPTUNA_SUFFIX}"


def base_method_key(method: str) -> str:
    if method.endswith(OPTUNA_SUFFIX):
        return method[:-len(OPTUNA_SUFFIX)]
    return method


def paired_method_order() -> list[str]:
    out: list[str] = []
    for method in MODELS + BASELINES:
        out.append(method)
        out.append(optuna_variant_key(method))
    return out


def method_line_style(method: str, kind: str | None = None) -> str:
    if method.endswith(OPTUNA_SUFFIX):
        return "-."
    if kind == "baseline" or base_method_key(method) in BASELINES:
        return "--"
    return "-"


def base_model_key(model: str) -> str:
    model = base_method_key(model)
    if model.startswith("dbgnn_k"):
        return "dbgnn"
    return model


def dbgnn_order_from_key(model: str) -> int:
    if not model.startswith("dbgnn_k"):
        return 2
    try:
        return int(model.removeprefix("dbgnn_k"))
    except ValueError as exc:
        raise ValueError(f"Invalid DBGNN model key '{model}'. Use dbgnn_k2 or dbgnn_k3.") from exc


def read_network_meta(network: str) -> dict[str, Any]:
    path = Path("nwk") / f"{network}.yml"
    if not path.exists():
        raise FileNotFoundError(f"Missing network metadata: {path}")
    with open(path) as f:
        meta = yaml.safe_load(f)
    t_max = meta.get("time_steps", meta.get("t_max"))
    if t_max is None:
        raise ValueError(f"Cannot determine t_max/time_steps for {network} from {path}")
    directed = meta.get("directed", False)
    if isinstance(directed, str):
        directed = directed.lower() in {"yes", "true", "1"}
    return {"t_max": int(t_max), "directed": bool(directed), **meta}


def scenario(network: str, r0_label: str) -> dict[str, Any]:
    return {
        "label": r0_label,
        "r0": R0_VALUES[r0_label],
        "beta": BETAS[network][r0_label],
        "mu": MUS[network],
    }


def artifact_name(network: str, r0_label: str) -> str:
    return f"thesis_final_{network}_{r0_label}"


def reduction_is_enabled(args: argparse.Namespace | None) -> bool:
    return bool(args is not None and getattr(args, "reduction", "none") != "none")


def reduction_config_for_network(
    network: str,
    args: argparse.Namespace | None,
    stats: NetworkStats | None = None,
) -> dict[str, Any] | None:
    """Return the default safe_1h reduction config for networks that need it."""
    if not reduction_is_enabled(args):
        return None
    meta = read_network_meta(network)
    if stats is None:
        stats = read_full_network_stats(network)

    needs_node = stats.n_nodes > 300
    needs_time = int(meta["t_max"]) > 1000
    if not needs_node and not needs_time:
        return None

    node_cfg = {
        "method": "balanced_activity_snowball",
        "apply_if_nodes_gt": 300,
        "target_nodes": int(getattr(args, "sample_target_nodes", 300)),
        "max_node_edge_cost": "auto_students_div72",
        "stratification_bins": 4,
        "seed": int(getattr(args, "reduction_seed", 42)),
        "min_nodes": 8,
    }
    time_cfg = {
        "method": "representative_window",
        "apply_if_time_steps_gt": 1000,
        "max_steps_days": 365,
        "candidate_windows": 32,
        "reindex_to_zero": True,
    }
    window_steps = str(getattr(args, "time_window_steps", "auto"))
    if window_steps != "auto":
        time_cfg["max_steps"] = int(window_steps)

    return {
        "enabled": "auto",
        "preset": "safe_1h",
        "runtime_target_s": int(getattr(args, "target_runtime_seconds", 3600)),
        "node": node_cfg,
        "time": time_cfg,
    }


def build_tsir_config(
    network: str,
    r0_label: str,
    preset: Preset,
    args: argparse.Namespace | None = None,
    stats: NetworkStats | None = None,
) -> dict[str, Any]:
    meta = read_network_meta(network)
    sc = scenario(network, r0_label)
    reduction_cfg = reduction_config_for_network(network, args, stats)
    nwk_cfg = {
        "type": "empirical",
        "name": network,
        "t_max": meta["t_max"],
        "directed": meta["directed"],
    }
    if reduction_cfg is not None:
        nwk_cfg["reduction"] = reduction_cfg
    sir_cfg: dict[str, Any] = {
        "beta": sc["beta"],
        "mu": sc["mu"],
        "start_t": 0,
        "end_t": meta["t_max"],
        "n_runs": preset.n_runs,
        "mc_runs": preset.mc_runs,
    }
    if args is not None and not bool(getattr(args, "use_full_betas", False)):
        # end_t is initialised to the full contact window (t_max above) and acts
        # as the upper bound for the target-infected calibration, which shrinks
        # it to the Sterchi-style ≈40%-infected snapshot.
        sir_cfg["calibration"] = {
            "enabled": True,
            "target_r0": sc["r0"],
            "target_infected": float(getattr(args, "target_infected", TARGET_INFECTED)),
            "target_infected_n_probe": int(getattr(args, "target_infected_n_probe", 64)),
            "target_infected_tolerance": 0.02,
            "output_dir": "results/calibration",
            "n_probe": 1,
            "max_iter": 8,
            "tolerance": 0.05,
            "seed": int(getattr(args, "reduction_seed", 42)),
        }
    return {
        "nwk": nwk_cfg,
        "sir": sir_cfg,
        "experiment": {
            "network": network,
            "r0_label": r0_label,
            **sc,
        },
    }


def _template_path(network: str, model: str) -> Path:
    model = base_model_key(model)
    direct = Path("exp") / network / f"{model}.yml"
    if direct.exists():
        return direct
    fallback = Path("exp/france_office") / f"{model}.yml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No model config template found for {network}/{model}")


def build_model_config(
    network: str,
    model: str,
    r0_label: str,
    preset: Preset,
    save_probs: bool = False,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    base_model = base_model_key(model)
    with open(_template_path(network, base_model)) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = base_model
    cfg["eval"] = {
        **cfg.get("eval", {}),
        "min_outbreak": MIN_OUTBREAK,
        "top_k": [1, 3, 5, 10],
        "credible_p": [0.80, 0.90],
        "inverse_rank_offset": [0],
        "n_truth": preset.n_truth,
        # Sterchi-exact: all reps are scored on the SAME held-out test window, so
        # the reported 95% CIs reflect training/initialisation noise (not test-set
        # sampling). Only needs n_truth runs, so n_truth can be large and cheap.
        "shared_eval_window": True,
    }
    cfg["train"] = {
        **cfg.get("train", {}),
        # Sterchi-aligned, FROZEN training hyperparameters. These are no longer
        # tuned by Optuna (see hpo/search_space.py); pinning them here makes the
        # untuned headline config and every HPO trial share the same optimiser
        # setup, so tuning can only change architecture. lr/weight_decay/test_size
        # were the knobs whose search degraded the strong models last run.
        "lr": STERCHI_TRAIN["lr"],
        "weight_decay": STERCHI_TRAIN["weight_decay"],
        "test_size": STERCHI_TRAIN["test_size"],
        "epochs": STERCHI_TRAIN["epochs"],
        "patience": STERCHI_TRAIN["patience"],
        "n_mc": preset.n_mc,
        "reps": preset.reps,
        "loss_guard": LOSS_GUARD,
        "grad_clip_norm": GRAD_CLIP_NORM,
    }
    # Batch size is frozen at Sterchi's 128 for the static models (small graphs);
    # the temporal models (backtracking/temporal_gnn/dbgnn) keep their
    # memory-appropriate template/cap values since 128 would OOM on full-network
    # samples. It is removed from the HPO search either way.
    if base_model in ("static_gnn", "static_mlp"):
        cfg["train"]["batch_size"] = STERCHI_TRAIN["batch_size"]
    cfg.setdefault("output", {})["save_probs"] = save_probs
    if base_model == "temporal_gnn":
        temporal_cfg = cfg.setdefault("temporal_gnn", {})
        temporal_cfg["group_by_time"] = TEMPORAL_GROUP_BY_TIME.get(network, 12)
        temporal_cfg.setdefault("residual", True)
        temporal_cfg.setdefault("layer_norm", True)
        temporal_cfg.setdefault("dropout_rate", 0.0)
        temporal_cfg.setdefault("readout", "jumping_mean")
    if base_model == "dbgnn":
        db_cfg = cfg.setdefault("dbgnn", {})
        order = dbgnn_order_from_key(model)
        safe_1h = reduction_is_enabled(args)
        db_cfg["order"] = order
        db_cfg["delta"] = db_cfg.get("delta", 24)
        if safe_1h:
            db_cfg["time_bin_size"] = int(db_cfg.get("time_bin_size") or 4)
            db_cfg["max_temporal_states"] = int(db_cfg.get("max_temporal_states") or 2_000_000)
            db_cfg["max_db_nodes"] = int(db_cfg.get("max_db_nodes") or 500_000)
            db_cfg["max_db_edges"] = int(db_cfg.get("max_db_edges") or 2_000_000)
        db_cfg["bipartite_agg"] = db_cfg.get("bipartite_agg", "sum")
        db_cfg["directed"] = read_network_meta(network)["directed"]
        hpo_cfg = cfg.setdefault("hpo", {})
        locked_params = list(hpo_cfg.get("locked_params") or [])
        if "dbgnn.order" not in locked_params:
            locked_params.append("dbgnn.order")
        hpo_cfg["locked_params"] = locked_params
        batch_cap = 8 if order >= 3 else 16
        cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", batch_cap)), batch_cap)
    apply_final_train_caps(cfg, args)
    cfg["experiment"] = {
        "network": network,
        "r0_label": r0_label,
        "model_variant": model,
        **scenario(network, r0_label),
    }
    if preset.n_runs < preset.reps * preset.n_truth:
        raise ValueError(
            f"Invalid preset: n_runs={preset.n_runs} < reps*n_truth={preset.reps * preset.n_truth}"
        )
    return cfg


def build_eval_config(
    network: str,
    r0_label: str,
    preset: Preset,
    truth_start: int = 0,
    n_truth: int | None = None,
) -> dict[str, Any]:
    sc = scenario(network, r0_label)
    eval_n_truth = int(n_truth) if n_truth is not None else preset.n_truth
    return {
        "eval": {
            "min_outbreak": MIN_OUTBREAK,
            "top_k": [1, 3, 5, 10],
            "credible_p": [0.80, 0.90],
            "inverse_rank_offset": [0],
            "n_truth": eval_n_truth,
            # Score baselines on the SAME held-out window as the HPO-paired GNN
            # finals (main_optuna shifts truth_start past the HPO-validation
            # runs). reps=1: the baselines are deterministic given the outbreak,
            # so a single pass over the aligned window is exact and there is no
            # run-to-run variance to average.
            "truth_start": int(truth_start),
            "reps": 1,
            "seed": 0,
        },
        "baselines": HEURISTIC_BASELINES,
        "baseline_params": {
            "default": {"chunk_size": 8192},
            "betweenness": {"normalized": True},
            "mc_mean_field": {
                "eps": 1e-6,
                "batch_size": 4096,
                "n_mc": preset.n_mc,
                "seed": 42,
            },
            "soft_margin": {
                # Artifact SME: use the same number of stored simulations the GNN
                # trains on (Sterchi: SME/MCMF share the GNN's outbreak substrate).
                "n_mc": min(preset.n_mc, preset.mc_runs),
            },
            "mcs_mean_field": {
                "n_mc": min(100, preset.mc_runs),
            },
        },
        "experiment": {
            "network": network,
            "r0_label": r0_label,
            **sc,
        },
    }


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_run_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.resume and args.run_name is None:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No previous run directories found under {root}")
        return candidates[-1]
    name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / name


def write_manifest(run_dir: Path, args: argparse.Namespace, networks: list[str], r0_labels: list[str]) -> None:
    preset = PRESETS[args.preset]
    network_stats: dict[str, NetworkStats] = {}
    reduction_policies: dict[str, dict[str, Any] | None] = {}
    for network in networks:
        try:
            network_stats[network] = read_full_network_stats(network)
            reduction_policies[network] = reduction_config_for_network(network, args, network_stats[network])
        except Exception:
            reduction_policies[network] = None
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "preset_values": preset.__dict__,
        "networks": networks,
        "models": args.models,
        "r0_labels": r0_labels,
        "betas": {n: BETAS[n] for n in networks if n in BETAS},
        "mus": {n: MUS[n] for n in networks if n in MUS},
        "baselines": BASELINES,
        "wandb_project": WANDB_PROJECT,
        "reduction": {
            "policy": getattr(args, "reduction", "none"),
            "target_runtime_seconds": getattr(args, "target_runtime_seconds", None),
            "sample_target_nodes": getattr(args, "sample_target_nodes", None),
            "time_window_steps": getattr(args, "time_window_steps", None),
            "seed": getattr(args, "reduction_seed", None),
            "reps": getattr(args, "reduction_reps", None),
            "use_full_betas": bool(getattr(args, "use_full_betas", False)),
            "policies": reduction_policies,
            "network_stats": {name: stats.__dict__ for name, stats in network_stats.items()},
        },
        "hpo": {
            "enabled": bool(getattr(args, "with_hpo", False)),
            "trials": int(getattr(args, "hpo_trials", 0)),
            "timeout": getattr(args, "hpo_timeout", None),
            "scope": getattr(args, "hpo_scope", None),
            "reference_r0": resolve_hpo_reference_r0(getattr(args, "hpo_reference_r0", "r0_10"), r0_labels),
            "sampler": getattr(args, "hpo_sampler", None),
            "pruner": getattr(args, "hpo_pruner", None),
            "enqueue_default": bool(getattr(args, "hpo_enqueue_default", True)),
            "n_truth": effective_hpo_n_truth(args, preset),
            "trial_n_mc": effective_hpo_n_mc(args, preset),
            "trial_epochs": _positive_cap(getattr(args, "hpo_epochs", None)),
            "trial_patience": _positive_cap(getattr(args, "hpo_patience", None)),
            "final_epoch_cap": _positive_cap(getattr(args, "max_train_epochs", None)),
            "final_patience_cap": _positive_cap(getattr(args, "max_train_patience", None)),
        },
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


STATUS_FIELDS = [
    "network", "r0_label", "stage", "model", "status", "run_id",
    "artifact", "returncode", "message", "log_path",
]
TERMINAL_STATUSES = {"success", "loss_guard_aborted", "skipped", "timeout_skipped"}


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def update_status(path: Path, row: dict[str, Any]) -> None:
    rows = read_status(path)
    key = (row.get("network", ""), row.get("r0_label", ""), row.get("stage", ""), row.get("model", ""))
    normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
    kept = [
        r for r in rows
        if (r.get("network", ""), r.get("r0_label", ""), r.get("stage", ""), r.get("model", "")) != key
    ]
    kept.append(normalized)
    write_status(path, kept)


def should_skip(status_path: Path, args: argparse.Namespace, network: str, r0_label: str, stage: str, model: str = "") -> bool:
    if args.force:
        return False
    if not args.resume:
        return False
    for row in read_status(status_path):
        if (
            row.get("network") == network
            and row.get("r0_label") == r0_label
            and row.get("stage") == stage
            and row.get("model", "") == model
            and row.get("status") in TERMINAL_STATUSES
        ):
            return True
    return False


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            proc.terminate()
        else:
            os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_command(
    cmd: list[str],
    log_path: Path,
    dry_run: bool = False,
    timeout_seconds: int | None = None,
) -> tuple[int, str]:
    label = " ".join(cmd)
    if dry_run:
        print(f"  [DRY] {label}")
        return 0, ""

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n$ {label}\n")
        if timeout_seconds is not None and timeout_seconds > 0:
            log_fh.write(f"# timeout_seconds={timeout_seconds}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=(os.name != "nt"),
        )
        assert proc.stdout is not None
        try:
            stdout, _ = proc.communicate(
                timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None
            )
            if stdout:
                sys.stdout.write(stdout)
                sys.stdout.flush()
                log_fh.write(stdout)
                captured.append(stdout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process(proc)
            stdout = exc.stdout or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            timeout_msg = f"\nTIMEOUT_SKIP: command exceeded {timeout_seconds} seconds\n"
            if stdout:
                sys.stdout.write(stdout)
                log_fh.write(stdout)
                captured.append(stdout)
            sys.stdout.write(timeout_msg)
            sys.stdout.flush()
            log_fh.write(timeout_msg)
            captured.append(timeout_msg)
            return 124, "".join(captured)
    return proc.returncode, "".join(captured)


def extract_run_id(stdout: str) -> str | None:
    for pattern in (
        r"wandb/(?:offline-)?run-\d{8}_\d{6}-([a-z0-9]{8})",
        r"/runs/([a-z0-9]{8})\b",
        r"run(?:\s+id)?[:\s]+([a-z0-9]{8})\b",
    ):
        m = re.search(pattern, stdout, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def stage_tsir(args: argparse.Namespace, run_dir: Path, status_path: Path, network: str, r0_label: str) -> str:
    art = artifact_name(network, r0_label)
    if args.skip_tsir:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "tsir", "status": "skipped", "artifact": art})
        return art
    if should_skip(status_path, args, network, r0_label, "tsir"):
        return art

    cfg_path = run_dir / "configs" / network / r0_label / "tsir.yml"
    write_yaml(cfg_path, build_tsir_config(network, r0_label, PRESETS[args.preset], args))
    log_path = run_dir / network / r0_label / "logs" / "tsir.log"
    cmd = [sys.executable, "main_tsir.py", "--cfg", str(cfg_path), "--data", art]
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.target_runtime_seconds)
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "tsir",
        "status": status, "artifact": art, "run_id": run_id, "returncode": rc,
        "message": "dry_run" if args.dry_run else "", "log_path": log_path,
    })
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"TSIR failed for {network}/{r0_label}; see {log_path}")
    return art


def stage_hpo(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    model: str,
    art: str,
) -> Path | None:
    if not args.with_hpo:
        return None
    tuned_model = optuna_variant_key(model)
    if should_skip(status_path, args, network, r0_label, "hpo", tuned_model):
        best = run_dir / "hpo" / f"{network}_{r0_label}_{model}" / "best_config.yml"
        return best if best.exists() else None

    preset = PRESETS[args.preset]
    base_cfg_path = run_dir / "configs" / network / r0_label / f"{model}.hpo_base.yml"
    hpo_base_cfg = build_model_config(network, model, r0_label, preset, args.save_probs, args)
    attach_hpo_budget(hpo_base_cfg, args, preset)
    write_yaml(base_cfg_path, hpo_base_cfg)
    study_name = f"{network}_{r0_label}_{model}"
    log_path = run_dir / network / r0_label / "logs" / f"hpo_{model}.log"
    cmd = [
        sys.executable,
        "main_optuna.py",
        "--cfg",
        str(base_cfg_path),
        "--data",
        f"{art}:latest",
        "--output-dir",
        str(run_dir / "hpo"),
        "--study-name",
        study_name,
        "--n-trials",
        str(args.hpo_trials),
        "--sampler",
        args.hpo_sampler,
        "--pruner",
        args.hpo_pruner,
        "--enqueue-default" if getattr(args, "hpo_enqueue_default", True) else "--no-enqueue-default",
    ]
    if args.hpo_timeout is not None:
        cmd.extend(["--timeout", str(args.hpo_timeout)])
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.target_runtime_seconds)
    best_cfg = run_dir / "hpo" / study_name / "best_config.yml"
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "hpo", "model": tuned_model,
        "status": status, "artifact": art, "returncode": rc,
        "message": str(best_cfg) if status == "success" else status,
        "log_path": log_path,
    })
    if args.dry_run:
        return best_cfg
    # Fail-soft + freshness: a non-timeout HPO crash on one scenario must not
    # abort the whole suite (it is recoverable via --resume), and we must never
    # silently reuse a stale best_config.yml from a previous run. Only return a
    # config the study (re)produced successfully this invocation; otherwise skip
    # this model's Optuna variant — the untuned final still runs.
    if status != "success":
        print(
            f"  WARNING: Optuna HPO {status} for {network}/{r0_label}/{model}; "
            f"skipping its Optuna variant (untuned final still runs). See {log_path}"
        )
        return None
    return best_cfg if best_cfg.exists() else None


def write_untuned_paired_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    model: str,
    best_cfg_path: Path | None,
) -> Path:
    """Write the untuned control config paired to an Optuna final window."""
    cfg = build_model_config(network, model, r0_label, PRESETS[args.preset], args.save_probs, args)
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        for key in ("truth_start", "n_truth"):
            if key in best_cfg.get("eval", {}):
                cfg["eval"][key] = best_cfg["eval"][key]
    elif bool(getattr(args, "with_hpo", False)):
        # HPO produced no config for this model (failed/skipped) but we are still
        # in the paired protocol: place the untuned final on the same shifted
        # held-out window the baselines use, so every method stays on one
        # substrate even in the degraded case.
        ts, nt = final_eval_window(args, PRESETS[args.preset])
        cfg["eval"]["truth_start"] = ts
        cfg["eval"]["n_truth"] = nt
    cfg.setdefault("experiment", {})["hpo_condition"] = "none"
    cfg["experiment"]["paired_optuna_variant"] = optuna_variant_key(model)
    cfg_path = run_dir / "configs" / network / r0_label / f"{model}.untuned.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def write_reused_optuna_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    model: str,
    reference_r0: str,
    best_cfg_path: Path | None,
) -> Path:
    """Write a scenario-local tuned config using network-scope Optuna params."""
    cfg = build_model_config(network, model, r0_label, PRESETS[args.preset], args.save_probs, args)
    params: dict[str, Any] = {}
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        params = dict(best_cfg.get("hpo_result", {}).get("params") or {})
        for locked in cfg.get("hpo", {}).get("locked_params", []):
            params.pop(locked, None)
        apply_trial_params(cfg, params)
        if base_model_key(model) == "dbgnn":
            cfg.setdefault("dbgnn", {})["order"] = dbgnn_order_from_key(model)
        for key in ("truth_start", "n_truth"):
            if key in best_cfg.get("eval", {}):
                cfg["eval"][key] = best_cfg["eval"][key]
        cfg["hpo_result"] = {
            **best_cfg.get("hpo_result", {}),
            "reused_from_r0": reference_r0,
            "reused_from_config": str(best_cfg_path),
        }
    elif not args.dry_run:
        raise FileNotFoundError(f"Cannot reuse missing Optuna config: {best_cfg_path}")

    cfg.setdefault("experiment", {})["hpo_condition"] = "optuna_reused"
    cfg["experiment"]["hpo_reference_r0"] = reference_r0
    cfg["experiment"]["model_variant"] = optuna_variant_key(model)
    cfg_path = run_dir / "configs" / network / r0_label / f"{model}.optuna.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def stage_train(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    model: str,
    art: str,
    cfg_override: Path | None = None,
    status_model: str | None = None,
) -> str | None:
    row_model = status_model or model
    if args.skip_train:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "train", "model": row_model, "status": "skipped", "artifact": art})
        return None
    if should_skip(status_path, args, network, r0_label, "train", row_model):
        rows = read_status(status_path)
        return next((r.get("run_id") for r in rows if r.get("network") == network and r.get("r0_label") == r0_label and r.get("stage") == "train" and r.get("model") == row_model), None)

    cfg_path = cfg_override
    if cfg_path is None:
        cfg_path = run_dir / "configs" / network / r0_label / f"{model}.yml"
        write_yaml(cfg_path, build_model_config(network, model, r0_label, PRESETS[args.preset], args.save_probs, args))
    log_path = run_dir / network / r0_label / "logs" / f"train_{row_model}.log"
    checkpoint_dir = run_dir / "checkpoints" / network / r0_label / row_model
    cmd = [
        sys.executable,
        "main_train.py",
        "--cfg",
        str(cfg_path),
        "--data",
        f"{art}:latest",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.save_probs:
        cmd.append("--save-probs")
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.target_runtime_seconds)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = (
        "success" if rc == 0
        else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout
        else "loss_guard_aborted" if rc == 88 or "LOSS_GUARD_ABORT" in stdout
        else "failed"
    )
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "train", "model": row_model,
        "status": status, "run_id": run_id, "artifact": art, "returncode": rc,
        "message": status, "log_path": log_path,
    })
    return run_id if status == "success" else None


def stage_eval(args: argparse.Namespace, run_dir: Path, status_path: Path, network: str, r0_label: str, art: str) -> str | None:
    if args.skip_eval:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "eval", "status": "skipped", "artifact": art})
        return None
    if should_skip(status_path, args, network, r0_label, "eval"):
        rows = read_status(status_path)
        return next((r.get("run_id") for r in rows if r.get("network") == network and r.get("r0_label") == r0_label and r.get("stage") == "eval"), None)

    truth_start, n_truth = final_eval_window(args, PRESETS[args.preset])
    cfg_path = run_dir / "configs" / network / r0_label / "eval.yml"
    write_yaml(cfg_path, build_eval_config(network, r0_label, PRESETS[args.preset], truth_start, n_truth))
    log_path = run_dir / network / r0_label / "logs" / "eval.log"
    cmd = [sys.executable, "main_eval.py", "--cfg", str(cfg_path), "--data", f"{art}:latest"]
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.target_runtime_seconds)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "eval",
        "status": status, "run_id": run_id, "artifact": art, "returncode": rc,
        "message": status, "log_path": log_path,
    })
    return run_id if status == "success" else None


def method_entries(status_rows: list[dict[str, str]], network: str, r0_label: str) -> list[dict[str, str]]:
    entries = []
    for row in status_rows:
        if row.get("network") != network or row.get("r0_label") != r0_label:
            continue
        if row.get("stage") == "train" and row.get("status") == "success" and row.get("run_id"):
            kind = "baseline" if base_method_key(row["model"]) in TRAINABLE_BASELINES else "model"
            entries.append({"method": row["model"], "kind": kind, "run_id": row["run_id"], "baseline": ""})
        if row.get("stage") == "eval" and row.get("status") == "success" and row.get("run_id"):
            for baseline in HEURISTIC_BASELINES:
                entries.append({"method": baseline, "kind": "baseline", "run_id": row["run_id"], "baseline": baseline})
    order = {m: i for i, m in enumerate(paired_method_order())}
    return sorted(entries, key=lambda e: order.get(e["method"], 999))


def infer_n_nodes(arrays: dict[str, np.ndarray]) -> int:
    vals = []
    if "true_sources" in arrays and len(arrays["true_sources"]):
        vals.append(int(np.max(arrays["true_sources"])) + 1)
    if "ranks" in arrays and len(arrays["ranks"]):
        vals.append(int(np.max(arrays["ranks"])))
    return max(vals) if vals else 1


def binned_topk(sizes: np.ndarray, ranks: np.ndarray, k: int, n_bins: int = 30):
    max_size = max(1.0, float(np.nanmax(sizes)))
    bins = np.linspace(1, math.ceil(max_size), n_bins + 1)
    cents = 0.5 * (bins[:-1] + bins[1:])
    vals, ses, counts = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        if not mask.any():
            vals.append(float("nan")); ses.append(float("nan")); counts.append(0)
            continue
        p = float(np.mean(ranks[mask] <= k))
        n = int(mask.sum())
        vals.append(p)
        ses.append(math.sqrt(p * (1 - p) / max(n, 1)))
        counts.append(n)
    return cents, np.array(vals), np.array(ses), np.array(counts)


def binned_rank(sizes: np.ndarray, ranks: np.ndarray, n_bins: int = 30):
    max_size = max(1.0, float(np.nanmax(sizes)))
    bins = np.linspace(1, math.ceil(max_size), n_bins + 1)
    cents = 0.5 * (bins[:-1] + bins[1:])
    means, p25, p75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        if not mask.any():
            means.append(float("nan")); p25.append(float("nan")); p75.append(float("nan"))
            continue
        vals = ranks[mask].astype(float)
        means.append(float(np.mean(vals)))
        p25.append(float(np.percentile(vals, 25)))
        p75.append(float(np.percentile(vals, 75)))
    return cents, np.array(means), np.array(p25), np.array(p75)


def add_outbreak_distribution_background(
    ax: plt.Axes,
    sizes: np.ndarray,
    n_bins: int = 40,
    height: float = 0.24,
) -> None:
    """Draw a grey outbreak-size distribution behind score/rank curves."""
    if sizes.size == 0:
        return
    max_size = max(1.0, float(np.nanmax(sizes)))
    counts, edges = np.histogram(sizes, bins=np.linspace(1, math.ceil(max_size), n_bins + 1))
    if counts.max(initial=0) == 0:
        return
    centers = 0.5 * (edges[:-1] + edges[1:])
    scaled = counts.astype(float) / counts.max() * height
    ax.fill_between(
        centers,
        0,
        scaled,
        color="0.82",
        alpha=0.65,
        step="mid",
        linewidth=0,
        zorder=0,
        label="_nolegend_",
    )


def write_plot_readme(plot_pdf: Path, title: str, description: str, params: dict[str, Any]) -> None:
    readme = plot_pdf.with_suffix(".README.md")
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Experiment Parameters",
        "",
    ]
    lines += [f"- `{k}`: `{v}`" for k, v in params.items()]
    readme.write_text("\n".join(lines) + "\n")


def plot_scenario_outputs(run_dir: Path, status_rows: list[dict[str, str]], network: str, r0_label: str) -> None:
    entries = method_entries(status_rows, network, r0_label)
    if not entries:
        return
    sc = scenario(network, r0_label)
    meta = read_network_meta(network)
    fig_dir = run_dir / network / r0_label / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    loaded = []
    for entry in entries:
        try:
            arrays = load_eval_arrays(entry["run_id"], "data", entry["baseline"] or None)
        except FileNotFoundError as exc:
            print(f"  WARNING: skipping plot series {entry['method']}: {exc}")
            continue
        n_nodes = infer_n_nodes(arrays)
        sel = arrays["sel"].astype(bool)
        loaded.append((entry, arrays, n_nodes, arrays["outbreak_sizes"][sel] * n_nodes, arrays["ranks"][sel]))
    if not loaded:
        return

    title_suffix = f"{network}, R0={sc['r0']}, beta={sc['beta']}, mu={sc['mu']}, end_t={meta['t_max']}"
    params = {"network": network, "r0": sc["r0"], "beta": sc["beta"], "mu": sc["mu"], "end_t": meta["t_max"], "min_outbreak": MIN_OUTBREAK}

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    add_outbreak_distribution_background(ax, loaded[0][3])
    for entry, _, _, sizes, ranks in loaded:
        style = model_style(entry["method"])
        cents, vals, ses, _ = binned_topk(sizes, ranks, 5)
        valid = ~np.isnan(vals)
        ax.fill_between(cents[valid], (vals - ses)[valid], (vals + ses)[valid], color=style["color"], alpha=0.12)
        ls = method_line_style(entry["method"], entry["kind"])
        ax.plot(cents[valid], vals[valid], color=style["color"], lw=2.2, ls=ls, label=style["label"])
    ax.set_title(f"Top-5 Score vs Outbreak Size: {title_suffix}")
    ax.set_xlabel("outbreak_size")
    ax.set_ylabel("top-5 score")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    out = fig_dir / "top5_vs_outbreak_compare.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, "Top-5 Score vs Outbreak Size", "Shows the fraction of valid outbreaks where the true source is ranked in the top 5, binned by absolute outbreak size. The grey background is the outbreak-size distribution for the same evaluation observations.", params)

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    for entry, _, n_nodes, sizes, ranks in loaded:
        style = model_style(entry["method"])
        cents, means, p25, p75 = binned_rank(sizes, ranks)
        valid = ~np.isnan(means)
        ax.fill_between(cents[valid], p25[valid], p75[valid], color=style["color"], alpha=0.12)
        ls = method_line_style(entry["method"], entry["kind"])
        ax.plot(cents[valid], means[valid], color=style["color"], lw=2.2, ls=ls, label=style["label"])
    ax.set_title(f"Rank vs Outbreak Size: {title_suffix}")
    ax.set_xlabel("outbreak_size")
    ax.set_ylabel("rank of true source")
    ax.legend(loc="best", fontsize=9)
    out = fig_dir / "rank_vs_outbreak_compare.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, "Rank vs Outbreak Size", "Shows mean rank of the true source with an interquartile band, binned by absolute outbreak size. Lower is better.", params)

    first_entry, first_arrays, n_nodes, _, _ = loaded[0]
    sel = first_arrays["sel"].astype(bool)
    sizes = first_arrays["outbreak_sizes"][sel] * n_nodes
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sizes, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    ax.set_title(f"Outbreak Size Distribution: {title_suffix}")
    ax.set_xlabel("outbreak_size")
    ax.set_ylabel("valid observations")
    out = fig_dir / "outbreak_size_distribution.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, "Outbreak Size Distribution", "Shows how many valid evaluation observations fall into each absolute outbreak-size bin. This contextualizes where model comparisons are well supported.", params)

    for row in status_rows:
        if row.get("network") == network and row.get("r0_label") == r0_label and row.get("stage") == "train" and row.get("status") == "success":
            plot_training_curve(run_dir, row)


def plot_training_curve(run_dir: Path, row: dict[str, str]) -> None:
    run_id = row.get("run_id", "")
    model = row.get("model", "")
    if not run_id:
        return
    run_data = Path("data") / run_id
    files = sorted(run_data.glob("loss_history_rep*.csv"))
    if not files:
        return
    fig_dir = run_dir / row["network"] / row["r0_label"] / "figures"
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    for path in files:
        epochs, train, val = [], [], []
        with open(path, newline="") as f:
            for rec in csv.DictReader(f):
                epochs.append(int(rec["epoch"]))
                train.append(float(rec["train_loss"]))
                val.append(float(rec["val_loss"]))
        rep = re.search(r"rep(\d+)", path.name)
        label = f"rep {rep.group(1)}" if rep else path.stem
        ax.plot(epochs, train, alpha=0.45, ls="--", label=f"{label} train")
        ax.plot(epochs, val, alpha=0.9, label=f"{label} val")
    ax.set_title(f"Training Curves: {model_style(model)['label']}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL loss")
    ax.legend(fontsize=8)
    out = fig_dir / f"training_curves_{model}.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, f"Training Curves: {model}", "Shows train and validation NLL per repetition. Flat curves near log(N) indicate near-uniform predictions.", row)


def metric_rows_from_status(status_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in status_rows:
        if row.get("status") != "success" or not row.get("run_id"):
            continue
        network = row["network"]
        r0_label = row["r0_label"]
        sc = scenario(network, r0_label)
        base = {"network": network, "r0_label": r0_label, "r0": sc["r0"], "beta": sc["beta"], "mu": sc["mu"], "run_id": row["run_id"]}
        if row.get("stage") == "train":
            run_data = Path("data") / row["run_id"]
            kind = "baseline" if base_method_key(row["model"]) in TRAINABLE_BASELINES else "model"
            summary_path = run_data / "metrics_summary.json"
            if summary_path.exists():
                payload = json.loads(summary_path.read_text())
                metrics = payload.get("metrics", {})
                rec = {**base, "method": row["model"], "kind": kind, "status": row["status"], **metrics}
                summary_rows.append(rec)
            for rep_path in sorted(run_data.glob("metrics_rep*.json")):
                payload = json.loads(rep_path.read_text())
                rep = payload.get("rep", "")
                for metric, value in payload.get("metrics", {}).items():
                    long_rows.append({**base, "method": row["model"], "kind": kind, "rep": rep, "metric": metric, "value": value})
        elif row.get("stage") == "eval":
            csv_path = Path("data") / row["run_id"] / "baseline_metrics.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, newline="") as f:
                for rec in csv.DictReader(f):
                    method = rec.get("model", "")
                    summary = {**base, "method": method, "kind": "baseline", "status": row["status"]}
                    for k, v in rec.items():
                        if k != "model" and v != "":
                            out_key = k if k.endswith(("_mean", "_std")) else f"{k}_mean"
                            summary[out_key] = float(v)
                    summary_rows.append(summary)
                    for k, v in rec.items():
                        if k != "model" and v != "":
                            long_rows.append({**base, "method": method, "kind": "baseline", "rep": "", "metric": k, "value": float(v)})
    return long_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_metrics_outputs(run_dir: Path, status_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    long_rows, summary_rows = metric_rows_from_status(status_rows)
    write_csv(run_dir / "metrics_long.csv", long_rows)
    write_csv(run_dir / "metrics_summary.csv", summary_rows)
    write_benchmark_table(run_dir, summary_rows)
    return summary_rows


def write_benchmark_table(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    tbl_dir = run_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    agg: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        agg.setdefault((row["network"], row["method"]), []).append(row)
    csv_rows = [["Network", "Method", "Mean MRR", "Mean Top-5", "Mean Norm-Brier"]]
    lines = [
        "% Benchmark table generated from local thesis-final metrics",
        "\\begin{tabular}{llccc}",
        "\\toprule",
        "Network & Method & MRR & Top-5 & Norm-Brier \\\\",
        "\\midrule",
    ]
    network_order = {network: i for i, network in enumerate(NETWORKS)}
    method_order = {method: i for i, method in enumerate(paired_method_order())}
    for (network, method), vals in sorted(
        agg.items(),
        key=lambda item: (
            network_order.get(item[0][0], 999),
            method_order.get(item[0][1], 999),
            item[0][1],
        ),
    ):
        mrr = np.nanmean([v.get("eval/mrr_mean", np.nan) for v in vals])
        top5 = np.nanmean([v.get("eval/top_5_mean", np.nan) for v in vals])
        brier = np.nanmean([v.get("eval/norm_brier_mean", np.nan) for v in vals])
        label = model_style(method)["label"].replace("_", "\\_")
        ntex = network.replace("_", "\\_")
        lines.append(f"{ntex} & {label} & {mrr:.4f} & {100 * top5:.1f} & {brier:.4f} \\\\")
        csv_rows.append([network, model_style(method)["label"], f"{mrr:.6g}", f"{top5:.6g}", f"{brier:.6g}"])
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tbl_dir / "benchmark_table.tex").write_text("\n".join(lines) + "\n")
    with open(tbl_dir / "benchmark_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)


def plot_metric_vs_r0(rows: list[dict[str, Any]], output: Path, metric: str, title: str, ylabel: str) -> None:
    networks = [n for n in NETWORKS if any(r["network"] == n and metric in r for r in rows)]
    if not networks:
        return
    methods = [m for m in paired_method_order() if any(r["method"] == m and metric in r for r in rows)]
    apply_style()
    ncols = 2
    nrows = math.ceil(len(networks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    for ax, network in zip(axes.ravel(), networks):
        for method in methods:
            vals = sorted((r["r0"], r[metric]) for r in rows if r["network"] == network and r["method"] == method and metric in r)
            if not vals:
                continue
            x, y = zip(*vals)
            style = model_style(method)
            ls = method_line_style(method)
            ax.plot(x, y, marker=style["marker"], color=style["color"], ls=ls, label=style["label"])
        ax.set_title(network)
        ax.set_xlabel("R0")
        ax.set_ylabel(ylabel)
    for ax in axes.ravel()[len(networks):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(title)
    fig.subplots_adjust(bottom=0.12)
    finish_fig(fig, str(output))
    write_plot_readme(output, title, f"Global thesis plot for `{metric}` across R0 values, faceted by network.", {"metric": metric, "min_outbreak": MIN_OUTBREAK})


def plot_top5_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    usable = [r for r in rows if "eval/top_5_mean" in r]
    if not usable:
        return
    row_keys = [(n, m) for n in NETWORKS for m in paired_method_order() if any(r["network"] == n and r["method"] == m for r in usable)]
    if not row_keys:
        return
    matrix = np.full((len(row_keys), len(R0_LABELS)), np.nan)
    for i, (network, method) in enumerate(row_keys):
        for j, label in enumerate(R0_LABELS):
            vals = [r["eval/top_5_mean"] for r in usable if r["network"] == network and r["method"] == method and r["r0_label"] == label]
            if vals:
                matrix[i, j] = vals[0]
    apply_style()
    fig, ax = plt.subplots(figsize=(9, max(6, 0.35 * len(row_keys))))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(R0_LABELS)), [str(R0_VALUES[l]) for l in R0_LABELS])
    ax.set_yticks(range(len(row_keys)), [f"{n} / {model_style(m)['label']}" for n, m in row_keys])
    ax.set_xlabel("R0")
    ax.set_title("Top-5 Accuracy Heatmap")
    fig.colorbar(im, ax=ax, label="Top-5 accuracy")
    finish_fig(fig, str(output))
    write_plot_readme(output, "Top-5 Accuracy Heatmap", "Compact comparison of Top-5 accuracy for every network/model/R0 combination.", {"metric": "eval/top_5_mean"})


def plot_valid_outbreaks(rows: list[dict[str, Any]], output: Path) -> None:
    usable = [r for r in rows if "eval/n_valid_mean" in r]
    if not usable:
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    for network in NETWORKS:
        vals = []
        for label in R0_LABELS:
            candidates = [r["eval/n_valid_mean"] for r in usable if r["network"] == network and r["r0_label"] == label]
            vals.append(candidates[0] if candidates else np.nan)
        if not all(np.isnan(vals)):
            ax.plot([R0_VALUES[l] for l in R0_LABELS], vals, marker="o", label=network)
    ax.set_title("Valid Outbreaks by Scenario")
    ax.set_xlabel("R0")
    ax.set_ylabel("valid observations")
    ax.legend()
    finish_fig(fig, str(output))
    write_plot_readme(output, "Valid Outbreaks by Scenario", f"Shows how many observations pass `min_outbreak={MIN_OUTBREAK}` for each network/R0 condition.", {"min_outbreak": MIN_OUTBREAK})


def load_scenario_method_arrays(
    status_rows: list[dict[str, str]],
    network: str,
    r0_label: str,
) -> list[tuple[dict[str, str], int, np.ndarray, np.ndarray]]:
    """Load (entry, n_nodes, absolute outbreak sizes, ranks) for one scenario."""
    loaded = []
    for entry in method_entries(status_rows, network, r0_label):
        try:
            arrays = load_eval_arrays(entry["run_id"], "data", entry["baseline"] or None)
        except FileNotFoundError:
            continue
        n_nodes = infer_n_nodes(arrays)
        sel = arrays["sel"].astype(bool)
        loaded.append((
            entry,
            n_nodes,
            arrays["outbreak_sizes"][sel] * n_nodes,
            arrays["ranks"][sel],
        ))
    return loaded


def plot_top5_outbreak_grid(run_dir: Path, status_rows: list[dict[str, str]]) -> None:
    """Grid plot: rows=networks, columns=R0, curves=methods, grey=outbreak distribution."""
    available = {
        (network, r0_label): load_scenario_method_arrays(status_rows, network, r0_label)
        for network in NETWORKS
        for r0_label in R0_LABELS
    }
    active_networks = [
        network for network in NETWORKS
        if any(available[(network, r0_label)] for r0_label in R0_LABELS)
    ]
    active_r0 = [
        r0_label for r0_label in R0_LABELS
        if any(available[(network, r0_label)] for network in NETWORKS)
    ]
    if not active_networks or not active_r0:
        return

    apply_style()
    fig, axes = plt.subplots(
        len(active_networks),
        len(active_r0),
        figsize=(4.2 * len(active_r0), 3.1 * len(active_networks)),
        squeeze=False,
        sharey=True,
    )
    legend_handles, legend_labels = None, None
    for row_idx, network in enumerate(active_networks):
        for col_idx, r0_label in enumerate(active_r0):
            ax = axes[row_idx][col_idx]
            loaded = available[(network, r0_label)]
            if not loaded:
                ax.axis("off")
                continue
            add_outbreak_distribution_background(ax, loaded[0][2], n_bins=28, height=0.25)
            for entry, _, sizes, ranks in loaded:
                style = model_style(entry["method"])
                cents, vals, _, _ = binned_topk(sizes, ranks, 5, n_bins=28)
                valid = ~np.isnan(vals)
                ls = method_line_style(entry["method"], entry["kind"])
                ax.plot(
                    cents[valid],
                    vals[valid],
                    color=style["color"],
                    lw=1.4,
                    ls=ls,
                    alpha=0.95,
                    label=style["label"],
                )
            sc = scenario(network, r0_label)
            ax.set_title(f"$R_0$={sc['r0']}", fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
            if col_idx == 0:
                ax.set_ylabel(f"{network}\nTop-5", fontsize=10)
            if row_idx == len(active_networks) - 1:
                ax.set_xlabel("outbreak_size")
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            ncol=min(5, len(legend_labels)),
            fontsize=8,
        )
    fig.suptitle("Top-5 Score vs Outbreak Size Across Networks and R0", fontsize=14)
    fig.subplots_adjust(bottom=0.10, hspace=0.35, wspace=0.18)
    output = run_dir / "figures" / "top5_outbreak_grid_network_r0.pdf"
    finish_fig(fig, str(output))
    write_plot_readme(
        output,
        "Top-5 Outbreak Grid by Network and R0",
        "Rows are networks, columns are R0 settings, and each panel overlays model/baseline Top-5 score against absolute outbreak size. The grey background shows the outbreak-size distribution for that panel.",
        {"metric": "eval/top_5", "min_outbreak": MIN_OUTBREAK},
    )


def plot_global_outputs(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    status_rows = read_status(run_dir / "status.csv")
    plot_metric_vs_r0(summary_rows, fig_dir / "mrr_vs_r0_by_network.pdf", "eval/mrr_mean", "MRR vs R0 by Network", "MRR")
    plot_metric_vs_r0(summary_rows, fig_dir / "top5_vs_r0_by_network.pdf", "eval/top_5_mean", "Top-5 vs R0 by Network", "Top-5 accuracy")
    plot_metric_vs_r0(summary_rows, fig_dir / "norm_brier_vs_r0.pdf", "eval/norm_brier_mean", "Norm-Brier vs R0", "Norm-Brier")
    plot_top5_outbreak_grid(run_dir, status_rows)
    plot_top5_heatmap(summary_rows, fig_dir / "top5_heatmap_network_model_r0.pdf")
    plot_valid_outbreaks(summary_rows, fig_dir / "valid_outbreaks_by_scenario.pdf")


def run_network_stats_table(args: argparse.Namespace, run_dir: Path, networks: list[str]) -> None:
    if args.skip_tables:
        return
    tbl_dir = run_dir / "tables"
    cmd = [sys.executable, "-m", "eval.tables", "network_stats", "--networks", *networks, "--output", str(tbl_dir)]
    run_command(cmd, run_dir / "logs" / "network_stats.log", args.dry_run)


def _refresh_result_bundle(
    run_dir: Path,
    status_path: Path,
    experiment_name: str = "thesis_final",
) -> Path:
    """Refresh the publication-facing result bundle."""
    return sync_publication_result(
        run_dir=run_dir,
        status_rows=read_status(status_path),
        experiment_name=experiment_name,
    )


def main(
    argv: list[str] | None = None,
    default_overrides: dict[str, Any] | None = None,
) -> None:
    args = parse_args(argv, default_overrides)
    global HEURISTIC_BASELINES, BASELINES
    # Bind the experiment name into every bundle refresh in this run. Shadows the
    # module-level helper for the body of main() and its nested closures only.
    refresh_result_bundle = partial(
        _refresh_result_bundle, experiment_name=args.experiment_name
    )
    networks = list(args.networks)
    if args.reverse_networks:
        networks.reverse()
    r0_labels = normalize_r0_labels(args.r0)
    args.models = normalize_model_keys(args.models)
    HEURISTIC_BASELINES = normalize_baseline_keys(
        args.baselines,
        exclude_expensive=bool(getattr(args, "exclude_expensive_baselines", False)),
    )
    BASELINES = TRAINABLE_BASELINES + HEURISTIC_BASELINES

    # Optional n_runs / n_truth overrides for tighter metric estimates without
    # touching the (expensive) training-side budgets. Mutate the preset entry in
    # place so every PRESETS[args.preset] read downstream picks up the values.
    if any(v is not None for v in (args.n_runs, args.mc_runs, args.n_mc, args.reps, args.n_truth)):
        base = PRESETS[args.preset]
        n_runs = int(args.n_runs) if args.n_runs is not None else base.n_runs
        mc_runs = int(args.mc_runs) if args.mc_runs is not None else base.mc_runs
        n_mc = int(args.n_mc) if args.n_mc is not None else base.n_mc
        reps = int(args.reps) if args.reps is not None else base.reps
        n_truth = int(args.n_truth) if args.n_truth is not None else base.n_truth
        if n_mc > mc_runs:
            raise ValueError(f"--n-mc ({n_mc}) cannot exceed --mc-runs ({mc_runs})")
        if reps * n_truth > n_runs:
            raise ValueError(
                f"--reps * --n-truth ({reps} * {n_truth} = {reps * n_truth}) "
                f"cannot exceed --n-runs ({n_runs})"
            )
        PRESETS[args.preset] = replace(
            base,
            n_runs=n_runs,
            mc_runs=mc_runs,
            n_mc=n_mc,
            reps=reps,
            n_truth=n_truth,
        )
    preset = PRESETS[args.preset]
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"
    hpo_reference_r0 = resolve_hpo_reference_r0(args.hpo_reference_r0, r0_labels)
    artifact_cache: dict[tuple[str, str], str] = {}
    hpo_best_cache: dict[tuple[str, str], Path | None] = {}

    def ensure_artifact(network: str, r0_label: str) -> str:
        key = (network, r0_label)
        if key not in artifact_cache:
            artifact_cache[key] = stage_tsir(args, run_dir, status_path, network, r0_label)
            refresh_result_bundle(run_dir, status_path)
        return artifact_cache[key]

    def resolve_hpo_config(network: str, r0_label: str, method: str, artifact: str) -> Path | None:
        if args.hpo_scope == "scenario":
            best_cfg = stage_hpo(args, run_dir, status_path, network, r0_label, method, artifact)
            refresh_result_bundle(run_dir, status_path)
            return best_cfg

        cache_key = (network, method)
        if cache_key not in hpo_best_cache:
            ref_artifact = artifact if r0_label == hpo_reference_r0 else ensure_artifact(network, hpo_reference_r0)
            hpo_best_cache[cache_key] = stage_hpo(
                args,
                run_dir,
                status_path,
                network,
                hpo_reference_r0,
                method,
                ref_artifact,
            )
            refresh_result_bundle(run_dir, status_path)

        best_cfg = hpo_best_cache[cache_key]
        if r0_label == hpo_reference_r0:
            return best_cfg

        tuned_cfg = write_reused_optuna_config(
            args,
            run_dir,
            network,
            r0_label,
            method,
            hpo_reference_r0,
            best_cfg,
        )
        update_status(status_path, {
            "network": network,
            "r0_label": r0_label,
            "stage": "hpo",
            "model": optuna_variant_key(method),
            "status": "success",
            "artifact": artifact,
            "returncode": 0,
            "message": f"reused network-scope HPO from {hpo_reference_r0}: {best_cfg}",
            "log_path": "",
        })
        refresh_result_bundle(run_dir, status_path)
        return tuned_cfg

    write_manifest(run_dir, args, networks, r0_labels)
    refresh_result_bundle(run_dir, status_path)
    print("=" * 72)
    print("Thesis Final Experiment Runner")
    print("=" * 72)
    print(f"Run dir  : {run_dir}")
    print(f"Preset   : {args.preset} ({preset})")
    print(f"Networks : {', '.join(networks)}")
    print(f"R0       : {', '.join(r0_labels)}")
    print(f"Models   : {', '.join(args.models)}")
    print(f"Baselines: {', '.join(BASELINES)}")
    print(
        "HPO      : "
        + (
            f"enabled, paired untuned/+Optuna finals, trials={args.hpo_trials}, "
            f"sampler={args.hpo_sampler}, pruner={args.hpo_pruner}, "
            f"scope={args.hpo_scope}, reference_r0={hpo_reference_r0}, "
            f"enqueue_default={bool(getattr(args, 'hpo_enqueue_default', True))}, "
            f"trial_epochs={_positive_cap(args.hpo_epochs)}, "
            f"trial_n_truth={effective_hpo_n_truth(args, preset)}, "
            f"trial_n_mc={effective_hpo_n_mc(args, preset)}"
            if args.with_hpo else "disabled"
        )
    )
    print(
        "Train cap: "
        f"epochs={_positive_cap(args.max_train_epochs)}, "
        f"patience={_positive_cap(args.max_train_patience)}"
    )
    print(
        "Reduction: "
        + (
            f"{args.reduction}, timeout={args.target_runtime_seconds}s, "
            f"target_nodes={args.sample_target_nodes}, "
            f"time_window={args.time_window_steps}, seed={args.reduction_seed}, "
            f"calibration={'off' if args.use_full_betas else 'on'}"
            if reduction_is_enabled(args) else "disabled"
        )
    )
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed")

    for network in networks:
        if network not in BETAS:
            raise ValueError(f"No beta matrix configured for network '{network}'")
        for r0_label in r0_labels:
            sc = scenario(network, r0_label)
            print(f"\n### {network} / {r0_label}  R0={sc['r0']} beta={sc['beta']} mu={sc['mu']}")
            try:
                art = ensure_artifact(network, r0_label)
            except Exception as exc:
                print(f"  FATAL TSIR stage failed, skipping scenario: {exc}")
                refresh_result_bundle(run_dir, status_path)
                continue

            for model in args.models:
                if model not in MODELS:
                    print(f"  SKIP unknown/out-of-scope model: {model}")
                    update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "train", "model": model, "status": "skipped", "message": "unknown_model"})
                    continue
                if args.with_hpo:
                    best_cfg = resolve_hpo_config(network, r0_label, model, art)
                    untuned_cfg = write_untuned_paired_config(args, run_dir, network, r0_label, model, best_cfg)
                    stage_train(args, run_dir, status_path, network, r0_label, model, art, untuned_cfg)
                    refresh_result_bundle(run_dir, status_path)
                    if best_cfg is not None or args.dry_run:
                        stage_train(
                            args,
                            run_dir,
                            status_path,
                            network,
                            r0_label,
                            model,
                            art,
                            best_cfg,
                            status_model=optuna_variant_key(model),
                        )
                        refresh_result_bundle(run_dir, status_path)
                    else:
                        update_status(status_path, {
                            "network": network,
                            "r0_label": r0_label,
                            "stage": "train",
                            "model": optuna_variant_key(model),
                            "status": "skipped",
                            "artifact": art,
                            "message": "missing_hpo_config",
                        })
                else:
                    stage_train(args, run_dir, status_path, network, r0_label, model, art)
                    refresh_result_bundle(run_dir, status_path)

            for baseline in TRAINABLE_BASELINES:
                if args.with_hpo:
                    best_cfg = resolve_hpo_config(network, r0_label, baseline, art)
                    untuned_cfg = write_untuned_paired_config(args, run_dir, network, r0_label, baseline, best_cfg)
                    stage_train(args, run_dir, status_path, network, r0_label, baseline, art, untuned_cfg)
                    refresh_result_bundle(run_dir, status_path)
                    if best_cfg is not None or args.dry_run:
                        stage_train(
                            args,
                            run_dir,
                            status_path,
                            network,
                            r0_label,
                            baseline,
                            art,
                            best_cfg,
                            status_model=optuna_variant_key(baseline),
                        )
                        refresh_result_bundle(run_dir, status_path)
                    else:
                        update_status(status_path, {
                            "network": network,
                            "r0_label": r0_label,
                            "stage": "train",
                            "model": optuna_variant_key(baseline),
                            "status": "skipped",
                            "artifact": art,
                            "message": "missing_hpo_config",
                        })
                else:
                    stage_train(args, run_dir, status_path, network, r0_label, baseline, art)
                    refresh_result_bundle(run_dir, status_path)

            stage_eval(args, run_dir, status_path, network, r0_label, art)
            refresh_result_bundle(run_dir, status_path)
            if not args.skip_viz and not args.dry_run:
                plot_scenario_outputs(run_dir, read_status(status_path), network, r0_label)
                refresh_result_bundle(run_dir, status_path)

    status_rows = read_status(status_path)
    summary_rows = write_metrics_outputs(run_dir, status_rows)
    refresh_result_bundle(run_dir, status_path)
    if not args.skip_viz and not args.dry_run:
        plot_global_outputs(run_dir, summary_rows)
        refresh_result_bundle(run_dir, status_path)
    run_network_stats_table(args, run_dir, networks)
    result_dir = refresh_result_bundle(run_dir, status_path)

    print("\nDone.")
    print(f"Status : {status_path}")
    print(f"Result : {result_dir}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
