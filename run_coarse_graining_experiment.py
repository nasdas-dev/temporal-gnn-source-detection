"""
H2 coarse-graining experiment runner.

Tests the second thesis hypothesis:

    H2: As temporal resolution is coarsened, the performance of all temporal
        representations should converge toward the StaticGNN baseline, because
        the ordering information available to the models is progressively
        removed.

    Practical question: among the temporal approaches, which gives the most
        reliable improvement over StaticGNN at a justifiable computational cost?

Methodology
-----------
Temporal coarse-graining is applied to ONE shared coarse-grained contact network
*before* model-specific graph construction (see
``gnn.graph_builder.coarsen_temporal_network`` and the ``coarsen.delta_t`` config
key honoured by ``main_train.py``).  For each Δt the same coarse-grained network
feeds the TemporalGNN snapshots, the BacktrackingNetwork edge textures, and the
DBGNN De Bruijn graph, so the only thing that changes across the sweep is the
temporal resolution.

The TSIR simulation and the StaticGNN / heuristic baselines are Δt-invariant
(the epidemic spreads on the full-resolution network; Δt only coarsens the
*model's view*).  So per network:

    * TSIR runs once (full resolution) at a single fixed R0.
    * Heuristic baselines run once (``main_eval.py``) as a reference.
    * StaticGNN runs once (paired untuned/Optuna) as the convergence target.
    * Each temporal model sweeps Δt (paired untuned/Optuna; HPO is tuned once at
      Δt=1 per (network, model) and reused at coarser Δt).

Δt is a per-network geometric span (1, 2, 4, 8, … up to a full-collapse bin), so
each network spans native resolution → static.  At the fully-collapsed Δt every
DBGNN order reduces to the SAME pure first-order static GCN: a single time bin has
no causal walk completions, so the higher-order De Bruijn branch is dropped for
all orders (see ``gnn.graph_builder.build_de_bruijn_graph``).  The k2 and k3
collapse points are therefore order-invariant by construction (any residual gap is
just training/init noise, as between two StaticGNN seeds).  Besides the usual
metrics, every
training run also logs graph-construction time, training time, peak memory, edge
counts, snapshot counts, edge-texture length, and De Bruijn node/edge counts (see
``main_train.py``), which answer the cost half of the practical question.

Every run maintains a publication bundle under:
    results/h2_coarse_graining/<run-name>/result/

Default run::

    python run_coarse_graining_experiment.py
        # paper_24h, networks lyon_ward/malawi/students/escort, R0=2.0,
        # all temporal models + StaticGNN + heuristic baselines, paired Optuna

Useful controls::

    python run_coarse_graining_experiment.py --dry-run --preset fast --networks lyon_ward --delta-t 1 4 97
    python run_coarse_graining_experiment.py --preset fast --no-hpo --networks lyon_ward
    python run_coarse_graining_experiment.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_all_experiments import (
    BETAS,
    GRAD_CLIP_NORM,
    LOSS_GUARD,
    MIN_OUTBREAK,
    MUS,
    R0_VALUES,
    STERCHI_TRAIN,
    TARGET_INFECTED,
    build_eval_config,
    extract_run_id,
    final_eval_window,
    normalize_r0_labels,
    read_network_meta,
    scenario,
)
from run_dbgnn_higher_order_experiment import (
    NetworkStats,
    apply_final_train_caps,
    attach_hpo_budget,
    compute_sample_budget,
    effective_hpo_n_mc,
    effective_hpo_n_truth,
    read_full_network_stats,
    resolve_run_dir,
    run_command,
    sampling_cfg_for_network,
    write_yaml,
    _positive_cap,
)
from hpo import apply_trial_params
from scripts.publication_bundle import sync_publication_result
from viz.style import apply_style, finish_fig


WANDB_PROJECT = "source-detection"
OPTUNA_SUFFIX = "_optuna"
# GRAD_CLIP_NORM is imported from run_all_experiments so all runners share one
# value; applied uniformly to every model here so the Δt comparison stays fair.

DEFAULT_NETWORKS = ["lyon_ward", "malawi", "students", "escort"]
DEFAULT_R0 = "r0_20"

# Temporal models that consume the shared coarse-grained network (sweep Δt).
TEMPORAL_MODELS = ["temporal_gnn", "backtracking", "dbgnn_k2", "dbgnn_k3"]
STATIC_MODEL = "static_gnn"

# model_key -> (template file stem, cfg["model"] value)
MODEL_TEMPLATE = {
    "temporal_gnn": "temporal_gnn",
    "backtracking": "backtracking",
    "dbgnn_k2": "dbgnn",
    "dbgnn_k3": "dbgnn",
    "static_gnn": "static_gnn",
}
MODEL_BASE = {
    "temporal_gnn": "temporal_gnn",
    "backtracking": "backtracking",
    "dbgnn_k2": "dbgnn",
    "dbgnn_k3": "dbgnn",
    "static_gnn": "static_gnn",
}
MODEL_LABEL = {
    "temporal_gnn": "TemporalGNN",
    "backtracking": "BacktrackingNet",
    "dbgnn_k2": "DBGNN (k=2)",
    "dbgnn_k3": "DBGNN (k=3)",
    "static_gnn": "StaticGNN",
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
    # Multi-seed publication preset mirroring run_all_experiments' tuner preset:
    # the final model is trained/evaluated 3 times (reps=3) with a shared held-out
    # test window, so each Δt point carries a Sterchi-style 95% CI over
    # training/init noise. n_runs leaves room for a disjoint HPO-validation window
    # plus the 3*250 held-out final-test window (final_stop=850 <= 1200).
    "tuner": Preset(n_runs=1200, mc_runs=500, n_mc=500, reps=3, n_truth=250),
}


STATUS_FIELDS = [
    "network",
    "r0_label",
    "stage",
    "model",
    "delta_t",
    "variant",
    "status",
    "run_id",
    "artifact",
    "returncode",
    "message",
    "log_path",
]
TERMINAL_STATUSES = {"success", "loss_guard_aborted", "skipped", "timeout_skipped"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper_24h",
                   help="Runtime/quality preset. Use max_quality for the full grid.")
    p.add_argument("--n-runs", type=int, default=None,
                   help="Override the preset's ground-truth simulation count (TSIR n_runs). "
                        "Raise this together with --n-truth for tighter metric estimates.")
    p.add_argument("--mc-runs", type=int, default=None,
                   help="Override the preset's Monte Carlo simulation count stored per source.")
    p.add_argument("--n-mc", type=int, default=None,
                   help="Override the per-repetition number of MC simulations used for training.")
    p.add_argument("--reps", type=int, default=None,
                   help="Override the number of independent train/eval repetitions.")
    p.add_argument("--n-truth", type=int, default=None,
                   help="Override the preset's evaluation sample count (final truth window). "
                        "Must be <= --n-runs; cheap (inference + metrics only, no extra training).")
    p.add_argument("--networks", nargs="+", default=DEFAULT_NETWORKS, help="Network names")
    p.add_argument("--r0", default=DEFAULT_R0, help="Single R0 label/number for the Δt sweep (default r0_20=2.0)")
    p.add_argument("--temporal-models", nargs="+", default=TEMPORAL_MODELS,
                   help="Subset of temporal models to sweep Δt over")
    p.add_argument("--delta-t", nargs="+", type=int, default=None,
                   help="Explicit Δt grid (overrides the per-network geometric span)")
    p.add_argument("--max-delta-levels", type=int, default=None,
                   help="Cap the number of Δt levels (endpoints are always kept)")
    p.add_argument("--output", default="results/h2_coarse_graining", help="Root results directory")
    p.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run directory")
    p.add_argument("--force", action="store_true", help="Rerun stages even when a terminal status exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    p.add_argument("--save-probs", action="store_true", help="Save probs_rep*.pt tensors from main_train.py")
    p.add_argument("--no-expensive-baselines", dest="exclude_expensive_baselines",
                   action="store_true",
                   help="Drop the per-observation baselines (subgraph betweenness/jordan_center "
                        "and the SME soft_margin). Cheap on lyon/malawi but available for parity.")
    # HPO
    p.add_argument("--with-hpo", dest="with_hpo", action="store_true",
                   help="Run paired untuned and Optuna-tuned finals (default)")
    p.add_argument("--no-hpo", dest="with_hpo", action="store_false",
                   help="Disable Optuna and run only the untuned configs")
    p.add_argument("--hpo-trials", type=int, default=5, help="Optuna trials per network/model")
    p.add_argument("--hpo-timeout", type=int, default=None, help="Optional Optuna study timeout in seconds")
    p.add_argument("--hpo-n-truth", type=int, default=100, help="Validation truth runs per HPO study")
    p.add_argument("--hpo-n-mc", type=int, default=80, help="MC samples used inside HPO trials only")
    p.add_argument("--hpo-epochs", type=int, default=120, help="Epoch cap inside HPO trials only")
    p.add_argument("--hpo-patience", type=int, default=12, help="Patience cap inside HPO trials only")
    p.add_argument("--max-train-epochs", type=int, default=500,
                   help="Cap final train.epochs; set 0 to keep template values. "
                        "Default 500 matches Sterchi (rely on early-stopping to converge).")
    p.add_argument("--max-train-patience", type=int, default=20,
                   help="Cap final train.patience; set 0 to keep template values")
    p.add_argument("--hpo-sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--hpo-pruner", choices=["hyperband", "median", "none"], default="hyperband")
    # Reduction / sampling (mirrors the other thesis runners)
    p.add_argument("--reduction", choices=["safe_1h", "none"], default="safe_1h",
                   help="Network reduction policy for TSIR artifacts. Default: safe_1h")
    p.add_argument("--no-reduction", dest="reduction", action="store_const", const="none",
                   help="Disable default network reduction.")
    p.add_argument("--no-sampling", action="store_true", help="Disable large-network sampling")
    p.add_argument("--sample-method", default="balanced_activity_snowball",
                   choices=["balanced_activity_snowball", "activity_snowball"])
    p.add_argument("--sample-node-threshold", type=int, default=300,
                   help="Only sample networks with more than this many nodes")
    p.add_argument("--sample-target-nodes", type=int, default=300,
                   help="Target sampled node count for networks above the threshold")
    p.add_argument("--sample-reference", default="students", help="Network used to derive the sampling budget")
    p.add_argument("--sample-budget-factor", type=float, default=72.0, help="Reference node*edge cost divisor")
    p.add_argument("--min-sample-nodes", type=int, default=8)
    p.add_argument("--stratification-bins", type=int, default=4)
    p.add_argument("--time-window-steps", default="auto", help="Temporal window length for safe_1h, or auto.")
    p.add_argument("--reduction-reps", type=int, default=1)
    p.add_argument("--use-full-betas", action="store_true",
                   help="Use static BETAS instead of per-artifact beta calibration.")
    p.add_argument("--timeout-seconds", dest="timeout_seconds", type=int, default=7200,
                   help="Skip any TSIR/train/eval command that runs longer than this. Default 2h.")
    p.add_argument("--target-runtime-seconds", dest="timeout_seconds", type=int,
                   help="Alias for --timeout-seconds")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reduction-seed", dest="seed", type=int, help="Alias for --seed")
    # DBGNN resource caps (mirror the higher-order runner)
    p.add_argument("--dbgnn-batch-size", type=int, default=8, help="Max train.batch_size for DBGNN")
    p.add_argument("--max-temporal-states", type=int, default=2_000_000)
    p.add_argument("--max-db-nodes", type=int, default=500_000)
    p.add_argument("--max-db-edges", type=int, default=2_000_000)
    # Skips
    p.add_argument("--skip-tsir", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-static", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--skip-tables", action="store_true")
    p.set_defaults(with_hpo=True)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Δt grid
# ---------------------------------------------------------------------------

def effective_t_max(network: str, args: argparse.Namespace) -> int:
    """Effective number of time steps after the reduction policy.

    The ``safe_1h`` reduction applies a representative temporal window (≈365
    steps) only when ``t_max > 1000`` (e.g. escort). Otherwise the full ``t_max``
    is used. The Δt grid is built from this effective span.
    """
    meta = read_network_meta(network)
    t_max = int(meta["t_max"])
    if getattr(args, "reduction", "safe_1h") != "none" and t_max > 1000:
        return min(t_max, 365)
    return t_max


def delta_t_grid(eff_t_max: int, explicit: list[int] | None = None,
                 max_levels: int | None = None) -> list[int]:
    """Per-network geometric Δt span: 1, 2, 4, 8, … up to a full-collapse bin."""
    if explicit:
        grid = sorted({int(d) for d in explicit if int(d) >= 1})
    else:
        grid = [1]
        d = 2
        while d <= eff_t_max:
            grid.append(d)
            d *= 2
        # Full collapse: a bin wider than the whole span maps everything to one slice.
        collapse = int(eff_t_max) + 1
        if collapse > grid[-1]:
            grid.append(collapse)
        grid = sorted(set(grid))
    if max_levels and len(grid) > max_levels:
        # Keep both endpoints, evenly subsample the interior.
        idx = np.linspace(0, len(grid) - 1, max_levels).round().astype(int)
        grid = sorted({grid[i] for i in idx})
    return grid


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def variant_name(model_key: str, delta_t: int | None = None) -> str:
    if model_key == STATIC_MODEL:
        return STATIC_MODEL
    return f"{model_key}_dt{delta_t}"


def optuna_variant_name(model_key: str, delta_t: int | None = None) -> str:
    return f"{variant_name(model_key, delta_t)}{OPTUNA_SUFFIX}"


def _template_path(network: str, base: str) -> Path:
    direct = Path("exp") / network / f"{base}.yml"
    if direct.exists():
        return direct
    fallback = Path("exp/france_office") / f"{base}.yml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No '{base}' config template found for {network}")


def build_model_config(
    network: str,
    r0_label: str,
    model_key: str,
    delta_t: int,
    preset: Preset,
    save_probs: bool,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    """Build a per-cell training config.

    Temporal models receive ``coarsen.delta_t``; StaticGNN does not (it is
    Δt-invariant). DBGNN's causal window ``delta`` is expressed in coarsened
    units so the real-time causal horizon stays roughly constant across Δt.
    """
    base = MODEL_TEMPLATE[model_key]
    with open(_template_path(network, base)) as f:
        cfg = yaml.safe_load(f) or {}

    cfg["model"] = MODEL_BASE[model_key]
    cfg["eval"] = {
        **cfg.get("eval", {}),
        "min_outbreak": MIN_OUTBREAK,
        "top_k": [1, 3, 5, 10],
        "credible_p": [0.80, 0.90],
        "inverse_rank_offset": [0],
        "n_truth": preset.n_truth,
        # Sterchi-exact: all reps scored on the same held-out window so the 95%
        # CIs reflect training/init noise (mirrors run_all_experiments).
        "shared_eval_window": True,
    }
    cfg["train"] = {
        **cfg.get("train", {}),
        # Frozen, Sterchi-aligned training hyperparameters — identical to H1, so
        # Optuna can only change architecture and the Δt curves are not confounded
        # by per-cell optimiser drift. Train to 500 epochs with early-stopping.
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
    if MODEL_BASE[model_key] in ("static_gnn", "static_mlp"):
        cfg["train"]["batch_size"] = STERCHI_TRAIN["batch_size"]
    cfg.setdefault("output", {})["save_probs"] = save_probs

    if model_key in TEMPORAL_MODELS:
        # Unified pre-construction coarsening shared by every temporal builder.
        cfg["coarsen"] = {"delta_t": int(delta_t)}

    dbgnn_delta_floored = False
    if base == "dbgnn":
        order = 2 if model_key == "dbgnn_k2" else 3
        db = cfg.setdefault("dbgnn", {})
        db["order"] = int(order)
        delta_real = int(db.get("delta", 24) or 24)
        # Scale delta with 1/Δt to keep the real-time causal window ~constant.
        # CAVEAT: delta is an integer number of (coarsened) bins with a floor of
        # 1, so once Δt grows past ~delta_real the scaled value saturates at 1
        # bin = Δt real steps and the horizon stops shrinking (it grows back with
        # Δt). The "constant real-time horizon" therefore only holds for the
        # finer half of the sweep; ``dbgnn_delta_floored`` is recorded in the
        # experiment block (and manifest) so the coarse-Δt DBGNN points are
        # transparently flagged as NOT the controlled comparison.
        if delta_t > 1:
            ideal = delta_real / int(delta_t)
            db["delta"] = max(1, round(ideal))
            dbgnn_delta_floored = ideal < 1.0
        else:
            db["delta"] = delta_real
        db["time_bin_size"] = 1  # binning handled centrally by coarsen.delta_t
        db["max_temporal_states"] = int(getattr(args, "max_temporal_states", 2_000_000))
        db["max_db_nodes"] = int(getattr(args, "max_db_nodes", 500_000))
        db["max_db_edges"] = int(getattr(args, "max_db_edges", 2_000_000))
        db["bipartite_agg"] = db.get("bipartite_agg", "sum")
        db["directed"] = read_network_meta(network)["directed"]
        cap = int(getattr(args, "dbgnn_batch_size", 8))
        cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", cap)), cap)

    apply_final_train_caps(cfg, args)

    sc = scenario(network, r0_label)
    cfg["experiment"] = {
        "name": "h2_coarse_graining",
        "network": network,
        "r0_label": r0_label,
        "model_key": model_key,
        "delta_t": int(delta_t),
        "variant": variant_name(model_key, delta_t),
        **sc,
    }
    if base == "dbgnn":
        cfg["experiment"]["dbgnn_delta"] = int(cfg["dbgnn"]["delta"])
        # True when 1/Δt scaling would want delta < 1 bin: the real-time causal
        # horizon is no longer held constant at this Δt (see build comment).
        cfg["experiment"]["dbgnn_delta_floored"] = bool(dbgnn_delta_floored)
    if preset.n_runs < preset.reps * preset.n_truth:
        raise ValueError(
            f"Invalid preset: n_runs={preset.n_runs} < reps*n_truth={preset.reps * preset.n_truth}"
        )
    return cfg


def write_untuned_paired_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    model_key: str,
    delta_t: int,
    best_cfg_path: Path | None,
) -> Path:
    """Untuned control paired to the same held-out final truth window as Optuna."""
    cfg = build_model_config(network, r0_label, model_key, delta_t, PRESETS[args.preset], args.save_probs, args)
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        for key in ("truth_start", "n_truth"):
            if key in best_cfg.get("eval", {}):
                cfg["eval"][key] = best_cfg["eval"][key]
    elif bool(getattr(args, "with_hpo", False)):
        # No Optuna config for this cell (failed/skipped): still place the
        # untuned final on the shifted held-out window the baselines use.
        ts, nt = final_eval_window(args, PRESETS[args.preset])
        cfg["eval"]["truth_start"] = ts
        cfg["eval"]["n_truth"] = nt
    cfg.setdefault("experiment", {})["hpo_condition"] = "none"
    cfg["experiment"]["paired_optuna_variant"] = optuna_variant_name(model_key, delta_t)
    cfg_path = run_dir / "configs" / network / f"{variant_name(model_key, delta_t)}.untuned.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def _scaled_dbgnn_delta(native_delta: Any, delta_t: int) -> tuple[int | None, bool]:
    """Scale a native-resolution DBGNN causal horizon into coarsened bins."""
    if native_delta is None:
        return None, False
    ideal = float(native_delta) / int(delta_t)
    return max(1, round(ideal)), ideal < 1.0


def write_reused_optuna_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    model_key: str,
    delta_t: int,
    best_cfg_path: Path | None,
) -> Path:
    """Tuned config at a coarser Δt reusing the Δt=1 Optuna params."""
    cfg = build_model_config(network, r0_label, model_key, delta_t, PRESETS[args.preset], args.save_probs, args)
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        params = dict(best_cfg.get("hpo_result", {}).get("params") or {})
        structural_params = {"coarsen.delta_t"}
        if MODEL_BASE[model_key] == "dbgnn":
            structural_params.update({
                "dbgnn.order",
                "dbgnn.delta",
                "dbgnn.time_bin_size",
            })
        for key in structural_params:
            params.pop(key, None)
        apply_trial_params(cfg, params)
        # Re-assert the structural variables that HPO must not change.
        if MODEL_BASE[model_key] == "dbgnn":
            db = cfg.setdefault("dbgnn", {})
            native_delta = (
                best_cfg.get("dbgnn", {}).get(
                    "delta",
                    (best_cfg.get("hpo_result", {}).get("params") or {}).get("dbgnn.delta"),
                )
            )
            scaled_delta, delta_floored = _scaled_dbgnn_delta(native_delta, delta_t)
            db["order"] = 2 if model_key == "dbgnn_k2" else 3
            db["delta"] = scaled_delta
            db["time_bin_size"] = 1
            cfg.setdefault("experiment", {})["dbgnn_delta"] = scaled_delta
            cfg["experiment"]["dbgnn_delta_floored"] = bool(delta_floored)
            cfg["experiment"]["dbgnn_native_delta"] = native_delta
        if model_key in TEMPORAL_MODELS:
            cfg["coarsen"] = {"delta_t": int(delta_t)}
        for key in ("truth_start", "n_truth"):
            if key in best_cfg.get("eval", {}):
                cfg["eval"][key] = best_cfg["eval"][key]
        cfg["hpo_result"] = {
            **best_cfg.get("hpo_result", {}),
            "reused_from_delta_t": 1,
            "reused_from_config": str(best_cfg_path),
        }
    elif not args.dry_run:
        raise FileNotFoundError(f"Cannot reuse missing Optuna config: {best_cfg_path}")

    cfg.setdefault("experiment", {})["hpo_condition"] = "optuna_reused"
    cfg["experiment"]["variant"] = optuna_variant_name(model_key, delta_t)
    cfg_path = run_dir / "configs" / network / f"{variant_name(model_key, delta_t)}.optuna.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def build_tsir_config(
    network: str,
    r0_label: str,
    preset: Preset,
    sample_cfg: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    meta = read_network_meta(network)
    sc = scenario(network, r0_label)
    nwk_cfg: dict[str, Any] = {
        "type": "empirical",
        "name": network,
        "t_max": meta["t_max"],
        "directed": meta["directed"],
    }
    if sample_cfg is not None:
        reduction_cfg: dict[str, Any] = {
            "enabled": "auto",
            "preset": "safe_1h",
            "runtime_target_s": int(getattr(args, "timeout_seconds", 3600) or 3600),
            "node": sample_cfg,
        }
        if int(meta["t_max"]) > 1000:
            time_cfg: dict[str, Any] = {
                "method": "representative_window",
                "apply_if_time_steps_gt": 1000,
                "max_steps_days": 365,
                "candidate_windows": 32,
                "reindex_to_zero": True,
            }
            window_steps = str(getattr(args, "time_window_steps", "auto"))
            if window_steps != "auto":
                time_cfg["max_steps"] = int(window_steps)
            reduction_cfg["time"] = time_cfg
        nwk_cfg["reduction"] = reduction_cfg
    sir_cfg: dict[str, Any] = {
        "beta": sc["beta"],
        "mu": sc["mu"],
        "start_t": 0,
        "end_t": meta["t_max"],
        "n_runs": preset.n_runs,
        "mc_runs": preset.mc_runs,
    }
    if not bool(getattr(args, "use_full_betas", False)):
        # end_t is calibrated to the Sterchi-style ≈40%-infected snapshot. The
        # SIR simulation is identical across Δt (Δt only changes the model's
        # temporal binning), so one calibrated end_t per network/R0 is shared
        # across the resolution sweep via the cache.
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
            "seed": int(getattr(args, "seed", 42)),
        }
    return {
        "nwk": nwk_cfg,
        "sir": sir_cfg,
        "experiment": {"name": "h2_coarse_graining", "network": network, "r0_label": r0_label, **sc},
    }


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

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


def _status_key(row: dict[str, Any]) -> tuple:
    return (
        row.get("network", ""),
        row.get("r0_label", ""),
        row.get("stage", ""),
        row.get("model", ""),
        str(row.get("delta_t", "")),
        row.get("variant", ""),
    )


def update_status(path: Path, row: dict[str, Any]) -> None:
    rows = read_status(path)
    key = _status_key(row)
    normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
    kept = [r for r in rows if _status_key(r) != key]
    kept.append(normalized)
    write_status(path, kept)


def should_skip(
    status_path: Path,
    args: argparse.Namespace,
    network: str,
    r0_label: str,
    stage: str,
    model: str = "",
    delta_t: int | str = "",
    variant: str = "",
) -> bool:
    if args.force or not args.resume:
        return False
    for row in read_status(status_path):
        if (
            row.get("network") == network
            and row.get("r0_label") == r0_label
            and row.get("stage") == stage
            and row.get("model", "") == model
            and row.get("delta_t", "") == str(delta_t)
            and row.get("variant", "") == variant
            and row.get("status") in TERMINAL_STATUSES
        ):
            return True
    return False


def artifact_name(network: str, r0_label: str) -> str:
    return f"h2_coarse_{network}_{r0_label}"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_tsir(args, run_dir, status_path, network, r0_label, sample_cfg) -> str | None:
    artifact = artifact_name(network, r0_label)
    if args.skip_tsir:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "tsir",
                                    "status": "skipped", "artifact": artifact})
        return artifact
    if should_skip(status_path, args, network, r0_label, "tsir"):
        return artifact
    cfg_path = run_dir / "configs" / network / "tsir.yml"
    write_yaml(cfg_path, build_tsir_config(network, r0_label, PRESETS[args.preset], sample_cfg, args))
    log_path = run_dir / network / "logs" / "tsir.log"
    cmd = ["python", "main_tsir.py", "--cfg", str(cfg_path), "--data", artifact]
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "tsir", "status": status,
        "artifact": artifact, "run_id": extract_run_id(stdout) or ("dryrun00" if args.dry_run else ""),
        "returncode": rc, "message": "dry_run" if args.dry_run else status, "log_path": log_path,
    })
    if status != "success":
        return artifact if args.dry_run else None
    return artifact


def stage_eval(args, run_dir, status_path, network, r0_label, artifact) -> None:
    if args.skip_eval:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "eval",
                                    "model": "baselines", "status": "skipped", "artifact": artifact})
        return
    if should_skip(status_path, args, network, r0_label, "eval", model="baselines"):
        return
    truth_start, n_truth = final_eval_window(args, PRESETS[args.preset])
    cfg_path = run_dir / "configs" / network / "eval.yml"
    write_yaml(cfg_path, build_eval_config(network, r0_label, PRESETS[args.preset], truth_start, n_truth))
    log_path = run_dir / network / "logs" / "eval.log"
    cmd = ["python", "main_eval.py", "--cfg", str(cfg_path), "--data", f"{artifact}:latest"]
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "eval", "model": "baselines",
        "status": status, "artifact": artifact,
        "run_id": extract_run_id(stdout) or ("dryrun00" if args.dry_run else ""),
        "returncode": rc, "message": status, "log_path": log_path,
    })


def stage_hpo(args, run_dir, status_path, network, r0_label, model_key, artifact) -> Path | None:
    """Tune once at Δt=1 (Δt-invariant for static) per (network, model_key)."""
    if not args.with_hpo:
        return None
    tuned_variant = optuna_variant_name(model_key, 1)
    study_name = f"{network}_{model_key}"
    best_cfg = run_dir / "hpo" / study_name / "best_config.yml"
    if should_skip(status_path, args, network, r0_label, "hpo", model=model_key, delta_t=1, variant=tuned_variant):
        return best_cfg if best_cfg.exists() else None

    preset = PRESETS[args.preset]
    cfg = build_model_config(network, r0_label, model_key, 1, preset, args.save_probs, args)
    attach_hpo_budget(cfg, args, preset)
    locked: list[str] = []
    if MODEL_BASE[model_key] == "dbgnn":
        locked.extend(["dbgnn.order", "dbgnn.time_bin_size"])
    if model_key in TEMPORAL_MODELS:
        locked.append("coarsen.delta_t")
    if locked:
        cfg.setdefault("hpo", {})["locked_params"] = locked
    base_cfg_path = run_dir / "configs" / network / f"{model_key}.hpo_base.yml"
    write_yaml(base_cfg_path, cfg)

    log_path = run_dir / network / "logs" / f"hpo_{model_key}.log"
    cmd = ["python", "main_optuna.py", "--cfg", str(base_cfg_path), "--data", f"{artifact}:latest",
           "--output-dir", str(run_dir / "hpo"), "--study-name", study_name,
           "--n-trials", str(args.hpo_trials), "--sampler", args.hpo_sampler, "--pruner", args.hpo_pruner]
    if args.hpo_timeout is not None:
        cmd.extend(["--timeout", str(args.hpo_timeout)])
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "hpo", "model": model_key,
        "delta_t": 1, "variant": tuned_variant, "status": status, "artifact": artifact,
        "returncode": rc, "message": str(best_cfg) if status == "success" else status, "log_path": log_path,
    })
    if args.dry_run:
        return best_cfg
    # Freshness guard: only reuse a best_config the study (re)produced
    # successfully this invocation. Otherwise a --resume/--force rerun whose new
    # study fails would silently train the Optuna variants from a stale config
    # left by a previous run. On failure we skip the Optuna variants (the
    # untuned Δt sweep still runs); the prior-success case is handled by the
    # should_skip early return above.
    if status != "success":
        print(
            f"      WARNING: Optuna HPO {status} for {network}/{model_key}; "
            f"skipping its Optuna Δt variants (untuned sweep still runs). See {log_path}"
        )
        return None
    return best_cfg if best_cfg.exists() else None


def stage_train(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact,
                cfg_override=None, status_variant=None) -> str | None:
    variant = status_variant or variant_name(model_key, delta_t)
    if args.skip_train:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "train",
                                    "model": model_key, "delta_t": delta_t, "variant": variant,
                                    "status": "skipped", "artifact": artifact})
        return None
    if should_skip(status_path, args, network, r0_label, "train", model=model_key, delta_t=delta_t, variant=variant):
        for r in read_status(status_path):
            if (r.get("network") == network and r.get("stage") == "train"
                    and r.get("model") == model_key and r.get("delta_t") == str(delta_t)
                    and r.get("variant") == variant):
                return r.get("run_id")
        return None

    cfg_path = cfg_override
    if cfg_path is None:
        cfg_path = run_dir / "configs" / network / f"{variant}.yml"
        write_yaml(cfg_path, build_model_config(network, r0_label, model_key, delta_t,
                                                PRESETS[args.preset], args.save_probs, args))
    log_path = run_dir / network / "logs" / f"train_{variant}.log"
    checkpoint_dir = run_dir / "checkpoints" / network / variant
    cmd = ["python", "main_train.py", "--cfg", str(cfg_path), "--data", f"{artifact}:latest",
           "--checkpoint-dir", str(checkpoint_dir)]
    if args.save_probs:
        cmd.append("--save-probs")
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    status = (
        "success" if rc == 0
        else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout
        else "loss_guard_aborted" if rc == 88 or "LOSS_GUARD_ABORT" in stdout
        else "failed"
    )
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "train", "model": model_key,
        "delta_t": delta_t, "variant": variant, "status": status,
        "run_id": extract_run_id(stdout) or ("dryrun00" if args.dry_run else ""),
        "artifact": artifact, "returncode": rc, "message": status, "log_path": log_path,
    })
    return extract_run_id(stdout) if status == "success" else None


def run_paired(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact, best_cfg):
    """Train the untuned + Optuna-tuned variants for one (model, Δt) cell."""
    if args.with_hpo:
        untuned_cfg = write_untuned_paired_config(args, run_dir, network, r0_label, model_key, delta_t, best_cfg)
        stage_train(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact,
                    untuned_cfg, status_variant=variant_name(model_key, delta_t))
        if best_cfg is not None or args.dry_run:
            optuna_cfg = best_cfg if delta_t == 1 else write_reused_optuna_config(
                args, run_dir, network, r0_label, model_key, delta_t, best_cfg)
            stage_train(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact,
                        optuna_cfg, status_variant=optuna_variant_name(model_key, delta_t))
    else:
        stage_train(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact)


# ---------------------------------------------------------------------------
# Harvest + outputs
# ---------------------------------------------------------------------------

def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _condition_from_variant(variant: str) -> str:
    return "optuna" if variant.endswith(OPTUNA_SUFFIX) else "untuned"


def metric_rows_from_status(status_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Collect run-level metrics + cost columns for every successful train run."""
    rows: list[dict[str, Any]] = []
    for row in status_rows:
        if row.get("stage") != "train" or row.get("status") != "success" or not row.get("run_id"):
            continue
        summary_path = Path("data") / row["run_id"] / "metrics_summary.json"
        if not summary_path.exists():
            continue
        sc = scenario(row["network"], row["r0_label"])
        variant = row.get("variant", "")
        delta_t = row.get("delta_t", "")
        payload = json.loads(summary_path.read_text())
        base = {
            "network": row["network"],
            "r0_label": row["r0_label"],
            "r0": sc["r0"],
            "model": row.get("model", ""),
            "variant": variant,
            "condition": _condition_from_variant(variant),
            "delta_t": int(delta_t) if str(delta_t).isdigit() else delta_t,
            "run_id": row["run_id"],
        }
        rows.append({**base, **payload.get("metrics", {})})
    return rows


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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _format_metric(value: float, percent: bool = False) -> str:
    if not np.isfinite(value):
        return "---"
    return f"{100 * value:.1f}" if percent else f"{value:.4f}"


def write_tables(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    tbl_dir = run_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    header = ["Network", "Model", "Delta_t", "Condition", "MRR", "Top-5", "Norm-Brier",
              "Norm-Entropy", "ConstructS", "TrainS", "PeakRSS_MB", "n_valid"]
    csv_rows = [header]
    for row in sorted(rows, key=lambda r: (r["network"], r["model"], _delta_sort(r), r["condition"])):
        csv_rows.append([
            row["network"], row["model"], row.get("delta_t", ""), row["condition"],
            _format_metric(_float_or_nan(row.get("eval/mrr_mean"))),
            _format_metric(_float_or_nan(row.get("eval/top_5_mean")), percent=True),
            _format_metric(_float_or_nan(row.get("eval/norm_brier_mean"))),
            _format_metric(_float_or_nan(row.get("eval/norm_entropy_mean"))),
            _format_metric(_float_or_nan(row.get("graph/construction_seconds"))),
            _format_metric(_float_or_nan(row.get("train/fit_seconds"))),
            _format_metric(_float_or_nan(row.get("resources/peak_rss_mb"))),
            _format_metric(_float_or_nan(row.get("eval/n_valid_mean"))),
        ])
    with open(tbl_dir / "coarse_graining_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)

    tex = [
        "% H2 coarse-graining results generated from local metrics",
        "\\begin{tabular}{llrcccccc}",
        "\\toprule",
        "Network & Model & $\\Delta t$ & Cond. & MRR & Top-5 & NBS & Constr.(s) & Train(s) \\\\",
        "\\midrule",
    ]
    for r in csv_rows[1:]:
        net = str(r[0]).replace("_", "\\_")
        mod = str(r[1]).replace("_", "\\_")
        tex.append(f"{net} & {mod} & {r[2]} & {r[3]} & {r[4]} & {r[5]} & {r[6]} & {r[8]} & {r[9]} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (tbl_dir / "coarse_graining_results.tex").write_text("\n".join(tex) + "\n")


def _delta_sort(row: dict[str, Any]) -> float:
    d = row.get("delta_t", "")
    try:
        return float(d)
    except (TypeError, ValueError):
        return float("inf")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _static_reference(rows: list[dict[str, Any]], network: str, metric: str, condition: str) -> float:
    vals = [
        _float_or_nan(r.get(metric))
        for r in rows
        if r["network"] == network and r["model"] == STATIC_MODEL and r["condition"] == condition
    ]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def plot_convergence(rows: list[dict[str, Any]], run_dir: Path, metric: str, ylabel: str,
                     condition: str = "optuna", percent: bool = False) -> None:
    networks = sorted({r["network"] for r in rows})
    networks = [n for n in networks if any(
        r["network"] == n and r["model"] in TEMPORAL_MODELS and np.isfinite(_float_or_nan(r.get(metric)))
        for r in rows)]
    if not networks:
        return
    # Fall back to untuned if the requested condition is missing everywhere.
    if not any(r["condition"] == condition for r in rows):
        condition = "untuned"
    apply_style()
    ncols = min(2, len(networks))
    nrows = int(np.ceil(len(networks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax, network in zip(axes.flat, networks):
        for model_key in TEMPORAL_MODELS:
            pts = [
                (float(r["delta_t"]), _float_or_nan(r.get(metric)))
                for r in rows
                if r["network"] == network and r["model"] == model_key and r["condition"] == condition
                and isinstance(r.get("delta_t"), (int, float)) and np.isfinite(_float_or_nan(r.get(metric)))
            ]
            if not pts:
                continue
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [100 * p[1] if percent else p[1] for p in pts]
            ax.plot(xs, ys, marker="o", lw=2.0, label=MODEL_LABEL[model_key])
        static_ref = _static_reference(rows, network, metric, condition)
        if np.isfinite(static_ref):
            ax.axhline(100 * static_ref if percent else static_ref, color="k", ls="--", lw=1.6,
                       label=MODEL_LABEL[STATIC_MODEL])
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Temporal bin width $\\Delta t$")
        ax.set_ylabel(ylabel)
        ax.set_title(network.replace("_", " "))
        ax.legend(fontsize=7)
    for ax in axes.flat[len(networks):]:
        ax.set_visible(False)
    out = run_dir / "figures" / f"convergence_{metric.replace('/', '_').replace('_mean', '')}_{condition}.pdf"
    finish_fig(fig, str(out))


def plot_cost_vs_delta(rows: list[dict[str, Any]], run_dir: Path, metric: str, ylabel: str,
                       condition: str = "optuna") -> None:
    networks = sorted({r["network"] for r in rows})
    networks = [n for n in networks if any(
        r["network"] == n and r["model"] in TEMPORAL_MODELS and np.isfinite(_float_or_nan(r.get(metric)))
        for r in rows)]
    if not networks:
        return
    if not any(r["condition"] == condition for r in rows):
        condition = "untuned"
    apply_style()
    ncols = min(2, len(networks))
    nrows = int(np.ceil(len(networks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.2 * nrows), squeeze=False)
    for ax, network in zip(axes.flat, networks):
        for model_key in TEMPORAL_MODELS:
            pts = [
                (float(r["delta_t"]), _float_or_nan(r.get(metric)))
                for r in rows
                if r["network"] == network and r["model"] == model_key and r["condition"] == condition
                and isinstance(r.get("delta_t"), (int, float)) and np.isfinite(_float_or_nan(r.get(metric)))
            ]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="s", lw=2.0, label=MODEL_LABEL[model_key])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Temporal bin width $\\Delta t$")
        ax.set_ylabel(ylabel)
        ax.set_title(network.replace("_", " "))
        ax.legend(fontsize=7)
    for ax in axes.flat[len(networks):]:
        ax.set_visible(False)
    out = run_dir / "figures" / f"cost_{metric.replace('/', '_').replace('_mean', '')}_{condition}.pdf"
    finish_fig(fig, str(out))


def plot_reliability_vs_cost(rows: list[dict[str, Any]], run_dir: Path, metric: str = "eval/mrr_mean",
                             condition: str = "optuna") -> None:
    """At the finest Δt=1: improvement over StaticGNN vs construction+training cost."""
    if not any(r["condition"] == condition for r in rows):
        condition = "untuned"
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for model_key in TEMPORAL_MODELS:
        improvements, costs = [], []
        for network in sorted({r["network"] for r in rows}):
            model_pts = [
                r for r in rows
                if r["network"] == network and r["model"] == model_key and r["condition"] == condition
                and r.get("delta_t") == 1
            ]
            if not model_pts:
                continue
            r = model_pts[0]
            static_ref = _static_reference(rows, network, metric, condition)
            mrr = _float_or_nan(r.get(metric))
            cost = _float_or_nan(r.get("graph/construction_seconds")) + _float_or_nan(r.get("train/fit_seconds"))
            if np.isfinite(mrr) and np.isfinite(static_ref) and np.isfinite(cost):
                improvements.append(mrr - static_ref)
                costs.append(cost)
        if not improvements:
            continue
        ax.scatter(costs, improvements, s=60, label=MODEL_LABEL[model_key])
        ax.scatter([np.mean(costs)], [np.mean(improvements)], s=200, marker="*", edgecolor="k", zorder=5)
    ax.axhline(0.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("Construction + training time at $\\Delta t=1$ (s)")
    ax.set_ylabel("MRR improvement over StaticGNN")
    ax.set_title("Reliability vs. cost (★ = mean over networks)")
    ax.legend(fontsize=8)
    finish_fig(fig, str(run_dir / "figures" / "reliability_vs_cost.pdf"))


def make_plots(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    has_optuna = any(r["condition"] == "optuna" for r in rows)
    condition = "optuna" if has_optuna else "untuned"
    plot_convergence(rows, run_dir, "eval/mrr_mean", "MRR", condition)
    plot_convergence(rows, run_dir, "eval/top_5_mean", "Top-5 accuracy (%)", condition, percent=True)
    plot_cost_vs_delta(rows, run_dir, "graph/construction_seconds", "Graph construction (s)", condition)
    plot_cost_vs_delta(rows, run_dir, "train/fit_seconds", "Training time (s)", condition)
    plot_cost_vs_delta(rows, run_dir, "resources/peak_rss_mb", "Peak RSS (MB)", condition)
    plot_reliability_vs_cost(rows, run_dir, "eval/mrr_mean", condition)


# ---------------------------------------------------------------------------
# Manifest + bundle
# ---------------------------------------------------------------------------

def write_manifest(run_dir, args, networks, r0_label, stats, grids, sample_budget) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "h2_coarse_graining",
        "hypothesis": "H2: temporal representations converge to StaticGNN as Δt grows.",
        "preset": args.preset,
        "preset_values": PRESETS[args.preset].__dict__,
        "networks": networks,
        "r0_label": r0_label,
        "r0": R0_VALUES[r0_label],
        "temporal_models": list(args.temporal_models),
        "static_model": STATIC_MODEL,
        "delta_t_grids": grids,
        "betas": {n: BETAS[n][r0_label] for n in networks if n in BETAS},
        "mus": {n: MUS[n] for n in networks if n in MUS},
        "with_hpo": bool(args.with_hpo),
        "hpo": {
            "scope": "delta_t (tuned at Δt=1, reused for coarser Δt)",
            "trials": args.hpo_trials,
            "sampler": args.hpo_sampler,
            "pruner": args.hpo_pruner,
            "n_truth": effective_hpo_n_truth(args, PRESETS[args.preset]),
            "trial_n_mc": effective_hpo_n_mc(args, PRESETS[args.preset]),
            "trial_epochs": _positive_cap(args.hpo_epochs),
            "trial_patience": _positive_cap(args.hpo_patience),
        },
        "reduction": getattr(args, "reduction", "safe_1h"),
        "sample_budget_node_edge": sample_budget,
        "network_stats": {n: stats[n].__dict__ for n in sorted(stats)},
        "wandb_project": WANDB_PROJECT,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def refresh_result_bundle(run_dir: Path, status_path: Path) -> Path:
    return sync_publication_result(
        run_dir=run_dir,
        status_rows=read_status(status_path),
        experiment_name="h2_coarse_graining",
    )


def write_outputs(run_dir: Path, status_path: Path, dry_run: bool) -> None:
    rows = metric_rows_from_status(read_status(status_path))
    write_csv(run_dir / "metrics_summary.csv", rows)
    if not dry_run and rows:
        write_tables(run_dir, rows)
        make_plots(run_dir, rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    networks = list(dict.fromkeys(args.networks))
    r0_label = normalize_r0_labels([args.r0])[0]
    temporal_models = [m for m in args.temporal_models if m in TEMPORAL_MODELS]

    # build_eval_config reads run_all_experiments.HEURISTIC_BASELINES (the paper
    # set). Honour --no-expensive-baselines by filtering that module global.
    import run_all_experiments as _rae
    _rae.HEURISTIC_BASELINES = _rae.normalize_baseline_keys(
        ["paper"], exclude_expensive=bool(getattr(args, "exclude_expensive_baselines", False))
    )

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

    for network in networks:
        if network not in BETAS or r0_label not in BETAS[network]:
            raise ValueError(f"No calibrated beta for {network}/{r0_label}")

    stats = {network: read_full_network_stats(network) for network in networks}
    networks = sorted(networks, key=lambda n: (stats[n].n_nodes, stats[n].n_edges, n))
    sample_budget = compute_sample_budget(stats, args.sample_reference, args.sample_budget_factor)
    grids = {n: delta_t_grid(effective_t_max(n, args), args.delta_t, args.max_delta_levels) for n in networks}

    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"
    write_manifest(run_dir, args, networks, r0_label, stats, grids, sample_budget)
    refresh_result_bundle(run_dir, status_path)

    print("=" * 72)
    print("H2 Coarse-Graining Experiment Runner")
    print("=" * 72)
    print(f"Run dir   : {run_dir}")
    print(f"Preset    : {args.preset} ({preset})")
    print(f"Networks  : {', '.join(networks)}")
    print(f"R0        : {r0_label} (R0={R0_VALUES[r0_label]})")
    print(f"Temporal  : {', '.join(temporal_models)} (+ {STATIC_MODEL} reference)")
    print(f"HPO       : {'paired untuned/Optuna, tuned at Δt=1 and reused' if args.with_hpo else 'disabled'}")
    for n in networks:
        print(f"  Δt[{n}] : {grids[n]}")
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed")

    for network in networks:
        sample_cfg = sampling_cfg_for_network(network, stats, sample_budget, args)
        tag = "sampled" if sample_cfg is not None else "full"
        print(f"\n### {network} ({tag}, n={stats[network].n_nodes})")

        artifact = stage_tsir(args, run_dir, status_path, network, r0_label, sample_cfg)
        refresh_result_bundle(run_dir, status_path)
        if artifact is None:
            print("    TSIR unavailable; skipping network")
            continue

        # Δt-invariant references: heuristic baselines + StaticGNN (run once).
        stage_eval(args, run_dir, status_path, network, r0_label, artifact)
        refresh_result_bundle(run_dir, status_path)

        if not args.skip_static:
            print(f"  --- {STATIC_MODEL} (reference)")
            best_static = stage_hpo(args, run_dir, status_path, network, r0_label, STATIC_MODEL, artifact)
            run_paired(args, run_dir, status_path, network, r0_label, STATIC_MODEL, 1, artifact, best_static)
            refresh_result_bundle(run_dir, status_path)

        # Temporal models sweep Δt; HPO tuned at Δt=1 and reused.
        for model_key in temporal_models:
            print(f"  --- {model_key}")
            best_cfg = stage_hpo(args, run_dir, status_path, network, r0_label, model_key, artifact)
            refresh_result_bundle(run_dir, status_path)
            for delta_t in grids[network]:
                print(f"      Δt={delta_t}")
                run_paired(args, run_dir, status_path, network, r0_label, model_key, delta_t, artifact, best_cfg)
                refresh_result_bundle(run_dir, status_path)

    write_outputs(run_dir, status_path, args.dry_run)
    result_dir = refresh_result_bundle(run_dir, status_path)
    print("\nDone.")
    print(f"Status : {status_path}")
    print(f"Result : {result_dir}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
