"""
DBGNN higher-order experiment runner.

Runs DBGNN on all calibrated thesis networks for the requested R0 values and
De Bruijn orders. The default matrix is:

    networks = lyon_ward, malawi, france_office, students, biasca, olten
    R0       = 0.8, 1.0, 1.5, 2.0, 2.5
    k        = 2, 3, 4, 5

Networks with more than 300 nodes are sampled before TSIR using a connected
activity/degree-stratified temporal snowball sampler. The sampler keeps the
sample in a temporally active connected region while preserving low, medium,
and high activity/degree strata as much as possible under the node/edge budget.

Every run maintains a publication bundle under:
    results/dbgnn_higher_order/<run-name>/result/

The bundle mirrors metrics, figures, tables, and lightweight run assets, and
includes latex_inputs.json for downstream LaTeX/report generation.

Default runs use the day-scale paired Optuna protocol: paper_24h, 5 HPO trials,
network-scope HPO reuse across R0s, short HPO trial budgets, capped final
training.  Use ``--hpo-scope scenario --preset max_quality --hpo-trials 30
--max-train-epochs 0 --max-train-patience 0`` for the exhaustive run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_all_experiments import (
    BETAS,
    LOSS_GUARD,
    MIN_OUTBREAK,
    MUS,
    NETWORKS,
    R0_VALUES,
    extract_run_id,
    normalize_r0_labels,
    read_network_meta,
    scenario,
)
from scripts.publication_bundle import sync_publication_result
from viz.style import apply_style, finish_fig
from hpo import apply_trial_params


DEFAULT_R0 = ["0.8", "1.0", "1.5", "2.0", "2.5"]
DEFAULT_ORDERS = [2, 3, 4, 5]
WANDB_PROJECT = "source-detection"
OPTUNA_SUFFIX = "_optuna"


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
}


@dataclass(frozen=True)
class NetworkStats:
    n_nodes: int
    n_edges: int
    n_contacts: int
    t_max: int
    directed: bool

    @property
    def node_edge_cost(self) -> int:
        return self.n_nodes * max(self.n_edges, 1)


STATUS_FIELDS = [
    "network",
    "r0_label",
    "stage",
    "variant",
    "order",
    "status",
    "run_id",
    "artifact",
    "returncode",
    "message",
    "log_path",
]
TERMINAL_STATUSES = {"success", "loss_guard_aborted", "skipped", "timeout_skipped"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper_24h",
                   help="Runtime/quality preset. Use max_quality for the full expensive grid.")
    p.add_argument("--networks", nargs="+", default=["all"], help="Network names, or all")
    p.add_argument("--r0", nargs="+", default=DEFAULT_R0, help="R0 labels/numbers, e.g. 1.0 1.1 r0_15")
    p.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS, help="DBGNN orders k")
    p.add_argument("--output", default="results/dbgnn_higher_order", help="Root results directory")
    p.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run directory")
    p.add_argument("--force", action="store_true", help="Rerun stages even when a terminal status exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    p.add_argument("--save-probs", action="store_true", help="Save probs_rep*.pt tensors from main_train.py")
    p.add_argument("--with-hpo", dest="with_hpo", action="store_true",
                   help="Run paired untuned and Optuna-tuned final evaluations (default)")
    p.add_argument("--no-hpo", dest="with_hpo", action="store_false",
                   help="Disable Optuna and run only untuned DBGNN order configs")
    p.add_argument("--hpo-trials", type=int, default=5,
                   help="Optuna trials per network/R0/order when HPO is enabled")
    p.add_argument("--hpo-timeout", type=int, default=None,
                   help="Optional Optuna study timeout in seconds")
    p.add_argument("--hpo-scope", choices=["network", "scenario"], default="network",
                   help="Tune once per network/order and reuse across R0s, or tune every scenario")
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
    p.add_argument("--max-train-epochs", type=int, default=250,
                   help="Cap final train.epochs for generated configs; set 0 to keep template values")
    p.add_argument("--max-train-patience", type=int, default=20,
                   help="Cap final train.patience for generated configs; set 0 to keep template values")
    p.add_argument("--hpo-sampler", choices=["tpe", "random"], default="tpe")
    p.add_argument("--hpo-pruner", choices=["hyperband", "median", "none"], default="hyperband")
    p.add_argument("--skip-tsir", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--reduction", choices=["safe_1h", "none"], default="safe_1h",
                   help="Network reduction policy for TSIR artifacts. Default: safe_1h")
    p.add_argument("--no-reduction", dest="reduction", action="store_const", const="none",
                   help="Disable default network reduction.")
    p.add_argument("--target-runtime-seconds", dest="timeout_seconds", type=int,
                   help="Alias for --timeout-seconds.")
    p.add_argument("--reduction-seed", dest="seed", type=int,
                   help="Alias for --seed used by reduction.")
    p.add_argument("--time-window-steps", default="auto",
                   help="Temporal window length for safe_1h, or auto.")
    p.add_argument("--reduction-reps", type=int, default=1)
    p.add_argument("--use-full-betas", action="store_true",
                   help="Use static BETAS instead of reduced-graph calibration.")
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
    p.add_argument("--stratification-bins", type=int, default=4,
                   help="Activity/degree quantile bins for balanced sampling")
    p.add_argument("--timeout-seconds", type=int, default=3600,
                   help="Skip any TSIR/train command that runs longer than this")
    p.add_argument("--base-batch-size", type=int, default=16,
                   help="Maximum train.batch_size for DBGNN k=2")
    p.add_argument("--order3-batch-size", type=int, default=8,
                   help="Maximum train.batch_size for DBGNN k=3")
    p.add_argument(
        "--high-order-delta",
        type=int,
        default=4,
        help="Causal time-window delta used for DBGNN k>=4. The paper uses delta=4 after aggregation.",
    )
    p.add_argument(
        "--high-order-time-bin-size",
        type=int,
        default=4,
        help="Aggregate contacts into this many original time steps for DBGNN k>=4.",
    )
    p.add_argument(
        "--very-high-order-time-bin-size",
        type=int,
        default=8,
        help="Minimum time-bin size for DBGNN k>=6, including k=10.",
    )
    p.add_argument(
        "--high-order-batch-size",
        type=int,
        default=4,
        help="Maximum train.batch_size for DBGNN k>=4.",
    )
    p.add_argument("--max-temporal-states", type=int, default=2_000_000)
    p.add_argument("--max-db-nodes", type=int, default=500_000)
    p.add_argument("--max-db-edges", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(with_hpo=True)
    return p.parse_args()


def _positive_cap(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def apply_final_train_caps(cfg: dict[str, Any], args: argparse.Namespace | None) -> None:
    """Bound definitive DBGNN training runs for day-scale execution."""
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


def resolve_networks(raw: list[str]) -> list[str]:
    if "all" in raw:
        return list(NETWORKS)
    return raw


def _directed_from_meta(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"yes", "true", "1"}
    return bool(value)


def read_full_network_stats(network: str) -> NetworkStats:
    meta = read_network_meta(network)
    t_max = int(meta["t_max"])
    directed = _directed_from_meta(meta.get("directed", False))
    graph = nx.DiGraph() if directed else nx.Graph()
    csv_path = Path("nwk") / f"{network}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing network CSV: {csv_path}")

    with open(csv_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) != 3:
                continue
            u, v, t = map(int, fields)
            if t > t_max or u == v:
                continue
            if graph.has_edge(u, v):
                if t not in graph.edges[u, v]["times"]:
                    graph.edges[u, v]["times"].append(t)
            else:
                graph.add_edge(u, v, times=[t])

    if graph.number_of_nodes() == 0:
        return NetworkStats(0, 0, 0, t_max, directed)
    components = nx.weakly_connected_components(graph) if directed else nx.connected_components(graph)
    largest = max(components, key=len)
    graph = graph.subgraph(largest).copy()
    contacts = sum(len(d.get("times", [])) for _, _, d in graph.edges(data=True))
    return NetworkStats(graph.number_of_nodes(), graph.number_of_edges(), contacts, t_max, directed)


def compute_sample_budget(stats: dict[str, NetworkStats], reference: str, factor: float) -> int:
    if reference not in stats:
        stats[reference] = read_full_network_stats(reference)
    if factor <= 0:
        raise ValueError("--sample-budget-factor must be positive")
    return max(1, int(stats[reference].node_edge_cost / factor))


def sampling_cfg_for_network(
    network: str,
    stats: dict[str, NetworkStats],
    budget: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.no_sampling or getattr(args, "reduction", "safe_1h") == "none":
        return None
    st = stats[network]
    if st.n_nodes <= int(args.sample_node_threshold):
        return None
    return {
        "method": args.sample_method,
        "target_nodes": min(int(args.sample_target_nodes), int(st.n_nodes)),
        "max_node_edge_cost": int(budget),
        "cost_metric": "node_edge",
        "seed": int(args.seed),
        "min_nodes": int(args.min_sample_nodes),
        "stratification_bins": int(args.stratification_bins),
        "sample_node_threshold": int(args.sample_node_threshold),
        "reference_network": args.sample_reference,
        "reference_budget_factor": float(args.sample_budget_factor),
        "original_node_edge_cost": int(st.node_edge_cost),
        "original_nodes": int(st.n_nodes),
    }


def artifact_name(network: str, r0_label: str) -> str:
    return f"dbgnn_higher_order_{network}_{r0_label}"


def variant_name(order: int) -> str:
    return f"dbgnn_k{order}"


def optuna_variant_name(order: int) -> str:
    return f"{variant_name(order)}{OPTUNA_SUFFIX}"


def higher_order_controls(order: int, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Return resource controls for the requested DBGNN order."""
    safe_1h = args is not None and getattr(args, "reduction", "safe_1h") != "none"
    controls: dict[str, Any] = {
        "time_bin_size": 4 if safe_1h else 1,
        "delta": None,
        "batch_size_cap": int(getattr(args, "base_batch_size", 16)),
    }
    if order >= 4:
        controls["delta"] = int(getattr(args, "high_order_delta", 4))
        controls["time_bin_size"] = int(getattr(args, "high_order_time_bin_size", 4))
        controls["batch_size_cap"] = int(getattr(args, "high_order_batch_size", 4))
        if order >= 6:
            controls["time_bin_size"] = max(
                controls["time_bin_size"],
                int(getattr(args, "very_high_order_time_bin_size", 8)),
            )
    elif order == 3:
        controls["batch_size_cap"] = int(getattr(args, "order3_batch_size", 8))
    return controls


def build_tsir_config(
    network: str,
    r0_label: str,
    preset: Preset,
    sample_cfg: dict[str, Any] | None,
    args: argparse.Namespace | None = None,
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
    if sample_cfg is not None and not bool(getattr(args, "use_full_betas", False)):
        sir_cfg["calibration"] = {
            "enabled": True,
            "target_r0": sc["r0"],
            "output_dir": "results/calibration",
            "n_probe": 1,
            "max_iter": 8,
            "tolerance": 0.05,
            "seed": int(getattr(args, "seed", 42)),
        }
    return {
        "nwk": nwk_cfg,
        "sir": sir_cfg,
        "experiment": {
            "name": "dbgnn_higher_order",
            "network": network,
            "r0_label": r0_label,
            **sc,
        },
    }


def _template_path(network: str) -> Path:
    direct = Path("exp") / network / "dbgnn.yml"
    if direct.exists():
        return direct
    fallback = Path("exp/france_office/dbgnn.yml")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No DBGNN config template found for {network}")


def build_dbgnn_config(
    network: str,
    r0_label: str,
    order: int,
    preset: Preset,
    save_probs: bool,
    args: argparse.Namespace | None = None,
) -> dict[str, Any]:
    if order < 2:
        raise ValueError(f"DBGNN order must be >= 2, got {order}")
    with open(_template_path(network)) as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = "dbgnn"
    cfg["eval"] = {
        **cfg.get("eval", {}),
        "min_outbreak": MIN_OUTBREAK,
        "top_k": [1, 3, 5, 10],
        "credible_p": [0.80, 0.90],
        "inverse_rank_offset": [0],
        "n_truth": preset.n_truth,
    }
    cfg["train"] = {
        **cfg.get("train", {}),
        "n_mc": preset.n_mc,
        "reps": preset.reps,
        "loss_guard": LOSS_GUARD,
    }
    controls = higher_order_controls(order, args)
    batch_cap = controls.get("batch_size_cap")
    if batch_cap is not None:
        cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", batch_cap)), int(batch_cap))
    cfg.setdefault("output", {})["save_probs"] = save_probs

    db_cfg = cfg.setdefault("dbgnn", {})
    db_cfg["order"] = int(order)
    db_cfg["delta"] = int(controls["delta"]) if controls["delta"] is not None else db_cfg.get("delta", 24)
    db_cfg["time_bin_size"] = int(controls["time_bin_size"])
    db_cfg["max_temporal_states"] = int(getattr(args, "max_temporal_states", 2_000_000))
    db_cfg["max_db_nodes"] = int(getattr(args, "max_db_nodes", 500_000))
    db_cfg["max_db_edges"] = int(getattr(args, "max_db_edges", 2_000_000))
    db_cfg["bipartite_agg"] = db_cfg.get("bipartite_agg", "sum")
    db_cfg["directed"] = read_network_meta(network)["directed"]
    apply_final_train_caps(cfg, args)

    cfg["experiment"] = {
        "name": "dbgnn_higher_order",
        "network": network,
        "r0_label": r0_label,
        "variant": variant_name(order),
        "dbgnn_order": int(order),
        **scenario(network, r0_label),
    }
    if preset.n_runs < preset.reps * preset.n_truth:
        raise ValueError(
            f"Invalid preset: n_runs={preset.n_runs} < reps*n_truth={preset.reps * preset.n_truth}"
        )
    return cfg


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
    return root / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    networks: list[str],
    r0_labels: list[str],
    orders: list[int],
    stats: dict[str, NetworkStats],
    sample_budget: int,
) -> None:
    sample_policies = {
        network: sampling_cfg_for_network(network, stats, sample_budget, args)
        for network in networks
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "dbgnn_higher_order",
        "preset": args.preset,
        "preset_values": PRESETS[args.preset].__dict__,
        "networks": networks,
        "r0_labels": r0_labels,
        "orders": orders,
        "betas": {n: BETAS[n] for n in networks if n in BETAS},
        "mus": {n: MUS[n] for n in networks if n in MUS},
        "sample_budget_node_edge": sample_budget,
        "reduction": {
            "policy": getattr(args, "reduction", "safe_1h"),
            "target_runtime_seconds": getattr(args, "timeout_seconds", None),
            "time_window_steps": getattr(args, "time_window_steps", None),
            "seed": getattr(args, "seed", None),
            "reps": getattr(args, "reduction_reps", None),
            "use_full_betas": bool(getattr(args, "use_full_betas", False)),
        },
        "sample_method": args.sample_method,
        "sample_node_threshold": args.sample_node_threshold,
        "sample_target_nodes": args.sample_target_nodes,
        "sample_reference": args.sample_reference,
        "sample_budget_factor": args.sample_budget_factor,
        "sample_policies": sample_policies,
        "timeout_seconds": args.timeout_seconds,
        "higher_order_controls": {
            "k_eq_2_batch_size_cap": args.base_batch_size,
            "k_eq_3_batch_size_cap": args.order3_batch_size,
            "k_ge_4_delta": args.high_order_delta,
            "k_ge_4_time_bin_size": args.high_order_time_bin_size,
            "k_ge_6_time_bin_size": args.very_high_order_time_bin_size,
            "k_ge_4_batch_size_cap": args.high_order_batch_size,
            "max_temporal_states": args.max_temporal_states,
            "max_db_nodes": args.max_db_nodes,
            "max_db_edges": args.max_db_edges,
        },
        "hpo": {
            "enabled": bool(getattr(args, "with_hpo", True)),
            "trials": int(getattr(args, "hpo_trials", 0)),
            "timeout": getattr(args, "hpo_timeout", None),
            "scope": getattr(args, "hpo_scope", None),
            "reference_r0": resolve_hpo_reference_r0(getattr(args, "hpo_reference_r0", "r0_10"), r0_labels),
            "sampler": getattr(args, "hpo_sampler", None),
            "pruner": getattr(args, "hpo_pruner", None),
            "n_truth": effective_hpo_n_truth(args, PRESETS[args.preset]),
            "trial_n_mc": effective_hpo_n_mc(args, PRESETS[args.preset]),
            "trial_epochs": _positive_cap(getattr(args, "hpo_epochs", None)),
            "trial_patience": _positive_cap(getattr(args, "hpo_patience", None)),
            "final_epoch_cap": _positive_cap(getattr(args, "max_train_epochs", None)),
            "final_patience_cap": _positive_cap(getattr(args, "max_train_patience", None)),
            "paired_final_evaluation": True,
            "locked_params": ["dbgnn.order"],
        },
        "network_stats": {n: stats[n].__dict__ for n in sorted(stats)},
        "wandb_project": WANDB_PROJECT,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


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
    key = (
        row.get("network", ""),
        row.get("r0_label", ""),
        row.get("stage", ""),
        row.get("variant", ""),
        str(row.get("order", "")),
    )
    normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
    kept = [
        r for r in rows
        if (
            r.get("network", ""),
            r.get("r0_label", ""),
            r.get("stage", ""),
            r.get("variant", ""),
            r.get("order", ""),
        ) != key
    ]
    kept.append(normalized)
    write_status(path, kept)


def should_skip(
    status_path: Path,
    args: argparse.Namespace,
    network: str,
    r0_label: str,
    stage: str,
    variant: str = "",
    order: int | str = "",
) -> bool:
    if args.force or not args.resume:
        return False
    for row in read_status(status_path):
        if (
            row.get("network") == network
            and row.get("r0_label") == r0_label
            and row.get("stage") == stage
            and row.get("variant", "") == variant
            and row.get("order", "") == str(order)
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
            stdout, _ = proc.communicate(timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
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


def stage_tsir(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    sample_cfg: dict[str, Any] | None,
) -> str:
    artifact = artifact_name(network, r0_label)
    if args.skip_tsir:
        update_status(
            status_path,
            {"network": network, "r0_label": r0_label, "stage": "tsir", "status": "skipped", "artifact": artifact},
        )
        return artifact
    if should_skip(status_path, args, network, r0_label, "tsir"):
        return artifact

    cfg_path = run_dir / "configs" / network / r0_label / "tsir.yml"
    write_yaml(cfg_path, build_tsir_config(network, r0_label, PRESETS[args.preset], sample_cfg, args))
    log_path = run_dir / network / r0_label / "logs" / "tsir.log"
    cmd = [sys.executable, "main_tsir.py", "--cfg", str(cfg_path), "--data", artifact]
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    status = "success" if rc == 0 else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout else "failed"
    update_status(
        status_path,
        {
            "network": network,
            "r0_label": r0_label,
            "stage": "tsir",
            "status": status,
            "artifact": artifact,
            "returncode": rc,
            "message": "dry_run" if args.dry_run else status,
            "log_path": log_path,
        },
    )
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"TSIR failed for {network}/{r0_label}; see {log_path}")
    return artifact


def stage_hpo(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    order: int,
    artifact: str,
) -> Path | None:
    if not args.with_hpo:
        return None

    base_variant = variant_name(order)
    tuned_variant = optuna_variant_name(order)
    if should_skip(status_path, args, network, r0_label, "hpo", tuned_variant, order):
        best = run_dir / "hpo" / f"{network}_{r0_label}_{base_variant}" / "best_config.yml"
        return best if best.exists() else None

    preset = PRESETS[args.preset]
    cfg = build_dbgnn_config(network, r0_label, order, preset, args.save_probs, args)
    attach_hpo_budget(cfg, args, preset)
    cfg.setdefault("hpo", {})["locked_params"] = ["dbgnn.order"]
    cfg["hpo"]["study_note"] = "DBGNN higher-order experiment fixes dbgnn.order as the independent variable."
    base_cfg_path = run_dir / "configs" / network / r0_label / f"{base_variant}.hpo_base.yml"
    write_yaml(base_cfg_path, cfg)

    study_name = f"{network}_{r0_label}_{base_variant}"
    log_path = run_dir / network / r0_label / "logs" / f"hpo_{base_variant}.log"
    cmd = [
        sys.executable,
        "main_optuna.py",
        "--cfg",
        str(base_cfg_path),
        "--data",
        f"{artifact}:latest",
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
    ]
    if args.hpo_timeout is not None:
        cmd.extend(["--timeout", str(args.hpo_timeout)])
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    best_cfg = run_dir / "hpo" / study_name / "best_config.yml"
    status = (
        "success" if rc == 0
        else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout
        else "failed"
    )
    update_status(
        status_path,
        {
            "network": network,
            "r0_label": r0_label,
            "stage": "hpo",
            "variant": tuned_variant,
            "order": order,
            "status": status,
            "artifact": artifact,
            "returncode": rc,
            "message": str(best_cfg) if status == "success" else status,
            "log_path": log_path,
        },
    )
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"Optuna HPO failed for {network}/{r0_label}/{base_variant}; see {log_path}")
    if args.dry_run:
        return best_cfg
    return best_cfg if best_cfg.exists() else None


def write_untuned_paired_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    order: int,
    best_cfg_path: Path | None,
) -> Path:
    """Write the untuned control config paired to an Optuna final window."""
    cfg = build_dbgnn_config(network, r0_label, order, PRESETS[args.preset], args.save_probs, args)
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        for key in ("truth_start", "n_truth"):
            if key in best_cfg.get("eval", {}):
                cfg["eval"][key] = best_cfg["eval"][key]
    cfg.setdefault("experiment", {})["hpo_condition"] = "none"
    cfg["experiment"]["paired_optuna_variant"] = optuna_variant_name(order)
    cfg_path = run_dir / "configs" / network / r0_label / f"{variant_name(order)}.untuned.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def write_reused_optuna_config(
    args: argparse.Namespace,
    run_dir: Path,
    network: str,
    r0_label: str,
    order: int,
    reference_r0: str,
    best_cfg_path: Path | None,
) -> Path:
    """Write a scenario-local tuned DBGNN config using network-scope Optuna params."""
    cfg = build_dbgnn_config(network, r0_label, order, PRESETS[args.preset], args.save_probs, args)
    if best_cfg_path is not None and best_cfg_path.exists():
        with open(best_cfg_path) as f:
            best_cfg = yaml.safe_load(f)
        params = dict(best_cfg.get("hpo_result", {}).get("params") or {})
        apply_trial_params(cfg, params)
        cfg.setdefault("dbgnn", {})["order"] = int(order)
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
    cfg["experiment"]["variant"] = optuna_variant_name(order)
    cfg_path = run_dir / "configs" / network / r0_label / f"{variant_name(order)}.optuna.yml"
    write_yaml(cfg_path, cfg)
    return cfg_path


def stage_train(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    order: int,
    artifact: str,
    cfg_override: Path | None = None,
    status_variant: str | None = None,
) -> str | None:
    variant = status_variant or variant_name(order)
    if args.skip_train:
        update_status(
            status_path,
            {
                "network": network,
                "r0_label": r0_label,
                "stage": "train",
                "variant": variant,
                "order": order,
                "status": "skipped",
                "artifact": artifact,
            },
        )
        return None
    if should_skip(status_path, args, network, r0_label, "train", variant, order):
        rows = read_status(status_path)
        return next(
            (
                r.get("run_id")
                for r in rows
                if r.get("network") == network
                and r.get("r0_label") == r0_label
                and r.get("stage") == "train"
                and r.get("variant") == variant
                and r.get("order") == str(order)
            ),
            None,
        )

    cfg_path = cfg_override
    if cfg_path is None:
        cfg_path = run_dir / "configs" / network / r0_label / f"{variant}.yml"
        write_yaml(cfg_path, build_dbgnn_config(network, r0_label, order, PRESETS[args.preset], args.save_probs, args))
    log_path = run_dir / network / r0_label / "logs" / f"train_{variant}.log"
    checkpoint_dir = run_dir / "checkpoints" / network / r0_label / variant
    cmd = [
        sys.executable,
        "main_train.py",
        "--cfg",
        str(cfg_path),
        "--data",
        f"{artifact}:latest",
        "--checkpoint-dir",
        str(checkpoint_dir),
    ]
    if args.save_probs:
        cmd.append("--save-probs")
    if args.force:
        cmd.append("--fresh")
    rc, stdout = run_command(cmd, log_path, args.dry_run, args.timeout_seconds)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = (
        "success" if rc == 0
        else "timeout_skipped" if rc == 124 or "TIMEOUT_SKIP" in stdout
        else "loss_guard_aborted" if rc == 88 or "LOSS_GUARD_ABORT" in stdout
        else "failed"
    )
    update_status(
        status_path,
        {
            "network": network,
            "r0_label": r0_label,
            "stage": "train",
            "variant": variant,
            "order": order,
            "status": status,
            "run_id": run_id,
            "artifact": artifact,
            "returncode": rc,
            "message": status,
            "log_path": log_path,
        },
    )
    return run_id if status == "success" else None


def write_summary_csv(run_dir: Path, status_rows: list[dict[str, str]]) -> None:
    out = run_dir / "run_matrix_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["network", "r0_label", "variant", "order", "status", "run_id", "artifact", "log_path"],
        )
        writer.writeheader()
        for row in status_rows:
            if row.get("stage") == "train":
                writer.writerow({
                    "network": row.get("network", ""),
                    "r0_label": row.get("r0_label", ""),
                    "variant": row.get("variant", ""),
                    "order": row.get("order", ""),
                    "status": row.get("status", ""),
                    "run_id": row.get("run_id", ""),
                    "artifact": row.get("artifact", ""),
                    "log_path": row.get("log_path", ""),
                })


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _condition_from_variant(variant: str) -> str:
    return "optuna" if variant.endswith(OPTUNA_SUFFIX) else "untuned"


def metric_rows_from_status(status_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect run-level and rep-level metrics for DBGNN order experiments."""
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for row in status_rows:
        if row.get("stage") != "train" or row.get("status") != "success" or not row.get("run_id"):
            continue
        run_data = Path("data") / row["run_id"]
        summary_path = run_data / "metrics_summary.json"
        if not summary_path.exists():
            continue
        sc = scenario(row["network"], row["r0_label"])
        variant = row.get("variant", "")
        base = {
            "network": row["network"],
            "r0_label": row["r0_label"],
            "r0": sc["r0"],
            "beta": sc["beta"],
            "mu": sc["mu"],
            "variant": variant,
            "condition": _condition_from_variant(variant),
            "order": int(row.get("order", 0) or 0),
            "run_id": row["run_id"],
            "status": row["status"],
        }
        payload = _read_json(summary_path)
        summary_rows.append({**base, **payload.get("metrics", {})})
        for rep_path in sorted(run_data.glob("metrics_rep*.json")):
            rep_payload = _read_json(rep_path)
            rep = rep_payload.get("rep", "")
            for metric, value in rep_payload.get("metrics", {}).items():
                long_rows.append({**base, "rep": rep, "metric": metric, "value": value})
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
            writer.writerow({key: _jsonable(row.get(key, "")) for key in fields})


def _format_metric(value: float, percent: bool = False) -> str:
    if not np.isfinite(value):
        return "---"
    if percent:
        return f"{100 * value:.1f}"
    return f"{value:.4f}"


def write_dbgnn_order_tables(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    tbl_dir = run_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = [[
        "Network", "R0", "Variant", "Order", "Condition", "MRR", "Top-5",
        "Norm-Brier", "Norm-Entropy", "n_valid",
    ]]
    detail_tex = [
        "% DBGNN order results generated from local metrics",
        "\\begin{tabular}{lllrccccc}",
        "\\toprule",
        "Network & $R_0$ & Variant & $k$ & Cond. & MRR & Top-5 & NBS & $n$ \\\\",
        "\\midrule",
    ]
    for row in sorted(rows, key=lambda r: (r["network"], r["r0"], r["order"], r["variant"])):
        mrr = _float_or_nan(row.get("eval/mrr_mean"))
        top5 = _float_or_nan(row.get("eval/top_5_mean"))
        brier = _float_or_nan(row.get("eval/norm_brier_mean"))
        entropy = _float_or_nan(row.get("eval/norm_entropy_mean"))
        n_valid = _float_or_nan(row.get("eval/n_valid_mean"))
        network_tex = row["network"].replace("_", "\\_")
        variant_tex = row["variant"].replace("_", "\\_")
        detail_csv.append([
            row["network"], row["r0"], row["variant"], row["order"], row["condition"],
            _format_metric(mrr), _format_metric(top5, percent=True),
            _format_metric(brier), _format_metric(entropy), _format_metric(n_valid),
        ])
        detail_tex.append(
            f"{network_tex} & {row['r0']} & "
            f"{variant_tex} & {row['order']} & {row['condition']} & "
            f"{_format_metric(mrr)} & {_format_metric(top5, percent=True)} & "
            f"{_format_metric(brier)} & {_format_metric(n_valid)} \\\\"
        )
    detail_tex += ["\\bottomrule", "\\end{tabular}"]
    (tbl_dir / "dbgnn_order_results.tex").write_text("\n".join(detail_tex) + "\n")
    with open(tbl_dir / "dbgnn_order_results.csv", "w", newline="") as f:
        csv.writer(f).writerows(detail_csv)

    agg: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        agg.setdefault((int(row["order"]), row["condition"]), []).append(row)
    agg_csv = [["Order", "Condition", "Mean MRR", "Std MRR", "Mean Top-5", "Std Top-5", "Cells"]]
    agg_tex = [
        "% DBGNN order aggregate table generated from local metrics",
        "\\begin{tabular}{lrcccr}",
        "\\toprule",
        "Cond. & $k$ & MRR & Top-5 & NBS & Cells \\\\",
        "\\midrule",
    ]
    for (order, condition), vals in sorted(agg.items()):
        mrr_vals = np.array([_float_or_nan(v.get("eval/mrr_mean")) for v in vals], dtype=float)
        top5_vals = np.array([_float_or_nan(v.get("eval/top_5_mean")) for v in vals], dtype=float)
        brier_vals = np.array([_float_or_nan(v.get("eval/norm_brier_mean")) for v in vals], dtype=float)
        mrr_vals = mrr_vals[np.isfinite(mrr_vals)]
        top5_vals = top5_vals[np.isfinite(top5_vals)]
        brier_vals = brier_vals[np.isfinite(brier_vals)]
        mrr_mean = float(np.mean(mrr_vals)) if len(mrr_vals) else float("nan")
        mrr_std = float(np.std(mrr_vals, ddof=1)) if len(mrr_vals) > 1 else 0.0
        top5_mean = float(np.mean(top5_vals)) if len(top5_vals) else float("nan")
        top5_std = float(np.std(top5_vals, ddof=1)) if len(top5_vals) > 1 else 0.0
        brier_mean = float(np.mean(brier_vals)) if len(brier_vals) else float("nan")
        agg_csv.append([
            order, condition, _format_metric(mrr_mean), _format_metric(mrr_std),
            _format_metric(top5_mean, percent=True), _format_metric(top5_std, percent=True), len(vals),
        ])
        agg_tex.append(
            f"{condition} & {order} & {_format_metric(mrr_mean)} $\\pm$ {_format_metric(mrr_std)} & "
            f"{_format_metric(top5_mean, percent=True)} $\\pm$ {_format_metric(top5_std, percent=True)} & "
            f"{_format_metric(brier_mean)} & {len(vals)} \\\\"
        )
    agg_tex += ["\\bottomrule", "\\end{tabular}"]
    (tbl_dir / "dbgnn_order_aggregate.tex").write_text("\n".join(agg_tex) + "\n")
    with open(tbl_dir / "dbgnn_order_aggregate.csv", "w", newline="") as f:
        csv.writer(f).writerows(agg_csv)


def write_metrics_outputs(run_dir: Path, status_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    long_rows, summary_rows = metric_rows_from_status(status_rows)
    write_csv(run_dir / "metrics_long.csv", long_rows)
    write_csv(run_dir / "metrics_summary.csv", summary_rows)
    write_dbgnn_order_tables(run_dir, summary_rows)
    return summary_rows


def plot_metric_vs_order(rows: list[dict[str, Any]], run_dir: Path, metric: str, ylabel: str) -> None:
    usable = [r for r in rows if metric in r and int(r.get("order", 0)) > 0]
    if not usable:
        return
    apply_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for condition, marker in [("untuned", "o"), ("optuna", "s")]:
        orders = sorted({int(r["order"]) for r in usable if r["condition"] == condition})
        if not orders:
            continue
        means, stds = [], []
        for order in orders:
            vals = np.array([
                _float_or_nan(r.get(metric))
                for r in usable
                if r["condition"] == condition and int(r["order"]) == order
            ], dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if len(vals) else np.nan)
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.plot(orders, means_arr, marker=marker, lw=2.2, label=condition)
        ax.fill_between(orders, means_arr - stds_arr, means_arr + stds_arr, alpha=0.15)
    ax.set_xlabel("DBGNN order $k$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs DBGNN order")
    ax.legend()
    out = run_dir / "figures" / f"dbgnn_order_{metric.replace('/', '_').replace('_mean', '')}.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, f"{ylabel} vs DBGNN order", f"Aggregates `{metric}` over all completed network/R0 cells.", {"metric": metric})


def plot_order_heatmap(rows: list[dict[str, Any]], run_dir: Path, metric: str, title: str) -> None:
    usable = [r for r in rows if metric in r and int(r.get("order", 0)) > 0]
    if not usable:
        return
    networks = sorted({r["network"] for r in usable})
    orders = sorted({int(r["order"]) for r in usable})
    if not networks or not orders:
        return
    matrix = np.full((len(networks), len(orders)), np.nan)
    for i, network in enumerate(networks):
        for j, order in enumerate(orders):
            vals = np.array([
                _float_or_nan(r.get(metric))
                for r in usable
                if r["network"] == network and int(r["order"]) == order
            ], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                matrix[i, j] = float(np.mean(vals))
    apply_style()
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(orders)), max(4, 0.5 * len(networks))))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(orders)), [str(order) for order in orders])
    ax.set_yticks(range(len(networks)), [n.replace("_", " ") for n in networks])
    ax.set_xlabel("DBGNN order $k$")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=metric)
    out = run_dir / "figures" / f"dbgnn_order_heatmap_{metric.replace('/', '_').replace('_mean', '')}.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, title, f"Network by order heatmap for `{metric}` averaged over completed R0/condition cells.", {"metric": metric})


def plot_dbgnn_outputs(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_order(summary_rows, run_dir, "eval/mrr_mean", "MRR")
    plot_metric_vs_order(summary_rows, run_dir, "eval/top_5_mean", "Top-5 accuracy")
    plot_metric_vs_order(summary_rows, run_dir, "eval/norm_brier_mean", "Norm-Brier")
    plot_order_heatmap(summary_rows, run_dir, "eval/mrr_mean", "Mean MRR by network and DBGNN order")
    plot_order_heatmap(summary_rows, run_dir, "eval/top_5_mean", "Mean Top-5 by network and DBGNN order")


def write_plot_readme(plot_pdf: Path, title: str, description: str, params: dict[str, Any]) -> None:
    readme = plot_pdf.with_suffix(".README.md")
    lines = [
        f"# {title}",
        "",
        description,
        "",
        "## Parameters",
        "",
    ]
    lines += [f"- `{key}`: `{value}`" for key, value in params.items()]
    readme.write_text("\n".join(lines) + "\n")


def refresh_result_bundle(run_dir: Path, status_path: Path) -> Path:
    """Refresh the publication-facing result bundle."""
    return sync_publication_result(
        run_dir=run_dir,
        status_rows=read_status(status_path),
        experiment_name="dbgnn_higher_order",
    )


def main() -> None:
    args = parse_args()
    networks = resolve_networks(args.networks)
    r0_labels = normalize_r0_labels(args.r0)
    orders = list(dict.fromkeys(args.orders))
    preset = PRESETS[args.preset]

    for network in networks:
        if network not in BETAS:
            raise ValueError(f"No calibrated beta matrix configured for network '{network}'")
    for order in orders:
        if order < 2:
            raise ValueError(f"DBGNN order must be >= 2, got {order}")

    stats = {network: read_full_network_stats(network) for network in networks}
    networks = sorted(networks, key=lambda n: (stats[n].n_nodes, stats[n].n_edges, n))
    sample_budget = compute_sample_budget(stats, args.sample_reference, args.sample_budget_factor)
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"
    hpo_reference_r0 = resolve_hpo_reference_r0(args.hpo_reference_r0, r0_labels)
    write_manifest(run_dir, args, networks, r0_labels, orders, stats, sample_budget)
    refresh_result_bundle(run_dir, status_path)

    print("=" * 72)
    print("DBGNN Higher-Order Experiment Runner")
    print("=" * 72)
    print(f"Run dir  : {run_dir}")
    print(f"Preset   : {args.preset} ({preset})")
    print(f"Networks : {', '.join(networks)}")
    print(f"R0       : {', '.join(r0_labels)}")
    print(f"Orders   : {', '.join(str(k) for k in orders)}")
    print(
        "HPO      : "
        + (
            f"enabled, paired untuned/+Optuna finals, trials={args.hpo_trials}, "
            f"sampler={args.hpo_sampler}, pruner={args.hpo_pruner}, "
            f"scope={args.hpo_scope}, reference_r0={hpo_reference_r0}, "
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
    if args.no_sampling:
        print("Sampling : disabled")
    else:
        print(
            f"Sampling : {args.sample_method} for networks with "
            f"n>{args.sample_node_threshold}; target n<={args.sample_target_nodes}; "
            f"node*edge budget {sample_budget:,} "
            f"({args.sample_reference}/{args.sample_budget_factor:g})"
        )
    print(f"Timeout  : {args.timeout_seconds}s per TSIR/train command")
    print(
        "Batch   : "
        f"k=2<={args.base_batch_size}, "
        f"k=3<={args.order3_batch_size}, "
        f"k>=4<={args.high_order_batch_size}"
    )
    print(
        "k>=4    : "
        f"delta={args.high_order_delta}, "
        f"time_bin={args.high_order_time_bin_size}, "
        f"batch<={args.high_order_batch_size}, "
        f"limits(states={args.max_temporal_states:,}, "
        f"nodes={args.max_db_nodes:,}, edges={args.max_db_edges:,})"
    )
    if any(order >= 6 for order in orders):
        print(f"k>=6    : time_bin>={args.very_high_order_time_bin_size}")
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed")

    for network in networks:
        sample_cfg = sampling_cfg_for_network(network, stats, sample_budget, args)
        if sample_cfg is None:
            print(f"\n### {network}: no sampling (n={stats[network].n_nodes}, cost={stats[network].node_edge_cost:,})")
        else:
            reduction = stats[network].node_edge_cost / sample_budget
            print(
                f"\n### {network}: sampled with {args.sample_method} to target "
                f"n<={sample_cfg['target_nodes']} and budget {sample_budget:,} "
                f"(expected node*edge reduction >= {reduction:.2f}x)"
            )

        artifact_cache: dict[str, str | None] = {}
        hpo_best_cache: dict[int, Path | None] = {}

        def ensure_artifact(r0_label: str) -> str | None:
            if r0_label not in artifact_cache:
                try:
                    artifact_cache[r0_label] = stage_tsir(args, run_dir, status_path, network, r0_label, sample_cfg)
                    refresh_result_bundle(run_dir, status_path)
                except Exception as exc:
                    artifact_cache[r0_label] = None
                    print(f"    FATAL TSIR stage failed, skipping scenario: {exc}")
                    refresh_result_bundle(run_dir, status_path)
            return artifact_cache[r0_label]

        def resolve_hpo_config(r0_label: str, order: int, artifact: str) -> Path | None:
            if args.hpo_scope == "scenario":
                best_cfg = stage_hpo(args, run_dir, status_path, network, r0_label, order, artifact)
                refresh_result_bundle(run_dir, status_path)
                return best_cfg

            if order not in hpo_best_cache:
                ref_artifact = artifact if r0_label == hpo_reference_r0 else ensure_artifact(hpo_reference_r0)
                if ref_artifact is None:
                    raise RuntimeError(f"Reference TSIR artifact unavailable for {network}/{hpo_reference_r0}")
                hpo_best_cache[order] = stage_hpo(
                    args,
                    run_dir,
                    status_path,
                    network,
                    hpo_reference_r0,
                    order,
                    ref_artifact,
                )
                refresh_result_bundle(run_dir, status_path)

            best_cfg = hpo_best_cache[order]
            if r0_label == hpo_reference_r0:
                return best_cfg

            tuned_cfg = write_reused_optuna_config(
                args,
                run_dir,
                network,
                r0_label,
                order,
                hpo_reference_r0,
                best_cfg,
            )
            update_status(
                status_path,
                {
                    "network": network,
                    "r0_label": r0_label,
                    "stage": "hpo",
                    "variant": optuna_variant_name(order),
                    "order": order,
                    "status": "success",
                    "artifact": artifact,
                    "returncode": 0,
                    "message": f"reused network-scope HPO from {hpo_reference_r0}: {best_cfg}",
                    "log_path": "",
                },
            )
            refresh_result_bundle(run_dir, status_path)
            return tuned_cfg

        for order in orders:
            print(f"\n--- {network} / DBGNN k={order}")
            for r0_label in r0_labels:
                sc = scenario(network, r0_label)
                print(f"  {r0_label}  R0={sc['r0']} beta={sc['beta']} mu={sc['mu']}")
                artifact = ensure_artifact(r0_label)
                if artifact is None:
                    print("    Skipping train because TSIR artifact is unavailable")
                    continue
                if args.with_hpo:
                    best_cfg = resolve_hpo_config(r0_label, order, artifact)
                    untuned_cfg = write_untuned_paired_config(args, run_dir, network, r0_label, order, best_cfg)
                    stage_train(args, run_dir, status_path, network, r0_label, order, artifact, untuned_cfg)
                    refresh_result_bundle(run_dir, status_path)
                    stage_train(
                        args,
                        run_dir,
                        status_path,
                        network,
                        r0_label,
                        order,
                        artifact,
                        best_cfg,
                        status_variant=optuna_variant_name(order),
                    )
                    refresh_result_bundle(run_dir, status_path)
                else:
                    stage_train(args, run_dir, status_path, network, r0_label, order, artifact)
                    refresh_result_bundle(run_dir, status_path)

    status_rows = read_status(status_path)
    write_summary_csv(run_dir, status_rows)
    summary_rows = write_metrics_outputs(run_dir, status_rows)
    if not args.dry_run:
        plot_dbgnn_outputs(run_dir, summary_rows)
    result_dir = refresh_result_bundle(run_dir, status_path)
    print("\nDone.")
    print(f"Status : {status_path}")
    print(f"Summary: {run_dir / 'run_matrix_summary.csv'}")
    print(f"Result : {result_dir}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
