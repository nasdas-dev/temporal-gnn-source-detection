"""
Thesis-final experiment pipeline.

Runs the complete source-detection sweep for the thesis networks, R0 values,
GNN models, heuristic baselines, result tables, and plots.

Default run:
    python run_all_experiments.py

Useful controls:
    python run_all_experiments.py --dry-run
    python run_all_experiments.py --preset fast --networks toy_holme --models static_gnn
    python run_all_experiments.py --resume --run-name 20260512_010000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from viz.rank_vs_outbreak import load_eval_arrays
from viz.style import MODEL_COLORS, MODEL_LABELS, apply_style, finish_fig, model_style


WANDB_PROJECT = "source-detection"

NETWORKS = ["lyon_ward", "malawi", "france_office", "biasca", "olten"]
MODELS = ["static_gnn", "temporal_gnn", "backtracking", "dbgnn"]
BASELINES = ["uniform", "random", "degree", "closeness", "betweenness", "jordan_center"]
R0_LABELS = ["r0_08", "r0_10", "r0_11", "r0_15", "r0_20", "r0_25"]

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
    "biasca":       {"r0_08": 0.043, "r0_10": 0.052, "r0_11": 0.056, "r0_15": 0.079, "r0_20": 0.117, "r0_25": 0.172},
    "olten":        {"r0_08": 0.079, "r0_10": 0.096, "r0_11": 0.104, "r0_15": 0.146, "r0_20": 0.217, "r0_25": 0.318},
}

TEMPORAL_GROUP_BY_TIME = {
    "lyon_ward": 1,
    "malawi": 10,
    "france_office": 6,
    "biasca": 12,
    "olten": 12,
}


@dataclass(frozen=True)
class Preset:
    n_runs: int
    mc_runs: int
    n_mc: int
    reps: int
    n_truth: int


PRESETS = {
    "max_quality": Preset(n_runs=1000, mc_runs=500, n_mc=500, reps=1, n_truth=1000),
    "balanced": Preset(n_runs=1000, mc_runs=500, n_mc=500, reps=1, n_truth=1000),
    "fast": Preset(n_runs=120, mc_runs=80, n_mc=80, reps=1, n_truth=40),
}

LOSS_GUARD = {
    "enabled": True,
    "warmup_epochs": 20,
    "divergence_factor": 1.5,
    "uniform_tolerance": 0.02,
    "uniform_window": 80,
    "min_improvement": 0.01,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="max_quality")
    p.add_argument("--networks", nargs="+", default=NETWORKS)
    p.add_argument("--models", nargs="+", default=MODELS)
    p.add_argument("--r0", nargs="+", default=["all"], help="R0 labels: r0_08 ... r0_25, numeric values, or all")
    p.add_argument("--output", default="results/thesis_final", help="Root results directory")
    p.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run directory and skip terminal stages")
    p.add_argument("--force", action="store_true", help="Rerun stages even when a terminal status exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    p.add_argument("--save-probs", action="store_true", help="Save probs_rep*.pt tensors from main_train.py")
    p.add_argument("--skip-tsir", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--skip-viz", action="store_true")
    p.add_argument("--skip-tables", action="store_true")
    return p.parse_args()


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
        "mu": 0.01,
    }


def artifact_name(network: str, r0_label: str) -> str:
    return f"thesis_final_{network}_{r0_label}"


def build_tsir_config(network: str, r0_label: str, preset: Preset) -> dict[str, Any]:
    meta = read_network_meta(network)
    sc = scenario(network, r0_label)
    return {
        "nwk": {
            "type": "empirical",
            "name": network,
            "t_max": meta["t_max"],
        },
        "sir": {
            "beta": sc["beta"],
            "mu": sc["mu"],
            "start_t": 0,
            "end_t": meta["t_max"],
            "n_runs": preset.n_runs,
            "mc_runs": preset.mc_runs,
        },
    }


def _template_path(network: str, model: str) -> Path:
    direct = Path("exp") / network / f"{model}.yml"
    if direct.exists():
        return direct
    fallback = Path("exp/france_office") / f"{model}.yml"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No model config template found for {network}/{model}")


def build_model_config(network: str, model: str, r0_label: str, preset: Preset, save_probs: bool = False) -> dict[str, Any]:
    with open(_template_path(network, model)) as f:
        cfg = yaml.safe_load(f)
    cfg["model"] = model
    cfg["eval"] = {
        **cfg.get("eval", {}),
        "min_outbreak": 1,
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
    cfg.setdefault("output", {})["save_probs"] = save_probs
    if model == "temporal_gnn":
        cfg.setdefault("temporal_gnn", {})["group_by_time"] = TEMPORAL_GROUP_BY_TIME.get(network, 12)
    if model == "dbgnn":
        cfg.setdefault("dbgnn", {})["delta"] = 24
    cfg["experiment"] = {
        "network": network,
        "r0_label": r0_label,
        **scenario(network, r0_label),
    }
    if preset.n_runs < preset.reps * preset.n_truth:
        raise ValueError(
            f"Invalid preset: n_runs={preset.n_runs} < reps*n_truth={preset.reps * preset.n_truth}"
        )
    return cfg


def build_eval_config(network: str, r0_label: str, preset: Preset) -> dict[str, Any]:
    return {
        "eval": {
            "min_outbreak": 1,
            "top_k": [1, 3, 5, 10],
            "credible_p": [0.80, 0.90],
            "inverse_rank_offset": [0],
            "n_truth": preset.n_truth,
        },
        "baselines": BASELINES,
        "experiment": {
            "network": network,
            "r0_label": r0_label,
            **scenario(network, r0_label),
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
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "preset": args.preset,
        "preset_values": preset.__dict__,
        "networks": networks,
        "models": args.models,
        "r0_labels": r0_labels,
        "betas": {n: BETAS[n] for n in networks if n in BETAS},
        "mu": 0.01,
        "baselines": BASELINES,
        "wandb_project": WANDB_PROJECT,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


STATUS_FIELDS = [
    "network", "r0_label", "stage", "model", "status", "run_id",
    "artifact", "returncode", "message", "log_path",
]
TERMINAL_STATUSES = {"success", "loss_guard_aborted", "skipped"}


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


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> tuple[int, str]:
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
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            captured.append(line)
        proc.wait()
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
    write_yaml(cfg_path, build_tsir_config(network, r0_label, PRESETS[args.preset]))
    log_path = run_dir / network / r0_label / "logs" / "tsir.log"
    cmd = [sys.executable, "main_tsir.py", "--cfg", str(cfg_path), "--data", art]
    rc, stdout = run_command(cmd, log_path, args.dry_run)
    status = "success" if rc == 0 else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "tsir",
        "status": status, "artifact": art, "returncode": rc,
        "message": "dry_run" if args.dry_run else "", "log_path": log_path,
    })
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"TSIR failed for {network}/{r0_label}; see {log_path}")
    return art


def stage_train(args: argparse.Namespace, run_dir: Path, status_path: Path, network: str, r0_label: str, model: str, art: str) -> str | None:
    if args.skip_train:
        update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "train", "model": model, "status": "skipped", "artifact": art})
        return None
    if should_skip(status_path, args, network, r0_label, "train", model):
        rows = read_status(status_path)
        return next((r.get("run_id") for r in rows if r.get("network") == network and r.get("r0_label") == r0_label and r.get("stage") == "train" and r.get("model") == model), None)

    cfg_path = run_dir / "configs" / network / r0_label / f"{model}.yml"
    write_yaml(cfg_path, build_model_config(network, model, r0_label, PRESETS[args.preset], args.save_probs))
    log_path = run_dir / network / r0_label / "logs" / f"train_{model}.log"
    cmd = [sys.executable, "main_train.py", "--cfg", str(cfg_path), "--data", f"{art}:latest"]
    if args.save_probs:
        cmd.append("--save-probs")
    rc, stdout = run_command(cmd, log_path, args.dry_run)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = "success" if rc == 0 else "loss_guard_aborted" if rc == 88 or "LOSS_GUARD_ABORT" in stdout else "failed"
    update_status(status_path, {
        "network": network, "r0_label": r0_label, "stage": "train", "model": model,
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

    cfg_path = run_dir / "configs" / network / r0_label / "eval.yml"
    write_yaml(cfg_path, build_eval_config(network, r0_label, PRESETS[args.preset]))
    log_path = run_dir / network / r0_label / "logs" / "eval.log"
    cmd = [sys.executable, "main_eval.py", "--cfg", str(cfg_path), "--data", f"{art}:latest"]
    rc, stdout = run_command(cmd, log_path, args.dry_run)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = "success" if rc == 0 else "failed"
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
            entries.append({"method": row["model"], "kind": "model", "run_id": row["run_id"], "baseline": ""})
        if row.get("stage") == "eval" and row.get("status") == "success" and row.get("run_id"):
            for baseline in BASELINES:
                entries.append({"method": baseline, "kind": "baseline", "run_id": row["run_id"], "baseline": baseline})
    order = {m: i for i, m in enumerate(MODELS + BASELINES)}
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
    params = {"network": network, "r0": sc["r0"], "beta": sc["beta"], "mu": sc["mu"], "end_t": meta["t_max"], "min_outbreak": 1}

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    for entry, _, _, sizes, ranks in loaded:
        style = model_style(entry["method"])
        cents, vals, ses, _ = binned_topk(sizes, ranks, 5)
        valid = ~np.isnan(vals)
        ax.fill_between(cents[valid], (vals - ses)[valid], (vals + ses)[valid], color=style["color"], alpha=0.12)
        ls = "-" if entry["kind"] == "model" else "--"
        ax.plot(cents[valid], vals[valid], color=style["color"], lw=2.2, ls=ls, label=style["label"])
    ax.set_title(f"Top-5 Score vs Outbreak Size: {title_suffix}")
    ax.set_xlabel("outbreak_size")
    ax.set_ylabel("top-5 score")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    out = fig_dir / "top5_vs_outbreak_compare.pdf"
    finish_fig(fig, str(out))
    write_plot_readme(out, "Top-5 Score vs Outbreak Size", "Shows the fraction of valid outbreaks where the true source is ranked in the top 5, binned by absolute outbreak size.", params)

    apply_style()
    fig, ax = plt.subplots(figsize=(11, 7))
    for entry, _, n_nodes, sizes, ranks in loaded:
        style = model_style(entry["method"])
        cents, means, p25, p75 = binned_rank(sizes, ranks)
        valid = ~np.isnan(means)
        ax.fill_between(cents[valid], p25[valid], p75[valid], color=style["color"], alpha=0.12)
        ls = "-" if entry["kind"] == "model" else "--"
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
    ax.set_title(f"Training Curves: {MODEL_LABELS.get(model, model)}")
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
            summary_path = run_data / "metrics_summary.json"
            if summary_path.exists():
                payload = json.loads(summary_path.read_text())
                metrics = payload.get("metrics", {})
                rec = {**base, "method": row["model"], "kind": "model", "status": row["status"], **metrics}
                summary_rows.append(rec)
            for rep_path in sorted(run_data.glob("metrics_rep*.json")):
                payload = json.loads(rep_path.read_text())
                rep = payload.get("rep", "")
                for metric, value in payload.get("metrics", {}).items():
                    long_rows.append({**base, "method": row["model"], "kind": "model", "rep": rep, "metric": metric, "value": value})
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
                            summary[f"{k}_mean"] = float(v)
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
    for (network, method), vals in sorted(agg.items()):
        mrr = np.nanmean([v.get("eval/mrr_mean", np.nan) for v in vals])
        top5 = np.nanmean([v.get("eval/top_5_mean", np.nan) for v in vals])
        brier = np.nanmean([v.get("eval/norm_brier_mean", np.nan) for v in vals])
        label = MODEL_LABELS.get(method, method).replace("_", "\\_")
        ntex = network.replace("_", "\\_")
        lines.append(f"{ntex} & {label} & {mrr:.4f} & {100 * top5:.1f} & {brier:.4f} \\\\")
        csv_rows.append([network, MODEL_LABELS.get(method, method), f"{mrr:.6g}", f"{top5:.6g}", f"{brier:.6g}"])
    lines += ["\\bottomrule", "\\end{tabular}"]
    (tbl_dir / "benchmark_table.tex").write_text("\n".join(lines) + "\n")
    with open(tbl_dir / "benchmark_table.csv", "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)


def plot_metric_vs_r0(rows: list[dict[str, Any]], output: Path, metric: str, title: str, ylabel: str) -> None:
    networks = [n for n in NETWORKS if any(r["network"] == n and metric in r for r in rows)]
    if not networks:
        return
    methods = [m for m in MODELS + BASELINES if any(r["method"] == m and metric in r for r in rows)]
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
            ls = "-" if method in MODELS else "--"
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
    write_plot_readme(output, title, f"Global thesis plot for `{metric}` across R0 values, faceted by network.", {"metric": metric, "min_outbreak": 1})


def plot_top5_heatmap(rows: list[dict[str, Any]], output: Path) -> None:
    usable = [r for r in rows if "eval/top_5_mean" in r]
    if not usable:
        return
    row_keys = [(n, m) for n in NETWORKS for m in MODELS if any(r["network"] == n and r["method"] == m for r in usable)]
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
    ax.set_yticks(range(len(row_keys)), [f"{n} / {MODEL_LABELS.get(m, m)}" for n, m in row_keys])
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
    write_plot_readme(output, "Valid Outbreaks by Scenario", "Shows how many observations pass `min_outbreak=1` for each network/R0 condition.", {"min_outbreak": 1})


def plot_global_outputs(run_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_metric_vs_r0(summary_rows, fig_dir / "mrr_vs_r0_by_network.pdf", "eval/mrr_mean", "MRR vs R0 by Network", "MRR")
    plot_metric_vs_r0(summary_rows, fig_dir / "top5_vs_r0_by_network.pdf", "eval/top_5_mean", "Top-5 vs R0 by Network", "Top-5 accuracy")
    plot_metric_vs_r0(summary_rows, fig_dir / "norm_brier_vs_r0.pdf", "eval/norm_brier_mean", "Norm-Brier vs R0", "Norm-Brier")
    plot_top5_heatmap(summary_rows, fig_dir / "top5_heatmap_network_model_r0.pdf")
    plot_valid_outbreaks(summary_rows, fig_dir / "valid_outbreaks_by_scenario.pdf")


def run_network_stats_table(args: argparse.Namespace, run_dir: Path, networks: list[str]) -> None:
    if args.skip_tables:
        return
    tbl_dir = run_dir / "tables"
    cmd = [sys.executable, "-m", "eval.tables", "network_stats", "--networks", *networks, "--output", str(tbl_dir)]
    run_command(cmd, run_dir / "logs" / "network_stats.log", args.dry_run)


def main() -> None:
    args = parse_args()
    networks = args.networks
    r0_labels = normalize_r0_labels(args.r0)
    preset = PRESETS[args.preset]
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"

    write_manifest(run_dir, args, networks, r0_labels)
    print("=" * 72)
    print("Thesis Final Experiment Runner")
    print("=" * 72)
    print(f"Run dir  : {run_dir}")
    print(f"Preset   : {args.preset} ({preset})")
    print(f"Networks : {', '.join(networks)}")
    print(f"R0       : {', '.join(r0_labels)}")
    print(f"Models   : {', '.join(args.models)}")
    print(f"Baselines: {', '.join(BASELINES)}")
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed")

    for network in networks:
        if network not in BETAS:
            raise ValueError(f"No beta matrix configured for network '{network}'")
        for r0_label in r0_labels:
            sc = scenario(network, r0_label)
            print(f"\n### {network} / {r0_label}  R0={sc['r0']} beta={sc['beta']} mu={sc['mu']}")
            try:
                art = stage_tsir(args, run_dir, status_path, network, r0_label)
            except Exception as exc:
                print(f"  FATAL TSIR stage failed, skipping scenario: {exc}")
                continue

            for model in args.models:
                if model not in MODELS:
                    print(f"  SKIP unknown/out-of-scope model: {model}")
                    update_status(status_path, {"network": network, "r0_label": r0_label, "stage": "train", "model": model, "status": "skipped", "message": "unknown_model"})
                    continue
                stage_train(args, run_dir, status_path, network, r0_label, model, art)

            stage_eval(args, run_dir, status_path, network, r0_label, art)
            if not args.skip_viz and not args.dry_run:
                plot_scenario_outputs(run_dir, read_status(status_path), network, r0_label)

    status_rows = read_status(status_path)
    summary_rows = write_metrics_outputs(run_dir, status_rows)
    if not args.skip_viz and not args.dry_run:
        plot_global_outputs(run_dir, summary_rows)
    run_network_stats_table(args, run_dir, networks)

    print("\nDone.")
    print(f"Status : {status_path}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
