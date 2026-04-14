"""
Full experiment pipeline — all networks × all SIR scenarios × all models.

Runs the complete source-detection evaluation pipeline:

    Stage 1  main_tsir.py   — SIR simulation → W&B artifact
    Stage 2  main_train.py  — GNN model training  (per model)
    Stage 3  main_eval.py   — Baseline evaluation
    Stage 4  viz scripts    — All evaluation plots
    Stage 5  eval.tables    — Network-stats + benchmark LaTeX/CSV tables

Output directory layout
-----------------------
results/
  <network>/
    scenario_r0_<X>/
      figures/
        rank_vs_outbreak.pdf
        topk_vs_outbreak.pdf
        training_curves_<model>.pdf
      tables/
        network_stats_table.{tex,csv}
        benchmark_table.{tex,csv}
  network_stats_table.{tex,csv}   ← single table covering all networks

Usage
-----
    # Full run (all networks, all scenarios)
    python run_all_experiments.py

    # Specific networks / scenarios
    python run_all_experiments.py --networks france_office karate_static --scenarios r0_25

    # Dry run (print commands, don't execute)
    python run_all_experiments.py --dry-run

    # Skip already-finished stages (re-uses W&B artifact from a previous run)
    python run_all_experiments.py --skip-tsir

Notes
-----
- W&B artifact names are tagged with the scenario: e.g. ``france_office_r0_25``
- All runs are tagged with ``full_experiment_sweep`` in W&B for easy filtering
- Logs (stdout + stderr) are written to results/<network>/scenario_r0_<X>/run.log
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Networks that have a complete exp/<name>/tsir.yml config.
# pig_data / test are excluded: no proper exp configs yet.
#
# NOTE: escort has 14 783 nodes — its tsir.yml uses n_runs=2 / mc_runs=5.
# The global N_RUNS=50 / MC_RUNS=1000 overrides applied below will produce
# arrays that are ~10 GB each and will almost certainly OOM.  escort is
# therefore listed last; use --networks to exclude it if memory is limited:
#   python run_all_experiments.py --networks france_office karate_static ...
NETWORKS: list[str] = [
    "france_office",
    "karate_static",
    "lyon_ward",
    "malawi",
    "students",
    "toy_holme",
    "escort",
]

# GNN models (each has exp/<network>/<model>.yml)
MODELS: list[str] = [
    "backtracking",
    "static_gnn",
    "temporal_gnn",
    "dbgnn",
    "dag_gnn",
]

# SIR scenarios: (label, beta, mu)
# R0 = beta/mu * <mean_contacts_per_time_step> — approximate labels
SCENARIOS: list[dict[str, Any]] = [
    {"label": "r0_25", "beta": 0.06, "mu": 0.01},
    {"label": "r0_20", "beta": 0.04, "mu": 0.01},
    {"label": "r0_15", "beta": 0.03, "mu": 0.01},
]

# Fixed simulation sizes for all runs
MC_RUNS = 1000
N_RUNS  = 50

WANDB_PROJECT = "source-detection"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    cmd: list[str],
    log_path: Path | None = None,
    dry_run: bool = False,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, str]:
    """Run *cmd*, optionally tee output to *log_path*.

    Returns (returncode, stdout_text).  On dry-run, prints the command and
    returns (0, "").
    """
    label = " ".join(cmd)
    if dry_run:
        print(f"  [DRY] {label}")
        return 0, ""

    env = {**os.environ, **(extra_env or {})}
    log_fh = open(log_path, "a") if log_path else None

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
            if log_fh:
                log_fh.write(line)
        proc.wait()
        return proc.returncode, "".join(captured)
    finally:
        if log_fh:
            log_fh.close()


def _extract_run_id(stdout: str) -> str | None:
    """Extract W&B run ID from main_train.py / main_eval.py stdout.

    W&B prints lines like:
        wandb: Run data is saved locally in wandb/run-20240101_120000-abc12345
    or
        W&B run : https://wandb.ai/user/project/runs/abc12345
    """
    # Pattern 1: run-<date>-<id>
    m = re.search(r"wandb/run-\d{8}_\d{6}-([a-z0-9]{8})", stdout)
    if m:
        return m.group(1)
    # Pattern 2: /runs/<id>
    m = re.search(r"/runs/([a-z0-9]{8})\b", stdout)
    if m:
        return m.group(1)
    # Pattern 3: "Run ID: <id>"
    m = re.search(r"run(?:\s+id)?[:\s]+([a-z0-9]{8})\b", stdout, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _make_temp_tsir_cfg(
    base_cfg_path: str,
    beta: float,
    mu: float,
    n_runs: int,
    mc_runs: int,
) -> str:
    """Write a temporary TSIR YAML with the scenario's SIR params.

    Returns the path to the temp file (caller must delete when done).
    """
    with open(base_cfg_path) as f:
        data = yaml.safe_load(f)

    data["sir"]["beta"]     = beta
    data["sir"]["mu"]       = mu
    data["sir"]["n_runs"]   = n_runs
    data["sir"]["mc_runs"]  = mc_runs

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False, prefix="tsir_tmp_"
    )
    yaml.dump(data, tmp)
    tmp.close()
    return tmp.name


def _log(msg: str, log_path: Path | None = None) -> None:
    print(msg)
    if log_path:
        with open(log_path, "a") as f:
            f.write(msg + "\n")


def _section(title: str, log_path: Path | None = None) -> None:
    bar = "=" * 60
    _log(f"\n{bar}\n{title}\n{bar}", log_path)


# ---------------------------------------------------------------------------
# Stage functions
# ---------------------------------------------------------------------------

def stage_tsir(
    network: str,
    scenario: dict[str, Any],
    artifact_name: str,
    out_dir: Path,
    dry_run: bool,
) -> int:
    """Run main_tsir.py for network × scenario."""
    base_cfg = f"exp/{network}/tsir.yml"
    if not os.path.exists(base_cfg):
        _log(f"  SKIP: no tsir config at {base_cfg}")
        return 1

    log_path = out_dir / "run.log"
    _section(f"Stage 1: SIR simulation  [{network} / {scenario['label']}]", log_path)

    tmp_cfg = _make_temp_tsir_cfg(
        base_cfg,
        beta=scenario["beta"],
        mu=scenario["mu"],
        n_runs=N_RUNS,
        mc_runs=MC_RUNS,
    )
    try:
        rc, _ = _run(
            ["python", "main_tsir.py", "--cfg", tmp_cfg, "--data", artifact_name],
            log_path=log_path,
            dry_run=dry_run,
        )
    finally:
        if not dry_run:
            os.unlink(tmp_cfg)

    return rc


def stage_train(
    network: str,
    model: str,
    scenario: dict[str, Any],
    artifact_name: str,
    out_dir: Path,
    dry_run: bool,
) -> str | None:
    """Run main_train.py for one model. Returns W&B run ID or None on failure."""
    cfg_path = f"exp/{network}/{model}.yml"
    if not os.path.exists(cfg_path):
        _log(f"  SKIP: no model config at {cfg_path}")
        return None

    log_path = out_dir / "run.log"
    _section(f"Stage 2: Training  [{network} / {scenario['label']} / {model}]", log_path)

    cmd = [
        "python", "main_train.py",
        "--cfg",  cfg_path,
        "--data", f"{artifact_name}:latest",
        "--override",
        f"sir.beta={scenario['beta']}",
        f"sir.mu={scenario['mu']}",
        f"sir.mc_runs={MC_RUNS}",
        f"sir.n_runs={N_RUNS}",
        f"train.n_mc={MC_RUNS}",
    ]

    rc, stdout = _run(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        _log(f"  ERROR: training failed (rc={rc})")
        return None

    if dry_run:
        return "dryrun00"

    run_id = _extract_run_id(stdout)
    if run_id:
        _log(f"  Captured run ID: {run_id}", log_path)
    else:
        _log("  WARNING: could not extract run ID from stdout", log_path)
    return run_id


def stage_eval(
    network: str,
    scenario: dict[str, Any],
    artifact_name: str,
    out_dir: Path,
    dry_run: bool,
) -> str | None:
    """Run main_eval.py (baselines). Returns W&B run ID or None on failure.

    Note: eval.yml contains only eval: and baselines: sections (no SIR params).
    SIR data is loaded from the W&B artifact, so no overrides are needed here.
    """
    cfg_path = f"exp/{network}/eval.yml"
    if not os.path.exists(cfg_path):
        _log(f"  SKIP: no eval config at {cfg_path}")
        return None

    log_path = out_dir / "run.log"
    _section(f"Stage 3: Baselines  [{network} / {scenario['label']}]", log_path)

    cmd = [
        "python", "main_eval.py",
        "--cfg",  cfg_path,
        "--data", f"{artifact_name}:latest",
    ]

    rc, stdout = _run(cmd, log_path=log_path, dry_run=dry_run)
    if rc != 0:
        _log(f"  ERROR: baseline eval failed (rc={rc})")
        return None

    if dry_run:
        return "dryrun00"

    run_id = _extract_run_id(stdout)
    return run_id


def stage_viz(
    network: str,
    scenario: dict[str, Any],
    model_run_ids: dict[str, str],
    eval_run_id: str | None,
    out_dir: Path,
    dry_run: bool,
) -> None:
    """Run all viz scripts. Saves figures into out_dir/figures/.

    rank_vs_outbreak / topk_vs_outbreak CLI convention
    ---------------------------------------------------
    Each series is one (run-id, label, baseline) tuple:
    - GNN model runs : --run-id <model_run_id>  --baseline None
    - Baseline series: --run-id <eval_run_id>   --baseline <name>
    Both scripts accept parallel lists of these three args.
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    _section(f"Stage 4: Visualisation  [{network} / {scenario['label']}]", log_path)

    BASELINE_NAMES = ["uniform", "random", "degree",
                      "closeness", "betweenness", "jordan_center"]

    # Build parallel lists for all series (GNN models + baselines)
    all_run_ids:  list[str] = list(model_run_ids.values())
    all_labels:   list[str] = list(model_run_ids.keys())
    all_baselines: list[str] = ["None"] * len(model_run_ids)

    if eval_run_id:
        for bname in BASELINE_NAMES:
            all_run_ids.append(eval_run_id)
            all_labels.append(bname)
            all_baselines.append(bname)

    if not all_run_ids:
        _log("  SKIP: no run IDs available for visualisation", log_path)
        return

    # ── rank_vs_outbreak ────────────────────────────────────────────────────
    cmd = (
        ["python", "viz/rank_vs_outbreak.py"]
        + ["--run-id"]   + all_run_ids
        + ["--label"]    + all_labels
        + ["--baseline"] + all_baselines
        + ["--no-scatter"]
        + ["--output", str(fig_dir / "rank_vs_outbreak.pdf")]
    )
    _run(cmd, log_path=log_path, dry_run=dry_run)

    # ── topk_vs_outbreak (k=5) ──────────────────────────────────────────────
    cmd = (
        ["python", "viz/topk_vs_outbreak.py"]
        + ["--run-id"]   + all_run_ids
        + ["--label"]    + all_labels
        + ["--baseline"] + all_baselines
        + ["--k", "5"]
        + ["--output", str(fig_dir / "topk_vs_outbreak.pdf")]
    )
    _run(cmd, log_path=log_path, dry_run=dry_run)

    # ── training curves (per model, W&B required) ───────────────────────────
    for model, rid in model_run_ids.items():
        cmd = [
            "python", "viz/training_curves.py",
            "--run-path", f"{WANDB_PROJECT}/{rid}",
            "--output", str(fig_dir / f"training_curves_{model}.pdf"),
        ]
        _run(cmd, log_path=log_path, dry_run=dry_run)


def stage_tables(
    network: str,
    scenario: dict[str, Any],
    artifact_name: str,
    out_dir: Path,
    dry_run: bool,
) -> None:
    """Generate LaTeX/CSV tables for this network × scenario."""
    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    _section(f"Stage 5: Tables  [{network} / {scenario['label']}]", log_path)

    # Benchmark table (requires W&B runs to exist)
    cmd = [
        "python", "-m", "eval.tables", "benchmark",
        "--data", artifact_name,
        "--project", WANDB_PROJECT,
        "--output", str(tbl_dir),
    ]
    _run(cmd, log_path=log_path, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Network-stats table (offline, all networks at once)
# ---------------------------------------------------------------------------

def global_network_stats_table(
    networks: list[str],
    results_root: Path,
    dry_run: bool,
) -> None:
    """Write a single network-stats table covering all requested networks."""
    tbl_dir = results_root / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)

    _section("Global: Network Statistics Table")

    cmd = (
        ["python", "-m", "eval.tables", "network_stats"]
        + ["--networks"] + networks
        + ["--output", str(tbl_dir)]
    )
    _run(cmd, log_path=None, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--networks", nargs="+", default=NETWORKS,
        metavar="NET",
        help="Networks to run (default: all configured networks)",
    )
    p.add_argument(
        "--scenarios", nargs="+",
        choices=[s["label"] for s in SCENARIOS] + ["all"],
        default=["all"],
        metavar="SCENARIO",
        help="Scenarios to run: r0_25 r0_20 r0_15 or 'all' (default: all)",
    )
    p.add_argument(
        "--models", nargs="+", default=MODELS,
        metavar="MODEL",
        help="GNN models to train (default: all)",
    )
    p.add_argument(
        "--output", default="results",
        metavar="DIR",
        help="Root output directory (default: results/)",
    )
    p.add_argument(
        "--skip-tsir", action="store_true",
        help="Skip Stage 1 (reuse existing W&B artifact from a previous run)",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Skip Stage 2 (GNN training)",
    )
    p.add_argument(
        "--skip-eval", action="store_true",
        help="Skip Stage 3 (baseline evaluation)",
    )
    p.add_argument(
        "--skip-viz", action="store_true",
        help="Skip Stage 4 (visualisation)",
    )
    p.add_argument(
        "--skip-tables", action="store_true",
        help="Skip Stage 5 (table generation)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve scenario list
    if "all" in args.scenarios:
        active_scenarios = SCENARIOS
    else:
        active_scenarios = [s for s in SCENARIOS if s["label"] in args.scenarios]

    results_root = Path(args.output)
    results_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Full Experiment Pipeline")
    print("=" * 70)
    print(f"  Networks   : {', '.join(args.networks)}")
    print(f"  Scenarios  : {', '.join(s['label'] for s in active_scenarios)}")
    print(f"  Models     : {', '.join(args.models)}")
    print(f"  MC runs    : {MC_RUNS}   N runs: {N_RUNS}")
    print(f"  Output     : {results_root}/")
    if args.dry_run:
        print("  [DRY RUN — commands printed, not executed]")
    print()

    # ── Global network stats table (offline, run once up-front) ────────────
    if not args.skip_tables:
        global_network_stats_table(args.networks, results_root, args.dry_run)

    # ── Per-network × per-scenario loop ────────────────────────────────────
    summary: list[dict[str, Any]] = []

    for network in args.networks:
        for scenario in active_scenarios:
            label    = scenario["label"]
            out_dir  = results_root / network / f"scenario_{label}"
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = out_dir / "run.log"

            # Artifact name encodes network + scenario for clean W&B namespacing
            artifact_name = f"{network}_{label}"

            _log(
                f"\n{'#' * 70}\n"
                f"  Network: {network}   Scenario: {label}  "
                f"(β={scenario['beta']}, μ={scenario['mu']})\n"
                f"{'#' * 70}",
                log_path,
            )

            # Stage 1 — SIR simulation
            if not args.skip_tsir:
                rc = stage_tsir(network, scenario, artifact_name, out_dir, args.dry_run)
                if rc != 0 and not args.dry_run:
                    _log(f"  FATAL: tsir failed for {network}/{label} — skipping", log_path)
                    continue

            # Stage 2 — GNN training (all models in sequence)
            model_run_ids: dict[str, str] = {}
            if not args.skip_train:
                for model in args.models:
                    run_id = stage_train(
                        network, model, scenario, artifact_name, out_dir, args.dry_run
                    )
                    if run_id:
                        model_run_ids[model] = run_id

            # Stage 3 — Baseline evaluation
            eval_run_id: str | None = None
            if not args.skip_eval:
                eval_run_id = stage_eval(
                    network, scenario, artifact_name, out_dir, args.dry_run
                )

            # Stage 4 — Visualisation
            if not args.skip_viz and (model_run_ids or eval_run_id):
                stage_viz(
                    network, scenario, model_run_ids, eval_run_id, out_dir, args.dry_run
                )

            # Stage 5 — Per-network/scenario benchmark table
            if not args.skip_tables:
                stage_tables(network, scenario, artifact_name, out_dir, args.dry_run)

            summary.append({
                "network":       network,
                "scenario":      label,
                "model_run_ids": model_run_ids,
                "eval_run_id":   eval_run_id,
            })

    # ── Final summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DONE — Summary")
    print("=" * 70)
    for entry in summary:
        models_done = ", ".join(entry["model_run_ids"].keys()) or "—"
        eval_done   = entry["eval_run_id"] or "—"
        print(
            f"  {entry['network']:<20} {entry['scenario']:<10}"
            f"  models=[{models_done}]  eval={eval_done}"
        )
    print(f"\nAll outputs saved under: {results_root}/")
    print()

    # Print directory tree hint
    print("Directory structure:")
    print(f"  {results_root}/")
    print(f"  ├── tables/                    ← global network stats table")
    for network in args.networks[:2]:
        print(f"  ├── {network}/")
        for scenario in active_scenarios[:1]:
            print(f"  │   └── scenario_{scenario['label']}/")
            print(f"  │       ├── run.log")
            print(f"  │       ├── figures/           ← PDF plots")
            print(f"  │       └── tables/            ← LaTeX + CSV")
    if len(args.networks) > 2:
        print(f"  └── ...  ({len(args.networks) - 2} more networks)")


if __name__ == "__main__":
    main()
