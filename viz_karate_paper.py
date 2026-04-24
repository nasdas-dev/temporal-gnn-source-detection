#!/usr/bin/env python3
"""
Generate all paper-replication figures and tables for karate_static / static GNN.

Queries W&B for the most recent karate_static runs automatically, or accepts
explicit run IDs.  Calls the existing viz/ scripts and eval/tables module via
subprocess, collecting output in a single directory.

Figures produced
----------------
top5_vs_outbreak_gnn.pdf       GNN-only top-5 accuracy vs outbreak size (Fig 3a style)
top5_vs_outbreak_compare.pdf   GNN + baselines overlay (Fig 4 style)
rank_vs_outbreak_compare.pdf   Reciprocal rank vs outbreak size (GNN + baselines)
training_curves.pdf            Training / validation NLL loss over epochs

Tables produced
---------------
tables/benchmark_karate.tex    Per-metric comparison table (mirrors Table 5)
tables/benchmark_karate.csv    Same data as CSV
tables/network_stats.tex       Karate network properties

Usage
-----
    # Auto-discover latest runs from W&B:
    python viz_karate_paper.py

    # Specify runs explicitly (useful when W&B is offline):
    python viz_karate_paper.py --gnn-run-id abc12345 --eval-run-id def67890

    # Change output directory or W&B artifact name:
    python viz_karate_paper.py --artifact karate_static --output figures/karate/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    print(f"\n[viz] {label}")
    print("    " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"    WARNING: '{label}' exited with code {result.returncode}")


def _find_runs(artifact: str, project: str, entity: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (gnn_run_id, eval_run_id, run_path_prefix) by querying W&B.

    Returns (None, None, None) if wandb is unavailable or no runs found.
    run_path_prefix is '<entity>/<project>' for use with training_curves.py.
    """
    try:
        import wandb
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        artifact_key = artifact.split(":")[0]

        # Most recent static_gnn training run for this artifact
        gnn_runs = api.runs(
            path,
            filters={
                "state": "finished",
                "config.model": "static_gnn",
            },
            order="-created_at",
        )
        gnn_id = None
        gnn_entity = None
        for r in gnn_runs:
            if str(r.config.get("data_name", "")).startswith(artifact_key):
                gnn_id = r.id
                gnn_entity = r.entity
                break

        # Most recent baseline eval run for this artifact (tagged "baselines")
        eval_runs = api.runs(
            path,
            filters={
                "state": "finished",
                "tags": "baselines",
            },
            order="-created_at",
        )
        eval_id = None
        for r in eval_runs:
            if str(r.config.get("data_name", "")).startswith(artifact_key):
                eval_id = r.id
                break

        resolved_entity = gnn_entity or entity
        prefix = f"{resolved_entity}/{project}" if resolved_entity else project
        return gnn_id, eval_id, prefix

    except Exception as exc:
        print(f"[viz] W&B auto-discovery failed ({exc}). Pass --gnn-run-id / --eval-run-id manually.")
        return None, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact",     default="karate_static",
                   help="W&B artifact / data name prefix (default: karate_static)")
    p.add_argument("--gnn-run-id",   default=None,
                   help="W&B run ID for the static_gnn training run")
    p.add_argument("--eval-run-id",  default=None,
                   help="W&B run ID for the baseline evaluation run")
    p.add_argument("--project",      default="source-detection")
    p.add_argument("--entity",       default=None,
                   help="W&B entity (username / team). Uses API default if omitted.")
    p.add_argument("--data-dir",     default="data",
                   help="Directory holding eval_arrays*.npz subdirs (default: data)")
    p.add_argument("--output",       default="figures/karate_replication",
                   help="Output directory for all figures and tables")
    args = p.parse_args()

    out = args.output
    tables_out = os.path.join(out, "tables")
    os.makedirs(out, exist_ok=True)
    os.makedirs(tables_out, exist_ok=True)

    # ── Discover run IDs ──────────────────────────────────────────────────────
    gnn_id   = args.gnn_run_id
    eval_id  = args.eval_run_id
    run_prefix = f"{args.entity}/{args.project}" if args.entity else args.project

    if gnn_id is None or eval_id is None:
        print("[viz] Querying W&B for latest karate_static runs …")
        disc_gnn, disc_eval, disc_prefix = _find_runs(
            args.artifact, args.project, args.entity
        )
        if gnn_id is None:
            gnn_id = disc_gnn
        if eval_id is None:
            eval_id = disc_eval
        if disc_prefix:
            run_prefix = disc_prefix

    if gnn_id is None and eval_id is None:
        print("[viz] No run IDs available — cannot generate figures. Exiting.")
        sys.exit(1)

    print(f"\n[viz] GNN run ID  : {gnn_id  or '(not found)'}")
    print(f"[viz] Eval run ID : {eval_id or '(not found)'}")
    print(f"[viz] Output dir  : {out}\n")

    # Baselines available in eval.yml (in display order for legend)
    BASELINES = ["random", "jordan_center", "betweenness", "degree", "closeness", "uniform"]
    BASELINE_LABELS = ["Random", "Jordan center", "Betweenness", "Degree",
                       "Closeness", "Uniform"]

    # ── Figure 1: GNN-only top-5 vs outbreak size ─────────────────────────────
    if gnn_id:
        _run(
            [
                sys.executable, "viz/topk_vs_outbreak.py",
                "--run-id",  gnn_id,
                "--label",   "GNN",
                "--baseline", "None",
                "--k",       "5",
                "--data-dir", args.data_dir,
                "--output",  os.path.join(out, "top5_vs_outbreak_gnn.pdf"),
            ],
            "Top-5 accuracy vs outbreak size (GNN only)"
        )

    # ── Figure 2: GNN + baselines top-5 comparison (Fig 4 style) ─────────────
    if gnn_id and eval_id:
        run_ids  = [gnn_id]  + [eval_id]  * len(BASELINES)
        labels   = ["GNN"]   + BASELINE_LABELS
        baselines = ["None"] + BASELINES

        _run(
            [
                sys.executable, "viz/topk_vs_outbreak.py",
                "--run-id",   *run_ids,
                "--label",    *labels,
                "--baseline", *baselines,
                "--k",        "5",
                "--data-dir", args.data_dir,
                "--output",   os.path.join(out, "top5_vs_outbreak_compare.pdf"),
            ],
            "Top-5 accuracy vs outbreak size (GNN + baselines)"
        )

    # ── Figure 3: Reciprocal rank vs outbreak size ────────────────────────────
    if gnn_id and eval_id:
        run_ids   = [gnn_id]  + [eval_id]  * len(BASELINES)
        labels    = ["GNN"]   + BASELINE_LABELS
        baselines = ["None"]  + BASELINES

        _run(
            [
                sys.executable, "viz/rank_vs_outbreak.py",
                "--run-id",   *run_ids,
                "--label",    *labels,
                "--baseline", *baselines,
                "--data-dir", args.data_dir,
                "--output",   os.path.join(out, "rank_vs_outbreak_compare.pdf"),
            ],
            "Reciprocal rank vs outbreak size (GNN + baselines)"
        )

    # ── Figure 4: Training curves ─────────────────────────────────────────────
    if gnn_id:
        run_path = f"{run_prefix}/{gnn_id}"
        _run(
            [
                sys.executable, "viz/training_curves.py",
                "--run-path", run_path,
                "--label",    "StaticGNN",
                "--output",   os.path.join(out, "training_curves.pdf"),
            ],
            "Training / validation loss curves"
        )

    # ── Table 1: Benchmark metrics (Table 5 equivalent) ──────────────────────
    _run(
        [
            sys.executable, "-m", "eval.tables", "benchmark",
            "--data",    args.artifact,
            "--project", args.project,
            *(["--entity", args.entity] if args.entity else []),
            "--output",  tables_out,
        ],
        "Benchmark metrics table (LaTeX + CSV)"
    )

    # ── Table 2: Network statistics ───────────────────────────────────────────
    _run(
        [
            sys.executable, "-m", "eval.tables", "network_stats",
            "--networks", "karate_static",
            "--output",   tables_out,
        ],
        "Network statistics table"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" Output summary")
    print("=" * 60)
    for root, _, files in os.walk(out):
        for f in sorted(files):
            rel = os.path.relpath(os.path.join(root, f), out)
            print(f"  {out}/{rel}")
    print()
    print(" Compare eval/top_5 in W&B against Table 5 target: 73.31% (±0.27%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
