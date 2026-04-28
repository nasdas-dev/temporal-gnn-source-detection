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
# Terminal table printer (paper Table 5 style)
# ---------------------------------------------------------------------------

_PAPER_TARGET = (
    "  Paper target (GNN, no DA, Table 5): "
    "Top-5=73.31%, Error=0.95, Rec.Rank=0.55, |90%CSS|=9.42, Res=0.2155"
)

# Baselines: (wandb_key, display_label, has_probability_distribution)
_BASELINE_ROWS = [
    ("random",         "Random",      False),
    ("jordan_center",  "Jordan",      False),
    ("betweenness",    "Betweenness", False),
    ("soft_margin",    "SME",         True),
    ("mcs_mean_field", "MCMF",        True),
]


def _print_paper_table(
    gnn_id: Optional[str],
    eval_id: Optional[str],
    project: str,
    entity: Optional[str],
) -> None:
    """Fetch W&B run summaries and print a paper-style Table 5 to stdout."""
    if not gnn_id and not eval_id:
        print("\n[table] No run IDs — skipping terminal table.")
        return

    try:
        import wandb
        api = wandb.Api()
        base = f"{entity}/{project}" if entity else project

        COL_W = 14   # column width

        header_row = (
            f"{'Method':<16} | "
            f"{'Top-5 Acc':>{COL_W}} | "
            f"{'Error Dist':>{COL_W}} | "
            f"{'Rec. Rank':>{COL_W}} | "
            f"{'|90% CSS|':>{COL_W}} | "
            f"{'Resistance':>{COL_W}}"
        )
        sep = "-" * len(header_row)

        print("\n" + "=" * len(header_row))
        print(f" Karate – Benchmark Results  (cf. Table 5, Sterchi et al. 2025)")
        print("=" * len(header_row))
        print(header_row)
        print(sep)

        def _row(label, top5, err, mrr, css, res, top5_std=None, mrr_std=None, res_std=None):
            def _f(v, decimals=4):
                return f"{v:.{decimals}f}" if v is not None else "-"
            def _t(v):
                return f"{100*v:.2f}%" if v is not None else "-"
            def _pm(v, s, decimals=4):
                if v is None:
                    return f"{'- ':>{COL_W}}"
                if s is not None:
                    return f"{v:.{decimals}f}(±{s:.{decimals}f})".rjust(COL_W)
                return f"{v:.{decimals}f}".rjust(COL_W)
            t5_cell = (
                f"{_t(top5)}(±{100*top5_std:.2f}%)".rjust(COL_W)
                if top5 is not None and top5_std is not None
                else f"{_t(top5)}".rjust(COL_W) if top5 is not None
                else f"{'-':>{COL_W}}"
            )
            return (
                f"{label:<16} | "
                f"{t5_cell} | "
                f"{_f(err, 4):>{COL_W}} | "
                f"{_pm(mrr, mrr_std, 4)} | "
                f"{_f(css, 2):>{COL_W}} | "
                f"{_pm(res, res_std, 4)}"
            )

        # --- Baselines ---
        if eval_id:
            eval_run = api.run(f"{base}/{eval_id}")
            es = dict(eval_run.summary)
            for key, label, has_probs in _BASELINE_ROWS:
                top5 = es.get(f"{key}/eval/top_5")
                err  = es.get(f"{key}/eval/error_dist")
                mrr  = es.get(f"{key}/eval/mrr")
                css  = es.get(f"{key}/eval/cred_set_size_90") if has_probs else None
                res  = es.get(f"{key}/eval/resistance")        if has_probs else None
                print(_row(label, top5, err, mrr, css, res))
        else:
            print("  (eval run not available — baseline rows skipped)")

        print(sep)

        # --- GNN ---
        if gnn_id:
            gr = api.run(f"{base}/{gnn_id}")
            gs = dict(gr.summary)
            print(_row(
                "GNN",
                gs.get("eval/top_5_mean"),    gs.get("eval/error_dist_mean"),
                gs.get("eval/mrr_mean"),
                gs.get("eval/cred_set_size_90_mean"),
                gs.get("eval/resistance_mean"),
                top5_std = gs.get("eval/top_5_std"),
                mrr_std  = gs.get("eval/mrr_std"),
                res_std  = gs.get("eval/resistance_std"),
            ))
        else:
            print("  (GNN run not available — GNN row skipped)")

        print("=" * len(header_row))
        print(_PAPER_TARGET)
        print("=" * len(header_row))

    except Exception as exc:
        print(f"\n[table] Could not generate terminal table: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str) -> None:
    print(f"\n[viz] {label}")
    print("    " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"    WARNING: '{label}' exited with code {result.returncode}")


def _is_run_name(s: str) -> bool:
    """Return True if s looks like a W&B display name (word-word-number) rather than a run ID."""
    import re
    return bool(re.fullmatch(r"[a-z]+-[a-z]+-\d+", s))


def _resolve_run_id(name_or_id: str, project: str, entity: Optional[str]) -> str:
    """If name_or_id is a W&B run name, query the API and return the real run ID.

    Run names look like 'daily-darkness-63'; run IDs look like 'abc1def2'.
    Returns the input unchanged if it already looks like an ID.
    """
    if not _is_run_name(name_or_id):
        return name_or_id
    try:
        import wandb
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"display_name": name_or_id})
        for r in runs:
            if r.name == name_or_id:
                print(f"[viz] Resolved '{name_or_id}' → run ID '{r.id}'")
                return r.id
        print(f"[viz] WARNING: could not resolve run name '{name_or_id}' to an ID — using as-is")
    except Exception as exc:
        print(f"[viz] WARNING: name resolution failed ({exc}) — using '{name_or_id}' as-is")
    return name_or_id


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

        # Fetch all finished runs and filter in Python — avoids W&B config
        # filter syntax quirks (config.model vs config.model.value etc.)
        all_runs = api.runs(path, filters={"state": "finished"}, order="-created_at")

        gnn_id = None
        eval_id = None
        gnn_entity = None

        for r in all_runs:
            cfg = dict(r.config)
            data_name = str(cfg.get("data_name", ""))
            if not data_name.startswith(artifact_key):
                continue
            tags = list(r.tags) if hasattr(r, "tags") else []
            model = cfg.get("model")

            if gnn_id is None and model == "static_gnn":
                gnn_id = r.id
                gnn_entity = r.entity

            if eval_id is None and "baselines" in tags:
                eval_id = r.id

            if gnn_id and eval_id:
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

    # Resolve run names → run IDs before any lookup
    if gnn_id is not None:
        gnn_id = _resolve_run_id(gnn_id, args.project, args.entity)
    if eval_id is not None:
        eval_id = _resolve_run_id(eval_id, args.project, args.entity)

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
    # Requires eval_arrays_rep*.npz in data/<gnn_id>/ — only produced by the
    # updated main_train.py.  Skip gracefully if the files are absent.
    if gnn_id and eval_id:
        import glob as _glob
        npz_present = bool(_glob.glob(
            os.path.join(args.data_dir, gnn_id, "eval_arrays_rep*.npz")
        ))
        if not npz_present:
            print(
                f"\n[viz] Skipping rank_vs_outbreak: no eval_arrays_rep*.npz in "
                f"{args.data_dir}/{gnn_id}/\n"
                "      Re-run the training pipeline to generate them."
            )
        else:
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

    _print_paper_table(gnn_id, eval_id, args.project, args.entity)


if __name__ == "__main__":
    main()
