"""
Combined comparison figures for one dataset: all models + all baselines on the
same axes.

Reads a manifest file written by run_all_experiments.sh (one entry per line):

    model=<key>   run_id=<id>   pipeline=<static|temporal>
    eval_run      run_id=<id>   pipeline=<static|temporal>

For ``model`` entries the rep arrays ``eval_arrays_rep*.npz`` are loaded.
For ``eval_run`` entries every ``eval_arrays_<baseline>.npz`` found in the
run directory is loaded automatically.

Outputs (all in ``--output-dir``):
    rank_vs_outbreak_all.pdf/.png        — rank vs. outbreak size, all methods
    top5_vs_outbreak_relative.pdf/.png   — Top-5 accuracy (fraction) vs. size
    top5_vs_outbreak_absolute.pdf/.png   — Top-5 hits (count)  vs. size

Usage
-----
::

    # After running the pipeline, generate all figures for one dataset:
    python viz/compare_all.py \\
        --dataset france_office \\
        --manifest logs/france_office_manifest.txt \\
        --output-dir figures/france_office/comparison

    # Skip individual per-model figures, just the comparison:
    python viz/compare_all.py \\
        --manifest logs/malawi_manifest.txt \\
        --output-dir figures/malawi/comparison
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import NamedTuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz.style import (apply_style, finish_fig,
                       MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, model_style)
from viz.rank_vs_outbreak import load_eval_arrays


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

class Entry(NamedTuple):
    kind:     str   # "model" or "eval_run"
    model:    str   # model key or baseline name
    run_id:   str
    pipeline: str   # "static" or "temporal"


def parse_manifest(path: str) -> list[Entry]:
    """Parse the manifest file and return a flat list of (kind, model, run_id, pipeline)."""
    entries: list[Entry] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
            if line.startswith("eval_run"):
                entries.append(Entry(
                    kind="eval_run",
                    model="",
                    run_id=kv.get("run_id", ""),
                    pipeline=kv.get("pipeline", "static"),
                ))
            elif "run_id" in kv:
                entries.append(Entry(
                    kind="model",
                    model=kv.get("model", ""),
                    run_id=kv.get("run_id", ""),
                    pipeline=kv.get("pipeline", "static"),
                ))
    return entries


def discover_baselines(run_dir: str) -> list[str]:
    """Return all baseline names found as eval_arrays_<name>.npz in run_dir."""
    if not os.path.isdir(run_dir):
        return []
    found = []
    for fname in sorted(os.listdir(run_dir)):
        m = re.match(r"eval_arrays_(.+)\.npz$", fname)
        if m and not m.group(1).startswith("rep"):
            found.append(m.group(1))
    return found


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

class MethodData(NamedTuple):
    label:    str
    key:      str
    arrays:   dict[str, np.ndarray]
    pipeline: str


def collect_methods(
    entries: list[Entry],
    data_dir: str,
    pipeline_filter: str | None = None,
) -> list[MethodData]:
    """Load eval arrays for every entry in the manifest."""
    methods: list[MethodData] = []
    seen_keys: set[str] = set()

    for e in entries:
        if not e.run_id:
            continue
        if pipeline_filter and e.pipeline != pipeline_filter:
            continue

        if e.kind == "model":
            key   = e.model
            label = MODEL_LABELS.get(key, key)
            try:
                arrays = load_eval_arrays(e.run_id, data_dir, baseline=None)
            except FileNotFoundError as exc:
                print(f"  WARNING: {exc}")
                continue
            if key not in seen_keys:
                methods.append(MethodData(label, key, arrays, e.pipeline))
                seen_keys.add(key)

        elif e.kind == "eval_run":
            run_dir    = os.path.join(data_dir, e.run_id)
            baselines  = discover_baselines(run_dir)
            for bl in baselines:
                key = bl
                if key in seen_keys:
                    continue
                label = MODEL_LABELS.get(key, key)
                try:
                    arrays = load_eval_arrays(e.run_id, data_dir, baseline=bl)
                except FileNotFoundError:
                    continue
                methods.append(MethodData(label, key, arrays, e.pipeline))
                seen_keys.add(key)

    # Sort by canonical model order
    order_map = {k: i for i, k in enumerate(MODEL_ORDER)}
    methods.sort(key=lambda m: order_map.get(m.key, len(MODEL_ORDER)))
    return methods


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

N_BINS = 15
BINS   = np.linspace(0, 1, N_BINS + 1)
CENTS  = 0.5 * (BINS[:-1] + BINS[1:])


def _binned_rank_stats(
    sizes: np.ndarray,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers, mean_rank, p25, p75) per outbreak-size bin."""
    means, p25, p75 = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        vals = ranks[mask].astype(float) if mask.any() else np.array([np.nan])
        means.append(float(np.nanmean(vals)))
        p25.append(float(np.nanpercentile(vals, 25)) if mask.any() else float("nan"))
        p75.append(float(np.nanpercentile(vals, 75)) if mask.any() else float("nan"))
    return CENTS, np.array(means), np.array(p25), np.array(p75)


def _binned_topk(
    sizes: np.ndarray,
    ranks: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (centers, fraction_topk, stderr, count_topk) per bin."""
    fracs, ses, counts = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        if not mask.any():
            fracs.append(float("nan"))
            ses.append(float("nan"))
            counts.append(0)
        else:
            r_bin = ranks[mask]
            p     = float(np.mean(r_bin <= k))
            n     = int(mask.sum())
            fracs.append(p)
            ses.append(float(np.sqrt(p * (1 - p) / max(n, 1))))
            counts.append(int(np.sum(r_bin <= k)))
    return CENTS, np.array(fracs), np.array(ses), np.array(counts)


# ---------------------------------------------------------------------------
# Plot 1: Rank vs. Outbreak Size (all methods)
# ---------------------------------------------------------------------------

def plot_rank_vs_outbreak(
    methods: list[MethodData],
    output_path: str,
    dataset_name: str = "",
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    n_nodes_max = 1
    for m in methods:
        sel  = m.arrays["sel"]
        r    = m.arrays["ranks"][sel].astype(float)
        s    = m.arrays["outbreak_sizes"][sel]
        n_nodes_max = max(n_nodes_max, int(r.max()))

        style = model_style(m.key)
        cents, means, p25, p75 = _binned_rank_stats(s, r)
        valid = ~np.isnan(means)

        ax.fill_between(cents[valid], p25[valid], p75[valid],
                        alpha=0.12, color=style["color"])
        ls = "-" if m.key in {"backtracking", "temporal_gnn", "static_gnn",
                               "dbgnn", "dag_gnn"} else "--"
        ax.plot(cents[valid], means[valid],
                color=style["color"], lw=2.2, ls=ls,
                marker=style["marker"], ms=4, zorder=5,
                label=style["label"])

    ax.set_xlabel("Outbreak size  (fraction of nodes infected)")
    ax.set_ylabel("Rank of true source")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, n_nodes_max + 0.5)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    title = "Rank vs. Outbreak Size — All Methods"
    if dataset_name:
        title += f"  ({dataset_name})"
    ax.set_title(title)
    finish_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 2: Top-k Accuracy vs. Outbreak Size — relative (fraction)
# ---------------------------------------------------------------------------

def plot_topk_relative(
    methods: list[MethodData],
    output_path: str,
    k: int = 5,
    dataset_name: str = "",
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        sel  = m.arrays["sel"]
        r    = m.arrays["ranks"][sel]
        s    = m.arrays["outbreak_sizes"][sel]

        style = model_style(m.key)
        cents, fracs, ses, _ = _binned_topk(s, r, k)
        valid = ~np.isnan(fracs)

        ax.fill_between(cents[valid],
                        (fracs - ses)[valid], (fracs + ses)[valid],
                        alpha=0.12, color=style["color"])
        ls = "-" if m.key in {"backtracking", "temporal_gnn", "static_gnn",
                               "dbgnn", "dag_gnn"} else "--"
        ax.plot(cents[valid], fracs[valid],
                color=style["color"], lw=2.2, ls=ls,
                marker=style["marker"], ms=4, zorder=5,
                label=style["label"])

    ax.set_xlabel("Outbreak size  (fraction of nodes infected)")
    ax.set_ylabel(f"Top-{k} accuracy  (fraction of scenarios)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    title = f"Top-{k} Accuracy vs. Outbreak Size — All Methods (Relative)"
    if dataset_name:
        title += f"  ({dataset_name})"
    ax.set_title(title)
    finish_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 3: Top-k hits vs. Outbreak Size — absolute count
# ---------------------------------------------------------------------------

def plot_topk_absolute(
    methods: list[MethodData],
    output_path: str,
    k: int = 5,
    dataset_name: str = "",
) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for m in methods:
        sel  = m.arrays["sel"]
        r    = m.arrays["ranks"][sel]
        s    = m.arrays["outbreak_sizes"][sel]

        style = model_style(m.key)
        cents, _, _, counts = _binned_topk(s, r, k)
        valid = counts > 0

        ls = "-" if m.key in {"backtracking", "temporal_gnn", "static_gnn",
                               "dbgnn", "dag_gnn"} else "--"
        ax.plot(cents[valid], np.array(counts)[valid],
                color=style["color"], lw=2.2, ls=ls,
                marker=style["marker"], ms=4, zorder=5,
                label=style["label"])

    ax.set_xlabel("Outbreak size  (fraction of nodes infected)")
    ax.set_ylabel(f"Number of scenarios with rank ≤ {k}")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    title = f"Top-{k} Hits vs. Outbreak Size — All Methods (Absolute)"
    if dataset_name:
        title += f"  ({dataset_name})"
    ax.set_title(title)
    finish_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Per-method individual plots
# ---------------------------------------------------------------------------

def plot_per_method(
    methods: list[MethodData],
    output_dir: str,
    k: int = 5,
    dataset_name: str = "",
) -> None:
    """Generate individual rank and top-k plots for each method."""
    for m in methods:
        method_dir = os.path.join(output_dir, m.key)
        os.makedirs(method_dir, exist_ok=True)

        sel  = m.arrays["sel"]
        r    = m.arrays["ranks"][sel].astype(float)
        s    = m.arrays["outbreak_sizes"][sel]
        n_nodes = max(1, int(r.max()))

        # Rank vs. outbreak
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        style = model_style(m.key)
        correct = r == 1
        ax.scatter(s[~correct], r[~correct], s=4, alpha=0.08,
                   color=style["color"], linewidths=0, rasterized=True)
        ax.scatter(s[correct], r[correct], s=4, alpha=0.20,
                   color="#e6554a", linewidths=0, rasterized=True, label="Rank 1")
        cents, means, p25, p75 = _binned_rank_stats(s, r)
        valid = ~np.isnan(means)
        ax.fill_between(cents[valid], p25[valid], p75[valid],
                        alpha=0.25, color=style["color"])
        ax.plot(cents[valid], means[valid], color=style["color"], lw=2.2,
                zorder=6, label="Binned mean")
        ax.set_xlabel("Outbreak size")
        ax.set_ylabel("Rank of true source")
        ax.set_xlim(0, 1)
        ax.set_ylim(0.5, n_nodes + 0.5)
        ax.set_title(f"Rank vs. Outbreak Size — {m.label}" +
                     (f"  ({dataset_name})" if dataset_name else ""))
        ax.legend(fontsize=8)
        finish_fig(fig, os.path.join(method_dir, "rank_vs_outbreak.pdf"))

        # Top-k vs. outbreak
        apply_style()
        fig, ax = plt.subplots(figsize=(7, 4))
        cents, fracs, ses, counts = _binned_topk(s, r.astype(int), k)
        valid = ~np.isnan(fracs)
        ax.fill_between(cents[valid], (fracs - ses)[valid], (fracs + ses)[valid],
                        alpha=0.20, color=style["color"])
        ax.plot(cents[valid], fracs[valid], color=style["color"], lw=2.2,
                marker="o", ms=4, zorder=5, label=f"Top-{k}")
        ax.set_xlabel("Outbreak size")
        ax.set_ylabel(f"Top-{k} accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(
            plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_title(f"Top-{k} Accuracy — {m.label}" +
                     (f"  ({dataset_name})" if dataset_name else ""))
        ax.legend(fontsize=8)
        finish_fig(fig, os.path.join(method_dir, f"top{k}_vs_outbreak.pdf"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest",    required=True,
                   help="Path to manifest file written by run_all_experiments.sh")
    p.add_argument("--output-dir",  default="figures/comparison",
                   help="Directory for comparison figures (default: figures/comparison)")
    p.add_argument("--dataset",     default="",
                   help="Dataset name for figure titles")
    p.add_argument("--pipeline",    default=None,
                   choices=["static", "temporal", None],
                   help="Only plot methods from this pipeline (default: all)")
    p.add_argument("--k",           type=int, default=5,
                   help="Top-k threshold (default: 5)")
    p.add_argument("--data-dir",    default="data",
                   help="Root data directory (default: data/)")
    p.add_argument("--per-method",  action="store_true",
                   help="Also generate individual per-method plots under output-dir/<key>/")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading manifest: {args.manifest}")
    entries = parse_manifest(args.manifest)
    print(f"  {len(entries)} entries found")

    methods = collect_methods(entries, args.data_dir, args.pipeline)
    if not methods:
        print("ERROR: No eval arrays found. "
              "Check that the pipeline has run and data/ is accessible.")
        sys.exit(1)

    print(f"  Methods loaded: {[m.label for m in methods]}")
    os.makedirs(args.output_dir, exist_ok=True)

    dn = args.dataset

    print("\nGenerating comparison figures…")
    plot_rank_vs_outbreak(
        methods,
        os.path.join(args.output_dir, "rank_vs_outbreak_all.pdf"),
        dataset_name=dn,
    )
    plot_topk_relative(
        methods,
        os.path.join(args.output_dir, f"top{args.k}_vs_outbreak_relative.pdf"),
        k=args.k,
        dataset_name=dn,
    )
    plot_topk_absolute(
        methods,
        os.path.join(args.output_dir, f"top{args.k}_vs_outbreak_absolute.pdf"),
        k=args.k,
        dataset_name=dn,
    )

    if args.per_method:
        print("\nGenerating per-method figures…")
        plot_per_method(methods, args.output_dir, k=args.k, dataset_name=dn)

    print(f"\nAll figures written to: {args.output_dir}")
    _print_index(args.output_dir)


def _print_index(output_dir: str) -> None:
    print("\nFigure index:")
    for root, dirs, files in os.walk(output_dir):
        dirs.sort()
        rel = os.path.relpath(root, output_dir)
        prefix = "" if rel == "." else f"  [{rel}] "
        for fname in sorted(f for f in files if f.endswith(".pdf")):
            print(f"  {prefix}{fname}")


if __name__ == "__main__":
    main()
