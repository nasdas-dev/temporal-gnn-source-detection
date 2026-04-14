"""
Top-k Accuracy vs. Outbreak Size plot.

Shows the fraction of runs where the true source is in the model's top-k
predictions, binned by outbreak size.  Multiple methods are overlaid as
lines for direct comparison.

Loads per-sample arrays from ``data/{run_id}/eval_arrays_rep*.npz`` (or
``eval_arrays_{baseline}.npz`` for baseline eval runs).

Usage
-----
::

    # Single model
    python viz/topk_vs_outbreak.py --run-id e6tw7k64 --k 5 \\
        --output figures/top5_vs_outbreak.pdf

    # Multi-model comparison
    python viz/topk_vs_outbreak.py \\
        --run-id e6tw7k64 abc123 <eval_run_id> <eval_run_id> \\
        --label  "BacktrackingNet" "StaticGNN" "Uniform" "Jordan Center" \\
        --baseline None None uniform jordan_center \\
        --k 5 --output figures/top5_vs_outbreak_compare.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz.style import apply_style, finish_fig, MODEL_COLORS, MODEL_LABELS
from viz.rank_vs_outbreak import load_eval_arrays


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id",   nargs="+", required=True,
                   help="One or more local data/ run IDs")
    p.add_argument("--label",    nargs="+", default=None,
                   help="Display labels for each run-id")
    p.add_argument("--baseline", nargs="+", default=None,
                   help="For eval runs: baseline name per run-id (use 'None' for GNN model runs)")
    p.add_argument("--k",        type=int,  default=5,
                   help="Top-k threshold (default: 5)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output",   default="figures/topk_vs_outbreak.pdf")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_style()

    k        = args.k
    run_ids  = args.run_id
    labels   = args.label or run_ids
    raw_bls  = args.baseline or ["None"] * len(run_ids)
    baselines = [None if b.lower() == "none" else b for b in raw_bls]

    if len(labels) < len(run_ids):
        labels = labels + run_ids[len(labels):]

    _cycle = list(MODEL_COLORS.values())
    colors = []
    for lbl in labels:
        key = next((k2 for k2, v in MODEL_LABELS.items() if v == lbl), lbl.lower().replace(" ", "_"))
        colors.append(MODEL_COLORS.get(key, _cycle[len(colors) % len(_cycle)]))

    bins  = np.linspace(0, 1, 16)
    cents = 0.5 * (bins[:-1] + bins[1:])

    fig, ax = plt.subplots(figsize=(7, 5))

    for run_id, lbl, col, bl in zip(run_ids, labels, colors, baselines):
        d    = load_eval_arrays(run_id, args.data_dir, bl)
        sel  = d["sel"]
        r    = d["ranks"][sel]
        s    = d["outbreak_sizes"][sel]

        top_k_per_bin, counts = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (s >= lo) & (s < hi)
            if mask.sum() == 0:
                top_k_per_bin.append(float("nan"))
                counts.append(0)
            else:
                top_k_per_bin.append(float(np.mean(r[mask] <= k)))
                counts.append(int(mask.sum()))

        vals = np.array(top_k_per_bin)
        # Standard error of proportion per bin: se = sqrt(p*(1-p)/n)
        se = np.array([
            np.sqrt(v * (1 - v) / max(c, 1)) if not np.isnan(v) else float("nan")
            for v, c in zip(top_k_per_bin, counts)
        ])

        valid = ~np.isnan(vals)
        ax.fill_between(cents[valid],
                        (vals - se)[valid], (vals + se)[valid],
                        alpha=0.15, color=col)
        ax.plot(cents[valid], vals[valid], color=col, lw=2.0,
                marker="o", ms=4, zorder=5, label=lbl)

    ax.axhline(k / (ax.get_ylim()[1] * 1e3 + 1), color="gray",
               lw=0.8, ls=":", alpha=0.0)   # invisible — set dynamically below
    ax.set_xlabel("Outbreak size  (fraction of N infected)")
    ax.set_ylabel(f"Top-{k} accuracy  (fraction of valid runs)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title(f"Top-{k} Accuracy vs. Outbreak Size")
    ax.legend(loc="upper right", fontsize=9)

    finish_fig(fig, args.output)


if __name__ == "__main__":
    main()
