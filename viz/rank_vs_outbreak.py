"""
Rank vs. Outbreak Size plot.

For each run, loads per-sample arrays from ``data/{run_id}/eval_arrays_rep*.npz``
(saved by main_train.py / main_eval.py).  Plots a scatter of (outbreak_size, rank)
with a binned-mean smoother and IQR band, one curve per method.

Usage
-----
::

    # Single model
    python viz/rank_vs_outbreak.py --run-id e6tw7k64 --output figures/rank_vs_outbreak.pdf

    # Multi-model comparison (one curve per run)
    python viz/rank_vs_outbreak.py \\
        --run-id e6tw7k64 abc123 \\
        --label "BacktrackingNet" "StaticGNN" \\
        --output figures/rank_vs_outbreak_compare.pdf

    # Baseline (arrays saved as eval_arrays_{baseline}.npz)
    python viz/rank_vs_outbreak.py \\
        --run-id <eval_run_id> --baseline uniform \\
        --output figures/rank_vs_outbreak_uniform.pdf
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
from viz.style import apply_style, finish_fig, MODEL_COLORS, MODEL_LABELS, REP_COLORS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id",   nargs="+", required=True,
                   help="One or more local data/ run IDs, e.g. e6tw7k64 abc123")
    p.add_argument("--label",    nargs="+", default=None,
                   help="Display labels for each run-id (default: run-id or model key)")
    p.add_argument("--baseline", nargs="+", default=None,
                   help="If given, load eval_arrays_{baseline}.npz instead of rep arrays")
    p.add_argument("--data-dir", default="data",
                   help="Root data directory (default: data/)")
    p.add_argument("--n-nodes",  type=int, default=None,
                   help="Number of nodes (for y-axis limit).  Inferred from arrays if omitted.")
    p.add_argument("--output",   default="figures/rank_vs_outbreak.pdf",
                   help="Output PDF path")
    p.add_argument("--no-scatter", action="store_true",
                   help="Skip scatter plot (only smoother lines) for cleaner comparison figures")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_arrays_from_npz(run_dir: str, baseline: str | None = None) -> dict[str, np.ndarray] | None:
    """Load eval arrays from .npz files saved by main_train.py / main_eval.py.

    Returns None if no array files exist for this run.
    """
    if baseline is not None and baseline != "None":
        path = os.path.join(run_dir, f"eval_arrays_{baseline}.npz")
        if os.path.exists(path):
            d = dict(np.load(path))
            return {k: [v] for k, v in d.items()}  # wrap in list (1 rep)
        return None

    import re as _re
    # Use os.listdir + filter instead of glob — glob can silently return []
    # on some systems even when matching files exist.
    try:
        all_files = os.listdir(run_dir)
    except OSError:
        return None
    rep_files = sorted(
        (f for f in all_files if _re.match(r"eval_arrays_rep\d+\.npz$", f)),
        key=lambda f: int(_re.search(r"rep(\d+)\.npz$", f).group(1)),
    )
    all_arrays: dict[str, list] = {}
    for fname in rep_files:
        d = np.load(os.path.join(run_dir, fname))
        for k, v in d.items():
            all_arrays.setdefault(k, []).append(v)

    return all_arrays if all_arrays else None


def load_eval_arrays(run_id: str, data_dir: str, baseline: str | None = None) -> dict[str, np.ndarray]:
    """Load and concatenate per-rep eval arrays for one run.

    Returns a dict with concatenated ``ranks``, ``outbreak_sizes``, ``sel``,
    ``true_sources`` arrays (across all reps).
    """
    run_dir = os.path.join(data_dir, run_id)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            "Make sure the run data is in your local data/ directory."
        )

    arrays = _load_arrays_from_npz(run_dir, baseline)
    if arrays is None:
        contents = sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []
        raise FileNotFoundError(
            f"No eval_arrays*.npz files found in {run_dir}.\n"
            f"Directory contents: {contents}\n"
            "Re-run the training/eval pipeline with the updated main_train.py / main_eval.py\n"
            "to generate lightweight .npz files alongside probs_rep*.pt."
        )

    return {k: np.concatenate(v) for k, v in arrays.items()}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _binned_stats(
    sizes: np.ndarray,
    ranks: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, mean, p25, p75 of ranks per outbreak-size bin."""
    bins  = np.linspace(0, 1, n_bins + 1)
    cents = 0.5 * (bins[:-1] + bins[1:])
    means, p25, p75 = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (sizes >= lo) & (sizes < hi)
        if mask.any():
            vals = ranks[mask].astype(float)
            means.append(float(np.mean(vals)))
            p25.append(float(np.percentile(vals, 25)))
            p75.append(float(np.percentile(vals, 75)))
        else:
            means.append(float("nan"))
            p25.append(float("nan"))
            p75.append(float("nan"))
    return cents, np.array(means), np.array(p25), np.array(p75)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_style()

    run_ids  = args.run_id
    labels   = args.label or run_ids
    baselines = args.baseline or ([None] * len(run_ids))

    if len(labels) < len(run_ids):
        labels = labels + run_ids[len(labels):]
    if len(baselines) < len(run_ids):
        baselines = baselines + [None] * (len(run_ids) - len(baselines))

    # Choose a colour per run from MODEL_COLORS if label matches, else cycle
    _cycle = list(MODEL_COLORS.values())
    colors = []
    for lbl in labels:
        key = next((k for k, v in MODEL_LABELS.items() if v == lbl), lbl.lower().replace(" ", "_"))
        colors.append(MODEL_COLORS.get(key, _cycle[len(colors) % len(_cycle)]))

    # Load all data
    all_data = []
    n_nodes_global = args.n_nodes or 1
    for run_id, bl in zip(run_ids, baselines):
        d = load_eval_arrays(run_id, args.data_dir, bl)
        if args.n_nodes is None:
            # Infer n_nodes from max rank
            n_nodes_global = max(n_nodes_global, int(d["ranks"].max()))
        all_data.append(d)

    single = len(run_ids) == 1

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (d, lbl, col) in enumerate(zip(all_data, labels, colors)):
        sel    = d["sel"]
        r      = d["ranks"][sel].astype(float)
        s_arr  = d["outbreak_sizes"][sel]

        # Scatter (for single-run or if not suppressed)
        if single and not args.no_scatter:
            correct = r == 1
            ax.scatter(s_arr[~correct], r[~correct], s=5, alpha=0.10,
                       color=col, linewidths=0, rasterized=True)
            ax.scatter(s_arr[correct],  r[correct],  s=5, alpha=0.25,
                       color="#e6554a", linewidths=0, rasterized=True,
                       label="Rank 1 (correct)")

        # Binned smoother
        cents, means, p2a, p7a = _binned_stats(s_arr, r, n_bins=15)
        valid = ~np.isnan(means)
        ax.fill_between(cents[valid], p2a[valid], p7a[valid],
                        alpha=0.15, color=col)
        ax.plot(cents[valid], means[valid], color=col, lw=2.2,
                zorder=5, label=lbl if not single else "Binned mean ± IQR")

    ax.set_xlabel(f"Outbreak size  (fraction of N infected)")
    ax.set_ylabel(f"Rank of true source")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, n_nodes_global + 0.5)
    ax.legend(loc="upper left", fontsize=9)

    title = "Rank vs. Outbreak Size"
    if single:
        title += f"  —  {labels[0]}"
    ax.set_title(title)

    # Marginal histogram of outbreak sizes
    if single:
        sel0  = all_data[0]["sel"]
        s_all = all_data[0]["outbreak_sizes"][sel0]
        ax_top = ax.inset_axes([0, 1.04, 1, 0.15])
        ax_top.hist(s_all, bins=np.linspace(0, 1, 16),
                    color=colors[0], alpha=0.7, edgecolor="none")
        ax_top.set_xlim(0, 1)
        ax_top.set_yticks([])
        ax_top.set_xticks([])
        ax_top.spines[["top", "right", "left", "bottom"]].set_visible(False)

    finish_fig(fig, args.output)


if __name__ == "__main__":
    main()
