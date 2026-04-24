"""
Training and validation loss curves from W&B.

Fetches per-epoch loss history for one or more runs and produces a
two-panel figure:
  - Left panel:  full training range (shows initialisation spike)
  - Right panel: zoomed view after the first few epochs

Works with any number of training repetitions (reps), colour-coded.

Usage
-----
::

    # Single run
    python viz/training_curves.py \\
        --run-path entity/source-detection/e6tw7k64 \\
        --output figures/training_curves.pdf

    # Multiple runs overlaid (e.g. different models for comparison)
    python viz/training_curves.py \\
        --run-path entity/source-detection/e6tw7k64 entity/source-detection/abc123 \\
        --label "BacktrackingNet" "StaticGNN" \\
        --output figures/training_curves_compare.pdf
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
from viz.style import apply_style, finish_fig, REP_COLORS, MODEL_COLORS, MODEL_LABELS
from viz.wandb_utils import fetch_run_history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-path", nargs="+", required=True,
                   help="W&B run path(s), e.g. entity/project/run_id")
    p.add_argument("--label",   nargs="+", default=None,
                   help="Display labels for each run path")
    p.add_argument("--output",  default="figures/training_curves.pdf")
    p.add_argument("--max-reps", type=int, default=5,
                   help="Maximum number of repetitions to look for per run (default: 5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_curves_for_run(
    ax_full: plt.Axes,
    ax_zoom: plt.Axes,
    history: dict[str, list],   # column-dict from fetch_run_history
    summary: dict,
    label: str,
    max_reps: int,
    color_offset: int = 0,
    zoom_skip: int = 3,
) -> None:
    """Plot train/val curves for one W&B run onto ax_full and ax_zoom."""
    epochs = np.array(history.get("epoch", []), dtype=float)

    for rep in range(max_reps):
        tr_key = f"train/loss_rep{rep}"
        va_key = f"val/loss_rep{rep}"

        if tr_key not in history:
            break

        tr_raw = np.array(history[tr_key], dtype=float)
        va_raw = np.array(history.get(va_key, [np.nan] * len(tr_raw)), dtype=float)

        # Drop rows where either series is NaN
        ep = epochs if len(epochs) == len(tr_raw) else np.arange(len(tr_raw), dtype=float)
        mask = np.isfinite(tr_raw) & np.isfinite(va_raw)
        if mask.sum() == 0:
            break

        ep, tr, va = ep[mask], tr_raw[mask], va_raw[mask]

        col = REP_COLORS[(rep + color_offset) % len(REP_COLORS)]
        rep_lbl = f"{label}  rep {rep + 1}" if label else f"Rep {rep + 1}"

        for ax in (ax_full, ax_zoom):
            ax.plot(ep, tr, color=col, lw=1.5, label=f"{rep_lbl} — train")
            ax.plot(ep, va, color=col, lw=1.5, ls="--", alpha=0.7,
                    label=f"{rep_lbl} — val")

        # Annotate best val loss as dotted horizontal
        best_val = summary.get(f"val/loss_rep{rep}")
        if best_val is not None:
            ax_zoom.axhline(float(best_val), color=col, lw=0.7, ls=":", alpha=0.5)


def make_figure(
    run_paths: list[str],
    labels: list[str],
    max_reps: int,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1, 2]})
    ax_full, ax_zoom = axes

    all_zoom_vals: list[float] = []
    color_offset = 0

    for run_path, lbl in zip(run_paths, labels):
        print(f"  Fetching {run_path}…")
        history, summary, cfg = fetch_run_history(run_path)
        reps = cfg.get("train", {}).get("reps", max_reps)

        plot_curves_for_run(
            ax_full, ax_zoom, history, summary,
            label=lbl if len(run_paths) > 1 else "",
            max_reps=min(reps, max_reps),
            color_offset=color_offset,
            zoom_skip=3,
        )
        color_offset += min(reps, max_reps)

        # Collect post-warmup values for zoom y-axis
        epochs_col = np.array(history.get("epoch", []), dtype=float)
        for rep in range(min(reps, max_reps)):
            for key in [f"train/loss_rep{rep}", f"val/loss_rep{rep}"]:
                if key not in history:
                    continue
                vals = np.array(history[key], dtype=float)
                ep_arr = epochs_col if len(epochs_col) == len(vals) else np.arange(len(vals), dtype=float)
                mask = (ep_arr > 3) & np.isfinite(vals)
                all_zoom_vals.extend(vals[mask].tolist())

    # Zoom y-axis: clip at 98th percentile × 1.15
    if all_zoom_vals:
        y_ceil = float(np.percentile(all_zoom_vals, 98)) * 1.15
        ax_zoom.set_ylim(0, y_ceil)

    ax_full.set_xlabel("Epoch");  ax_full.set_ylabel("NLL loss")
    ax_zoom.set_xlabel("Epoch");  ax_zoom.set_ylabel("NLL loss")
    ax_full.set_xlim(left=0);     ax_zoom.set_xlim(left=0)
    ax_full.set_title("Full range")
    ax_zoom.set_title("Zoomed — after initialisation spike")

    # Deduplicated legend on zoom panel
    handles, lbls = ax_zoom.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax_zoom.legend(by_label.values(), by_label.keys(),
                   ncol=2, fontsize=8, loc="upper right")

    title = "Training & Validation Loss"
    if labels and labels[0]:
        title += "  —  " + "  vs.  ".join(labels)
    fig.suptitle(title, fontsize=12)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_style()

    run_paths = args.run_path
    labels    = args.label or [""] * len(run_paths)
    if len(labels) < len(run_paths):
        labels = labels + [""] * (len(run_paths) - len(labels))

    print("Plotting training curves…")
    fig = make_figure(run_paths, labels, args.max_reps)
    finish_fig(fig, args.output)


if __name__ == "__main__":
    main()
