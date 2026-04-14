"""
Training Set Size Scaling — performance vs. number of MC training samples.

Queries W&B for runs from a training-size sweep (produced by
``run_training_size_sweep.sh``) and plots how MRR / Top-1 improves as more
simulated training data is provided.

Requires that sweep runs were tagged with ``n_mc:{value}`` or that
``config.train.n_mc`` is set (this is done automatically by the sweep script).

Usage
-----
::

    python viz/training_size_scaling.py \\
        --artifact france_office \\
        --metric mrr top_1 \\
        --output figures/training_size_scaling.pdf

    python viz/training_size_scaling.py \\
        --artifact france_office \\
        --models backtracking static_gnn temporal_gnn \\
        --output figures/training_size_scaling.pdf
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
from viz.style import apply_style, finish_fig, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, model_style
from viz.wandb_utils import fetch_runs_for_artifact


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact", required=True,
                   help="Artifact name prefix, e.g. france_office")
    p.add_argument("--metric",   nargs="+", default=["mrr", "top_1"],
                   help="Metrics to plot (default: mrr top_1)")
    p.add_argument("--models",   nargs="+", default=None,
                   help="Model keys to include (default: all GNN models found)")
    p.add_argument("--project",  default="source-detection")
    p.add_argument("--entity",   default=None)
    p.add_argument("--output",   default="figures/training_size_scaling.pdf")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate_by_n_mc(
    runs: list[dict],
    metric_key: str,
    gnn_only: bool = True,
) -> dict[str, dict[int, list[float]]]:
    """Group run values by model and n_mc (number of MC training samples).

    Only includes models that have a ``train.n_mc`` config key (i.e. GNN
    training runs, not baseline eval runs).
    """
    grouped: dict[str, dict[int, list[float]]] = {}
    for run in runs:
        model = run["model"]
        n_mc  = run["config"].get("train", {}).get("n_mc")
        if n_mc is None:
            continue
        n_mc = int(n_mc)
        val = run["summary"].get(metric_key)
        if val is None:
            val = run["summary"].get(metric_key.replace("_mean", ""))
        if val is None:
            continue
        grouped.setdefault(model, {}).setdefault(n_mc, []).append(float(val))
    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_style()

    print(f"Fetching W&B runs for artifact: {args.artifact}")
    runs = fetch_runs_for_artifact(args.artifact, args.project, args.entity)
    if not runs:
        print("No finished runs found.  Run experiments first.")
        return
    print(f"  Found {len(runs)} runs")

    gnn_models = [m for m in MODEL_ORDER if m not in
                  {"uniform", "random", "degree", "closeness",
                   "betweenness", "jordan_center", "soft_margin", "mcs_mean_field"}]
    include_models = args.models or gnn_models

    n_metrics = len(args.metric)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5),
                             squeeze=False)
    axes = axes[0]

    for ax, mkey in zip(axes, args.metric):
        wandb_key = f"eval/{mkey}_mean"
        grouped   = aggregate_by_n_mc(runs, wandb_key)

        for model in include_models:
            if model not in grouped:
                continue
            n_mc_vals = sorted(grouped[model].keys())
            y_mean    = [float(np.mean(grouped[model][n])) for n in n_mc_vals]
            y_std     = [float(np.std(grouped[model][n]))  for n in n_mc_vals]

            sty = model_style(model)
            ax.errorbar(
                n_mc_vals, y_mean, yerr=y_std,
                color=sty["color"], marker=sty["marker"],
                label=sty["label"],
                lw=1.8, ms=7, capsize=4,
            )

        ax.set_xlabel("Number of MC training samples  (n_mc)")
        metric_label = {
            "mrr":   "MRR",
            "top_1": "Top-1 Accuracy",
            "top_3": "Top-3 Accuracy",
            "top_5": "Top-5 Accuracy",
        }.get(mkey, mkey)
        ax.set_ylabel(metric_label)

        if mkey.startswith("top_"):
            ax.yaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
            )
        ax.set_title(metric_label)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle(f"Training Set Size Scaling  —  {args.artifact}",
                 fontsize=12, fontweight="bold")
    finish_fig(fig, args.output)


if __name__ == "__main__":
    main()
