"""
Performance vs. Network Size (N) — scaling analysis.

Queries W&B for all runs whose ``data_name`` matches a given artifact prefix
(e.g. ``exp_1_vary_n``), groups by network size N (from ``config.nwk.n``),
and plots how each model's MRR / Top-k degrades as the network grows.

Usage
-----
::

    python viz/perf_vs_n.py \\
        --artifact-prefix exp_1_vary_n \\
        --metric mrr top_1 \\
        --output figures/perf_vs_n.pdf

    python viz/perf_vs_n.py \\
        --artifact-prefix exp_1_vary_n \\
        --project source-detection --entity myteam \\
        --models backtracking temporal_gnn static_gnn uniform jordan_center \\
        --output figures/scaling_mrr.pdf
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
    p.add_argument("--artifact-prefix", required=True,
                   help="Artifact name prefix to match, e.g. exp_1_vary_n")
    p.add_argument("--metric",   nargs="+", default=["mrr"],
                   help="Metric(s) to plot from eval/ (default: mrr). "
                        "Use: mrr top_1 top_3 top_5")
    p.add_argument("--models",   nargs="+", default=None,
                   help="Model keys to include (default: all found). "
                        "E.g. backtracking temporal_gnn uniform jordan_center")
    p.add_argument("--project",  default="source-detection")
    p.add_argument("--entity",   default=None)
    p.add_argument("--output",   default="figures/perf_vs_n.pdf")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data aggregation
# ---------------------------------------------------------------------------

def aggregate_by_n(
    runs: list[dict],
    metric_key: str,
) -> dict[str, dict[int, list[float]]]:
    """Group run summary values by model key and network size N.

    Parameters
    ----------
    runs:
        Output of ``fetch_runs_for_artifact``.
    metric_key:
        W&B summary key, e.g. ``"eval/mrr_mean"`` or ``"eval/top_1_mean"``.

    Returns
    -------
    Nested dict: model → {N → [values…]}
    """
    grouped: dict[str, dict[int, list[float]]] = {}
    for run in runs:
        model = run["model"]
        cfg   = run["config"]
        n     = cfg.get("nwk", {}).get("n") or cfg.get("n")
        if n is None:
            continue
        n = int(n)
        val = run["summary"].get(metric_key)
        if val is None:
            # Try without _mean suffix for baseline runs
            val = run["summary"].get(metric_key.replace("_mean", ""))
        if val is None:
            continue
        grouped.setdefault(model, {}).setdefault(n, []).append(float(val))
    return grouped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    apply_style()

    print(f"Fetching W&B runs for artifact prefix: {args.artifact_prefix}")
    runs = fetch_runs_for_artifact(args.artifact_prefix, args.project, args.entity)
    if not runs:
        print("No finished runs found.  Run experiments first.")
        return
    print(f"  Found {len(runs)} runs covering models: "
          f"{sorted(set(r['model'] for r in runs))}")

    n_metrics = len(args.metric)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5),
                             sharey=False, squeeze=False)
    axes = axes[0]

    include_models = args.models or MODEL_ORDER

    for ax, mkey in zip(axes, args.metric):
        wandb_key = f"eval/{mkey}_mean"
        grouped   = aggregate_by_n(runs, wandb_key)

        plotted_any = False
        for model in include_models:
            if model not in grouped:
                continue
            n_vals = sorted(grouped[model].keys())
            y_mean = [float(np.mean(grouped[model][n])) for n in n_vals]
            y_std  = [float(np.std(grouped[model][n]))  for n in n_vals]

            sty = model_style(model)
            ax.errorbar(
                n_vals, y_mean, yerr=y_std,
                color=sty["color"], marker=sty["marker"],
                label=sty["label"],
                lw=1.8, ms=7, capsize=4,
            )
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        ax.set_xscale("log")
        ax.set_xlabel("Network size  N  (nodes)")

        metric_label = {
            "mrr":   "Mean Reciprocal Rank (MRR)",
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
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(f"Performance vs. Network Size  —  {args.artifact_prefix}",
                 fontsize=12, fontweight="bold")
    finish_fig(fig, args.output)


if __name__ == "__main__":
    main()
