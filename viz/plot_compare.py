"""
Compare all models and baselines for a single network experiment.

Fetches W&B runs linked to a given TSIR artifact and produces:
  - Bar chart: Top-k accuracy per model (grouped bars)
  - Bar chart: Rank score per model
  - LaTeX table: All metrics (stdout)

Usage
-----
::

    python viz/plot_compare.py --data toy_holme --output figures/toy_holme.pdf
    python viz/plot_compare.py --data france_office --metric top_1 top_3 top_5
    python viz/plot_compare.py --data toy_holme --no-wandb  # offline: load from data/ dir
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",   required=True,
                   help="Artifact name, e.g. toy_holme or france_office")
    p.add_argument("--output", default=None,
                   help="Output PDF/PNG path. Default: figures/<data>_compare.pdf")
    p.add_argument("--project", default="source-detection",
                   help="W&B project name (default: source-detection)")
    p.add_argument("--entity",  default=None,
                   help="W&B entity (username/team). Uses default if not set.")
    p.add_argument("--metric",  nargs="+",
                   default=["top_1", "top_3", "top_5"],
                   help="Metrics to plot (default: top_1 top_3 top_5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B data fetching
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "static_gnn", "temporal_gnn", "dbgnn", "dag_gnn", "backtracking",
    "uniform", "random", "degree", "closeness", "betweenness", "jordan_center",
]

MODEL_LABELS = {
    "static_gnn":   "StaticGNN",
    "temporal_gnn": "TemporalGNN",
    "dbgnn":        "DBGNN",
    "dag_gnn":      "DAG-GNN",
    "backtracking": "BacktrackingNet",
    "uniform":      "Uniform",
    "random":       "Random",
    "degree":       "Degree",
    "closeness":    "Closeness",
    "betweenness":  "Betweenness",
    "jordan_center":"Jordan Center",
}

MODEL_COLORS = {
    "static_gnn":   "#4C72B0",
    "temporal_gnn": "#DD8452",
    "dbgnn":        "#55A868",
    "dag_gnn":      "#C44E52",
    "backtracking": "#8172B2",
    "uniform":      "#BBBBBB",
    "random":       "#AAAAAA",
    "degree":       "#999999",
    "closeness":    "#888888",
    "betweenness":  "#777777",
    "jordan_center":"#666666",
}


def fetch_results(data_name: str, project: str, entity: str | None) -> dict[str, dict]:
    """Pull summary metrics from W&B for all runs using this artifact."""
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("W&B not available. Install wandb or use --no-wandb.")
        sys.exit(1)

    prefix = entity + "/" if entity else ""
    runs = api.runs(f"{prefix}{project}")

    results: dict[str, dict] = {}

    for run in runs:
        cfg = run.config
        summary = run.summary

        # Match runs that used this artifact
        run_data = cfg.get("data_name", "")
        if not run_data.startswith(data_name.split(":")[0]):
            continue

        if run.state != "finished":
            continue

        model = cfg.get("model", None)
        if model is None:
            continue

        metrics: dict[str, float] = {}
        for k, v in summary.items():
            if k.startswith("eval/"):
                metrics[k] = float(v) if v is not None else float("nan")

        if not metrics:
            continue

        # For GNN models: use mean over repetitions from summary
        # For baselines: summary keys are also eval/*
        if model not in results:
            results[model] = metrics
        else:
            # Keep the most recent run (last wins)
            results[model] = metrics

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(
    results: dict[str, dict],
    data_name: str,
    metric_keys: list[str],
    output_path: str,
) -> None:
    """Plot grouped bar charts for the requested metrics."""

    # Filter to models that have results, in canonical order
    present_models = [m for m in MODEL_ORDER if m in results]
    if not present_models:
        print("No results found. Check --data argument and W&B run history.")
        return

    n_models   = len(present_models)
    n_metrics  = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics + 1, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, mkey in zip(axes, metric_keys):
        wandb_key = f"eval/{mkey}_mean" if f"eval/{mkey}_mean" in list(results.values())[0] else f"eval/{mkey}"

        values = []
        errors = []
        colors = []
        labels = []

        for m in present_models:
            val = results[m].get(wandb_key, results[m].get(f"eval/{mkey}", float("nan")))
            std = results[m].get(f"eval/{mkey}_std", 0.0)
            values.append(float(val) * 100)  # percentage
            errors.append(float(std) * 100)
            colors.append(MODEL_COLORS.get(m, "#333333"))
            labels.append(MODEL_LABELS.get(m, m))

        x = np.arange(n_models)
        bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.5,
                      capsize=3, width=0.7)

        # Error bars
        ax.errorbar(x, values, yerr=errors, fmt="none", ecolor="black",
                    elinewidth=1, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Accuracy (%)")
        k = mkey.split("_")[-1]
        ax.set_title(f"Top-{k} Accuracy" if mkey.startswith("top") else mkey)
        ax.set_ylim(0, min(100, max(values) * 1.25 + 5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(f"Source Detection — {data_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_latex_table(results: dict[str, dict], metric_keys: list[str]) -> None:
    """Print a LaTeX-formatted results table to stdout."""
    present = [m for m in MODEL_ORDER if m in results]

    header_cols = " & ".join([f"Top-{k.split('_')[-1]}" for k in metric_keys])
    print(f"\n\\begin{{tabular}}{{l{'c' * len(metric_keys)}}}")
    print("\\toprule")
    print(f"Method & {header_cols} \\\\")
    print("\\midrule")

    for m in present:
        r = results[m]
        row = [MODEL_LABELS.get(m, m)]
        for mkey in metric_keys:
            wk_mean = f"eval/{mkey}_mean"
            wk_std  = f"eval/{mkey}_std"
            val = r.get(wk_mean, r.get(f"eval/{mkey}", float("nan")))
            std = r.get(wk_std, 0.0)
            if not np.isnan(val):
                row.append(f"{val*100:.1f} $\\pm$ {std*100:.1f}")
            else:
                row.append("---")
        print(" & ".join(row) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output = args.output or f"figures/{args.data}_compare.pdf"

    print(f"Fetching W&B results for artifact: {args.data}")
    results = fetch_results(args.data, args.project, args.entity)

    if not results:
        print("No finished runs found for this artifact. Run experiments first.")
        return

    print(f"Found results for: {list(results.keys())}")

    plot_comparison(results, args.data, args.metric, output)
    print_latex_table(results, args.metric)


if __name__ == "__main__":
    main()
