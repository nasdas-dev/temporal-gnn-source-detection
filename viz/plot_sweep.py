"""
Plot sweep results: model performance vs. a parameter (beta or observation time).

Fetches W&B runs for a sweep experiment directory and produces:
  - Line plot: metric vs. parameter, one line per model
  - LaTeX table: results at each parameter value

Usage
-----
::

    python viz/plot_sweep.py --sweep sweep_vary_beta --x-param beta \\
        --x-values 0.10 0.20 0.30 0.50 0.70 \\
        --x-labels 0.10 0.20 0.30 0.50 0.70 \\
        --artifact-prefix france_office.sweep_vary_beta \\
        --metric top_1 --output figures/sweep_beta.pdf

    python viz/plot_sweep.py --sweep sweep_vary_observation --x-param obs_frac \\
        --x-values 25 50 75 100 \\
        --x-labels 25% 50% 75% 100% \\
        --artifact-prefix france_office.sweep_vary_observation \\
        --metric top_1 --output figures/sweep_obs.pdf
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
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep",   required=True,
                   help="Sweep name, e.g. sweep_vary_beta")
    p.add_argument("--x-param", required=True,
                   help="Parameter name for x-axis label")
    p.add_argument("--x-values", nargs="+", required=True, type=float,
                   help="Numeric x-axis values in order")
    p.add_argument("--x-labels", nargs="+", required=True,
                   help="Display labels for x-axis values")
    p.add_argument("--artifact-prefix", required=True,
                   help="Prefix shared by all artifact names in this sweep")
    p.add_argument("--metric",  default="top_1",
                   help="Metric to plot (default: top_1)")
    p.add_argument("--output",  default=None,
                   help="Output path. Default: figures/<sweep>_<metric>.pdf")
    p.add_argument("--project", default="source-detection",
                   help="W&B project name")
    p.add_argument("--entity",  default=None,
                   help="W&B entity")
    p.add_argument("--models",  nargs="+",
                   default=["static_gnn", "temporal_gnn", "dbgnn", "dag_gnn", "backtracking"],
                   help="Models to include in the plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B fetching
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "static_gnn":   "StaticGNN",
    "temporal_gnn": "TemporalGNN",
    "dbgnn":        "DBGNN",
    "dag_gnn":      "DAG-GNN",
    "backtracking": "BacktrackingNet",
}

MODEL_COLORS = {
    "static_gnn":   "#4C72B0",
    "temporal_gnn": "#DD8452",
    "dbgnn":        "#55A868",
    "dag_gnn":      "#C44E52",
    "backtracking": "#8172B2",
}

MODEL_MARKERS = {
    "static_gnn":   "o",
    "temporal_gnn": "s",
    "dbgnn":        "^",
    "dag_gnn":      "D",
    "backtracking": "v",
}


def fetch_sweep_results(
    artifact_prefix: str,
    x_values: list[float],
    models: list[str],
    metric: str,
    project: str,
    entity: str | None,
) -> dict[str, list[tuple[float, float]]]:
    """Return {model_name: [(x, y), ...]} for all x_values where data exists."""
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("W&B not available. Install wandb.")
        sys.exit(1)

    prefix = entity + "/" if entity else ""
    runs = api.runs(f"{prefix}{project}")

    # Group runs by (artifact_name, model)
    run_map: dict[tuple[str, str], float] = {}
    for run in runs:
        if run.state != "finished":
            continue
        cfg     = run.config
        summary = run.summary
        model   = cfg.get("model")
        data    = cfg.get("data_name", "")

        if model not in models:
            continue
        if not data.startswith(artifact_prefix.split(":")[0]):
            continue

        wk = f"eval/{metric}_mean" if f"eval/{metric}_mean" in summary else f"eval/{metric}"
        val = summary.get(wk)
        if val is not None:
            run_map[(data, model)] = float(val)

    # Map artifact names to x_values by position (assuming alphabetical order matches)
    # Collect unique artifact names matching prefix
    artifact_names = sorted({k[0] for k in run_map if k[0].startswith(artifact_prefix)})

    results: dict[str, list[tuple[float, float]]] = {m: [] for m in models}

    for art, x in zip(artifact_names, x_values):
        for m in models:
            val = run_map.get((art, m))
            if val is not None:
                results[m].append((x, val))

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(
    results: dict[str, list[tuple[float, float]]],
    x_param: str,
    x_labels: list[str],
    x_values: list[float],
    metric: str,
    sweep_name: str,
    output_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for model, pts in results.items():
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] * 100 for p in pts]   # percentage
        ax.plot(
            xs, ys,
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=2,
            markersize=6,
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_param.replace("_", " ").title())
    k = metric.split("_")[-1]
    ax.set_ylabel(f"Top-{k} Accuracy (%)" if metric.startswith("top") else metric)
    ax.set_title(f"Source Detection — {sweep_name}")
    ax.legend(loc="best")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def print_latex_table(
    results: dict[str, list[tuple[float, float]]],
    x_labels: list[str],
    metric: str,
) -> None:
    models = [m for m, pts in results.items() if pts]
    print(f"\n\\begin{{tabular}}{{l{'c' * len(x_labels)}}}")
    print("\\toprule")
    header = " & ".join(x_labels)
    print(f"Method & {header} \\\\")
    print("\\midrule")
    for m in models:
        pts_dict = dict(results[m])
        row = [MODEL_LABELS.get(m, m)]
        for xl in x_labels:
            v = pts_dict.get(xl)
            row.append(f"{v*100:.1f}" if v is not None else "---")
        print(" & ".join(row) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output = args.output or f"figures/{args.sweep}_{args.metric}.pdf"

    print(f"Fetching sweep results: {args.sweep}")
    results = fetch_sweep_results(
        artifact_prefix = args.artifact_prefix,
        x_values        = args.x_values,
        models          = args.models,
        metric          = args.metric,
        project         = args.project,
        entity          = args.entity,
    )

    if all(not pts for pts in results.values()):
        print("No results found. Run sweep experiments first.")
        return

    plot_sweep(results, args.x_param, args.x_labels, args.x_values,
               args.metric, args.sweep, output)
    print_latex_table(results, args.x_labels, args.metric)


if __name__ == "__main__":
    main()
