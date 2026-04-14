"""
Table generation for thesis evaluation chapters.

Produces two canonical tables as LaTeX (booktabs style) + CSV:

1. **Benchmark Table** — rows: methods; columns: MRR, Top-1/3/5/10, Norm-Brier,
   Norm-Entropy, Cred-90; segmented by network.  Requires W&B runs.

2. **Network Statistics Table** — N, E, mean degree, clustering coefficient,
   diameter, time span T, burstiness B.  Fully offline — reads from nwk/.

Usage
-----
::

    # Benchmark table (requires finished W&B runs)
    python -m eval.tables benchmark \\
        --data france_office karate_static toy_holme \\
        --output figures/tables/

    # Network stats table (offline, no wandb needed)
    python -m eval.tables network_stats \\
        --networks france_office karate_static lyon_ward malawi toy_holme \\
        --output figures/tables/

    # Both
    python -m eval.tables all \\
        --data france_office karate_static \\
        --networks france_office karate_static lyon_ward malawi \\
        --output figures/tables/
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import textwrap
from typing import Any

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    # --- benchmark ---
    bm = sub.add_parser("benchmark", help="Generate benchmark comparison table")
    bm.add_argument("--data",    nargs="+", required=True,
                    help="Artifact names, e.g. france_office karate_static")
    bm.add_argument("--project", default="source-detection")
    bm.add_argument("--entity",  default=None)
    bm.add_argument("--output",  default="figures/tables/",
                    help="Output directory for .tex and .csv files")
    bm.add_argument("--metrics", nargs="+",
                    default=["mrr", "top_1", "top_3", "top_5", "top_10",
                             "norm_brier", "norm_entropy", "cred_cov_90"],
                    help="Metric keys to include (eval/ prefix added automatically)")

    # --- network_stats ---
    ns = sub.add_parser("network_stats", help="Generate network statistics table")
    ns.add_argument("--networks", nargs="+", required=True,
                    help="Network names matching nwk/<name>.yml and nwk/<name>.csv")
    ns.add_argument("--output",  default="figures/tables/")

    # --- all ---
    al = sub.add_parser("all", help="Generate both tables")
    al.add_argument("--data",    nargs="+", required=True)
    al.add_argument("--networks", nargs="+", required=True)
    al.add_argument("--project", default="source-detection")
    al.add_argument("--entity",  default=None)
    al.add_argument("--output",  default="figures/tables/")
    al.add_argument("--metrics", nargs="+",
                    default=["mrr", "top_1", "top_3", "top_5", "top_10",
                             "norm_brier", "norm_entropy", "cred_cov_90"])

    return p.parse_args()


# ---------------------------------------------------------------------------
# Shared LaTeX helpers
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "backtracking", "temporal_gnn", "static_gnn", "dbgnn", "dag_gnn",
    "jordan_center", "betweenness", "closeness", "degree",
    "soft_margin", "mcs_mean_field", "uniform", "random",
]

MODEL_LABELS = {
    "backtracking":  "BacktrackingNet",
    "temporal_gnn":  "TemporalGNN",
    "static_gnn":    "StaticGNN",
    "dbgnn":         "DBGNN",
    "dag_gnn":       "DAG-GNN",
    "jordan_center": "Jordan Center",
    "betweenness":   "Betweenness",
    "closeness":     "Closeness",
    "degree":        "Degree",
    "soft_margin":   "Soft Margin",
    "mcs_mean_field":"MCS MF",
    "uniform":       "Uniform",
    "random":        "Random",
}

METRIC_LABELS = {
    "mrr":          "MRR",
    "top_1":        "Top-1",
    "top_3":        "Top-3",
    "top_5":        "Top-5",
    "top_10":       "Top-10",
    "norm_brier":   "NBS",
    "norm_entropy": "Entropy",
    "cred_cov_90":  "Cred-90",
}

GNN_MODELS = {"backtracking", "temporal_gnn", "static_gnn", "dbgnn", "dag_gnn"}


def _bold_best(values: list[str], raw: list[float | None],
               higher_is_better: list[bool]) -> list[str]:
    """Bold the best value in each column."""
    result = list(values)
    n = len(raw)
    for col in range(len(higher_is_better)):
        col_vals = [raw[row][col] if raw[row] is not None else None
                    for row in range(n)]
        valid = [v for v in col_vals if v is not None]
        if not valid:
            continue
        best = max(valid) if higher_is_better[col] else min(valid)
        for row in range(n):
            if col_vals[row] is not None and abs(col_vals[row] - best) < 1e-9:
                # Bold only the value cell — parse formatted string
                parts = result[row].split(" & ")
                parts[col + 1] = r"\textbf{" + parts[col + 1] + "}"
                result[row] = " & ".join(parts)
    return result


def _write_files(
    lines: list[str],
    csv_rows: list[list[str]],
    output_dir: str,
    stem: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, f"{stem}.tex")
    csv_path = os.path.join(output_dir, f"{stem}.csv")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"  Saved  {tex_path}")
    print(f"  Saved  {csv_path}")


# ---------------------------------------------------------------------------
# Phase 8a: Network statistics table (offline)
# ---------------------------------------------------------------------------

def _burstiness(G: nx.Graph) -> float:
    """Compute Fano-factor-based burstiness B = (σ - μ) / (σ + μ).

    Collects all inter-event times across all edges.  Returns nan if fewer
    than 2 events exist.
    """
    iet: list[float] = []
    for u, v, data in G.edges(data=True):
        times = sorted(data.get("times", []))
        for dt in np.diff(times):
            if dt > 0:
                iet.append(float(dt))
    if len(iet) < 2:
        return float("nan")
    mu  = float(np.mean(iet))
    sig = float(np.std(iet))
    return (sig - mu) / (sig + mu) if (sig + mu) > 0 else float("nan")


def _network_stats(name: str) -> dict[str, Any]:
    """Compute statistics for network `name` by loading nwk/<name>.yml and .csv."""
    import yaml
    from setup.read_network import read_networkx

    yml_path = f"nwk/{name}.yml"
    csv_path = f"nwk/{name}.csv"

    if not os.path.exists(yml_path):
        raise FileNotFoundError(f"Missing: {yml_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing: {csv_path}")

    with open(yml_path) as f:
        meta = yaml.safe_load(f)

    t_max = meta.get("time_steps", meta.get("t_max", None))
    if t_max is None:
        raise ValueError(f"Cannot determine t_max for network '{name}' from {yml_path}")

    print(f"  Loading {name}…", end=" ", flush=True)
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        G = read_networkx(csv_path, t_max=t_max,
                          directed=meta.get("directed", False))
    print("done")

    N = G.number_of_nodes()
    E = G.number_of_edges()

    # Static projection for topological metrics
    H_static = nx.Graph()
    H_static.add_nodes_from(G.nodes())
    H_static.add_edges_from(G.edges())

    mean_deg = 2.0 * E / N if N > 0 else 0.0

    clust = nx.average_clustering(H_static)

    try:
        if nx.is_connected(H_static):
            diam = nx.diameter(H_static)
        else:
            giant = H_static.subgraph(max(nx.connected_components(H_static), key=len))
            diam = nx.diameter(giant)
    except Exception:
        diam = float("nan")

    B = _burstiness(G)

    return {
        "name":     name,
        "N":        N,
        "E":        E,
        "mean_deg": mean_deg,
        "clust":    clust,
        "diam":     diam,
        "T":        t_max,
        "B":        B,
    }


def network_stats_table(networks: list[str], output_dir: str) -> None:
    """Compute and write the network statistics table."""
    print("=" * 60)
    print("Network Statistics Table")
    print("=" * 60)

    rows_data = []
    for name in networks:
        try:
            stats = _network_stats(name)
            rows_data.append(stats)
        except Exception as exc:
            print(f"  WARNING: skipping '{name}': {exc}")

    if not rows_data:
        print("  No network data collected.")
        return

    # ── LaTeX ──────────────────────────────────────────────────────────────
    col_spec = "l" + "r" * 7
    header   = ("Network & $N$ & $E$ & $\\langle k \\rangle$ "
                "& $C$ & $d$ & $T$ & $B$")
    lines = [
        "% Network statistics table — generated by eval/tables.py",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"{header} \\\\",
        "\\midrule",
    ]
    csv_rows = [["Network", "N", "E", "mean_deg", "clustering", "diameter", "T", "burstiness"]]

    for s in rows_data:
        diam_str = str(int(s["diam"])) if not math.isnan(s["diam"]) else r"\textemdash"
        B_str    = f"{s['B']:.3f}" if not math.isnan(s["B"]) else r"\textemdash"
        name_tex = s["name"].replace("_", "\\_")
        row_tex  = (
            f"{name_tex} & "
            f"{s['N']} & {s['E']} & "
            f"{s['mean_deg']:.2f} & "
            f"{s['clust']:.3f} & "
            f"{diam_str} & "
            f"{s['T']} & "
            f"{B_str} \\\\"
        )
        lines.append(row_tex)
        csv_rows.append([
            s["name"], s["N"], s["E"],
            f"{s['mean_deg']:.4f}",
            f"{s['clust']:.4f}",
            str(int(s["diam"])) if not math.isnan(s["diam"]) else "nan",
            s["T"],
            f"{s['B']:.4f}" if not math.isnan(s["B"]) else "nan",
        ])

    lines += ["\\bottomrule", "\\end{tabular}"]
    _write_files(lines, csv_rows, output_dir, "network_stats_table")


# ---------------------------------------------------------------------------
# Phase 8b: Benchmark table (requires W&B)
# ---------------------------------------------------------------------------

def _fetch_all_results(
    data_names: list[str],
    project: str,
    entity: str | None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Return {data_name → {model → {metric → value}}}."""
    from viz.wandb_utils import fetch_runs_for_artifact, best_run_per_model

    result: dict[str, dict[str, dict[str, float]]] = {}
    for data_name in data_names:
        print(f"  Fetching runs for: {data_name}…")
        runs = fetch_runs_for_artifact(data_name, project, entity)
        best = best_run_per_model(runs, metric="eval/mrr_mean")
        result[data_name] = {}
        for model, run in best.items():
            result[data_name][model] = dict(run["summary"])
    return result


def benchmark_table(
    data_names: list[str],
    metric_keys: list[str],
    project: str,
    entity: str | None,
    output_dir: str,
) -> None:
    """Fetch W&B results and write the benchmark comparison table."""
    print("=" * 60)
    print("Benchmark Table")
    print("=" * 60)

    all_results = _fetch_all_results(data_names, project, entity)

    # Which metrics are higher-is-better?
    higher = {
        "mrr": True, "top_1": True, "top_3": True, "top_5": True, "top_10": True,
        "norm_brier": False, "norm_entropy": False, "cred_cov_90": True,
    }
    hib = [higher.get(m, True) for m in metric_keys]
    metric_header = " & ".join(METRIC_LABELS.get(m, m) for m in metric_keys)

    lines = [
        "% Benchmark results table — generated by eval/tables.py",
        f"\\begin{{tabular}}{{ll{'c' * len(metric_keys)}}}",
        "\\toprule",
        f"Dataset & Method & {metric_header} \\\\",
        "\\midrule",
    ]
    csv_header = ["Dataset", "Method"] + [METRIC_LABELS.get(m, m) for m in metric_keys]
    csv_rows   = [csv_header]

    for di, data_name in enumerate(data_names):
        if di > 0:
            lines.append("\\midrule")  # horizontal rule between datasets

        res = all_results.get(data_name, {})
        present_models = [m for m in MODEL_ORDER if m in res]

        tex_rows_raw:  list[str]               = []
        raw_vals:      list[list[float | None]] = []

        for model in present_models:
            summary = res[model]
            vals: list[float | None] = []
            cells: list[str]         = []

            for mkey in metric_keys:
                # Try _mean first, fall back to plain key
                v = summary.get(f"eval/{mkey}_mean",
                    summary.get(f"eval/{mkey}"))
                if v is not None:
                    v = float(v)
                    vals.append(v)
                    if mkey.startswith("top_") or mkey.startswith("cred_"):
                        cells.append(f"{v * 100:.1f}")
                    else:
                        cells.append(f"{v:.4f}")
                else:
                    vals.append(None)
                    cells.append("---")

            label   = MODEL_LABELS.get(model, model).replace("_", "\\_")
            dn_cell = data_name.replace("_", "\\_") if model == present_models[0] else ""
            row_tex = f"{dn_cell} & {label} & " + " & ".join(cells) + " \\\\"
            tex_rows_raw.append(row_tex)
            raw_vals.append(vals)
            csv_rows.append([data_name, MODEL_LABELS.get(model, model)] + [
                str(v) if v is not None else "" for v in vals
            ])

        # Bold best per column within this network segment
        bolded = _bold_best(tex_rows_raw, raw_vals, hib)
        lines.extend(bolded)

    lines += ["\\bottomrule", "\\end{tabular}"]
    _write_files(lines, csv_rows, output_dir, "benchmark_table")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.command in ("network_stats", "all"):
        network_stats_table(args.networks, args.output)

    if args.command in ("benchmark", "all"):
        benchmark_table(
            data_names  = args.data,
            metric_keys = args.metrics,
            project     = args.project,
            entity      = args.entity,
            output_dir  = args.output,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
