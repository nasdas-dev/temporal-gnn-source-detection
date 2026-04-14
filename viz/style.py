"""
Shared visual style, model registry, and figure helpers for all viz scripts.

Import at the top of every viz script:

    from viz.style import apply_style, finish_fig, MODEL_COLORS, MODEL_LABELS, MODEL_ORDER, REP_COLORS

``apply_style()`` must be called before creating any figure.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Thesis-quality rcParams
# ---------------------------------------------------------------------------

RCPARAMS: dict = {
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "grid.linewidth":      0.5,
    "axes.axisbelow":      True,
    "lines.linewidth":     1.6,
    "patch.linewidth":     0.5,
    "legend.framealpha":   0.85,
    "legend.edgecolor":    "0.7",
    "figure.constrained_layout.use": False,
}


def apply_style() -> None:
    """Apply thesis rcParams globally.  Call once per script before any plot."""
    plt.rcParams.update(RCPARAMS)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Canonical display order (GNN models first, then baselines)
MODEL_ORDER: list[str] = [
    "backtracking",
    "temporal_gnn",
    "static_gnn",
    "dbgnn",
    "dag_gnn",
    "jordan_center",
    "betweenness",
    "closeness",
    "degree",
    "soft_margin",
    "mcs_mean_field",
    "uniform",
    "random",
]

#: Display names for each model key
MODEL_LABELS: dict[str, str] = {
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

#: Hex colours per model — GNNs use saturated blues/oranges, baselines use greys
MODEL_COLORS: dict[str, str] = {
    "backtracking":  "#8172B2",   # purple
    "temporal_gnn":  "#DD8452",   # orange
    "static_gnn":    "#4C72B0",   # blue
    "dbgnn":         "#55A868",   # green
    "dag_gnn":       "#C44E52",   # red
    "jordan_center": "#555555",
    "betweenness":   "#666666",
    "closeness":     "#777777",
    "degree":        "#888888",
    "soft_margin":   "#999999",
    "mcs_mean_field":"#AAAAAA",
    "uniform":       "#BBBBBB",
    "random":        "#CCCCCC",
}

#: Marker styles per model (for line plots)
MODEL_MARKERS: dict[str, str] = {
    "backtracking":  "o",
    "temporal_gnn":  "s",
    "static_gnn":    "^",
    "dbgnn":         "D",
    "dag_gnn":       "v",
    "jordan_center": "x",
    "betweenness":   "+",
    "closeness":     "*",
    "degree":        "P",
    "soft_margin":   "h",
    "mcs_mean_field":"H",
    "uniform":       ".",
    "random":        ",",
}

#: Colours for training repetitions 0/1/2/3/…
REP_COLORS: list[str] = [
    "#2f74c0",   # rep 0 — blue
    "#e6554a",   # rep 1 — red
    "#2ecc71",   # rep 2 — green
    "#f39c12",   # rep 3 — amber
    "#9b59b6",   # rep 4 — violet
]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def finish_fig(fig: plt.Figure, path: str) -> None:
    """Apply tight layout, save (PDF + PNG), close, and print confirmation."""
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    plt.savefig(path)
    # Also save PNG at same location
    png_path = os.path.splitext(path)[0] + ".png"
    if path.endswith(".pdf"):
        plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Saved  {path}")


def model_style(key: str) -> dict:
    """Return a dict of plot kwargs (color, marker, label) for a given model key."""
    return {
        "color":  MODEL_COLORS.get(key, "#333333"),
        "marker": MODEL_MARKERS.get(key, "o"),
        "label":  MODEL_LABELS.get(key, key),
    }
