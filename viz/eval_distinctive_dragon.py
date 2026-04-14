"""
Evaluation plots for W&B run 'distinctive-dragon-30' (e6tw7k64).
Network: france_office (92 nodes)  |  Model: BacktrackingNetwork
Data  : france_office_b024:v0  (β=0.24, μ=0.01, n_runs=5000, mc_runs=500)

Plots generated (all from W&B data — no retraining):
  1. training_curves.pdf     — Train/val NLL loss for all 3 reps
  2. topk_accuracy.pdf       — Top-k accuracy mean ± std with per-rep scatter
  3. rank_score.pdf          — Inverse rank score mean ± std
  4. metrics_overview.pdf    — All final metrics on one summary figure

NOTE — rank vs. outbreak size:
  This plot requires per-sample model predictions, which were saved to the
  training machine's local disk (data/e6tw7k64/probs_rep*.pt) but were not
  uploaded to W&B.  Copy those .pt files here and run  viz/rank_vs_outbreak.py
  (generated at the bottom of this script).

Usage:
    cd <project_root>
    python viz/eval_distinctive_dragon.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import wandb

# ── Config ──────────────────────────────────────────────────────────────────
RUN_PATH = "nasdas-dev/source-detection/e6tw7k64"
OUT_DIR  = "viz/eval_dragon"
N_NODES  = 92
REPS     = 3
TOP_K    = [1, 3, 5]

# Thesis-quality style
plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.5,
})

REP_COLORS = ["#2f74c0", "#e6554a", "#2ecc71"]

os.makedirs(OUT_DIR, exist_ok=True)

# ── Fetch W&B data ───────────────────────────────────────────────────────────

def fetch_run():
    print("Fetching W&B run…")
    api = wandb.Api()
    run = api.run(RUN_PATH)
    h   = run.history(samples=5000)
    s   = dict(run.summary)
    cfg = dict(run.config)
    print(f"  Name   : {run.name}")
    print(f"  Network: {cfg.get('nwk', {}).get('name', 'france_office')}")
    print(f"  Epochs : {s.get('epoch', '?')} (early-stop)")
    print(f"  Params : {s.get('model/n_params', '?'):,}")
    return h, s, cfg


# ── Plot helpers ─────────────────────────────────────────────────────────────

def _finish(fig, path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)
    print(f"  Saved  {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. Training Curves
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves(h, s, path):
    print("Plotting training curves…")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={"width_ratios": [1, 2]})

    all_series = {}
    for rep, col in enumerate(REP_COLORS):
        tr_key = f"train/loss_rep{rep}"
        va_key = f"val/loss_rep{rep}"
        sub    = h[["epoch", tr_key, va_key]].dropna()
        if sub.empty:
            continue
        all_series[rep] = (sub["epoch"].values,
                           sub[tr_key].values,
                           sub[va_key].values,
                           col)

    # ── Left panel: full range (shows initialization spike) ───────────────
    ax0 = axes[0]
    for rep, (ep, tr, va, col) in all_series.items():
        ax0.plot(ep, tr, color=col, lw=1.4, label=f"Rep {rep+1} train")
        ax0.plot(ep, va, color=col, lw=1.4, ls="--", alpha=0.6)
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("NLL loss")
    ax0.set_title("Full range")
    ax0.set_xlim(left=0)

    # ── Right panel: clipped at p95 of values after epoch 3 ───────────────
    ax1 = axes[1]
    all_vals = []
    for rep, (ep, tr, va, col) in all_series.items():
        mask = ep > 3
        all_vals.extend(tr[mask].tolist())
        all_vals.extend(va[mask].tolist())
    y_ceil = float(np.percentile(all_vals, 98)) * 1.15 if all_vals else 10

    for rep, (ep, tr, va, col) in all_series.items():
        ax1.plot(ep, tr, color=col, lw=1.6, label=f"Rep {rep+1} — train")
        ax1.plot(ep, va, color=col, lw=1.6, ls="--", alpha=0.7,
                 label=f"Rep {rep+1} — val")

    # Annotate final val losses
    for rep, col in enumerate(REP_COLORS):
        vl = s.get(f"val/loss_rep{rep}")
        if vl is not None:
            ax1.axhline(vl, color=col, lw=0.7, ls=":", alpha=0.55)

    ax1.set_ylim(0, y_ceil)
    ax1.set_xlim(left=0)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL loss")
    ax1.set_title("Zoomed — after initialisation spike")
    ax1.legend(ncol=2, fontsize=8, loc="upper right")

    fig.suptitle("Training & Validation Loss\n"
                 "BacktrackingNetwork — france_office  (3 independent reps)",
                 fontsize=12)
    _finish(fig, path)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Top-k Accuracy
# ═══════════════════════════════════════════════════════════════════════════

def plot_topk_accuracy(s, path):
    print("Plotting top-k accuracy…")
    fig, ax = plt.subplots(figsize=(5.5, 4))

    x     = np.arange(len(TOP_K))
    means = [s[f"eval/top_{k}_mean"] for k in TOP_K]
    stds  = [s[f"eval/top_{k}_std"]  for k in TOP_K]

    # Bar for mean
    bars = ax.bar(x, means, width=0.45, color="#2f74c0", alpha=0.85,
                  yerr=stds, capsize=5, error_kw={"lw": 1.5},
                  label="Mean ± std  (3 reps)", zorder=3)

    # Per-rep scatter on top
    for rep, col in enumerate(REP_COLORS):
        vals = [s[f"eval/top_{k}_rep{rep}"] for k in TOP_K]
        ax.scatter(x, vals, color=col, s=55, zorder=5,
                   label=f"Rep {rep+1}", edgecolors="white", linewidths=0.5)

    # Uniform-random baseline
    baselines = [k / N_NODES for k in TOP_K]
    ax.plot(x, baselines, color="gray", lw=1.2, ls="--",
            marker="x", markersize=6, label=f"Uniform random (n={N_NODES})", zorder=4)

    # Annotate mean values
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{m:.1%}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in TOP_K], fontsize=10)
    ax.set_ylabel("Detection accuracy")
    ax.set_ylim(0, max(means) * 1.22)
    ax.set_title("Top-k Source Detection Accuracy\n"
                 "BacktrackingNetwork — france_office")
    ax.legend(fontsize=8, loc="upper left")
    _finish(fig, path)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Rank Score
# ═══════════════════════════════════════════════════════════════════════════

def plot_rank_score(s, path):
    print("Plotting rank score…")
    fig, ax = plt.subplots(figsize=(4.5, 4))

    rep_vals = [s[f"eval/rank_score_off0_rep{r}"] for r in range(REPS)]
    mean_val = s["eval/rank_score_off0_mean"]
    std_val  = s["eval/rank_score_off0_std"]
    random_baseline = 1 / N_NODES  # E[1/rank] for uniform random ≈ H_N/N

    # Jitter strip
    jitter = np.random.default_rng(0).uniform(-0.08, 0.08, REPS)
    for rep, (v, j, col) in enumerate(zip(rep_vals, jitter, REP_COLORS)):
        ax.scatter(1 + j, v, s=100, color=col, zorder=5,
                   edgecolors="white", linewidths=0.6,
                   label=f"Rep {rep+1}: {v:.4f}")

    # Mean ± std
    ax.errorbar(1, mean_val, yerr=std_val, fmt="D", color="black",
                ms=8, capsize=6, lw=2, zorder=6, label=f"Mean ± std")
    ax.axhline(random_baseline, color="gray", ls="--", lw=1.1,
               label=f"Uniform random ({random_baseline:.4f})")

    ax.set_xlim(0.7, 1.3)
    ax.set_xticks([])
    ax.set_ylabel("Inverse rank score  1/rank  (higher = better)")
    ax.set_title("Rank Score\n"
                 f"BacktrackingNetwork — france_office\n"
                 f"mean = {mean_val:.4f}  ±  {std_val:.4f}")
    ax.legend(fontsize=9)
    _finish(fig, path)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Metrics Overview (summary panel)
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_overview(s, cfg, path):
    print("Plotting metrics overview…")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: all top-k per rep as grouped scatter+line ──────────────────
    ax = axes[0]
    for rep, col in enumerate(REP_COLORS):
        vals = [s[f"eval/top_{k}_rep{rep}"] for k in TOP_K]
        ax.plot(TOP_K, vals, color=col, marker="o", ms=7, lw=1.5,
                label=f"Rep {rep+1}")

    # Mean line
    means = [s[f"eval/top_{k}_mean"] for k in TOP_K]
    stds  = [s[f"eval/top_{k}_std"]  for k in TOP_K]
    ax.plot(TOP_K, means, color="black", marker="D", ms=7, lw=2,
            ls="--", label="Mean", zorder=5)
    ax.fill_between(TOP_K,
                    [m - sd for m, sd in zip(means, stds)],
                    [m + sd for m, sd in zip(means, stds)],
                    color="black", alpha=0.10, label="± std")

    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy")
    ax.set_xticks(TOP_K)
    ax.set_title("Top-k Accuracy per Repetition")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # ── Right: summary table ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    rows = []
    # Top-k rows
    for k in TOP_K:
        m = s[f"eval/top_{k}_mean"]
        sd = s[f"eval/top_{k}_std"]
        reps_str = "  ".join(f"{s[f'eval/top_{k}_rep{r}']:.1%}" for r in range(REPS))
        rows.append([f"Top-{k}", f"{m:.1%}", f"{sd:.1%}", reps_str])
    # Rank score
    m  = s["eval/rank_score_off0_mean"]
    sd = s["eval/rank_score_off0_std"]
    reps_str = "  ".join(f"{s[f'eval/rank_score_off0_rep{r}']:.4f}" for r in range(REPS))
    rows.append(["Rank score", f"{m:.4f}", f"{sd:.4f}", reps_str])
    # Final losses
    for split in ["train", "val"]:
        for rep in range(REPS):
            v = s.get(f"{split}/loss_rep{rep}", float("nan"))
            rows.append([f"{split} loss rep{rep}", f"{v:.4f}", "—", "—"])

    col_labels = ["Metric", "Mean", "Std", "Rep 0 / Rep 1 / Rep 2"]
    table = ax2.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.15, 1.45)
    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#d0dff2")
        table[(0, j)].set_text_props(weight="bold")

    ax2.set_title("Summary — distinctive-dragon-30\n"
                  f"france_office  |  BN  |  n={N_NODES} nodes  "
                  f"|  params={s.get('model/n_params', '?'):,}",
                  pad=12)

    _finish(fig, path)


# ═══════════════════════════════════════════════════════════════════════════
#  Generate companion script for rank-vs-outbreak (requires saved probs)
# ═══════════════════════════════════════════════════════════════════════════

RANK_VS_OUTBREAK_SCRIPT = '''\
"""
Rank vs Outbreak Size plot.

Requires per-sample predictions saved during training:
    data/e6tw7k64/probs_rep{0,1,2}.pt

Copy those files from the training machine, then run:
    cd <project_root>
    python viz/rank_vs_outbreak.py
"""
import os, sys, pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval import compute_ranks

DATA_DIR  = "data/2d25y02o"
PROBS_DIR = "data/e6tw7k64"
OUT_DIR   = "viz/eval_dragon"
N_NODES   = 92
N_RUNS    = 5000
N_TRUTH   = 1000
MIN_OUTBREAK = 2
REPS      = 3

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})

# Check probs exist
missing = [r for r in range(REPS)
           if not os.path.exists(f"{PROBS_DIR}/probs_rep{r}.pt")]
if missing:
    raise FileNotFoundError(
        f"Missing probs for reps {missing}.\\n"
        f"Copy data/e6tw7k64/probs_rep*.pt from the training machine first."
    )

# Load ground-truth data
gt = {s: np.fromfile(f"{DATA_DIR}/ground_truth_{s}.bin", dtype=np.int8)
         .reshape(N_NODES, N_RUNS, N_NODES) for s in "SIR"}
possible = np.fromfile(f"{DATA_DIR}/possible_sources.bin", dtype=np.int8
                       ).reshape(N_NODES, N_RUNS, N_NODES)

all_ranks, all_sizes, all_src, all_sel = [], [], [], []
for rep in range(REPS):
    probs = torch.load(f"{PROBS_DIR}/probs_rep{rep}.pt",
                       map_location="cpu").numpy()
    sel_t  = np.arange(rep * N_TRUTH, (rep + 1) * N_TRUTH) % N_RUNS

    S_flat = gt["S"][:, sel_t, :].reshape(-1, N_NODES)
    I_flat = gt["I"][:, sel_t, :].reshape(-1, N_NODES)
    R_flat = gt["R"][:, sel_t, :].reshape(-1, N_NODES)
    poss   = possible[:, sel_t, :].reshape(-1, N_NODES)
    lik    = np.where(poss == 1, 0.0, np.inf)

    infected = (I_flat + R_flat).sum(axis=1)
    sel      = infected >= MIN_OUTBREAK
    sizes    = infected / N_NODES
    sources  = np.repeat(np.arange(N_NODES), N_TRUTH)

    log_p  = np.log(np.clip(probs, 1e-12, 1.0)) - lik
    ranks  = compute_ranks(log_p, n_nodes=N_NODES, n_runs=N_TRUTH)

    all_ranks.append(ranks);  all_sizes.append(sizes)
    all_src.append(sources);  all_sel.append(sel)

ranks   = np.concatenate(all_ranks)
sizes   = np.concatenate(all_sizes)
sources = np.concatenate(all_src)
sel     = np.concatenate(all_sel)

r = ranks[sel].astype(float)
s = sizes[sel]

fig, ax = plt.subplots(figsize=(6.5, 4.5))

correct = r == 1
ax.scatter(s[~correct], r[~correct], s=6, alpha=0.12, color="#5b7fa6",
           linewidths=0, rasterized=True, label="Rank > 1")
ax.scatter(s[correct],  r[correct],  s=6, alpha=0.30, color="#e6554a",
           linewidths=0, rasterized=True, label="Rank 1 (correct)")

bins = np.linspace(0, 1, 16)
cents = 0.5 * (bins[:-1] + bins[1:])
means, p25, p75 = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (s >= lo) & (s < hi)
    vals = r[mask] if mask.sum() > 0 else np.array([np.nan])
    means.append(np.nanmean(vals))
    p25.append(np.nanpercentile(vals, 25) if mask.sum() > 0 else np.nan)
    p75.append(np.nanpercentile(vals, 75) if mask.sum() > 0 else np.nan)

ca = np.array(cents); ma = np.array(means); p2a = np.array(p25); p7a = np.array(p75)
v = ~np.isnan(ma)
ax.fill_between(ca[v], p2a[v], p7a[v], alpha=0.20, color="black", label="IQR (25–75 %)")
ax.plot(ca[v], ma[v], color="black", lw=2.2, zorder=5, label="Binned mean")

ax.set_xlabel("Outbreak size  (fraction of N={N_NODES} nodes infected)".replace("{N_NODES}", str(N_NODES)))
ax.set_ylabel(f"Rank of true source  (out of {N_NODES})")
ax.set_title("Rank vs Outbreak Size\\nBacktrackingNetwork — france_office  (β=0.24, μ=0.01)")
ax.set_xlim(0, 1); ax.set_ylim(0.5, N_NODES + 0.5)
ax.legend(loc="upper right")

# Marginal histogram
ax_top = ax.inset_axes([0, 1.04, 1, 0.16])
ax_top.hist(s, bins=bins, color="#5b7fa6", alpha=0.7, edgecolor="none")
ax_top.set_xlim(0, 1); ax_top.set_yticks([]); ax_top.set_xticks([])
ax_top.spines[["top","right","left","bottom"]].set_visible(False)

os.makedirs(OUT_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/rank_vs_outbreak.pdf")
print(f"Saved {OUT_DIR}/rank_vs_outbreak.pdf")
'''


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Eval plots — distinctive-dragon-30  (W&B data only)")
    print("=" * 60)

    h, s, cfg = fetch_run()

    print()
    plot_training_curves(h, s, f"{OUT_DIR}/training_curves.pdf")
    plot_topk_accuracy(s, f"{OUT_DIR}/topk_accuracy.pdf")
    plot_rank_score(s, f"{OUT_DIR}/rank_score.pdf")
    plot_metrics_overview(s, cfg, f"{OUT_DIR}/metrics_overview.pdf")

    # Write companion script for rank-vs-outbreak
    script_path = "viz/rank_vs_outbreak.py"
    with open(script_path, "w") as f:
        f.write(RANK_VS_OUTBREAK_SCRIPT)
    print(f"\n  Companion script written: {script_path}")
    print("  → Copy data/e6tw7k64/probs_rep*.pt from the training machine,")
    print("    then run:  python viz/rank_vs_outbreak.py")

    print(f"\nDone. Plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
