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
        f"Missing probs for reps {missing}.\n"
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
ax.set_title("Rank vs Outbreak Size\nBacktrackingNetwork — france_office  (β=0.24, μ=0.01)")
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
