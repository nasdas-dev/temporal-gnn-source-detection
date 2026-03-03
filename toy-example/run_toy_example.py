#!/usr/bin/env python3
"""
Toy example: SIR epidemic source detection on a small temporal network.

Demonstrates the full pipeline — SIR data generation (via the C tsir simulator)
followed by two source-detection methods — without requiring wandb, config files,
or pre-built artifacts.

Usage (run from project root):
    python toy-example/run_toy_example.py

Output:
    toy-example/data/   binary simulation results (gitignored)
    Summary table printed to stdout
"""

import sys
import os

# Make sure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
import networkx as nx

from tsir.read_run import make_c_readable_from_networkx, run as tsir_run
from mc.monte_carlo import monte_carlo
from sm.soft_margin import jaccard_similarity_numpy, soft_margin_numpy
from eval.factorized_likelihood import log_likelihood, source_probabilities
from eval.ranks import compute_expected_ranks
from eval.scores import top_k_score, rank_score

# ─── Parameters ───────────────────────────────────────────────────────────────
N        = 10      # number of nodes
T_MAX    = 20      # network time horizon
BETA     = 0.30    # per-contact infection probability
MU       = 0.20    # per-timestep recovery probability
N_RUNS   = 200     # ground-truth SIR runs per source node
MC_RUNS  = 2000    # Monte Carlo SIR runs per source node
SM_A     = 0.2     # bandwidth for Soft Margin kernel
OUT_DIR  = "toy-example/data"

os.makedirs(OUT_DIR, exist_ok=True)

# ─── 1. Build a small Barabasi–Albert temporal network ────────────────────────
print("=" * 60)
print("1. Network")
print("=" * 60)

G = nx.barabasi_albert_graph(N, m=2, seed=42)
# Convert static graph to temporal: every edge is active at every integer timestep
for u, v in G.edges():
    G[u][v]["times"] = list(range(T_MAX + 1))

print(f"   Barabasi-Albert graph: {N} nodes, {G.number_of_edges()} edges")
print(f"   Each edge active at t = 0, 1, …, {T_MAX}")
print(f"   SIR parameters: beta={BETA}, mu={MU}")

H_cread = make_c_readable_from_networkx(G, t_max=T_MAX, directed=False)

# ─── 2. Ground-truth SIR simulations ──────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Ground-truth SIR simulations")
print("=" * 60)
print(f"   {N_RUNS} runs per source node  ({N * N_RUNS} total)")

seed_gt = random.getrandbits(64)
R0, avg_os, sd, se = tsir_run(
    H_cread,
    beta=BETA, mu=MU,
    start_t=0, end_t=T_MAX,
    n=N_RUNS, seed=seed_gt,
    path=f"{OUT_DIR}/ground_truth_{{}}.bin",
    log=f"{OUT_DIR}/ground_truth.txt",
)
print(f"   R0 = {R0:.2f}  |  avg. outbreak size = {100 * avg_os / N:.1f}%  "
      f"(sd = {sd:.2f})")

# Shape: (n_sources, n_runs, n_nodes)
truth_S, truth_I, truth_R = (
    np.fromfile(f"{OUT_DIR}/ground_truth_{s}.bin", dtype=np.int8).reshape(N, N_RUNS, N)
    for s in "SIR"
)

# ─── 3. Maximal outbreak (beta=1, mu=0) — used for add-one smoothing ──────────
print("\n   Computing maximal reachable sets (beta=1, mu=0) …")
tsir_run(
    H_cread,
    beta=1.0, mu=0.0,
    start_t=0, end_t=T_MAX,
    n=1, seed=0,
    path=f"{OUT_DIR}/maximal_outbreak_{{}}.bin",
    log=f"{OUT_DIR}/maximal_outbreak.txt",
)
# maximal_outbreak[s, v] = 1 if node v can ever be infected when s is the source
maximal_outbreak = np.fromfile(f"{OUT_DIR}/maximal_outbreak_I.bin", dtype=np.int8).reshape(N, N)

# ─── 4. Monte Carlo simulations ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. Monte Carlo simulations")
print("=" * 60)
print(f"   {MC_RUNS} runs per source node  ({N * MC_RUNS} total)")

seed_mc = random.getrandbits(64)
tsir_run(
    H_cread,
    beta=BETA, mu=MU,
    start_t=0, end_t=T_MAX,
    n=MC_RUNS, seed=seed_mc,
    path=f"{OUT_DIR}/monte_carlo_{{}}.bin",
    log=f"{OUT_DIR}/monte_carlo.txt",
)

# Shape: (n_sources, n_mc_runs, n_nodes)
mc_S, mc_I, mc_R = (
    np.fromfile(f"{OUT_DIR}/monte_carlo_{s}.bin", dtype=np.int8).reshape(N, MC_RUNS, N)
    for s in "SIR"
)

# Flatten truth arrays for method input: (n_sources * n_runs, n_nodes)
truth_S_flat = truth_S.reshape(-1, N)
truth_I_flat = truth_I.reshape(-1, N)
truth_R_flat = truth_R.reshape(-1, N)
# sel: only evaluate on outbreaks that infected at least 2 nodes
sel = (1 - truth_S_flat).sum(axis=1) >= 2

# ─── 5. Monte Carlo Mean Field ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. Monte Carlo Mean Field")
print("=" * 60)

# log-probability tables: (n_sources, n_nodes)
mc_log_S, mc_log_I, mc_log_R = monte_carlo(mc_S, mc_I, mc_R, MC_RUNS, N, maximal_outbreak)

# Log-likelihood of each observation under each candidate source
# mc_log_lik shape: (n_sources * n_runs, n_sources)
mc_log_lik = log_likelihood(
    truth_S_flat.astype(float), truth_I_flat.astype(float), truth_R_flat.astype(float),
    mc_log_S, mc_log_I, mc_log_R,
)

mc_ranks = compute_expected_ranks(mc_log_lik, n_nodes=N, n_runs=N_RUNS)
mc_top1  = top_k_score(mc_ranks, sel, k=1)
mc_top3  = top_k_score(mc_ranks, sel, k=3)
mc_ir    = rank_score(mc_ranks, sel, offset=0)
print(f"   Mean rank : {mc_ranks[sel].mean():.2f}  (random = {(N + 1) / 2:.1f})")
print(f"   Top-1     : {100 * mc_top1:.1f}%")
print(f"   Top-3     : {100 * mc_top3:.1f}%")
print(f"   Inv. rank : {mc_ir:.3f}  (random = {1 / N:.3f})")

# ─── 6. Soft Margin (Jaccard) ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. Soft Margin (Jaccard)")
print("=" * 60)

# jaccard shape: (n_sources, n_mc_runs, n_sources * n_runs)
jaccard = jaccard_similarity_numpy(mc_S, truth_S_flat, N)
# sm_probs shape: (n_sources * n_runs, n_sources)
sm_probs = soft_margin_numpy(jaccard, a=SM_A)

sm_ranks = compute_expected_ranks(sm_probs, n_nodes=N, n_runs=N_RUNS)
sm_top1  = top_k_score(sm_ranks, sel, k=1)
sm_top3  = top_k_score(sm_ranks, sel, k=3)
sm_ir    = rank_score(sm_ranks, sel, offset=0)
print(f"   Mean rank : {sm_ranks[sel].mean():.2f}  (random = {(N + 1) / 2:.1f})")
print(f"   Top-1     : {100 * sm_top1:.1f}%")
print(f"   Top-3     : {100 * sm_top3:.1f}%")
print(f"   Inv. rank : {sm_ir:.3f}  (random = {1 / N:.3f})")

# ─── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
hdr = f"{'Method':<22}  {'Mean rank':>9}  {'Top-1':>6}  {'Top-3':>6}  {'Inv. rank':>9}"
print(hdr)
print("-" * len(hdr))
print(f"{'Random baseline':<22}  {(N + 1) / 2:>9.2f}  {100 / N:>5.1f}%  {300 / N:>5.1f}%  {1 / N:>9.3f}")
print(f"{'MC Mean Field':<22}  {mc_ranks[sel].mean():>9.2f}  {100 * mc_top1:>5.1f}%  {100 * mc_top3:>5.1f}%  {mc_ir:>9.3f}")
print(f"{'Soft Margin (a=' + str(SM_A) + ')':<22}  {sm_ranks[sel].mean():>9.2f}  {100 * sm_top1:>5.1f}%  {100 * sm_top3:>5.1f}%  {sm_ir:>9.3f}")
print()
print(f"Network: {N} nodes, beta={BETA}, mu={MU}, "
      f"{N_RUNS} GT runs/source, {MC_RUNS} MC runs/source")
print(f"Evaluation on {sel.sum()} outbreaks (>= 2 infected nodes) "
      f"out of {len(sel)} total.")
