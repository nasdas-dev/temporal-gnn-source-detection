"""
Stage 3 — Non-ML baseline evaluation.

Evaluates classical heuristic baselines on a TSIR artifact, logging the
same metrics as ``main_train.py`` for direct comparison in W&B.

Usage
-----
::

    python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest
    python main_eval.py --cfg exp/france_office/eval.yml --data france_office:latest

The ``--cfg`` YAML must contain an ``eval`` section (same keys as model
configs) and a ``baselines`` list selecting which heuristics to run.
"""

from __future__ import annotations

import argparse
from typing import Iterator

import networkx as nx
import numpy as np
import wandb
import yaml

from eval import compute_ranks, top_k_score, rank_score
from setup import setup_methods_run, load_tsir_data


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg",  required=True,
                   help="Eval config YAML, e.g. exp/toy_holme/eval.yml")
    p.add_argument("--data", required=True,
                   help="W&B artifact reference, e.g. toy_holme:latest")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infected_subgraph(H_static: nx.Graph, I_snap: np.ndarray, R_snap: np.ndarray) -> nx.Graph:
    """Return the subgraph of H_static induced by infected + recovered nodes."""
    infected_nodes = list(np.where((I_snap + R_snap) > 0)[0])
    if not infected_nodes:
        return nx.Graph()
    return H_static.subgraph(infected_nodes).copy()


def _scores_to_probs(scores: dict[int, float], n_nodes: int, poss: np.ndarray) -> np.ndarray:
    """Convert {node: score} → probability vector [n_nodes], masked by ``poss``."""
    vec = np.zeros(n_nodes, dtype=np.float64)
    for node, score in scores.items():
        if 0 <= node < n_nodes:
            vec[node] = max(0.0, score)

    vec = vec * poss.astype(np.float64)
    total = vec.sum()
    if total > 0:
        return (vec / total).astype(np.float32)
    # Fallback: uniform over possible
    u = poss.astype(np.float64)
    s = u.sum()
    return (u / s if s > 0 else np.ones(n_nodes) / n_nodes).astype(np.float32)


def _batch_iter(
    truth_S: np.ndarray,  # [n_nodes, n_truth, n_nodes]
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    possible: np.ndarray,
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (s_idx, r_idx, S_snap, I_snap, R_snap, poss_vec) for every pair."""
    n_nodes, n_truth, _ = truth_S.shape
    for s_idx in range(n_nodes):
        for r_idx in range(n_truth):
            yield (
                s_idx,
                r_idx,
                truth_S[s_idx, r_idx],
                truth_I[s_idx, r_idx],
                truth_R[s_idx, r_idx],
                possible[s_idx, r_idx],
            )


# ---------------------------------------------------------------------------
# Per-baseline probability computation
# ---------------------------------------------------------------------------

def compute_baseline_probs(
    baseline: str,
    H_static: nx.Graph,
    truth_S: np.ndarray,   # [n_nodes, n_truth, n_nodes]
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    possible: np.ndarray,  # [n_nodes, n_truth, n_nodes]
    n_nodes: int,
    n_truth: int,
) -> np.ndarray:
    """Compute probability distributions for one baseline.

    Parameters
    ----------
    baseline:
        One of ``uniform``, ``random``, ``degree``, ``closeness``,
        ``betweenness``, ``jordan_center``.
    H_static:
        Static (undirected) projection of the temporal network.

    Returns
    -------
    probs : ndarray [n_nodes * n_truth, n_nodes]
        Row ``s * n_truth + r`` contains the probability distribution for
        observation (source=s, run=r).
    """
    n_total = n_nodes * n_truth
    probs = np.zeros((n_total, n_nodes), dtype=np.float32)

    for s_idx, r_idx, S_snap, I_snap, R_snap, poss in _batch_iter(
        truth_S, truth_I, truth_R, possible
    ):
        flat_idx = s_idx * n_truth + r_idx

        # --- Uniform & Random: no subgraph needed ---
        if baseline == "uniform":
            u = poss.astype(np.float32)
            s = u.sum()
            probs[flat_idx] = u / s if s > 0 else np.ones(n_nodes, dtype=np.float32) / n_nodes
            continue

        if baseline == "random":
            u = poss.astype(np.float32)
            s = u.sum()
            p = u / s if s > 0 else np.ones(n_nodes, dtype=np.float32) / n_nodes
            chosen = int(np.random.choice(n_nodes, p=p))
            probs[flat_idx, chosen] = 1.0
            continue

        # --- Topology-based: need infected subgraph ---
        G_sub = _infected_subgraph(H_static, I_snap, R_snap)

        if len(G_sub.nodes) == 0:
            u = poss.astype(np.float32)
            s = u.sum()
            probs[flat_idx] = u / s if s > 0 else np.ones(n_nodes, dtype=np.float32) / n_nodes
            continue

        scores_dict: dict[int, float] = {}

        if baseline == "degree":
            scores_dict = {n: float(d) for n, d in G_sub.degree()}

        elif baseline == "closeness":
            try:
                scores_dict = {n: float(v) for n, v in nx.closeness_centrality(G_sub).items()}
            except Exception:
                scores_dict = {n: 1.0 for n in G_sub.nodes()}

        elif baseline == "betweenness":
            try:
                scores_dict = {n: float(v) for n, v in nx.betweenness_centrality(G_sub).items()}
            except Exception:
                scores_dict = {n: 1.0 for n in G_sub.nodes()}

        elif baseline == "jordan_center":
            # Jordan center = node with minimum eccentricity
            try:
                if not nx.is_connected(G_sub):
                    G_cc = G_sub.subgraph(
                        max(nx.connected_components(G_sub), key=len)
                    ).copy()
                else:
                    G_cc = G_sub
                ecc = nx.eccentricity(G_cc)
                max_ecc = max(ecc.values()) + 1
                scores_dict = {n: float(max_ecc - e) for n, e in ecc.items()}
            except Exception:
                scores_dict = {n: 1.0 for n in G_sub.nodes()}

        else:
            raise ValueError(f"Unknown baseline: '{baseline}'")

        probs[flat_idx] = _scores_to_probs(scores_dict, n_nodes, poss)

    return probs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -----------------------------------------------------------------------
    # 1. Load config
    # -----------------------------------------------------------------------
    with open(args.cfg) as f:
        cfg_dict = yaml.safe_load(f)

    eval_cfg  = cfg_dict["eval"]
    baselines = cfg_dict["baselines"]
    n_truth   = eval_cfg["n_truth"]
    top_k_vals = eval_cfg["top_k"]
    offsets    = eval_cfg["inverse_rank_offset"]

    # -----------------------------------------------------------------------
    # 2. W&B initialisation
    # -----------------------------------------------------------------------
    setup_methods_run(job_type="eval")
    wandb.config.update({"data_name": args.data, **cfg_dict})
    wandb.run.tags += ("baselines",)
    print(f"\nW&B run : {wandb.run.url}")
    print(f"Data    : {args.data}\n")

    # -----------------------------------------------------------------------
    # 3. Load TSIR artifact
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("Loading TSIR data")
    print("=" * 60)
    H, data = load_tsir_data(args.data)
    n_nodes = data.n_nodes
    print(f"  n_nodes  : {n_nodes}")
    print(f"  n_runs   : {data.n_runs}  (ground-truth)")

    if n_truth > data.n_runs:
        raise ValueError(
            f"n_truth={n_truth} requested but artifact only has {data.n_runs} runs."
        )

    # Build static projection of H for topology-based baselines
    H_static = nx.Graph()
    H_static.add_nodes_from(range(n_nodes))
    for u, v in H.edges():
        H_static.add_edge(int(u), int(v))

    # Select ground-truth slice
    select_truth = np.arange(n_truth)
    truth_S = data.truth_S[:, select_truth, :]   # [n_nodes, n_truth, n_nodes]
    truth_I = data.truth_I[:, select_truth, :]
    truth_R = data.truth_R[:, select_truth, :]
    possible = data.possible[:, select_truth, :]
    lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, n_nodes)

    truth_S_flat = truth_S.reshape(-1, n_nodes)
    sel = (1 - truth_S_flat).sum(axis=1) >= eval_cfg["min_outbreak"]
    print(f"  Valid outbreaks: {sel.sum()} / {len(sel)}\n")

    # -----------------------------------------------------------------------
    # 4. Evaluate each baseline
    # -----------------------------------------------------------------------
    summary_rows: list[dict] = []

    for baseline in baselines:
        print("=" * 60)
        print(f"Baseline: {baseline}")
        print("=" * 60)

        probs = compute_baseline_probs(
            baseline  = baseline,
            H_static  = H_static,
            truth_S   = truth_S,
            truth_I   = truth_I,
            truth_R   = truth_R,
            possible  = possible,
            n_nodes   = n_nodes,
            n_truth   = n_truth,
        )   # [n_nodes * n_truth, n_nodes]

        log_probs = np.log(np.clip(probs, 1e-12, 1.0)) - lik_possible
        ranks = compute_ranks(log_probs, n_nodes=n_nodes, n_runs=n_truth)

        metrics: dict[str, float] = {"model": baseline}
        for k in top_k_vals:
            score = float(top_k_score(ranks, sel, k))
            metrics[f"eval/top_{k}"] = score
            print(f"  top-{k}: {100 * score:.1f}%")
        for o in offsets:
            rs = float(rank_score(ranks, sel, o))
            metrics[f"eval/rank_score_off{o}"] = rs
            print(f"  rank_score (offset={o}): {rs:.4f}")

        # Log this baseline as its own W&B step (keyed by baseline name)
        wandb.log({
            **{k: v for k, v in metrics.items() if k != "model"},
            "baseline": baseline,
        })

        summary_rows.append(metrics)

    # -----------------------------------------------------------------------
    # 5. Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    header = ["baseline"] + [f"top_{k}" for k in top_k_vals] + [f"rs_off{o}" for o in offsets]
    print("  " + "  ".join(f"{h:>12}" for h in header))
    for row in summary_rows:
        vals = [row["model"]]
        for k in top_k_vals:
            vals.append(f"{100 * row[f'eval/top_{k}']:.1f}%")
        for o in offsets:
            vals.append(f"{row[f'eval/rank_score_off{o}']:.4f}")
        print("  " + "  ".join(f"{v:>12}" for v in vals))

    # Log comparison table to wandb
    table = wandb.Table(columns=header)
    for row in summary_rows:
        table_row = [row["model"]]
        for k in top_k_vals:
            table_row.append(row[f"eval/top_{k}"])
        for o in offsets:
            table_row.append(row[f"eval/rank_score_off{o}"])
        table.add_data(*table_row)
    wandb.log({"baselines_comparison": table})

    wandb.summary["data/name"] = args.data
    wandb.summary["n_valid_outbreaks"] = int(sel.sum())
    wandb.summary["n_total"] = len(sel)

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
