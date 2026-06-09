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
import csv
import json
import os
from typing import Iterator

import networkx as nx
import numpy as np
import wandb
import yaml

from eval import compute_all_metrics, per_sample_arrays, precompute_graph_metric_context
from eval.benchmark import (
    mc_mean_field as _mc_mean_field,
    mcs_mean_field as _mcs_mean_field,
    soft_margin as _soft_margin,
)
from setup import setup_methods_run, load_tsir_data


BASELINE_METHOD_NOTES: dict[str, str] = {
    "uniform": "Uniform distribution over feasible non-susceptible source candidates.",
    "random": "One feasible source candidate sampled uniformly with a fixed seed.",
    "degree": "Exact observed-subgraph degree, computed by matrix multiplication.",
    "closeness": "Static-graph closeness centrality prior, masked to feasible candidates.",
    "betweenness": "Static-graph betweenness centrality prior, masked to feasible candidates.",
    "jordan_center": "Jordan center using static shortest-path distances among observed non-susceptible nodes.",
    "soft_margin": "Soft-margin simulation baseline with parameters from baseline_params/TSIR metadata.",
    "mcs_mean_field": "Monte Carlo simulation mean-field baseline with parameters from baseline_params/TSIR metadata.",
    "mc_mean_field": "Artifact Monte Carlo mean-field baseline using stored simulations.",
}


PREFERRED_METRIC_KEYS = [
    "eval/mrr",
    "eval/mean_rank",
    "eval/median_rank",
    "eval/top_1",
    "eval/top_3",
    "eval/top_5",
    "eval/top_10",
    "eval/rank_score_off0",
    "eval/brier",
    "eval/norm_brier",
    "eval/log_score",
    "eval/norm_log_score",
    "eval/norm_entropy",
    "eval/mean_true_prob",
    "eval/map_confidence",
    "eval/cred_cov_80",
    "eval/cred_cov_90",
    "eval/cred_set_size_80",
    "eval/cred_set_size_90",
    "eval/error_dist",
    "eval/resistance",
    "eval/n_valid",
    "eval/n_total",
    "eval/valid_frac",
    "eval/rank_scope_candidate",
    "eval/mean_candidate_count",
    "eval/median_candidate_count",
]


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _baseline_param_map(cfg_dict: dict, data_config: dict) -> dict[str, dict]:
    """Return merged per-baseline parameters from config and TSIR metadata."""
    raw = cfg_dict.get("baseline_params", {}) or {}
    default_params = dict(raw.get("default", {}))
    experiment = cfg_dict.get("experiment", {}) or {}
    sir_cfg = (data_config or {}).get("sir", {}) or {}

    sim_defaults: dict[str, float | int] = {}
    if "beta" in sir_cfg or "beta" in experiment:
        sim_defaults["beta"] = float(sir_cfg.get("beta", experiment.get("beta")))
    if "mu" in sir_cfg or "mu" in experiment:
        sim_defaults["mu"] = float(sir_cfg.get("mu", experiment.get("mu")))
    start_t = sir_cfg.get("start_t", 0)
    end_t = sir_cfg.get("end_t", sir_cfg.get("t_max", None))
    if end_t is not None:
        sim_defaults["n_steps"] = max(1, int(end_t) - int(start_t))

    merged: dict[str, dict] = {}
    for baseline in cfg_dict.get("baselines", []):
        params = dict(default_params)
        if baseline in {"soft_margin", "mcs_mean_field"}:
            params = {**sim_defaults, **params}
        params.update(raw.get(baseline, {}) or {})
        merged[baseline] = params
    return merged


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
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Override config values, e.g. --override eval.n_truth=50")
    return p.parse_args()


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> None:
    """Apply ``key.subkey=value`` overrides to a nested config dict in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value or key.subkey=value format")
        key_path, raw_val = item.split("=", 1)
        keys = key_path.strip().split(".")
        for cast in (int, float):
            try:
                raw_val = cast(raw_val)
                break
            except ValueError:
                pass
        else:
            if raw_val.lower() in ("true", "false"):
                raw_val = raw_val.lower() == "true"
        node = cfg_dict
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = raw_val


def _truth_indices(eval_cfg: dict, n_eval_runs: int, n_runs: int) -> np.ndarray:
    """Return the held-out truth window requested by eval config."""
    truth_start = int(eval_cfg.get("truth_start", 0))
    if truth_start < 0:
        raise ValueError(f"eval.truth_start must be non-negative, got {truth_start}")
    truth_stop = truth_start + n_eval_runs
    if truth_stop > n_runs:
        raise ValueError(
            f"eval.truth_start + eval.reps * n_truth = {truth_start} + "
            f"{n_eval_runs} = {truth_stop} exceeds n_runs={n_runs}. Reduce "
            "eval.n_truth/eval.reps, lower eval.truth_start, or regenerate the "
            "artifact with more ground-truth runs."
        )
    return np.arange(truth_start, truth_stop)


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


def _uniform_prob_chunk(poss_chunk: np.ndarray) -> np.ndarray:
    """Return row-wise uniform probabilities over feasible candidates."""
    probs = poss_chunk.astype(np.float32, copy=True)
    totals = probs.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    if np.any(valid):
        probs[valid] /= totals[valid]
    if np.any(~valid):
        probs[~valid] = 1.0 / probs.shape[1]
    return probs


def _scores_matrix_to_probs(scores: np.ndarray, poss_chunk: np.ndarray) -> np.ndarray:
    """Mask non-negative score rows and normalise, with uniform fallback."""
    probs = np.maximum(scores.astype(np.float32, copy=False), 0.0)
    probs = probs * poss_chunk.astype(np.float32, copy=False)
    totals = probs.sum(axis=1, keepdims=True)
    valid = totals[:, 0] > 0
    if np.any(valid):
        probs[valid] /= totals[valid]
    if np.any(~valid):
        probs[~valid] = _uniform_prob_chunk(poss_chunk[~valid])
    return probs.astype(np.float32, copy=False)


def _random_prob_chunk(poss_chunk: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one feasible candidate per row without a Python loop."""
    weights = poss_chunk.astype(np.float64, copy=False)
    totals = weights.sum(axis=1)
    probs = np.zeros_like(poss_chunk, dtype=np.float32)

    valid = totals > 0
    if np.any(valid):
        cdf = np.cumsum(weights[valid], axis=1)
        draws = rng.random(np.sum(valid)) * totals[valid]
        chosen = np.sum(cdf < draws[:, None], axis=1)
        probs[np.where(valid)[0], chosen] = 1.0
    if np.any(~valid):
        chosen = rng.integers(0, poss_chunk.shape[1], size=np.sum(~valid))
        probs[np.where(~valid)[0], chosen] = 1.0

    return probs


def _static_centrality_scores(
    baseline: str,
    H_static: nx.Graph,
    n_nodes: int,
    baseline_params: dict,
    seed: int,
) -> np.ndarray:
    """Return one static centrality vector used as a prior baseline."""
    if baseline == "closeness":
        values = nx.closeness_centrality(H_static)
    elif baseline == "betweenness":
        k = baseline_params.get("k", None)
        values = nx.betweenness_centrality(
            H_static,
            k=None if k is None else int(k),
            normalized=bool(baseline_params.get("normalized", True)),
            seed=seed,
        )
    else:
        raise ValueError(f"No static centrality implementation for '{baseline}'")
    return np.array([float(values.get(i, 0.0)) for i in range(n_nodes)], dtype=np.float32)


def _static_shortest_path_distances(H_static: nx.Graph, n_nodes: int) -> np.ndarray:
    """Return all-pairs static graph distances with finite disconnected penalty."""
    dist_dict = dict(nx.all_pairs_shortest_path_length(H_static))
    finite = [
        d
        for src_dists in dist_dict.values()
        for d in src_dists.values()
        if d > 0
    ]
    unreachable = (max(finite) + 1) if finite else n_nodes
    dist = np.full((n_nodes, n_nodes), float(unreachable), dtype=np.float32)
    np.fill_diagonal(dist, 0.0)
    for src, src_dists in dist_dict.items():
        if not 0 <= int(src) < n_nodes:
            continue
        for dst, d in src_dists.items():
            if 0 <= int(dst) < n_nodes:
                dist[int(src), int(dst)] = float(d)
    return dist


def _jordan_center_score(mask: np.ndarray, dist_matrix: np.ndarray, n_nodes: int) -> np.ndarray:
    """Score feasible nodes by inverse eccentricity to the observed node set."""
    nodes = np.flatnonzero(mask)
    scores = np.zeros(n_nodes, dtype=np.float32)
    if nodes.size == 0:
        return scores
    if nodes.size == 1:
        scores[nodes[0]] = 1.0
        return scores

    # Standard Jordan-center heuristic: minimize maximum distance to observed
    # infected/recovered nodes.  Distances come from the static contact graph,
    # which avoids rebuilding an induced NetworkX graph per outbreak snapshot.
    eccentricity = dist_matrix[np.ix_(nodes, nodes)].max(axis=1)
    max_ecc = float(eccentricity.max())
    scores[nodes] = (max_ecc + 1.0) - eccentricity
    return scores


def _fast_baseline_probs(
    baseline: str,
    H_static: nx.Graph,
    possible: np.ndarray,
    n_nodes: int,
    n_truth: int,
    baseline_params: dict,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Fast exact/standard heuristic baselines, or ``None`` if unsupported."""
    n_total = n_nodes * n_truth
    poss_flat = possible.reshape(n_total, n_nodes)
    probs = np.zeros((n_total, n_nodes), dtype=np.float32)
    chunk_size = int(baseline_params.get("chunk_size", 8192))

    if baseline == "uniform":
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            probs[start:end] = _uniform_prob_chunk(poss_flat[start:end])
        return probs

    if baseline == "random":
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            probs[start:end] = _random_prob_chunk(poss_flat[start:end], rng)
        return probs

    if baseline == "degree":
        adjacency = nx.to_numpy_array(
            H_static,
            nodelist=list(range(n_nodes)),
            dtype=np.float32,
            weight=None,
        )
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            poss_chunk = poss_flat[start:end]
            scores = poss_chunk.astype(np.float32, copy=False) @ adjacency
            probs[start:end] = _scores_matrix_to_probs(scores, poss_chunk)
        return probs

    if baseline in {"closeness", "betweenness"}:
        static_scores = _static_centrality_scores(
            baseline, H_static, n_nodes, baseline_params, int(baseline_params.get("seed", 0))
        )
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            poss_chunk = poss_flat[start:end]
            scores = np.broadcast_to(static_scores, poss_chunk.shape)
            probs[start:end] = _scores_matrix_to_probs(scores, poss_chunk)
        return probs

    if baseline == "jordan_center":
        dist_matrix = _static_shortest_path_distances(H_static, n_nodes)
        cache: dict[bytes, np.ndarray] = {}
        max_cache_size = int(baseline_params.get("cache_size", 4096))
        for start in range(0, n_total, chunk_size):
            end = min(start + chunk_size, n_total)
            poss_chunk = poss_flat[start:end]
            scores = np.zeros((end - start, n_nodes), dtype=np.float32)
            for local_idx, mask in enumerate(poss_chunk.astype(bool, copy=False)):
                key = np.packbits(mask).tobytes()
                row_scores = cache.get(key)
                if row_scores is None:
                    row_scores = _jordan_center_score(mask, dist_matrix, n_nodes)
                    if len(cache) < max_cache_size:
                        cache[key] = row_scores
                scores[local_idx] = row_scores
            probs[start:end] = _scores_matrix_to_probs(scores, poss_chunk)
        return probs

    return None


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
    baseline_params: dict | None = None,
    rng: np.random.Generator | None = None,
    mc_S: np.ndarray | None = None,
    mc_I: np.ndarray | None = None,
    mc_R: np.ndarray | None = None,
) -> np.ndarray:
    """Compute probability distributions for one baseline.

    Parameters
    ----------
    baseline:
        One of ``uniform``, ``random``, ``degree``, ``closeness``,
        ``betweenness``, ``jordan_center``, ``soft_margin``,
        ``mcs_mean_field``.
    H_static:
        Static (undirected) projection of the temporal network.

    Returns
    -------
    probs : ndarray [n_nodes * n_truth, n_nodes]
        Row ``s * n_truth + r`` contains the probability distribution for
        observation (source=s, run=r).
    """
    if rng is None:
        rng = np.random.default_rng()
    baseline_params = baseline_params or {}

    if baseline == "mc_mean_field":
        if mc_S is None or mc_I is None or mc_R is None:
            raise ValueError("mc_mean_field baseline requires mc_S, mc_I, and mc_R arrays")
        return _mc_mean_field(
            mc_S=mc_S,
            mc_I=mc_I,
            mc_R=mc_R,
            truth_S=truth_S,
            truth_I=truth_I,
            truth_R=truth_R,
            possible=possible,
            eps=float(baseline_params.get("eps", 1e-6)),
            batch_size=int(baseline_params.get("batch_size", 4096)),
        )

    fast_probs = _fast_baseline_probs(
        baseline=baseline,
        H_static=H_static,
        possible=possible,
        n_nodes=n_nodes,
        n_truth=n_truth,
        baseline_params=baseline_params,
        rng=rng,
    )
    if fast_probs is not None:
        return fast_probs

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
            chosen = int(rng.choice(n_nodes, p=p))
            probs[flat_idx, chosen] = 1.0
            continue

        if baseline == "soft_margin":
            kwargs = {
                k: v for k, v in baseline_params.items()
                if k in {"beta", "mu", "n_steps", "n_mc", "a_sq_candidates"}
            }
            probs[flat_idx] = _soft_margin(
                H_static=H_static,
                truth_S=S_snap,
                truth_I=I_snap,
                truth_R=R_snap,
                possible=poss,
                rng=rng,
                **kwargs,
            )
            continue

        if baseline == "mcs_mean_field":
            kwargs = {
                k: v for k, v in baseline_params.items()
                if k in {"beta", "mu", "n_steps", "n_mc"}
            }
            probs[flat_idx] = _mcs_mean_field(
                H_static=H_static,
                truth_S=S_snap,
                truth_I=I_snap,
                truth_R=R_snap,
                possible=poss,
                rng=rng,
                **kwargs,
            )
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

    _apply_overrides(cfg_dict, args.override)

    eval_cfg  = cfg_dict["eval"]
    baselines = cfg_dict["baselines"]
    n_truth   = eval_cfg["n_truth"]
    eval_reps = int(eval_cfg.get("reps", 1))
    n_eval_runs = n_truth * eval_reps
    seed = int(eval_cfg.get("seed", cfg_dict.get("seed", 0)))
    eval_cfg["seed"] = seed

    # -----------------------------------------------------------------------
    # 2. W&B initialisation
    # -----------------------------------------------------------------------
    setup_methods_run(job_type="eval")
    wandb.config.update({"data_name": args.data, **cfg_dict})
    wandb.run.tags += ("baselines",)
    print(f"\nW&B run : {wandb.run.url}")
    print(f"Data    : {args.data}\n")
    run_dir = f"data/{wandb.run.id}"
    os.makedirs(run_dir, exist_ok=True)

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

    select_truth = _truth_indices(eval_cfg, n_eval_runs=n_eval_runs, n_runs=data.n_runs)

    # Build static projection of H for topology-based baselines
    H_static = nx.Graph()
    H_static.add_nodes_from(range(n_nodes))
    for u, v in H.edges():
        H_static.add_edge(int(u), int(v))
    graph_metric_context = precompute_graph_metric_context(H_static, n_nodes)

    # Select the same held-out truth window used by model evaluation.
    truth_S = data.truth_S[:, select_truth, :]   # [n_nodes, n_eval_runs, n_nodes]
    truth_I = data.truth_I[:, select_truth, :]
    truth_R = data.truth_R[:, select_truth, :]
    possible = data.possible[:, select_truth, :]
    lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, n_nodes)

    truth_S_flat = truth_S.reshape(-1, n_nodes)
    sel = (1 - truth_S_flat).sum(axis=1) >= eval_cfg["min_outbreak"]
    print(f"  Valid outbreaks: {sel.sum()} / {len(sel)}\n")
    baseline_params = _baseline_param_map(cfg_dict, data.config)

    # -----------------------------------------------------------------------
    # 4. Evaluate each baseline
    # -----------------------------------------------------------------------
    summary_rows: list[dict] = []

    for baseline_idx, baseline in enumerate(baselines):
        print("=" * 60)
        print(f"Baseline: {baseline}")
        print("=" * 60)
        rng = np.random.default_rng(seed + baseline_idx)
        np.random.seed(seed + baseline_idx)
        params = dict(baseline_params.get(baseline, {}))
        params.setdefault("seed", seed + baseline_idx)
        if params:
            print(f"  Params: {params}")
        if baseline in BASELINE_METHOD_NOTES:
            print(f"  Method: {BASELINE_METHOD_NOTES[baseline]}")

        probs = compute_baseline_probs(
            baseline        = baseline,
            H_static        = H_static,
            truth_S         = truth_S,
            truth_I         = truth_I,
            truth_R         = truth_R,
            possible        = possible,
            n_nodes         = n_nodes,
            n_truth         = n_eval_runs,
            baseline_params = params,
            rng             = rng,
            mc_S            = data.mc_S,
            mc_I            = data.mc_I,
            mc_R            = data.mc_R,
        )   # [n_nodes * n_truth, n_nodes]

        metrics = compute_all_metrics(
            probs        = probs,
            lik_possible = lik_possible,
            truth_S_flat = truth_S_flat,
            eval_cfg     = eval_cfg,
            n_nodes      = n_nodes,
            n_runs       = n_eval_runs,
            graph_metric_context = graph_metric_context,
        )
        metrics["model"] = baseline

        top_k_vals = eval_cfg["top_k"]
        offsets    = eval_cfg["inverse_rank_offset"]
        n_valid    = int(metrics["eval/n_valid"])
        print(f"  Valid outbreaks: {n_valid} / {n_nodes * n_eval_runs}")
        print(f"  MRR           : {metrics['eval/mrr']:.4f}")
        for k in top_k_vals:
            print(f"  top-{k:<2}         : {100 * metrics[f'eval/top_{k}']:.1f}%")
        print(f"  Norm. Brier   : {metrics['eval/norm_brier']:.4f}")
        print(f"  Norm. Entropy : {metrics['eval/norm_entropy']:.4f}")

        # Save per-sample arrays for viz scripts
        arrays = per_sample_arrays(
            probs        = probs,
            lik_possible = lik_possible,
            truth_S_flat = truth_S_flat,
            eval_cfg     = eval_cfg,
            n_nodes      = n_nodes,
            n_runs       = n_eval_runs,
        )
        np.savez_compressed(
            f"{run_dir}/eval_arrays_{baseline}.npz",
            **arrays,
        )
        _write_json(
            f"{run_dir}/metrics_{baseline}.json",
            {
                "baseline": baseline,
                "data": args.data,
                "method": BASELINE_METHOD_NOTES.get(baseline, ""),
                "params": params,
                "metrics": {k: v for k, v in metrics.items() if k != "model"},
            },
        )

        # Log this baseline as its own W&B step (keyed by baseline name)
        wandb.log({
            **{k: v for k, v in metrics.items() if k != "model"},
            "baseline": baseline,
        })

        summary_rows.append(metrics)

        # Log per-baseline summary metrics so viz_karate_paper.py can fetch them
        for metric_key, val in metrics.items():
            if metric_key not in ("model", "eval/n_valid"):
                wandb.summary[f"{baseline}/{metric_key}"] = val

    # -----------------------------------------------------------------------
    # 5. Summary table
    # -----------------------------------------------------------------------
    top_k_vals = eval_cfg["top_k"]
    offsets    = eval_cfg["inverse_rank_offset"]
    credible_ps = eval_cfg.get("credible_p", [0.90])

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    col_keys = (
        ["model"]
        + [f"eval/mrr"]
        + [f"eval/top_{k}" for k in top_k_vals]
        + [f"eval/rank_score_off{o}" for o in offsets]
        + ["eval/norm_brier", "eval/norm_entropy"]
        + [f"eval/cred_cov_{int(round(p*100))}" for p in credible_ps]
    )
    header = (
        ["baseline", "MRR"]
        + [f"top_{k}" for k in top_k_vals]
        + [f"rs_off{o}" for o in offsets]
        + ["norm_brier", "norm_entropy"]
        + [f"cred_{int(round(p*100))}" for p in credible_ps]
    )
    print("  " + "  ".join(f"{h:>12}" for h in header))
    for row in summary_rows:
        vals = [row["model"]]
        for key in col_keys[1:]:
            v = row.get(key, float("nan"))
            if "top_" in key:
                vals.append(f"{100 * v:.1f}%")
            else:
                vals.append(f"{v:.4f}")
        print("  " + "  ".join(f"{v:>12}" for v in vals))

    # Log expanded comparison table to wandb
    table = wandb.Table(columns=header)
    for row in summary_rows:
        table_row = [row["model"]]
        for key in col_keys[1:]:
            table_row.append(row.get(key, float("nan")))
        table.add_data(*table_row)
    wandb.log({"baselines_comparison": table})

    wandb.summary["data/name"] = args.data
    wandb.summary["eval/n_truth_per_rep"] = n_truth
    wandb.summary["eval/reps"] = eval_reps
    wandb.summary["eval/truth_start"] = int(eval_cfg.get("truth_start", 0))
    wandb.summary["eval/truth_stop"] = int(select_truth[-1]) + 1 if len(select_truth) else int(eval_cfg.get("truth_start", 0))
    wandb.summary["baseline/method_notes"] = {
        b: BASELINE_METHOD_NOTES.get(b, "") for b in baselines
    }
    wandb.summary["n_valid_outbreaks"] = int(sel.sum())
    wandb.summary["n_total"] = len(sel)
    wandb.summary["run/status"] = "success"

    all_metric_keys = []
    for key in PREFERRED_METRIC_KEYS:
        if any(key in row for row in summary_rows):
            all_metric_keys.append(key)
    for row in summary_rows:
        for key in sorted(k for k in row if k.startswith("eval/")):
            if key not in all_metric_keys:
                all_metric_keys.append(key)

    _write_json(
        f"{run_dir}/baseline_metrics.json",
        {
            "status": "success",
            "data": args.data,
            "truth_start": int(eval_cfg.get("truth_start", 0)),
            "truth_stop": int(select_truth[-1]) + 1 if len(select_truth) else int(eval_cfg.get("truth_start", 0)),
            "baseline_method_notes": {
                b: BASELINE_METHOD_NOTES.get(b, "") for b in baselines
            },
            "baselines": summary_rows,
        },
    )
    with open(f"{run_dir}/baseline_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + all_metric_keys)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: row.get(key, "") for key in ["model"] + all_metric_keys})

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
