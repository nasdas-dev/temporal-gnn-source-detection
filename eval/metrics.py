"""
Centralised metric computation for source detection evaluation.

Both ``main_train.py`` (GNN models) and ``main_eval.py`` (baselines) call
``compute_all_metrics`` to produce a standardised, fully-populated metrics dict.
``per_sample_arrays`` returns the raw per-sample arrays needed by visualisation
scripts without requiring them to re-load large probability tensors.

Metric suite
------------
- MRR (Mean Reciprocal Rank) via ``eval/mrr``
- Top-k accuracy          via ``eval/top_{k}`` for each k in eval_cfg
- Rank score with offset  via ``eval/rank_score_off{o}`` for each offset
- Proper Brier score      via ``eval/brier``
- Normalised Brier score  via ``eval/norm_brier`` (divided by uniform baseline)
- Normalised entropy      via ``eval/norm_entropy``
- Credible set coverage   via ``eval/cred_cov_{p_int}`` for each p

By default, every metric follows the Sterchi et al. benchmark convention:
each method is scored on the infected subgraph, i.e. its probability vector is
restricted to feasible (non-susceptible) candidates and renormalised before
both ranking and calibration metrics are computed. This makes models that mask
susceptible nodes internally (e.g. BacktrackingNetwork, DBGNN) and those that
do not (StaticGNN, TemporalGNN) directly comparable. Set ``eval.rank_scope``
(or the backwards-compatible alias ``eval.ranking_scope``) to ``all_nodes`` to
score over every node without applying ``lik_possible``.
"""

from __future__ import annotations

import numpy as np

from .ranks import compute_ranks
from .scores import (
    credible_set,
    credible_set_size_mean,
    error_distance,
    logarithmic_score,
    normalized_entropy,
    proper_brier_score,
    rank_score,
    top_k_score,
)


def precompute_graph_metric_context(H_static, n_nodes: int) -> dict[str, np.ndarray]:
    """Precompute graph objects shared by every evaluated method.

    Baseline runs evaluate several predictors on the same static projection.
    Computing all-pairs distances and the resistance-distance matrix once keeps
    the metrics identical while avoiding repeated NetworkX and linear-algebra
    work for every baseline.
    """
    import networkx as nx

    dist_matrix = _shortest_path_distance_matrix(H_static, n_nodes)

    A = nx.to_numpy_array(H_static, nodelist=sorted(H_static.nodes()))
    D_deg = np.diag(A.sum(axis=1))
    L = D_deg - A
    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    omega = diag[:, None] + diag[None, :] - 2.0 * L_pinv

    return {
        "dist_matrix": dist_matrix,
        "omega": omega.astype(np.float64, copy=False),
    }


def _rng_from_eval_cfg(eval_cfg: dict) -> np.random.Generator:
    """Build a deterministic RNG for stochastic tie-breaking in metrics."""
    return np.random.default_rng(int(eval_cfg.get("seed", 0)))


def _rank_scope(eval_cfg: dict) -> str:
    """Return the configured rank scope normalised to an internal value."""
    raw = str(eval_cfg.get("rank_scope", eval_cfg.get("ranking_scope", "candidate")))
    scope = raw.lower().replace("-", "_")
    aliases = {
        "sterchi": "candidate",
        "feasible": "candidate",
        "possible": "candidate",
        "infected": "candidate",
        "infected_subgraph": "candidate",
        "outbreak": "candidate",
        "outbreak_subgraph": "candidate",
        "all": "all_nodes",
        "allnodes": "all_nodes",
        "unbiased": "all_nodes",
    }
    scope = aliases.get(scope, scope)
    if scope not in {"candidate", "all_nodes"}:
        raise ValueError(
            "eval.rank_scope must be one of 'candidate'/'sterchi' or "
            f"'all_nodes', got {raw!r}"
        )
    return scope


def _candidate_mask(lik_possible: np.ndarray) -> np.ndarray:
    """Return True for nodes allowed by the feasible-source mask."""
    return np.isfinite(lik_possible) & (lik_possible <= 0)


def _restrict_and_renormalize(
    probs: np.ndarray, candidate_mask: np.ndarray
) -> np.ndarray:
    """Project each probability row onto the feasible candidate set.

    Non-candidate (susceptible) nodes are zeroed and every row is renormalised
    to sum to one, so that all methods are scored on the infected subgraph
    exactly as in Sterchi et al. Rows with no probability mass on candidates
    fall back to a uniform distribution over their candidates; rows with no
    candidates at all (degenerate) fall back to uniform over all nodes.
    """
    probs_eval = probs.astype(np.float64, copy=True)
    probs_eval[~candidate_mask] = 0.0

    totals = probs_eval.sum(axis=1, keepdims=True)
    positive = totals[:, 0] > 0
    probs_eval[positive] /= totals[positive]

    zero_mass = ~positive
    if np.any(zero_mass):
        cand_counts = candidate_mask.sum(axis=1, keepdims=True)
        has_cand = zero_mass & (cand_counts[:, 0] > 0)
        if np.any(has_cand):
            probs_eval[has_cand] = (
                candidate_mask[has_cand].astype(np.float64)
                / cand_counts[has_cand]
            )
        no_cand = zero_mass & (cand_counts[:, 0] == 0)
        if np.any(no_cand):
            probs_eval[no_cand] = 1.0 / probs.shape[1]
    return probs_eval


def _prepare_eval_distribution(
    probs: np.ndarray, lik_possible: np.ndarray, eval_cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(probs_eval, rank_values)`` for the configured rank scope.

    With the default Sterchi-style ``candidate`` scope, ``probs_eval`` is the
    probability vector restricted to the infected subgraph and renormalised;
    it is used for *every* metric so masking models and non-masking models are
    compared on identical footing. ``rank_values`` carries ``-inf`` on
    non-candidate nodes so they always rank below feasible candidates. With
    ``all_nodes`` scope both arrays are the raw probabilities.
    """
    if _rank_scope(eval_cfg) == "candidate":
        cand = _candidate_mask(lik_possible)
        probs_eval = _restrict_and_renormalize(probs, cand)
        rank_values = np.where(cand, probs_eval, -np.inf)
        return probs_eval, rank_values
    probs_eval = probs.astype(np.float64, copy=False)
    return probs_eval, probs_eval


def _shortest_path_distance_matrix(H_static, n_nodes: int) -> np.ndarray:
    """Return graph distances, penalising disconnected pairs explicitly."""
    import networkx as nx

    dist_dict = dict(nx.all_pairs_shortest_path_length(H_static))
    finite_distances = [
        d
        for src_dists in dist_dict.values()
        for d in src_dists.values()
        if d > 0
    ]
    unreachable = (max(finite_distances) + 1) if finite_distances else n_nodes
    dist_matrix = np.full((n_nodes, n_nodes), float(unreachable), dtype=np.float64)
    np.fill_diagonal(dist_matrix, 0.0)
    for i in range(n_nodes):
        for j, d in dist_dict.get(i, {}).items():
            if 0 <= j < n_nodes:
                dist_matrix[i, j] = float(d)
    return dist_matrix


def per_sample_arrays(
    probs: np.ndarray,
    lik_possible: np.ndarray,
    truth_S_flat: np.ndarray,
    eval_cfg: dict,
    n_nodes: int,
    n_runs: int,
) -> dict[str, np.ndarray]:
    """Return per-sample arrays needed by visualisation scripts.

    Computes ranks and outbreak sizes once so that viz scripts can load
    lightweight ``.npz`` files instead of multi-GB probability tensors.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_nodes)
        Predicted probability distribution (non-negative, sums to 1).
    lik_possible : np.ndarray, shape (n_samples, n_nodes)
        Feasible-source log mask. With the default Sterchi-style rank scope,
        nodes with ``+inf`` in this array are excluded from rank-based metrics.
        Set ``eval.rank_scope: all_nodes`` to ignore it for ranking.
    truth_S_flat : np.ndarray, shape (n_samples, n_nodes), int8
        Susceptible-state matrix; row ``s * n_runs + r`` corresponds to
        source *s*, run *r*.
    eval_cfg : dict
        Eval section of the experiment YAML.  Must contain ``min_outbreak``.
    n_nodes : int
        Number of nodes in the network.
    n_runs : int
        Number of evaluation runs per source node (n_truth).

    Returns
    -------
    dict with numpy arrays:

    ``ranks``          — int [n_samples], 1-indexed rank of the true source
    ``outbreak_sizes`` — float32 [n_samples], fraction of infected nodes
    ``sel``            — bool [n_samples], valid outbreak mask
    ``true_sources``   — int [n_samples], true source node index per sample
    """
    min_outbreak = eval_cfg["min_outbreak"]

    infected_counts = (1 - truth_S_flat).sum(axis=1)
    sel = infected_counts >= min_outbreak
    outbreak_sizes = (infected_counts / n_nodes).astype(np.float32)

    # true source for row s * n_runs + r is node s
    true_sources = np.repeat(np.arange(n_nodes), n_runs)

    rng = _rng_from_eval_cfg(eval_cfg)

    # Sterchi-style evaluation: every method is scored on the infected subgraph.
    # ``rank_values`` ranks the true source among feasible candidates only.
    _, rank_values = _prepare_eval_distribution(probs, lik_possible, eval_cfg)
    ranks = compute_ranks(rank_values, n_nodes=n_nodes, n_runs=n_runs, rng=rng)

    return {
        "ranks":          ranks,
        "outbreak_sizes": outbreak_sizes,
        "sel":            sel,
        "true_sources":   true_sources,
    }


def compute_all_metrics(
    probs: np.ndarray,
    lik_possible: np.ndarray,
    truth_S_flat: np.ndarray,
    eval_cfg: dict,
    n_nodes: int,
    n_runs: int,
    H_static=None,          # nx.Graph | None — optional, enables graph metrics
    graph_metric_context: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute the full evaluation metric suite for one set of predictions.

    Wraps ``per_sample_arrays`` and then applies all scoring functions from
    ``eval/scores.py``.  The returned dict can be passed directly to
    ``wandb.log`` or ``wandb.summary``.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_nodes)
        Predicted probability distribution (not log-probs).  Values must be
        non-negative and sum to 1 over axis 1.
    lik_possible : np.ndarray, shape (n_samples, n_nodes)
        Feasible-source log mask. By default it restricts every metric to the
        infected subgraph (Sterchi-style candidate evaluation): probabilities
        are renormalised over feasible candidates before scoring.
        Set ``eval.rank_scope: all_nodes`` for strict all-node evaluation.
    truth_S_flat : np.ndarray, shape (n_samples, n_nodes), int8
        Susceptible-state matrix from TSIR simulation.
    eval_cfg : dict
        Eval section of the experiment YAML.  Expected keys:

        - ``min_outbreak``         — int, minimum infected nodes for valid run
        - ``top_k``                — list[int], k values for top-k accuracy
        - ``inverse_rank_offset``  — list[int], offsets for rank score
        - ``credible_p``           — list[float], optional, default [0.90]
        - ``rank_scope``           — ``candidate`` (default) or ``all_nodes``
    n_nodes : int
        Number of nodes in the network.
    n_runs : int
        Number of evaluation runs per source (n_truth).

    Returns
    -------
    dict[str, float]
        Flat metrics dict.  Keys:

        - ``eval/mrr``                  — Mean Reciprocal Rank (= rank_score at offset=0)
        - ``eval/top_{k}``              — top-k accuracy for each configured k
        - ``eval/rank_score_off{o}``    — rank score with offset o
        - ``eval/brier``                — proper Brier score (lower is better)
        - ``eval/norm_brier``           — Brier / Brier(uniform), 1.0 = uniform baseline
        - ``eval/norm_entropy``         — entropy / log(n_nodes), 0..1
        - ``eval/cred_cov_{p_int}``     — credible set coverage at level p
        - ``eval/n_valid``              — number of valid (non-trivial) outbreaks
    """
    arrays = per_sample_arrays(
        probs, lik_possible, truth_S_flat, eval_cfg, n_nodes, n_runs
    )
    ranks        = arrays["ranks"]
    sel          = arrays["sel"]
    true_sources = arrays["true_sources"]

    # Score every method on the same distribution the ranks use: with the
    # default candidate scope this restricts probabilities to the infected
    # subgraph and renormalises, so calibration metrics (Brier, entropy,
    # credible set, resistance, log-score) are comparable across models that
    # mask susceptible nodes internally and those that do not.
    probs_eval, _ = _prepare_eval_distribution(probs, lik_possible, eval_cfg)

    top_k_vals  = eval_cfg["top_k"]
    offsets     = eval_cfg["inverse_rank_offset"]
    credible_ps = eval_cfg.get("credible_p", [0.90])

    metrics: dict[str, float] = {}
    n_total = float(len(sel))
    n_valid = float(sel.sum())

    # --- Rank-based ---
    metrics["eval/mrr"] = float(rank_score(ranks, sel, offset=0))
    metrics["eval/mean_rank"] = float(np.mean(ranks[sel])) if np.any(sel) else float("nan")
    metrics["eval/median_rank"] = float(np.median(ranks[sel])) if np.any(sel) else float("nan")
    for k in top_k_vals:
        metrics[f"eval/top_{k}"] = float(top_k_score(ranks, sel, k))
    for o in offsets:
        metrics[f"eval/rank_score_off{o}"] = float(rank_score(ranks, sel, o))

    # --- Calibration: Brier score ---
    brier_raw = float(proper_brier_score(probs_eval, true_sources, n_nodes, sel))
    metrics["eval/brier"] = brier_raw
    # Normalise: uniform predictor baseline = (n_nodes - 1) / n_nodes
    brier_uniform = (n_nodes - 1) / n_nodes
    metrics["eval/norm_brier"] = brier_raw / brier_uniform if brier_uniform > 0 else float("nan")

    log_raw = float(logarithmic_score(probs_eval, true_sources, sel))
    metrics["eval/log_score"] = log_raw
    metrics["eval/norm_log_score"] = log_raw / np.log(n_nodes) if n_nodes > 1 else float("nan")

    if np.any(sel):
        idx = np.arange(len(true_sources))
        true_probs = probs_eval[idx, true_sources.astype(int)]
        metrics["eval/mean_true_prob"] = float(np.mean(true_probs[sel]))
        metrics["eval/map_confidence"] = float(np.mean(np.max(probs_eval[sel], axis=1)))
    else:
        metrics["eval/mean_true_prob"] = float("nan")
        metrics["eval/map_confidence"] = float("nan")

    # --- Calibration: entropy ---
    metrics["eval/norm_entropy"] = float(normalized_entropy(probs_eval, n_nodes, sel))

    # --- Credible set coverage ---
    for p in credible_ps:
        p_int = int(round(p * 100))
        metrics[f"eval/cred_cov_{p_int}"] = float(
            credible_set(probs_eval, sel, p, n_nodes, n_runs)
        )
        metrics[f"eval/cred_set_size_{p_int}"] = float(
            credible_set_size_mean(probs_eval, sel, p)
        )

    metrics["eval/n_valid"] = n_valid
    metrics["eval/n_total"] = n_total
    metrics["eval/valid_frac"] = n_valid / n_total if n_total > 0 else float("nan")
    candidate_counts = _candidate_mask(lik_possible).sum(axis=1)
    metrics["eval/rank_scope_candidate"] = 1.0 if _rank_scope(eval_cfg) == "candidate" else 0.0
    metrics["eval/mean_candidate_count"] = float(np.mean(candidate_counts[sel])) if np.any(sel) else float("nan")
    metrics["eval/median_candidate_count"] = float(np.median(candidate_counts[sel])) if np.any(sel) else float("nan")

    if H_static is not None or graph_metric_context is not None:
        if graph_metric_context is None:
            graph_metric_context = precompute_graph_metric_context(H_static, n_nodes)

        # Error distance (MAP prediction vs true source)
        dist_matrix = graph_metric_context["dist_matrix"]
        metrics["eval/error_dist"] = float(
            error_distance(probs_eval, true_sources, dist_matrix, sel, rng=_rng_from_eval_cfg(eval_cfg))
        )

        # Resistance distance scoring rule: S(p,i) = (Ω@p)[i] - 0.5 p^T Ω p
        Omega = graph_metric_context["omega"]
        probs_valid = probs_eval[sel].astype(np.float64)
        true_src_valid = true_sources[sel].astype(int)
        n_valid_int = len(probs_valid)
        if n_valid_int:
            chunk_size = int(eval_cfg.get("graph_metric_chunk_size", 4096))
            total = 0.0
            for start in range(0, n_valid_int, chunk_size):
                end = min(start + chunk_size, n_valid_int)
                chunk = probs_valid[start:end]
                omega_p = chunk @ Omega
                idx = np.arange(end - start)
                expected_res = omega_p[idx, true_src_valid[start:end]]
                regularisation = 0.5 * np.sum(omega_p * chunk, axis=1)
                total += float(np.sum(expected_res - regularisation))
            metrics["eval/resistance"] = total / n_valid_int
        else:
            metrics["eval/resistance"] = float("nan")

    return metrics
