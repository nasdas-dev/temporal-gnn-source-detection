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

The "possible" filter (``lik_possible``) is applied *only* for ranking —
impossible nodes are ranked last via ``-inf`` log-probability.  Brier and
entropy metrics use the raw predicted probabilities.
"""

from __future__ import annotations

import numpy as np

from .ranks import compute_ranks
from .scores import (
    credible_set,
    credible_set_size_mean,
    error_distance,
    normalized_entropy,
    proper_brier_score,
    rank_score,
    top_k_score,
)


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
        Masking array — ``0`` for possible source nodes, ``np.inf`` for
        impossible ones.  Subtracted from log-probs before ranking.
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

    # Apply lik_possible masking before ranking (impossible nodes → -inf)
    log_probs = np.log(np.clip(probs, 1e-12, 1.0)) - lik_possible
    ranks = compute_ranks(log_probs, n_nodes=n_nodes, n_runs=n_runs)

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
        Masking array — ``0`` for possible source nodes, ``np.inf`` for
        impossible ones.  Used for ranking only; does not affect calibration
        metrics (Brier, entropy).
    truth_S_flat : np.ndarray, shape (n_samples, n_nodes), int8
        Susceptible-state matrix from TSIR simulation.
    eval_cfg : dict
        Eval section of the experiment YAML.  Expected keys:

        - ``min_outbreak``         — int, minimum infected nodes for valid run
        - ``top_k``                — list[int], k values for top-k accuracy
        - ``inverse_rank_offset``  — list[int], offsets for rank score
        - ``credible_p``           — list[float], optional, default [0.90]
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

    top_k_vals  = eval_cfg["top_k"]
    offsets     = eval_cfg["inverse_rank_offset"]
    credible_ps = eval_cfg.get("credible_p", [0.90])

    metrics: dict[str, float] = {}

    # --- Rank-based ---
    metrics["eval/mrr"] = float(rank_score(ranks, sel, offset=0))
    for k in top_k_vals:
        metrics[f"eval/top_{k}"] = float(top_k_score(ranks, sel, k))
    for o in offsets:
        metrics[f"eval/rank_score_off{o}"] = float(rank_score(ranks, sel, o))

    # --- Calibration: Brier score ---
    brier_raw = float(proper_brier_score(probs, true_sources, n_nodes, sel))
    metrics["eval/brier"] = brier_raw
    # Normalise: uniform predictor baseline = (n_nodes - 1) / n_nodes
    brier_uniform = (n_nodes - 1) / n_nodes
    metrics["eval/norm_brier"] = brier_raw / brier_uniform if brier_uniform > 0 else float("nan")

    # --- Calibration: entropy ---
    metrics["eval/norm_entropy"] = float(normalized_entropy(probs, n_nodes, sel))

    # --- Credible set coverage ---
    for p in credible_ps:
        p_int = int(round(p * 100))
        metrics[f"eval/cred_cov_{p_int}"] = float(
            credible_set(probs, sel, p, n_nodes, n_runs)
        )

    metrics["eval/n_valid"] = float(sel.sum())

    if H_static is not None:
        import networkx as nx

        # All-pairs shortest path → distance matrix
        dist_dict = dict(nx.all_pairs_shortest_path_length(H_static))
        dist_matrix = np.array(
            [[dist_dict.get(i, {}).get(j, 0) for j in range(n_nodes)] for i in range(n_nodes)],
            dtype=np.float64,
        )

        # Error distance (MAP prediction vs true source)
        metrics["eval/error_dist"] = float(
            error_distance(probs, true_sources, dist_matrix, sel)
        )

        # Credible set size (mean number of nodes) for each configured p
        for p in credible_ps:
            p_int = int(round(p * 100))
            metrics[f"eval/cred_set_size_{p_int}"] = float(
                credible_set_size_mean(probs, sel, p)
            )

        # Resistance distance scoring rule: S(p,i) = (Ω@p)[i] - 0.5 p^T Ω p
        A = nx.to_numpy_array(H_static, nodelist=sorted(H_static.nodes()))
        D_deg = np.diag(A.sum(axis=1))
        L = D_deg - A
        L_pinv = np.linalg.pinv(L)
        diag = np.diag(L_pinv)
        Omega = diag[:, None] + diag[None, :] - 2.0 * L_pinv

        probs_valid = probs[sel].astype(np.float64)
        true_src_valid = true_sources[sel].astype(int)
        n_valid_int = len(probs_valid)
        OmegaP = Omega @ probs_valid.T               # [n_nodes, n_valid]
        expected_res = OmegaP[true_src_valid, np.arange(n_valid_int)]
        regularisation = 0.5 * np.sum((probs_valid @ Omega) * probs_valid, axis=1)
        metrics["eval/resistance"] = float(np.mean(expected_res - regularisation))

    return metrics
