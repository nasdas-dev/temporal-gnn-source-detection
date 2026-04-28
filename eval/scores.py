import numpy as np
from typing import Optional

def rank_score(ranks, sel, offset = 0):
    return np.mean((1 + offset) / (ranks[sel] + offset))

def top_k_score(ranks, sel, k=5):
    return np.mean(ranks[sel] <= k)

def normalized_brier_score(states, probs, n_nodes, sel):
    brier_scores = np.mean((states - probs) ** 2, axis=1)
    normalized_score = brier_scores / (2 / n_nodes)
    return np.mean(normalized_score[sel])

def normalized_entropy(probs, n_nodes, sel):
    log_probs = np.log(probs)
    log_probs[np.isneginf(log_probs)] = 0
    entropy = -np.sum(probs * log_probs, axis=1)
    normalized_entropy = entropy / np.log(n_nodes)
    return np.mean(normalized_entropy[sel])

def credible_set(probs, sel, p, n_nodes, n_runs):
    """Computes the credible set evaluation metric, but penalizes if credible sets are very large."""
    sorted_indices = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=1)
    cumulative_probs = np.cumsum(sorted_probs, axis=1)
    credible_set_sizes = np.argmax(cumulative_probs >= p, axis=1) + 1
    true_source_indeces = np.repeat(np.arange(n_nodes), n_runs)
    positions = (sorted_indices == true_source_indeces[:, None]).argmax(axis=1)
    return np.mean((positions < credible_set_sizes)[sel])


def credible_set_size_mean(probs: np.ndarray, sel: np.ndarray, p: float) -> float:
    """Mean number of nodes in the smallest set accumulating probability mass >= p."""
    sorted_probs = -np.sort(-probs, axis=1)   # descending
    sizes = np.argmax(np.cumsum(sorted_probs, axis=1) >= p, axis=1) + 1
    return float(np.mean(sizes[sel]))


# Ported from gnn/static_source_detection_gnn/sourcedet/evaluate.py
def error_distance(
    probs: np.ndarray,
    true_sources: np.ndarray,
    distances: np.ndarray,
    sel: np.ndarray,
) -> float:
    """Compute the mean error distance between the predicted top-1 node and the true source.

    For each valid sample the top-1 predicted node is found (ties broken uniformly
    at random) and the precomputed shortest-path distance to the true source is
    looked up.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_nodes)
        Predicted probability distribution over nodes for each sample.
    true_sources : np.ndarray, shape (n_samples,)
        Index of the true source node for each sample.
    distances : np.ndarray, shape (n_nodes, n_nodes)
        Precomputed all-pairs shortest-path distances.
    sel : np.ndarray, shape (n_samples,)
        Boolean mask selecting valid outbreaks.

    Returns
    -------
    float
        Mean error distance over all valid samples.
    """
    probs_sel = probs[sel]
    true_sources_sel = true_sources[sel]

    error_dists = np.empty(len(probs_sel), dtype=np.float64)
    for i in range(len(probs_sel)):
        p = probs_sel[i]
        max_val = np.nanmax(p)
        # All nodes achieving the maximum probability (handles ties)
        max_nodes = np.where(p == max_val)[0]
        predicted = int(np.random.choice(max_nodes))
        src = int(true_sources_sel[i])
        error_dists[i] = distances[src, predicted]

    return float(np.mean(error_dists))


# Ported from gnn/static_source_detection_gnn/propnetscore/node_selection.py
def proper_brier_score(
    probs: np.ndarray,
    true_sources: np.ndarray,
    n_nodes: int,
    sel: np.ndarray,
) -> float:
    """Compute the proper Brier scoring rule averaged over valid samples.

    Computes ||p - e_i||^2 where e_i is the one-hot vector at the true source
    index, then averages over all valid samples.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_nodes)
        Predicted probability distribution over nodes.
    true_sources : np.ndarray, shape (n_samples,)
        Index of the true source node for each sample.
    n_nodes : int
        Total number of nodes.
    sel : np.ndarray, shape (n_samples,)
        Boolean mask selecting valid outbreaks.

    Returns
    -------
    float
        Mean proper Brier score over valid samples.
    """
    probs_sel = probs[sel]
    true_sources_sel = true_sources[sel].astype(int)

    n_sel = len(probs_sel)
    # Build one-hot matrix for all samples at once
    one_hot = np.zeros((n_sel, n_nodes), dtype=np.float64)
    one_hot[np.arange(n_sel), true_sources_sel] = 1.0

    residuals = one_hot - probs_sel.astype(np.float64)
    brier_scores = np.sum(residuals ** 2, axis=1)
    return float(np.mean(brier_scores))


# Ported from gnn/static_source_detection_gnn/propnetscore/node_selection.py
def logarithmic_score(
    probs: np.ndarray,
    true_sources: np.ndarray,
    sel: np.ndarray,
) -> float:
    """Compute the proper logarithmic scoring rule averaged over valid samples.

    Returns the mean of -log(p_i) where i is the true source index.  A lower
    value indicates better calibration.

    Parameters
    ----------
    probs : np.ndarray, shape (n_samples, n_nodes)
        Predicted probability distribution over nodes.
    true_sources : np.ndarray, shape (n_samples,)
        Index of the true source node for each sample.
    sel : np.ndarray, shape (n_samples,)
        Boolean mask selecting valid outbreaks.

    Returns
    -------
    float
        Mean negative log-probability of the true source over valid samples.
    """
    probs_sel = probs[sel]
    true_sources_sel = true_sources[sel].astype(int)

    # Look up the predicted probability for the true source of each sample
    p_true = probs_sel[np.arange(len(probs_sel)), true_sources_sel]
    # Clip to avoid log(0)
    p_true = np.clip(p_true, 1e-12, 1.0)
    return float(np.mean(-np.log(p_true)))
