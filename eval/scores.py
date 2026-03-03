import numpy as np

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
