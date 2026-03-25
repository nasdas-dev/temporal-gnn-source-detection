import numpy as np

def compute_ranks(values, n_nodes, n_runs):
    # values can be the log-likelihood, source probabilities, or some other values
    source_values = values[np.arange(n_nodes * n_runs), np.repeat(np.arange(n_nodes), n_runs)]
    rank = np.sum(values > source_values[:, None], axis=1)
    # deal with ties
    ties = np.sum(values == source_values[:, None], axis=1)
    rank = rank + np.random.randint(ties) + 1
    return rank

def compute_expected_ranks(values, n_nodes, n_runs):
    # values can be the log-likelihood, source probabilities, or some other values
    source_values = values[np.arange(n_nodes * n_runs), np.repeat(np.arange(n_nodes), n_runs)]
    rank = np.sum(values > source_values[:, None], axis=1) + (np.sum(values == source_values[:, None], axis=1) + 1) / 2
    return rank
