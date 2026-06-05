from __future__ import annotations

import numpy as np


def _true_source_indices(n_nodes: int, n_runs: int) -> np.ndarray:
    """Return row-aligned true source indices for ``source * n_runs + run``."""
    return np.repeat(np.arange(n_nodes), n_runs)


def compute_ranks(
    values: np.ndarray,
    n_nodes: int,
    n_runs: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute 1-indexed ranks with reproducible random tie-breaking.

    ``values`` can be log-likelihoods, source probabilities, or any other
    node score matrix with shape ``(n_nodes * n_runs, n_nodes)``. If several
    nodes tie with the true source, one of the tied ranks is sampled uniformly.
    Passing an explicit RNG makes uniform/random-baseline metrics repeatable.
    """
    if rng is None:
        rng = np.random.default_rng()

    true_sources = _true_source_indices(n_nodes, n_runs)
    source_values = values[np.arange(n_nodes * n_runs), true_sources]
    strictly_better = np.sum(values > source_values[:, None], axis=1)
    ties = np.sum(values == source_values[:, None], axis=1)
    tie_offsets = rng.integers(0, ties)
    return strictly_better + tie_offsets + 1


def compute_expected_ranks(values: np.ndarray, n_nodes: int, n_runs: int) -> np.ndarray:
    """Compute 1-indexed average ranks under uniform tie-breaking."""
    true_sources = _true_source_indices(n_nodes, n_runs)
    source_values = values[np.arange(n_nodes * n_runs), true_sources]
    strictly_better = np.sum(values > source_values[:, None], axis=1)
    ties = np.sum(values == source_values[:, None], axis=1)
    return strictly_better + (ties + 1) / 2
