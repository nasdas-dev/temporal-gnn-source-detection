from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from main_eval import _truth_indices, compute_baseline_probs


def _sir_from_possible(possible: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build simple S/I/R tensors whose non-susceptible mask is ``possible``."""
    truth_S = (1 - possible).astype(np.int8)
    truth_I = possible.astype(np.int8)
    truth_R = np.zeros_like(truth_S, dtype=np.int8)
    return truth_S, truth_I, truth_R


def test_degree_baseline_is_exact_observed_subgraph_degree():
    H = nx.path_graph(3)
    possible = np.array(
        [
            [[1, 1, 0], [1, 1, 1]],
            [[0, 1, 1], [1, 0, 1]],
            [[1, 1, 1], [0, 0, 1]],
        ],
        dtype=np.int8,
    )
    truth_S, truth_I, truth_R = _sir_from_possible(possible)

    probs = compute_baseline_probs(
        "degree",
        H,
        truth_S,
        truth_I,
        truth_R,
        possible,
        n_nodes=3,
        n_truth=2,
        baseline_params={"chunk_size": 2},
        rng=np.random.default_rng(0),
    )

    expected = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.25, 0.5, 0.25],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.25, 0.5, 0.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(probs, expected, atol=1e-7)


def test_random_baseline_is_seeded_and_feasible():
    H = nx.path_graph(4)
    possible = np.array([[[1, 0, 1, 0], [0, 1, 0, 1]]] * 4, dtype=np.int8)
    truth_S, truth_I, truth_R = _sir_from_possible(possible)

    kwargs = dict(
        baseline="random",
        H_static=H,
        truth_S=truth_S,
        truth_I=truth_I,
        truth_R=truth_R,
        possible=possible,
        n_nodes=4,
        n_truth=2,
        baseline_params={"chunk_size": 3},
    )
    probs_a = compute_baseline_probs(**kwargs, rng=np.random.default_rng(123))
    probs_b = compute_baseline_probs(**kwargs, rng=np.random.default_rng(123))

    np.testing.assert_array_equal(probs_a, probs_b)
    assert np.allclose(probs_a.sum(axis=1), 1.0)
    assert np.all((probs_a == 0.0) | (possible.reshape(-1, 4) == 1))


def test_static_centrality_baseline_masks_to_feasible_candidates():
    H = nx.path_graph(4)
    possible = np.array([[[1, 1, 0, 0], [0, 1, 1, 1]]] * 4, dtype=np.int8)
    truth_S, truth_I, truth_R = _sir_from_possible(possible)

    probs = compute_baseline_probs(
        "closeness",
        H,
        truth_S,
        truth_I,
        truth_R,
        possible,
        n_nodes=4,
        n_truth=2,
        baseline_params={"chunk_size": 2},
        rng=np.random.default_rng(0),
    )

    closeness = nx.closeness_centrality(H)
    scores = np.array([closeness[i] for i in range(4)], dtype=np.float32)
    expected_first = scores * np.array([1, 1, 0, 0], dtype=np.float32)
    expected_first /= expected_first.sum()
    np.testing.assert_allclose(probs[0], expected_first, atol=1e-7)
    assert np.all(probs[possible.reshape(-1, 4) == 0] == 0.0)


def test_jordan_center_uses_infected_subgraph():
    # jordan_center is computed on the per-outbreak *infected subgraph*, not the
    # full static graph (see compute_baseline_probs G_sub path).
    H = nx.path_graph(3)
    possible = np.array([[[1, 1, 1], [1, 0, 1]]] * 3, dtype=np.int8)
    truth_S, truth_I, truth_R = _sir_from_possible(possible)

    probs = compute_baseline_probs(
        "jordan_center",
        H,
        truth_S,
        truth_I,
        truth_R,
        possible,
        n_nodes=3,
        n_truth=2,
        baseline_params={"chunk_size": 2},
        rng=np.random.default_rng(0),
    )

    # Obs 0: the infected subgraph is the connected path {0,1,2} -> centre node 1.
    assert probs[0, 1] > probs[0, 0]
    assert probs[0, 1] > probs[0, 2]
    # Obs 1: node 1 is susceptible, so the infected subgraph {0,2} is two isolated
    # nodes. Jordan center is only defined on a connected graph, so the
    # implementation restricts to the largest connected component (ties broken by
    # iteration order), concentrating all mass on a single component centre.
    # The uninfected node 1 gets zero probability.
    assert probs[1, 1] == 0.0
    assert np.isclose(probs[1].sum(), 1.0)
    assert np.isclose(probs[1].max(), 1.0)


def test_eval_truth_indices_respect_truth_start():
    indices = _truth_indices({"truth_start": 4}, n_eval_runs=5, n_runs=20)
    np.testing.assert_array_equal(indices, np.array([4, 5, 6, 7, 8]))

    with pytest.raises(ValueError, match="truth_start"):
        _truth_indices({"truth_start": -1}, n_eval_runs=5, n_runs=20)

    with pytest.raises(ValueError, match="exceeds"):
        _truth_indices({"truth_start": 18}, n_eval_runs=5, n_runs=20)
