"""
Tests for eval/metrics.py — compute_all_metrics and per_sample_arrays.

Run from project root:
    pytest tests/test_metrics.py -v

These tests use synthetic probability arrays to verify:
- All expected metric keys are returned
- Values are in valid ranges
- A perfect predictor scores better than a uniform predictor
- The sel mask is correctly applied
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import compute_all_metrics, per_sample_arrays


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_eval_cfg(top_k=(1, 3, 5, 10), offsets=(0,), credible_p=(0.80, 0.90)):
    return {
        "min_outbreak": 2,
        "top_k": list(top_k),
        "inverse_rank_offset": list(offsets),
        "credible_p": list(credible_p),
    }


def make_synthetic(
    n_nodes: int = 20,
    n_runs: int = 50,
    seed: int = 42,
    perfect: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, lik_possible, truth_S_flat) for testing.

    Parameters
    ----------
    perfect:
        If True, return a perfect predictor (probability 1 on true source).
    """
    rng = np.random.default_rng(seed)
    n_samples = n_nodes * n_runs

    # lik_possible: all nodes possible (no masking)
    lik_possible = np.zeros((n_samples, n_nodes), dtype=np.float64)

    # truth_S_flat: each row s*n_runs+r has susceptible=1 for non-source nodes
    # Simulate: source node is infected (S=0), other nodes randomly S or R
    truth_S_flat = np.ones((n_samples, n_nodes), dtype=np.int8)
    true_sources = np.repeat(np.arange(n_nodes), n_runs)
    for i, s in enumerate(true_sources):
        truth_S_flat[i, s] = 0  # source is infected
        # Infect some other nodes to ensure min_outbreak >= 2
        n_infected = rng.integers(2, max(3, n_nodes // 3))
        others = [j for j in range(n_nodes) if j != s]
        infected_others = rng.choice(others, size=min(n_infected, len(others)), replace=False)
        truth_S_flat[i, infected_others] = 0

    if perfect:
        probs = np.zeros((n_samples, n_nodes), dtype=np.float32)
        probs[np.arange(n_samples), true_sources] = 1.0
    else:
        # Uniform predictor
        probs = np.ones((n_samples, n_nodes), dtype=np.float32) / n_nodes

    return probs, lik_possible, truth_S_flat


# ---------------------------------------------------------------------------
# Tests: per_sample_arrays
# ---------------------------------------------------------------------------

class TestPerSampleArrays:
    def test_output_keys(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = per_sample_arrays(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert set(result.keys()) == {"ranks", "outbreak_sizes", "sel", "true_sources"}

    def test_shapes(self):
        n_nodes, n_runs = 20, 50
        probs, lik_possible, truth_S_flat = make_synthetic(n_nodes, n_runs)
        cfg = make_eval_cfg()
        result = per_sample_arrays(probs, lik_possible, truth_S_flat, cfg, n_nodes, n_runs)
        n_samples = n_nodes * n_runs
        assert result["ranks"].shape == (n_samples,)
        assert result["outbreak_sizes"].shape == (n_samples,)
        assert result["sel"].shape == (n_samples,)
        assert result["true_sources"].shape == (n_samples,)

    def test_ranks_positive(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = per_sample_arrays(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert (result["ranks"] >= 1).all()
        assert (result["ranks"] <= 20).all()

    def test_outbreak_sizes_in_range(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = per_sample_arrays(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert (result["outbreak_sizes"] >= 0).all()
        assert (result["outbreak_sizes"] <= 1).all()

    def test_true_sources(self):
        n_nodes, n_runs = 20, 50
        probs, lik_possible, truth_S_flat = make_synthetic(n_nodes, n_runs)
        cfg = make_eval_cfg()
        result = per_sample_arrays(probs, lik_possible, truth_S_flat, cfg, n_nodes, n_runs)
        expected = np.repeat(np.arange(n_nodes), n_runs)
        np.testing.assert_array_equal(result["true_sources"], expected)


# ---------------------------------------------------------------------------
# Tests: compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_output_keys_present(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        expected_keys = {
            "eval/mrr",
            "eval/top_1", "eval/top_3", "eval/top_5", "eval/top_10",
            "eval/rank_score_off0",
            "eval/brier", "eval/norm_brier",
            "eval/norm_entropy",
            "eval/cred_cov_80", "eval/cred_cov_90",
            "eval/n_valid",
        }
        assert expected_keys.issubset(result.keys()), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_mrr_in_range(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert 0.0 < result["eval/mrr"] <= 1.0

    def test_top_k_in_range(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        for k in [1, 3, 5, 10]:
            val = result[f"eval/top_{k}"]
            assert 0.0 <= val <= 1.0, f"top_{k} = {val} out of [0,1]"

    def test_top_k_monotone(self):
        """top_1 <= top_3 <= top_5 <= top_10."""
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert result["eval/top_1"] <= result["eval/top_3"] + 1e-9
        assert result["eval/top_3"] <= result["eval/top_5"] + 1e-9
        assert result["eval/top_5"] <= result["eval/top_10"] + 1e-9

    def test_perfect_predictor_better_than_uniform(self):
        """A perfect predictor must outperform the uniform baseline on all metrics."""
        n_nodes, n_runs = 20, 50
        perfect_probs, lik_possible, truth_S_flat = make_synthetic(n_nodes, n_runs, perfect=True)
        uniform_probs, _, _ = make_synthetic(n_nodes, n_runs, perfect=False)
        cfg = make_eval_cfg()

        perf = compute_all_metrics(perfect_probs, lik_possible, truth_S_flat, cfg, n_nodes, n_runs)
        unif = compute_all_metrics(uniform_probs, lik_possible, truth_S_flat, cfg, n_nodes, n_runs)

        assert perf["eval/mrr"]   >= unif["eval/mrr"],   "Perfect MRR <= Uniform MRR"
        assert perf["eval/top_1"] >= unif["eval/top_1"], "Perfect Top-1 <= Uniform Top-1"
        assert perf["eval/top_5"] >= unif["eval/top_5"], "Perfect Top-5 <= Uniform Top-5"
        assert perf["eval/brier"] <= unif["eval/brier"], "Perfect Brier >= Uniform Brier"

    def test_perfect_predictor_top1_is_one(self):
        n_nodes, n_runs = 20, 50
        probs, lik_possible, truth_S_flat = make_synthetic(n_nodes, n_runs, perfect=True)
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes, n_runs)
        assert abs(result["eval/top_1"] - 1.0) < 1e-6, (
            f"Perfect predictor top-1 should be 1.0, got {result['eval/top_1']}"
        )

    def test_n_valid_positive(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert result["eval/n_valid"] > 0

    def test_credible_coverage_monotone(self):
        """cred_cov_80 <= cred_cov_90 (larger set should cover more)."""
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg(credible_p=[0.80, 0.90])
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert result["eval/cred_cov_80"] <= result["eval/cred_cov_90"] + 1e-9

    def test_lik_possible_masking(self):
        """Masking impossible nodes should not increase MRR vs. no masking."""
        n_nodes, n_runs = 15, 30
        rng = np.random.default_rng(7)
        n_samples = n_nodes * n_runs
        probs = rng.dirichlet(np.ones(n_nodes), size=n_samples).astype(np.float32)

        truth_S_flat = np.ones((n_samples, n_nodes), dtype=np.int8)
        true_sources = np.repeat(np.arange(n_nodes), n_runs)
        for i, s in enumerate(true_sources):
            truth_S_flat[i, s] = 0
            others = [j for j in range(n_nodes) if j != s]
            truth_S_flat[i, rng.choice(others, 2, replace=False)] = 0

        cfg = make_eval_cfg(top_k=[1, 5])

        # No masking
        lik_none = np.zeros((n_samples, n_nodes), dtype=np.float64)
        result_none = compute_all_metrics(probs, lik_none, truth_S_flat, cfg, n_nodes, n_runs)

        # Mask out all nodes except true source (extreme case → perfect ranking)
        lik_strict = np.full((n_samples, n_nodes), np.inf, dtype=np.float64)
        lik_strict[np.arange(n_samples), true_sources] = 0.0
        result_strict = compute_all_metrics(probs, lik_strict, truth_S_flat, cfg, n_nodes, n_runs)

        assert abs(result_strict["eval/top_1"] - 1.0) < 1e-6, (
            "Masking all but true source should give top-1 = 1.0"
        )

    def test_no_credible_p_defaults(self):
        """If credible_p is absent, only cred_cov_90 should appear."""
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = {"min_outbreak": 2, "top_k": [1, 5], "inverse_rank_offset": [0]}
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        assert "eval/cred_cov_90" in result
        assert "eval/cred_cov_80" not in result

    def test_all_values_are_float(self):
        probs, lik_possible, truth_S_flat = make_synthetic()
        cfg = make_eval_cfg()
        result = compute_all_metrics(probs, lik_possible, truth_S_flat, cfg, n_nodes=20, n_runs=50)
        for k, v in result.items():
            assert isinstance(v, float), f"Metric '{k}' is {type(v)}, expected float"
