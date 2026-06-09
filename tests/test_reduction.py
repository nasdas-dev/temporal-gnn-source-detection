from __future__ import annotations

import argparse

import networkx as nx

import run_all_experiments as runner
from setup.reduction import apply_network_reduction, sample_temporal_network


def _path_graph(n: int) -> nx.Graph:
    H = nx.Graph()
    for node in range(n):
        H.add_node(node, old_id=100 + node)
    for node in range(n - 1):
        H.add_edge(node, node + 1, times=[node, node + 5])
    return H


def _args(**overrides):
    base = {
        "reduction": "safe_1h",
        "target_runtime_seconds": 3600,
        "sample_target_nodes": 300,
        "time_window_steps": "auto",
        "reduction_seed": 42,
        "reduction_reps": 1,
        "use_full_betas": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_representative_time_window_reindexes_to_zero():
    H = _path_graph(12)
    cfg = {
        "name": "toy",
        "t_max": 20,
        "time_granularity": "days",
        "reduction": {
            "enabled": "auto",
            "preset": "safe_1h",
            "time": {
                "method": "representative_window",
                "apply_if_time_steps_gt": 5,
                "max_steps": 4,
                "candidate_windows": 4,
                "reindex_to_zero": True,
            },
        },
    }

    reduced, report = apply_network_reduction(H, cfg)

    assert report["time"]["window_steps"] == 4
    assert report["reduced"]["t_max"] == 3
    assert all(0 <= t <= 3 for _, _, data in reduced.edges(data=True) for t in data["times"])
    assert report["reduction_id"].startswith("toy_safe_1h")


def test_node_sampling_preserves_original_labels_and_reports_distances():
    H = _path_graph(20)
    cfg = {
        "name": "toy",
        "t_max": 20,
        "reduction": {
            "enabled": "auto",
            "preset": "safe_1h",
            "node": {
                "method": "balanced_activity_snowball",
                "apply_if_nodes_gt": 5,
                "target_nodes": 8,
                "max_node_edge_cost": 1000,
                "seed": 7,
                "stratification_bins": 3,
            },
        },
    }

    reduced, report = apply_network_reduction(H, cfg)

    assert reduced.number_of_nodes() <= 8
    assert "old_id" in next(iter(dict(reduced.nodes(data=True)).values()))
    assert report["node"]["sampled_nodes"] == reduced.number_of_nodes()
    assert "degree_ks" in report["node"]
    assert len(report["node"]["original_ids"]) == reduced.number_of_nodes()


def test_legacy_nwk_sample_alias_still_works():
    H = _path_graph(12)
    sampled = sample_temporal_network(
        H,
        {
            "method": "activity_snowball",
            "target_nodes": 5,
            "seed": 1,
            "max_node_edge_cost": 1000,
        },
    )

    assert sampled.number_of_nodes() <= 5
    assert sampled.graph["sample"]["method"] == "activity_snowball"


def test_runner_safe_1h_defaults_for_large_and_small_networks():
    preset = runner.PRESETS["paper_24h"]

    students = runner.build_tsir_config("students", "r0_10", preset, _args())
    escort = runner.build_tsir_config("escort", "r0_10", preset, _args())
    pig = runner.build_tsir_config("pig_data", "r0_10", preset, _args())
    malawi = runner.build_tsir_config("malawi", "r0_10", preset, _args())

    assert students["nwk"]["reduction"]["node"]["target_nodes"] == 300
    assert students["sir"]["calibration"]["target_r0"] == 1.0
    assert escort["nwk"]["reduction"]["time"]["max_steps_days"] == 365
    assert pig["nwk"]["reduction"]["time"]["reindex_to_zero"] is True
    assert "reduction" not in malawi["nwk"]
    assert malawi["sir"]["calibration"]["target_r0"] == 1.0


def test_dbgnn_safe_1h_locks_order_and_sets_caps():
    cfg = runner.build_model_config(
        "students",
        "dbgnn_k3",
        "r0_10",
        runner.PRESETS["paper_24h"],
        args=_args(),
    )

    assert cfg["dbgnn"]["order"] == 3
    assert cfg["dbgnn"]["time_bin_size"] == 4
    assert cfg["dbgnn"]["max_temporal_states"] == 2_000_000
    assert cfg["dbgnn"]["max_db_nodes"] == 500_000
    assert cfg["dbgnn"]["max_db_edges"] == 2_000_000
    assert "dbgnn.order" in cfg["hpo"]["locked_params"]
