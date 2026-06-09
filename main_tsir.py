"""
Stage 1 — SIR simulation pipeline.

Generates ground-truth and Monte-Carlo SIR simulations for a given temporal
network and logs all results as a versioned W&B artifact.

Usage
-----
::

    python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme

The ``--data`` argument becomes the W&B artifact name.  Downstream training
runs reference it as ``--data toy_holme:latest`` (or a specific version).

Output artifact contents (``data/<wandb_run_id>/``)
----------------------------------------------------
``ground_truth_{S,I,R}.bin``   — shape [n_nodes * n_runs, n_nodes] int8
``monte_carlo_{S,I,R}.bin``    — shape [n_nodes * mc_runs, n_nodes] int8
``maximal_outbreak_{S,I,R}.bin``— shape [n_nodes, n_nodes] int8
``possible_sources.bin``       — shape [n_nodes, n_runs, n_nodes] int8
``network.gpickle``            — NetworkX temporal graph
``ground_truth.txt``           — per-source SIR log
``monte_carlo.txt``            — Monte Carlo SIR log/
``maximal_outbreak.txt``       — maximal outbreak SIR log
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import wandb
import yaml

from setup import setup_tsir_run
from setup.read_network import load_network, make_array_from_networkx
from tsir.read_run import (
    make_c_readable_from_networkx,
    run as sir_probe_run,
    sir_ground_truth,
    sir_monte_carlo,
    sir_maximal_outbreak,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg",  required=True,
                   help="Path to TSIR YAML config, e.g. exp/toy_holme/tsir.yml")
    p.add_argument("--data", required=True,
                   help="W&B artifact name, e.g. toy_holme")
    return p.parse_args()


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: str | Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _summary_reduction(report: dict) -> None:
    if not report:
        return
    for section in ("original", "reduced", "time", "node"):
        values = report.get(section, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                wandb.summary[f"reduction/{section}/{key}"] = value
    for key in ("enabled", "preset", "runtime_target_s", "network", "reduction_id", "node_edge_cost_reduction"):
        if key in report:
            wandb.summary[f"reduction/{key}"] = report[key]


def _calibration_enabled(cfg) -> bool:
    calibration = _cfg_get(cfg.sir, "calibration", None)
    if calibration is None:
        return False
    enabled = _cfg_get(calibration, "enabled", False)
    return enabled not in (False, "false", "False", "none", "None", "off", "disabled")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _write_beta_cache(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_yaml(path)
    merged = {**existing, **payload}
    with open(path, "w") as f:
        yaml.safe_dump(_jsonable(merged), f, sort_keys=False)


def _calibrate_beta_if_requested(cfg, H, local_folder: str, reduction_report: dict) -> dict:
    """Optionally calibrate beta on the reduced graph before final TSIR runs."""
    if not _calibration_enabled(cfg):
        return {}

    calibration = cfg.sir.calibration
    target_r0 = _cfg_get(calibration, "target_r0", None)
    if target_r0 is None and hasattr(cfg, "experiment"):
        target_r0 = _cfg_get(cfg.experiment, "r0", None)
    if target_r0 is None:
        return {}
    target_r0 = float(target_r0)

    n_nodes = H.number_of_nodes()
    n_probe = int(_cfg_get(calibration, "n_probe", 1))
    max_iter = int(_cfg_get(calibration, "max_iter", 8))
    tolerance = float(_cfg_get(calibration, "tolerance", 0.05))
    low = float(_cfg_get(calibration, "beta_min", 1e-5))
    high = float(_cfg_get(calibration, "beta_max", 1.0))
    seed = int(_cfg_get(calibration, "seed", 20260609))
    out_dir = Path(_cfg_get(calibration, "output_dir", "results/calibration"))
    reduction_id = reduction_report.get("reduction_id", f"{cfg.nwk.name}_full")
    cache_path = out_dir / str(reduction_id) / "betas.yml"
    r0_label = _cfg_get(getattr(cfg, "experiment", None), "r0_label", _cfg_get(getattr(cfg, "experiment", None), "label", "r0"))

    cache = _read_yaml(cache_path)
    cached = (((cache.get("betas") or {}).get(str(r0_label)) or {}) if isinstance(cache, dict) else {})
    if cached.get("target_r0") == target_r0 and cached.get("beta") is not None:
        cfg.sir.beta = float(cached["beta"])
        wandb.summary["calibration/cache_hit"] = True
        return dict(cached)

    print("\n" + "=" * 60)
    print(f"Calibrating beta on reduced graph (target R0={target_r0:.3g})")
    print("=" * 60)
    H_cread = make_c_readable_from_networkx(H, t_max=cfg.nwk.t_max, directed=cfg.nwk.directed)
    probe_dir = Path(local_folder) / "calibration"
    probe_dir.mkdir(parents=True, exist_ok=True)

    best = {"beta": float(cfg.sir.beta), "estimated_r0": float("nan"), "error": float("inf")}
    for i in range(max_iter):
        beta = 0.5 * (low + high)
        R0, avg_os, sd, se = sir_probe_run(
            H_cread,
            beta=beta,
            mu=cfg.sir.mu,
            start_t=cfg.sir.start_t,
            end_t=cfg.sir.end_t,
            n=n_probe,
            seed=seed + i,
            path=str(probe_dir / f"probe_{i}_{{}}.bin"),
            log=str(probe_dir / f"probe_{i}.txt"),
        )
        error = abs(R0 - target_r0)
        print(f"  iter {i + 1:02d}: beta={beta:.6g}, R0={R0:.4g}, |err|={error:.4g}")
        if error < best["error"]:
            best = {
                "beta": float(beta),
                "estimated_r0": float(R0),
                "error": float(error),
                "avg_outbreak_size": float(avg_os / max(n_nodes, 1)),
                "n_probe": int(n_probe),
                "max_iter": int(max_iter),
                "target_r0": float(target_r0),
            }
        if error <= tolerance:
            break
        if R0 < target_r0:
            low = beta
        else:
            high = beta

    cfg.sir.beta = float(best["beta"])
    payload = {
        "reduction_id": reduction_id,
        "network": cfg.nwk.name,
        "betas": {
            str(r0_label): {
                **best,
                "mu": float(cfg.sir.mu),
                "method": "tsir_bisection_on_reduced_graph",
            }
        },
    }
    _write_beta_cache(cache_path, payload)
    wandb.summary["calibration/cache_hit"] = False
    wandb.summary["calibration/target_r0"] = target_r0
    wandb.summary["calibration/beta"] = float(best["beta"])
    wandb.summary["calibration/estimated_r0"] = float(best["estimated_r0"])
    wandb.summary["calibration/error"] = float(best["error"])
    print(f"  selected beta={best['beta']:.6g} (estimated R0={best['estimated_r0']:.4g})")
    return payload["betas"][str(r0_label)]


def _outbreak_report(truth_S: np.ndarray, n_nodes: int, n_runs: int, min_outbreak: int = 2) -> dict:
    sizes = (1 - truth_S.reshape(n_nodes, n_runs, n_nodes)).sum(axis=2).reshape(-1)
    hist, edges = np.histogram(sizes, bins=min(20, max(1, n_nodes)))
    return {
        "mean": float(np.mean(sizes)),
        "std": float(np.std(sizes)),
        "min": int(np.min(sizes)),
        "max": int(np.max(sizes)),
        "valid_fraction": float(np.mean(sizes >= min_outbreak)),
        "hist_counts": hist.astype(int).tolist(),
        "hist_edges": edges.astype(float).tolist(),
    }


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------
    # 1. W&B initialisation + config loading
    # ---------------------------------------------------------------
    cfg = setup_tsir_run(args.cfg)
    local_folder = f"data/{wandb.run.id}"
    os.makedirs(local_folder, exist_ok=True)

    print(f"\nW&B run : {wandb.run.url}")
    print(f"Data dir: {local_folder}\n")

    # ---------------------------------------------------------------
    # 2. Load temporal network
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Network")
    print("=" * 60)
    H = load_network(cfg)
    n_nodes = H.number_of_nodes()
    n_edges = H.number_of_edges()
    n_contacts = sum(len(d["times"]) for _, _, d in H.edges(data=True))
    sample_meta = H.graph.get("sample", {})
    reduction_report = H.graph.get("reduction_report", {})
    calibration_meta = _calibrate_beta_if_requested(cfg, H, local_folder, reduction_report)
    if reduction_report:
        reduction_report["calibration"] = calibration_meta
        _summary_reduction(reduction_report)

    print(f"  Nodes    : {n_nodes}")
    print(f"  Edges    : {n_edges}  (undirected pairs)")
    print(f"  Contacts : {n_contacts}  (total timestamped events)")
    print(f"  t_max    : {cfg.nwk.t_max}")
    print(f"  beta     : {cfg.sir.beta}")

    wandb.summary["n_nodes"]   = n_nodes
    wandb.summary["n_edges"]   = n_edges
    wandb.summary["n_contacts"] = n_contacts
    wandb.summary["t_max"]     = cfg.nwk.t_max
    wandb.summary["network"]   = cfg.nwk.name
    wandb.summary["beta"]      = cfg.sir.beta
    wandb.summary["mu"]        = cfg.sir.mu
    for key, value in sample_meta.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            wandb.summary[f"sample/{key}"] = value

    # Persist the graph so downstream runs can load it without re-reading CSV
    with open(f"{local_folder}/network.gpickle", "wb") as f:
        pickle.dump(H, f)
    if reduction_report:
        _write_json(f"{local_folder}/reduction_report.json", reduction_report)

    # ---------------------------------------------------------------
    # 3. Build C-readable network representation
    # ---------------------------------------------------------------
    H_cread = make_c_readable_from_networkx(
        H, t_max=cfg.nwk.t_max, directed=cfg.nwk.directed
    )

    # ---------------------------------------------------------------
    # 4. Ground-truth SIR simulations
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Ground-truth SIR  ({cfg.sir.n_runs} runs × {n_nodes} sources)")
    print("=" * 60)
    truth_S, truth_I, truth_R = sir_ground_truth(cfg, H_cread, n_nodes, local_folder)
    if reduction_report:
        reduction_report["outbreak_distribution"] = _outbreak_report(
            truth_S,
            n_nodes,
            cfg.sir.n_runs,
            min_outbreak=int(_cfg_get(getattr(cfg, "eval", None), "min_outbreak", 2)),
        )
        reduction_report["observed"] = {
            "R0_ground_truth": wandb.summary.get("R0_ground_truth"),
            "avg_outbreak_size_ground_truth": wandb.summary.get("avg_outbreak_size_ground_truth"),
        }
        _write_json(f"{local_folder}/reduction_report.json", reduction_report)

    # ---------------------------------------------------------------
    # 5. Monte-Carlo SIR simulations  (training data for GNN)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Monte-Carlo SIR  ({cfg.sir.mc_runs} runs × {n_nodes} sources)")
    print("=" * 60)
    sir_monte_carlo(cfg, H_cread, n_nodes, local_folder)

    # ---------------------------------------------------------------
    # 6. Maximal-outbreak SIR  (β=1, μ=0: determines reachable nodes)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Maximal-outbreak SIR  (β=1, μ=0)")
    print("=" * 60)
    sir_maximal_outbreak(cfg, H_cread, n_nodes, local_folder)

    # ---------------------------------------------------------------
    # 7. Possible-sources mask
    # truth_S: [n_runs*n_nodes, n_nodes] → reshape to [n_nodes, n_runs, n_nodes]
    # possible[s, r, v] = 1  iff node v is non-susceptible in run r from source s
    # (any infected/recovered node is a feasible source candidate)
    # ---------------------------------------------------------------
    truth_S_3d = truth_S.reshape(n_nodes, cfg.sir.n_runs, n_nodes)
    possible   = (1 - truth_S_3d).astype(np.int8)
    possible.tofile(f"{local_folder}/possible_sources.bin")
    print(f"\nPossible-sources mask saved  "
          f"(avg. {possible.mean(axis=(1,2)).mean():.3f} feasible per run)")

    # ---------------------------------------------------------------
    # 8. Log as W&B artifact
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Logging artifact '{args.data}'")
    print("=" * 60)
    artifact = wandb.Artifact(
        name        = args.data,
        type        = "tsir-data",
        description = (
            f"SIR simulations on '{cfg.nwk.name}'  "
            f"(β={cfg.sir.beta}, μ={cfg.sir.mu}, "
            f"n_runs={cfg.sir.n_runs}, mc_runs={cfg.sir.mc_runs})"
        ),
        metadata = {
            "network":  cfg.nwk.name,
            "n_nodes":  n_nodes,
            "t_max":    cfg.nwk.t_max,
            "beta":     cfg.sir.beta,
            "mu":       cfg.sir.mu,
            "n_runs":   cfg.sir.n_runs,
            "mc_runs":  cfg.sir.mc_runs,
            "sample":   sample_meta,
            "reduction": reduction_report,
            "calibration": calibration_meta,
        },
    )
    artifact.add_dir(local_folder)
    wandb.log_artifact(artifact)
    print(f"Artifact logged.  Reference downstream runs with: --data {args.data}:latest")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
