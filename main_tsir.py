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


def _summary_get(key: str):
    try:
        return wandb.summary.get(key)
    except Exception:
        return None


def _observed_r0_report() -> dict:
    keys = [
        "R0_ground_truth",
        "avg_outbreak_size_ground_truth",
        "R0_monte_carlo",
        "avg_outbreak_size_monte_carlo",
        "R0_maximal_outbreak",
        "avg_outbreak_size_maximal_outbreak",
    ]
    return {key: _summary_get(key) for key in keys if _summary_get(key) is not None}


def _network_provenance(
    args: argparse.Namespace,
    cfg,
    H,
    reduction_report: dict,
    calibration_meta: dict,
) -> dict:
    experiment = getattr(cfg, "experiment", None)
    target_r0 = _cfg_get(experiment, "r0", None)
    calibration_target = calibration_meta.get("target_r0") if isinstance(calibration_meta, dict) else None
    calibration_error = calibration_meta.get("error") if isinstance(calibration_meta, dict) else None
    tolerance = _cfg_get(_cfg_get(cfg.sir, "calibration", {}), "tolerance", None)
    observed = _observed_r0_report()
    return {
        "artifact_name": args.data,
        "network": {
            "name": cfg.nwk.name,
            "type": _cfg_get(cfg.nwk, "type", None),
            "directed": bool(cfg.nwk.directed),
            "n_nodes": int(H.number_of_nodes()),
            "n_edges": int(H.number_of_edges()),
            "n_contacts": int(sum(len(d.get("times", [])) for _, _, d in H.edges(data=True))),
            "t_max": int(cfg.nwk.t_max),
        },
        "sir": {
            "beta": float(cfg.sir.beta),
            "mu": float(cfg.sir.mu),
            "start_t": int(cfg.sir.start_t),
            "end_t": int(cfg.sir.end_t),
            "n_runs": int(cfg.sir.n_runs),
            "mc_runs": int(cfg.sir.mc_runs),
        },
        "r0": {
            "label": _cfg_get(experiment, "r0_label", _cfg_get(experiment, "label", None)),
            "target": float(target_r0) if target_r0 is not None else None,
            "canonical_targets": [0.8, 1.0, 1.1, 1.5, 2.0, 2.5],
            "calibration_target": float(calibration_target) if calibration_target is not None else None,
            "calibration_error": float(calibration_error) if calibration_error is not None else None,
            "calibration_tolerance": float(tolerance) if tolerance is not None else None,
            "within_calibration_tolerance": (
                bool(float(calibration_error) <= float(tolerance))
                if calibration_error is not None and tolerance is not None
                else None
            ),
            "observed": observed,
        },
        "calibration": calibration_meta or {},
        "reduction": reduction_report or {},
    }


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


def _close(a, b, rel: float = 1e-6, abs_: float = 1e-9) -> bool:
    """Return True if two scalars are equal within tolerance (None-safe)."""
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


def _write_label_cache(
    cache_path: Path, section: str, label: str, entry: dict, base: dict
) -> None:
    """Merge ``entry`` into ``cache_path[section][label]`` without dropping
    sibling labels already present in the cache file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_yaml(cache_path)
    if not isinstance(existing, dict):
        existing = {}
    merged = {**existing, **base}
    section_map = dict(existing.get(section) or {})
    section_map[str(label)] = entry
    merged[section] = section_map
    with open(cache_path, "w") as f:
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
    # A cached beta is only valid for the same target R0 *and* the same mu:
    # beta calibrates R0 = beta/mu * structure, so changing mu invalidates it.
    if (
        cached.get("target_r0") == target_r0
        and cached.get("beta") is not None
        and _close(cached.get("mu"), cfg.sir.mu)
    ):
        cfg.sir.beta = float(cached["beta"])
        wandb.summary["calibration/cache_hit"] = True
        wandb.summary["calibration/target_r0"] = target_r0
        wandb.summary["calibration/beta"] = float(cached["beta"])
        if cached.get("estimated_r0") is not None:
            wandb.summary["calibration/estimated_r0"] = float(cached["estimated_r0"])
        if cached.get("error") is not None:
            wandb.summary["calibration/error"] = float(cached["error"])
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


def _calibrate_end_t_if_requested(
    cfg, H, local_folder: str, reduction_report: dict
) -> dict:
    """Calibrate the observation time ``end_t`` to a target infected fraction.

    Runs *after* beta/R0 calibration, so it uses the R0-calibrated beta. The
    mean outbreak fraction at ``end_t`` is monotone non-decreasing in ``end_t``,
    so a bisection over the integer observation time converges to the smallest
    ``end_t`` whose mean outbreak size reaches ``sir.calibration.target_infected``
    (default ≈ 0.40 of nodes infected at the snapshot).

    If the full contact window (``end_t = t_max``) still does not reach the
    target — e.g. R0≈2 on a fragmented network saturates below 40% — ``end_t``
    is pinned to ``t_max`` and the shortfall is logged rather than silently
    overshooting the available simulation horizon.
    """
    if not _calibration_enabled(cfg):
        return {}
    calibration = cfg.sir.calibration
    target = _cfg_get(calibration, "target_infected", None)
    if target is None:
        return {}
    target = float(target)

    n_nodes = H.number_of_nodes()
    t_max = int(cfg.nwk.t_max)
    n_probe = int(_cfg_get(calibration, "target_infected_n_probe", 64))
    max_iter = int(_cfg_get(calibration, "target_infected_max_iter", 12))
    tolerance = float(_cfg_get(calibration, "target_infected_tolerance", 0.02))
    seed = int(_cfg_get(calibration, "seed", 20260609))
    out_dir = Path(_cfg_get(calibration, "output_dir", "results/calibration"))
    reduction_id = reduction_report.get("reduction_id", f"{cfg.nwk.name}_full")
    cache_path = out_dir / str(reduction_id) / "end_t.yml"
    r0_label = _cfg_get(
        getattr(cfg, "experiment", None),
        "r0_label",
        _cfg_get(getattr(cfg, "experiment", None), "label", "r0"),
    )

    # Cache hit only when beta, mu, target and t_max all match.
    cache = _read_yaml(cache_path)
    cached = (((cache.get("end_t") or {}).get(str(r0_label)) or {}) if isinstance(cache, dict) else {})
    if (
        _close(cached.get("target_infected"), target)
        and _close(cached.get("beta"), cfg.sir.beta)
        and _close(cached.get("mu"), cfg.sir.mu)
        and cached.get("t_max") == t_max
        and cached.get("end_t") is not None
    ):
        cfg.sir.end_t = int(cached["end_t"])
        wandb.summary["calibration/end_t_cache_hit"] = True
        wandb.summary["calibration/end_t"] = int(cached["end_t"])
        wandb.summary["calibration/end_t_outbreak"] = cached.get("outbreak_fraction")
        wandb.summary["calibration/end_t_target"] = target
        return dict(cached)

    H_cread = make_c_readable_from_networkx(H, t_max=t_max, directed=cfg.nwk.directed)
    probe_dir = Path(local_folder) / "calibration_end_t"
    probe_dir.mkdir(parents=True, exist_ok=True)

    def outbreak_fraction(end_t: int, it: int) -> tuple[float, int]:
        end_t = max(1, min(int(end_t), t_max))
        _, avg_os, _, _ = sir_probe_run(
            H_cread,
            beta=cfg.sir.beta,
            mu=cfg.sir.mu,
            start_t=cfg.sir.start_t,
            end_t=end_t,
            n=n_probe,
            seed=seed + it,
            path=str(probe_dir / f"probe_{it}_{{}}.bin"),
            log=str(probe_dir / f"probe_{it}.txt"),
        )
        return float(avg_os) / max(n_nodes, 1), end_t

    print("\n" + "=" * 60)
    print(f"Calibrating end_t for target infected fraction ≈ {target:.0%}")
    print("=" * 60)

    f_max, _ = outbreak_fraction(t_max, 0)
    print(f"  full window (end_t={t_max}): outbreak={f_max:.3f}")
    if f_max < target:
        cfg.sir.end_t = t_max
        best = {
            "end_t": t_max,
            "outbreak_fraction": float(f_max),
            "reached": False,
        }
        print(
            f"  WARNING: target {target:.0%} unreachable — final outbreak only "
            f"{f_max:.1%}. Pinning end_t={t_max}."
        )
    else:
        low, high = 1, t_max
        best = {"end_t": t_max, "outbreak_fraction": float(f_max), "err": abs(f_max - target)}
        for it in range(1, max_iter + 1):
            mid = (low + high) // 2
            f_mid, mid = outbreak_fraction(mid, it)
            err = abs(f_mid - target)
            print(f"  iter {it:02d}: end_t={mid}, outbreak={f_mid:.3f}, |err|={err:.3f}")
            if err < best.get("err", float("inf")):
                best = {"end_t": int(mid), "outbreak_fraction": float(f_mid), "err": float(err)}
            if err <= tolerance or (high - low) <= 1:
                break
            if f_mid < target:
                low = mid
            else:
                high = mid
        best["reached"] = True
        cfg.sir.end_t = int(best["end_t"])

    entry = {
        "end_t": int(cfg.sir.end_t),
        "outbreak_fraction": float(best["outbreak_fraction"]),
        "target_infected": target,
        "reached": bool(best.get("reached", True)),
        "beta": float(cfg.sir.beta),
        "mu": float(cfg.sir.mu),
        "t_max": t_max,
        "n_probe": n_probe,
        "method": "tsir_bisection_on_end_t",
    }
    _write_label_cache(
        cache_path,
        section="end_t",
        label=str(r0_label),
        entry=entry,
        base={"reduction_id": reduction_id, "network": cfg.nwk.name},
    )
    wandb.summary["calibration/end_t_cache_hit"] = False
    wandb.summary["calibration/end_t"] = int(cfg.sir.end_t)
    wandb.summary["calibration/end_t_outbreak"] = float(best["outbreak_fraction"])
    wandb.summary["calibration/end_t_target"] = target
    print(f"  selected end_t={cfg.sir.end_t} (outbreak={best['outbreak_fraction']:.3f})")
    return entry


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
    # Calibrate the observation time so ~target_infected of nodes are infected
    # at the snapshot, using the just-calibrated beta.
    end_t_meta = _calibrate_end_t_if_requested(cfg, H, local_folder, reduction_report)
    if end_t_meta:
        calibration_meta = {**(calibration_meta or {}), "end_t": end_t_meta}
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
    observed_r0 = _observed_r0_report()
    if reduction_report:
        reduction_report.setdefault("observed", {}).update(observed_r0)
        _write_json(f"{local_folder}/reduction_report.json", reduction_report)

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
    network_provenance = _network_provenance(args, cfg, H, reduction_report, calibration_meta)
    _write_json(f"{local_folder}/network_provenance.json", network_provenance)

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
            "network_provenance": network_provenance,
        },
    )
    artifact.add_dir(local_folder)
    wandb.log_artifact(artifact)
    print(f"Artifact logged.  Reference downstream runs with: --data {args.data}:latest")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
