"""
Stage 2 — Unified trainable-model training and evaluation.

Trains any registered source-detection model on a TSIR artifact produced by
``main_tsir.py``, evaluates on ground-truth simulations, and logs all
metrics to W&B.

Usage
-----
::

    # BacktrackingNetwork on toy_holme
    python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest

    # StaticGNN on karate_static
    python main_train.py --cfg exp/karate_static/static_gnn.yml --data karate_static:latest

    # TemporalGNN on france_office
    python main_train.py --cfg exp/france_office/temporal_gnn.yml --data france_office:latest

    # Graph-free MLP baseline on france_office
    python main_train.py --cfg exp/france_office/static_mlp.yml --data france_office:latest

The ``--cfg`` YAML must contain a top-level ``model:`` key matching a name
in ``MODEL_REGISTRY`` (e.g. ``backtracking``, ``static_gnn``, ``static_mlp``).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time

# Prevent OpenMP/MKL deadlock when wandb spawns background threads alongside
# PyTorch's multi-threaded CPU kernels (especially scatter_add_ and Linear).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import networkx as nx
import numpy as np
import torch
torch.set_num_threads(1)
import wandb
import yaml

from eval import compute_all_metrics, per_sample_arrays
from gnn import MODEL_REGISTRY, get_model_spec
from gnn.graph_builder import coarsen_temporal_network
from setup import setup_methods_run, load_tsir_data
from training import CheckpointError, LossGuardAbort, SIRDataset, Trainer, fit_compatibility_metadata
from training.checkpointing import (
    assert_compatible,
    atomic_json_dump,
    checkpoint_timestamp,
    compatibility_hash,
    load_json,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg",  required=True,
                   help="Model config YAML, e.g. exp/toy_holme/backtracking.yml")
    p.add_argument("--data", required=True,
                   help="W&B artifact reference, e.g. toy_holme:latest")
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="Override config values, e.g. --override train.n_mc=100 train.reps=1")
    p.add_argument("--save-probs", action="store_true",
                   help="Save probs_rep*.pt tensors. Eval arrays and metrics are always saved.")
    p.add_argument("--checkpoint-dir", default=None,
                   help="Local checkpoint root. Defaults to data/checkpoints/<data>/<model>/<hash>.")
    p.add_argument("--resume-from", default=None,
                   help="Alias for --checkpoint-dir when resuming from an existing local checkpoint root.")
    p.add_argument("--fresh", action="store_true",
                   help="Start a fresh attempt and ignore existing local checkpoints.")
    return p.parse_args()


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> None:
    """Apply ``key.subkey=value`` overrides to a nested config dict in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value or key.subkey=value format")
        key_path, raw_val = item.split("=", 1)
        keys = key_path.strip().split(".")
        # Try to cast value to int/float/bool, fall back to str
        for cast in (int, float):
            try:
                raw_val = cast(raw_val)
                break
            except ValueError:
                pass
        else:
            if raw_val.lower() in ("true", "false"):
                raw_val = raw_val.lower() == "true"
        node = cfg_dict
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = raw_val


def _builder_kwargs(model_name: str, model_cfg: dict) -> dict:
    """Extract graph-builder keyword arguments from the model config section."""
    if model_name == "static_gnn":
        return {"use_edge_weights": model_cfg.get("use_edge_weights", False)}
    if model_name == "temporal_gnn":
        return {"group_by_time": model_cfg.get("group_by_time", 1)}
    if model_name == "dag_gnn":
        return {"delta_t": model_cfg.get("delta_t", None)}
    if model_name == "dbgnn":
        return {
            "order": model_cfg.get("order", 2),
            "delta": model_cfg.get("delta", 24),
            "directed": model_cfg.get("directed", None),
            "time_bin_size": model_cfg.get("time_bin_size", 1),
            "max_temporal_states": model_cfg.get("max_temporal_states", None),
            "max_db_nodes": model_cfg.get("max_db_nodes", None),
            "max_db_edges": model_cfg.get("max_db_edges", None),
        }
    if model_name == "backtracking":
        return {
            "directed": model_cfg.get("directed", None),
            "dense_edge_attr": model_cfg.get("dense_edge_attr", False),
        }
    return {}


def _graph_structure_stats(model_name: str, graph_data: dict, H_static: nx.Graph) -> dict:
    """Collect model-specific structural counts for the H2 cost analysis."""
    stats: dict[str, float] = {"n_edges_static": int(H_static.number_of_edges())}
    if model_name == "temporal_gnn":
        edge_indeces = graph_data.get("edge_indeces", {}) or {}
        per_snapshot = [int(t.size(1)) for t in edge_indeces.values()]
        stats["num_snapshots"] = int(graph_data.get("num_snapshots", len(per_snapshot)))
        stats["edges_total"] = int(sum(per_snapshot))
        stats["edges_per_snapshot_mean"] = float(np.mean(per_snapshot)) if per_snapshot else 0.0
    elif model_name == "backtracking":
        eti = graph_data.get("edge_time_index")
        stats["edge_texture_length"] = int(graph_data.get("T", 0))
        stats["edge_texture_nnz"] = int(eti.numel()) if eti is not None else 0
        stats["n_edges"] = int(graph_data.get("n_edges", 0))
    elif model_name == "dbgnn":
        db_stats = graph_data.get("db_stats", {}) or {}
        db_edge_index = graph_data.get("db_edge_index")
        static_edge_index = graph_data.get("static_edge_index")
        stats["n_db_nodes"] = int(graph_data.get("n_db_nodes", db_stats.get("n_db_nodes", 0)) or 0)
        stats["n_db_edges"] = int(db_edge_index.size(1)) if db_edge_index is not None else 0
        stats["n_static_edges"] = int(static_edge_index.size(1)) if static_edge_index is not None else 0
    elif model_name in ("static_gnn", "static_mlp"):
        edge_index = graph_data.get("edge_index")
        stats["n_edges"] = int(edge_index.size(1)) if edge_index is not None else 0
    return stats


def _peak_rss_mb() -> float:
    """Peak resident set size in MiB (ru_maxrss is bytes on macOS, KiB on Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(ru) / (1024.0 * 1024.0)
    return float(ru) / 1024.0


def _jsonable(value):
    """Convert numpy scalars/arrays to JSON-friendly Python values."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(_jsonable(payload), f, indent=2, sort_keys=True)


def _write_loss_history(path: str, train_losses: list[float], val_losses: list[float]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, tl, vl])


def _safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unknown"


def _checkpoint_settings(cfg_dict: dict, args: argparse.Namespace) -> dict:
    raw = cfg_dict.get("checkpoint") or {}
    resume_requested = bool(raw.get("resume", True)) or bool(args.resume_from)
    return {
        "enabled": bool(raw.get("enabled", True)),
        "resume": resume_requested and not bool(args.fresh),
        "save_every": int(raw.get("save_every", 1)),
        "dir": raw.get("dir"),
    }


def _resolve_checkpoint_root(
    *,
    args: argparse.Namespace,
    cfg_dict: dict,
    output_cfg: dict,
    checkpoint_cfg: dict,
    model_name: str,
    data_name: str,
    n_nodes: int,
) -> Path:
    explicit = args.checkpoint_dir or args.resume_from or output_cfg.get("checkpoint_dir") or checkpoint_cfg.get("dir")
    if explicit:
        return Path(explicit)
    key = compatibility_hash({
        "model": model_name,
        "data": data_name,
        "cfg": cfg_dict,
        "n_nodes": int(n_nodes),
    })[:16]
    return Path("data") / "checkpoints" / _safe_part(data_name) / _safe_part(model_name) / key


def _mc_indices_by_rep(mc_runs: int, n_mc: int, reps: int, seed: int) -> list[np.ndarray]:
    rng = np.random.RandomState(seed)
    return [
        np.asarray(rng.choice(mc_runs, n_mc, replace=False), dtype=np.int64)
        for _ in range(reps)
    ]


def _copy_if_exists(src: Path, dest: Path) -> None:
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _write_rep_state(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    atomic_json_dump(
        {
            "updated_at": checkpoint_timestamp(),
            **payload,
        },
        path,
    )


def _restore_rep_outputs(rep_dir: Path, run_dir: Path, rep: int, save_probs: bool) -> dict | None:
    metrics_src = rep_dir / f"metrics_rep{rep}.json"
    arrays_src = rep_dir / f"eval_arrays_rep{rep}.npz"
    if not metrics_src.exists() or not arrays_src.exists():
        return None
    _copy_if_exists(rep_dir / f"loss_history_rep{rep}.csv", run_dir / f"loss_history_rep{rep}.csv")
    _copy_if_exists(metrics_src, run_dir / f"metrics_rep{rep}.json")
    _copy_if_exists(arrays_src, run_dir / f"eval_arrays_rep{rep}.npz")
    if save_probs:
        _copy_if_exists(rep_dir / f"probs_rep{rep}.pt", run_dir / f"probs_rep{rep}.pt")
    with open(metrics_src) as f:
        return json.load(f).get("metrics")


def _compact_fit_info(info: dict) -> dict:
    """Drop large split arrays from state JSON; they remain in .pt checkpoints."""
    return {
        key: value
        for key, value in info.items()
        if key not in {"train_indices", "val_indices"}
    }


def _sample_std(vals: list[float]) -> float:
    """Return the sample standard deviation used for cross-repetition summaries."""
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


# Two-sided t critical values t_{0.975, n-1} for small repetition counts, used
# for 95% confidence intervals across reps (Sterchi et al. report 95% CIs over 3
# runs). Falls back to the normal approximation (1.96) for larger n.
_T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
        7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def _ci95_halfwidth(vals: list[float]) -> float:
    """95% confidence-interval half-width of the mean across repetitions."""
    n = len(vals)
    if n < 2:
        return 0.0
    t_crit = _T95.get(n, 1.96)
    return t_crit * _sample_std(vals) / (n ** 0.5)


def _truth_indices_for_rep(
    eval_cfg: dict,
    rep: int,
    n_truth: int,
    n_runs: int,
    reps: int,
) -> np.ndarray:
    """Return the held-out truth-run indices for one repetition.

    Two protocols are supported via ``eval.shared_eval_window``:

    - ``True`` (Sterchi-exact): every repetition is evaluated on the *same*
      held-out test window ``[truth_start, truth_start + n_truth)``. Only the
      training seed and weight initialisation differ across reps, so the
      cross-rep spread isolates training/initialisation noise — which is exactly
      what the reported 95% CIs then mean. Needs only ``n_truth`` truth runs.
    - ``False`` (default, legacy): each rep gets a disjoint window
      ``[truth_start + rep*n_truth, …)``, which uses more data but conflates
      training noise with test-set sampling noise. Needs ``reps * n_truth`` runs.
    """
    truth_start = int(eval_cfg.get("truth_start", 0))
    if truth_start < 0:
        raise ValueError(f"eval.truth_start must be non-negative, got {truth_start}")
    if bool(eval_cfg.get("shared_eval_window", False)):
        truth_stop = truth_start + n_truth
        if truth_stop > n_runs:
            raise ValueError(
                f"eval.truth_start + n_truth = {truth_start} + {n_truth} = "
                f"{truth_stop} exceeds n_runs={n_runs}. Lower n_truth/truth_start "
                "or regenerate the artifact with more ground-truth runs."
            )
        return np.arange(truth_start, truth_start + n_truth)
    truth_stop = truth_start + reps * n_truth
    if truth_stop > n_runs:
        raise ValueError(
            f"eval.truth_start + reps * n_truth = {truth_start} + {reps} * "
            f"{n_truth} = {truth_stop} exceeds n_runs={n_runs}. Reduce reps or "
            "n_truth, lower eval.truth_start, or regenerate the artifact with "
            "more ground-truth runs."
        )
    start = truth_start + rep * n_truth
    return np.arange(start, start + n_truth)


def main() -> None:
    args = parse_args()

    # ---------------------------------------------------------------
    # 1. Load YAML config
    # ---------------------------------------------------------------
    with open(args.cfg) as f:
        cfg_dict = yaml.safe_load(f)

    _apply_overrides(cfg_dict, args.override)

    model_name = cfg_dict["model"]
    train_cfg  = cfg_dict["train"]
    eval_cfg   = cfg_dict["eval"]
    model_cfg  = cfg_dict[model_name]     # model-specific section
    output_cfg = cfg_dict.get("output", {})
    save_probs = bool(args.save_probs or output_cfg.get("save_probs", False))
    checkpoint_cfg = _checkpoint_settings(cfg_dict, args)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Registered: {sorted(MODEL_REGISTRY.keys())}"
        )

    # ---------------------------------------------------------------
    # 2. W&B initialisation
    # ---------------------------------------------------------------
    setup_methods_run(job_type="train")
    wandb.config.update({
        "model":     model_name,
        "data_name": args.data,
        **cfg_dict,
    })
    wandb.run.tags += (f"model:{model_name}",)
    print(f"\nW&B run : {wandb.run.url}")
    print(f"Model   : {model_name}")
    print(f"Data    : {args.data}\n")
    run_dir = f"data/{wandb.run.id}"
    os.makedirs(run_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 3. Load TSIR artifact
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Loading TSIR data")
    print("=" * 60)
    H, data = load_tsir_data(args.data)
    n_nodes = data.n_nodes

    # ---------------------------------------------------------------
    # 3b. Unified temporal coarse-graining (H2): bin the shared contact
    #     network once, BEFORE any model-specific graph construction, so every
    #     temporal representation is derived from the same coarse-grained graph.
    # ---------------------------------------------------------------
    coarsen_cfg = cfg_dict.get("coarsen", {}) or {}
    delta_t = int(coarsen_cfg.get("delta_t", 1))
    coarsen_stats: dict | None = None
    if delta_t > 1:
        H, coarsen_stats = coarsen_temporal_network(H, delta_t)
        print(
            f"  Coarsen  : delta_t={delta_t}  "
            f"t_max {coarsen_stats['t_max_before']}->{coarsen_stats['t_max_after']}, "
            f"contacts {coarsen_stats['contacts_before']}->{coarsen_stats['contacts_after']}"
        )
        for key, value in coarsen_stats.items():
            wandb.summary[f"coarsen/{key}"] = value

    H_static = nx.Graph()
    H_static.add_nodes_from(range(n_nodes))
    for u, v in H.edges():
        H_static.add_edge(int(u), int(v))
    print(f"  n_nodes  : {n_nodes}")
    print(f"  n_runs   : {data.n_runs}  (ground-truth)")
    print(f"  mc_runs  : {data.mc_runs}  (Monte Carlo)")

    if train_cfg["n_mc"] > data.mc_runs:
        raise ValueError(
            f"n_mc={train_cfg['n_mc']} requested but artifact only has "
            f"{data.mc_runs} MC runs. Reduce n_mc or regenerate the artifact."
        )
    if eval_cfg["n_truth"] > data.n_runs:
        raise ValueError(
            f"n_truth={eval_cfg['n_truth']} requested but artifact only has "
            f"{data.n_runs} ground-truth runs."
        )

    # ---------------------------------------------------------------
    # 4. Build model-specific graph representation
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Building graph inputs")
    print("=" * 60)
    spec = get_model_spec(model_name)
    bkw  = _builder_kwargs(model_name, model_cfg)
    if delta_t > 1:
        # The shared coarsening already binned the network; disable any
        # per-model native binning so time is not coarsened twice.
        if "group_by_time" in bkw:
            bkw["group_by_time"] = 1
        if "time_bin_size" in bkw:
            bkw["time_bin_size"] = 1
    _build_t0 = time.perf_counter()
    graph_data = spec.builder_fn(H, **bkw)
    construction_seconds = time.perf_counter() - _build_t0
    graph_data["n_nodes"] = n_nodes   # ensure key present for forward fns
    wandb.summary["graph/construction_seconds"] = construction_seconds
    print(f"  Graph construction: {construction_seconds:.3f}s")

    structure_stats = _graph_structure_stats(model_name, graph_data, H_static)
    for key, value in structure_stats.items():
        wandb.summary[f"graph/{key}"] = value

    for k, v in graph_data.items():
        if hasattr(v, "shape"):
            print(f"  {k:20s}: {tuple(v.shape)}")
        elif isinstance(v, dict):
            preview = ", ".join(f"{dk}={dv}" for dk, dv in v.items())
            print(f"  {k:20s}: {preview}")
        elif not isinstance(v, dict):
            print(f"  {k:20s}: {v}")

    db_stats = graph_data.get("db_stats")
    if isinstance(db_stats, dict):
        for key, value in db_stats.items():
            if isinstance(value, (int, float, bool)) or value is None:
                wandb.summary[f"graph/{key}"] = value

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device  : {device}")
    checkpoint_root = _resolve_checkpoint_root(
        args=args,
        cfg_dict=cfg_dict,
        output_cfg=output_cfg,
        checkpoint_cfg=checkpoint_cfg,
        model_name=model_name,
        data_name=args.data,
        n_nodes=n_nodes,
    )
    if checkpoint_cfg["enabled"]:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(
            {
                "status": "active",
                "model": model_name,
                "data": args.data,
                "run_id": wandb.run.id,
                "checkpoint_root": str(checkpoint_root),
                "resume": checkpoint_cfg["resume"],
                "fresh": bool(args.fresh),
                "updated_at": checkpoint_timestamp(),
            },
            checkpoint_root / "manifest.json",
        )
        wandb.summary["checkpoint/root"] = str(checkpoint_root)
        print(f"  Checkpoints: {checkpoint_root}")
        if args.fresh:
            print("  Fresh run: existing checkpoints will be ignored")

    # ---------------------------------------------------------------
    # 5. Training repetitions
    # ---------------------------------------------------------------
    top_k_vals = eval_cfg["top_k"]
    offsets    = eval_cfg["inverse_rank_offset"]
    n_truth    = eval_cfg["n_truth"]
    reps       = train_cfg["reps"]
    truth_start = int(eval_cfg.get("truth_start", 0))

    # Aggregation buffers keyed by metric name (filled per rep, averaged in summary)
    rep_metric_lists: dict[str, list[float]] = {}

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])
    mc_selects = _mc_indices_by_rep(
        mc_runs=data.mc_runs,
        n_mc=int(train_cfg["n_mc"]),
        reps=int(reps),
        seed=int(train_cfg["seed"]),
    )
    n_params = 0
    fit_seconds_total = 0.0

    for rep in range(reps):
        print("\n" + "=" * 60)
        print(f"Repetition {rep + 1}/{reps}")
        print("=" * 60)

        # --- Sample MC runs ---
        select = mc_selects[rep]
        dataset = SIRDataset(
            data.mc_S[:, select, :],
            data.mc_I[:, select, :],
            data.mc_R[:, select, :],
        )
        rep_dir = checkpoint_root / f"rep{rep}" if checkpoint_cfg["enabled"] else None
        state_path = rep_dir / "state.json" if rep_dir is not None else None
        final_model_path = rep_dir / f"final_model_rep{rep}.pt" if rep_dir is not None else None
        rep_metadata = {
            "model": model_name,
            "data": args.data,
            "rep": int(rep),
            "cfg": cfg_dict,
            "n_nodes": int(n_nodes),
            "graph_builder_kwargs": bkw,
            "selected_mc_indices": select.tolist(),
            "truth_start": truth_start,
        }
        fit_metadata, _, _, fit_hash = fit_compatibility_metadata(
            dataset=dataset,
            batch_size=int(train_cfg["batch_size"]),
            epochs=int(train_cfg["epochs"]),
            patience=int(train_cfg["patience"]),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            test_size=float(train_cfg["test_size"]),
            seed=int(train_cfg["seed"]) + rep,
            checkpoint_metadata=rep_metadata,
        )

        rep_state = load_json(state_path) if state_path is not None else None
        if (
            rep_state is not None
            and rep_state.get("compatibility_hash") != fit_hash
            and checkpoint_cfg["resume"]
        ):
            raise CheckpointError(
                f"Incompatible repetition state {state_path}: expected {fit_hash}, "
                f"found {rep_state.get('compatibility_hash')}. Use --fresh to start over."
            )

        if (
            rep_dir is not None
            and checkpoint_cfg["resume"]
            and not args.fresh
            and rep_state is not None
            and rep_state.get("status") == "evaluated"
        ):
            restored_metrics = _restore_rep_outputs(rep_dir, Path(run_dir), rep, save_probs)
            if restored_metrics is not None:
                print(f"  Restored evaluated repetition from {rep_dir}")
                n_params = int(rep_state.get("n_params", n_params))
                wandb.log({f"{k}_rep{rep}": v for k, v in restored_metrics.items()})
                for metric_key, val in restored_metrics.items():
                    rep_metric_lists.setdefault(metric_key, []).append(float(val))
                continue

        # --- Build fresh model ---
        torch.manual_seed(int(train_cfg["seed"]) + rep)
        model = spec.build_fn(model_cfg, n_nodes, graph_data)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # --- Train or restore final model ---
        trainer = Trainer(model, spec.forward_fn, graph_data, device)
        final_loaded = False
        if (
            final_model_path is not None
            and checkpoint_cfg["resume"]
            and not args.fresh
            and final_model_path.exists()
        ):
            payload = torch_load(final_model_path, map_location=device)
            assert_compatible(payload, fit_hash, final_model_path)
            trainer.model.load_state_dict(payload["model_state"])
            train_losses = list(payload.get("train_losses", []))
            val_losses = list(payload.get("val_losses", []))
            trainer.last_fit_info = {
                "compatibility_hash": fit_hash,
                "metadata": fit_metadata,
                "best_epoch": payload.get("best_epoch"),
                "best_val": payload.get("best_val"),
                "epochs_trained": payload.get("epochs_trained", len(train_losses)),
                "checkpoint_dir": str(rep_dir),
                "final_model": str(final_model_path),
                "resumed": True,
            }
            final_loaded = True
            print(f"  Restored trained model from {final_model_path}")
        else:
            try:
                _fit_t0 = time.perf_counter()
                train_losses, val_losses = trainer.fit(
                    dataset       = dataset,
                    batch_size    = train_cfg["batch_size"],
                    epochs        = train_cfg["epochs"],
                    patience      = train_cfg["patience"],
                    lr            = train_cfg["lr"],
                    weight_decay  = train_cfg["weight_decay"],
                    test_size     = train_cfg["test_size"],
                    seed          = train_cfg["seed"] + rep,
                    grad_clip_norm = train_cfg.get("grad_clip_norm"),
                    wandb_run     = wandb.run,
                    rep           = rep,
                    loss_guard    = train_cfg.get("loss_guard"),
                    checkpoint_dir = rep_dir,
                    checkpoint_metadata = rep_metadata,
                    checkpoint_enabled = checkpoint_cfg["enabled"],
                    checkpoint_resume = checkpoint_cfg["resume"],
                    checkpoint_fresh = bool(args.fresh),
                    checkpoint_save_every = checkpoint_cfg["save_every"],
                    final_model_path = final_model_path,
                )
                fit_seconds_total += time.perf_counter() - _fit_t0
            except LossGuardAbort as exc:
                print(f"LOSS_GUARD_ABORT: {exc.reason} at epoch {exc.epoch}")
                wandb.summary["run/status"] = "loss_guard_aborted"
                wandb.summary["run/abort_reason"] = exc.reason
                wandb.summary["run/abort_epoch"] = exc.epoch
                if state_path is not None:
                    _write_rep_state(
                        state_path,
                        {
                            "status": "loss_guard_aborted",
                            "compatibility_hash": fit_hash,
                            "reason": exc.reason,
                            "epoch": exc.epoch,
                            "train_loss": exc.train_loss,
                            "val_loss": exc.val_loss,
                            "n_params": n_params,
                        },
                    )
                _write_json(
                    f"{run_dir}/abort.json",
                    {
                        "status": "loss_guard_aborted",
                        "reason": exc.reason,
                        "epoch": exc.epoch,
                        "train_loss": exc.train_loss,
                        "val_loss": exc.val_loss,
                        "model": model_name,
                        "data": args.data,
                    },
                )
                wandb.finish(exit_code=88)
                raise SystemExit(88)

        _write_loss_history(
            f"{run_dir}/loss_history_rep{rep}.csv", train_losses, val_losses
        )
        if rep_dir is not None:
            _write_loss_history(
                str(rep_dir / f"loss_history_rep{rep}.csv"), train_losses, val_losses
            )
            _write_rep_state(
                state_path,
                {
                    "status": "trained",
                    "compatibility_hash": fit_hash,
                    "model": model_name,
                    "data": args.data,
                    "rep": rep,
                    "n_params": n_params,
                    "final_model": str(final_model_path),
                    "fit_info": _compact_fit_info(trainer.last_fit_info),
                    "final_loaded": final_loaded,
                },
            )
            wandb.summary[f"checkpoint/rep{rep}_dir"] = str(rep_dir)
            wandb.summary[f"model/final_rep{rep}_path"] = str(final_model_path)

        # --- Inference on ground truth ---
        print("\n  Running inference on ground truth…")
        select_truth = _truth_indices_for_rep(
            eval_cfg, rep=rep, n_truth=n_truth, n_runs=data.n_runs, reps=reps
        )
        probs = trainer.predict_from_tensor(
            truth_S    = data.truth_S[:, select_truth, :],
            truth_I    = data.truth_I[:, select_truth, :],
            truth_R    = data.truth_R[:, select_truth, :],
            batch_size = 32,
        )   # [n_nodes * n_truth, n_nodes]

        # --- Compute all metrics ---
        lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, n_nodes)
        truth_S_flat = data.truth_S[:, select_truth, :].reshape(-1, n_nodes)

        rep_metrics = compute_all_metrics(
            probs        = probs,
            lik_possible = lik_possible,
            truth_S_flat = truth_S_flat,
            eval_cfg     = eval_cfg,
            n_nodes      = n_nodes,
            n_runs       = n_truth,
            H_static     = H_static,
        )

        n_valid = int(rep_metrics["eval/n_valid"])
        print(f"\n  Valid outbreaks: {n_valid} / {n_nodes * n_truth}")
        print(f"  MRR           : {rep_metrics['eval/mrr']:.4f}")
        for k in top_k_vals:
            print(f"  top-{k:<2}         : {100 * rep_metrics[f'eval/top_{k}']:.1f}%")
        print(f"  Norm. Brier   : {rep_metrics['eval/norm_brier']:.4f}")
        print(f"  Norm. Entropy : {rep_metrics['eval/norm_entropy']:.4f}")

        wandb.log({f"{k}_rep{rep}": v for k, v in rep_metrics.items()})

        # Accumulate for cross-rep summary. eval/n_valid is included so the
        # summary carries eval/n_valid_mean (consumed by the runners' valid-
        # outbreak plot and the H2 cost table).
        for metric_key, val in rep_metrics.items():
            rep_metric_lists.setdefault(metric_key, []).append(val)

        # Save raw model outputs + lightweight eval arrays for viz scripts
        if save_probs:
            torch.save(
                torch.tensor(probs),
                f"{run_dir}/probs_rep{rep}.pt",
            )
            if rep_dir is not None:
                torch.save(
                    torch.tensor(probs),
                    rep_dir / f"probs_rep{rep}.pt",
                )
        arrays = per_sample_arrays(
            probs        = probs,
            lik_possible = lik_possible,
            truth_S_flat = truth_S_flat,
            eval_cfg     = eval_cfg,
            n_nodes      = n_nodes,
            n_runs       = n_truth,
        )
        np.savez_compressed(
            f"{run_dir}/eval_arrays_rep{rep}.npz",
            **arrays,
        )
        if rep_dir is not None:
            np.savez_compressed(
                rep_dir / f"eval_arrays_rep{rep}.npz",
                **arrays,
            )
        metrics_payload = {
            "rep": rep,
            "model": model_name,
            "data": args.data,
            "metrics": rep_metrics,
        }
        _write_json(f"{run_dir}/metrics_rep{rep}.json", metrics_payload)
        if rep_dir is not None:
            _write_json(str(rep_dir / f"metrics_rep{rep}.json"), metrics_payload)
            _write_rep_state(
                state_path,
                {
                    "status": "evaluated",
                    "compatibility_hash": fit_hash,
                    "model": model_name,
                    "data": args.data,
                    "rep": rep,
                    "n_params": n_params,
                    "final_model": str(final_model_path),
                    "metrics_path": str(rep_dir / f"metrics_rep{rep}.json"),
                    "eval_arrays_path": str(rep_dir / f"eval_arrays_rep{rep}.npz"),
                    "fit_info": _compact_fit_info(trainer.last_fit_info),
                },
            )

    # ---------------------------------------------------------------
    # 6. Summary (averaged over reps)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Summary (mean ± 95% CI over {len(next(iter(rep_metric_lists.values()), []))} reps)")
    print("=" * 60)
    for metric_key, vals in sorted(rep_metric_lists.items()):
        mean = float(np.mean(vals))
        std  = _sample_std(vals)
        ci95 = _ci95_halfwidth(vals)
        wandb.summary[f"{metric_key}_mean"] = mean
        wandb.summary[f"{metric_key}_std"]  = std
        wandb.summary[f"{metric_key}_ci95"] = ci95
        # Human-friendly output: percentages for top_k, 4dp for scalars
        if "top_" in metric_key:
            print(f"  {metric_key}: {100 * mean:.1f}% ± {100 * ci95:.1f}% (95% CI)")
        else:
            print(f"  {metric_key}: {mean:.4f} ± {ci95:.4f} (95% CI)")

    wandb.summary["model/n_params"] = n_params
    wandb.summary["model/name"]     = model_name
    wandb.summary["data/name"]      = args.data
    wandb.summary["run/status"]     = "success"

    # --- Cost + structure metrics for the H2 coarse-graining analysis ---
    cost_stats: dict[str, float] = {
        "graph/construction_seconds": float(construction_seconds),
        "train/fit_seconds": float(fit_seconds_total),
        "resources/peak_rss_mb": _peak_rss_mb(),
    }
    if torch.cuda.is_available():
        cost_stats["resources/peak_gpu_mb"] = float(torch.cuda.max_memory_allocated()) / 1e6
    cost_stats.update({f"graph/{k}": v for k, v in structure_stats.items()})
    if coarsen_stats is not None:
        cost_stats.update({f"coarsen/{k}": v for k, v in coarsen_stats.items()})
    cost_stats["coarsen/delta_t"] = float(delta_t)
    for key, value in cost_stats.items():
        wandb.summary[key] = value

    summary_payload = {
        "status": "success",
        "model": model_name,
        "data": args.data,
        "n_params": n_params,
        "save_probs": save_probs,
        "truth_start": truth_start,
        "metrics": {
            f"{metric_key}_mean": float(np.mean(vals))
            for metric_key, vals in sorted(rep_metric_lists.items())
        } | {
            f"{metric_key}_std": _sample_std(vals)
            for metric_key, vals in sorted(rep_metric_lists.items())
        } | {
            f"{metric_key}_ci95": _ci95_halfwidth(vals)
            for metric_key, vals in sorted(rep_metric_lists.items())
        } | {
            key: value for key, value in cost_stats.items()
        },
    }
    _write_json(f"{run_dir}/metrics_summary.json", summary_payload)
    if checkpoint_cfg["enabled"]:
        atomic_json_dump(
            {
                "status": "success",
                "model": model_name,
                "data": args.data,
                "run_id": wandb.run.id,
                "checkpoint_root": str(checkpoint_root),
                "n_params": n_params,
                "summary": summary_payload,
                "updated_at": checkpoint_timestamp(),
            },
            checkpoint_root / "manifest.json",
        )

    with open(f"{run_dir}/metrics_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std"])
        for metric_key, vals in sorted(rep_metric_lists.items()):
            writer.writerow([metric_key, float(np.mean(vals)), float(np.std(vals))])

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
