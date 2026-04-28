"""
Stage 2 — Unified GNN training and evaluation.

Trains any registered GNN model on a TSIR artifact produced by
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

The ``--cfg`` YAML must contain a top-level ``model:`` key matching a name
in ``MODEL_REGISTRY`` (e.g. ``backtracking``, ``static_gnn``, ``temporal_gnn``).
"""

from __future__ import annotations

import argparse
import os

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
from setup import setup_methods_run, load_tsir_data
from training import SIRDataset, Trainer


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
    return {}


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

    # ---------------------------------------------------------------
    # 3. Load TSIR artifact
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Loading TSIR data")
    print("=" * 60)
    H, data = load_tsir_data(args.data)
    n_nodes = data.n_nodes
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
    graph_data = spec.builder_fn(H, **bkw)
    graph_data["n_nodes"] = n_nodes   # ensure key present for forward fns

    for k, v in graph_data.items():
        if hasattr(v, "shape"):
            print(f"  {k:20s}: {tuple(v.shape)}")
        elif not isinstance(v, dict):
            print(f"  {k:20s}: {v}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device  : {device}")

    # ---------------------------------------------------------------
    # 5. Training repetitions
    # ---------------------------------------------------------------
    top_k_vals = eval_cfg["top_k"]
    offsets    = eval_cfg["inverse_rank_offset"]
    n_truth    = eval_cfg["n_truth"]
    reps       = train_cfg["reps"]

    # Aggregation buffers keyed by metric name (filled per rep, averaged in summary)
    rep_metric_lists: dict[str, list[float]] = {}

    torch.manual_seed(train_cfg["seed"])
    np.random.seed(train_cfg["seed"])

    for rep in range(reps):
        print("\n" + "=" * 60)
        print(f"Repetition {rep + 1}/{reps}")
        print("=" * 60)

        # --- Sample MC runs ---
        n_mc   = train_cfg["n_mc"]
        select = np.random.choice(data.mc_runs, n_mc, replace=False)
        dataset = SIRDataset(
            data.mc_S[:, select, :],
            data.mc_I[:, select, :],
            data.mc_R[:, select, :],
        )

        # --- Build fresh model ---
        model = spec.build_fn(model_cfg, n_nodes, graph_data)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        # --- Train ---
        trainer = Trainer(model, spec.forward_fn, graph_data, device)
        trainer.fit(
            dataset       = dataset,
            batch_size    = train_cfg["batch_size"],
            epochs        = train_cfg["epochs"],
            patience      = train_cfg["patience"],
            lr            = train_cfg["lr"],
            weight_decay  = train_cfg["weight_decay"],
            test_size     = train_cfg["test_size"],
            seed          = train_cfg["seed"] + rep,
            wandb_run     = wandb.run,
            rep           = rep,
        )

        # --- Inference on ground truth ---
        print("\n  Running inference on ground truth…")
        select_truth = np.arange(rep * n_truth, (rep + 1) * n_truth) % data.n_runs
        probs = trainer.predict_from_tensor(
            truth_S    = data.truth_S[:, select_truth, :],
            truth_I    = data.truth_I[:, select_truth, :],
            truth_R    = data.truth_R[:, select_truth, :],
            batch_size = 256,
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

        # Accumulate for cross-rep summary
        for metric_key, val in rep_metrics.items():
            if metric_key != "eval/n_valid":
                rep_metric_lists.setdefault(metric_key, []).append(val)

        # Save raw model outputs + lightweight eval arrays for viz scripts
        os.makedirs(f"data/{wandb.run.id}", exist_ok=True)
        torch.save(
            torch.tensor(probs),
            f"data/{wandb.run.id}/probs_rep{rep}.pt",
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
            f"data/{wandb.run.id}/eval_arrays_rep{rep}.npz",
            **arrays,
        )

    # ---------------------------------------------------------------
    # 6. Summary (averaged over reps)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary (mean ± std over reps)")
    print("=" * 60)
    for metric_key, vals in sorted(rep_metric_lists.items()):
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        wandb.summary[f"{metric_key}_mean"] = mean
        wandb.summary[f"{metric_key}_std"]  = std
        # Human-friendly output: percentages for top_k, 4dp for scalars
        if "top_" in metric_key:
            print(f"  {metric_key}: {100 * mean:.1f}% ± {100 * std:.1f}%")
        else:
            print(f"  {metric_key}: {mean:.4f} ± {std:.4f}")

    wandb.summary["model/n_params"] = n_params
    wandb.summary["model/name"]     = model_name
    wandb.summary["data/name"]      = args.data

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
