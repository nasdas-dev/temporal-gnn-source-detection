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

import numpy as np
import torch
import wandb
import yaml

from eval import compute_ranks, top_k_score, rank_score
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
    return p.parse_args()


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

    all_top_k   = {k: [] for k in top_k_vals}
    all_rs      = {o: [] for o in offsets}

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

        # --- Compute ranks & metrics ---
        lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, n_nodes)
        truth_S_flat = data.truth_S[:, select_truth, :].reshape(-1, n_nodes)
        sel = (1 - truth_S_flat).sum(axis=1) >= eval_cfg["min_outbreak"]

        log_probs = np.log(np.clip(probs, 1e-12, 1.0)) - lik_possible
        ranks = compute_ranks(log_probs, n_nodes=n_nodes, n_runs=n_truth)

        print(f"\n  Valid outbreaks: {sel.sum()} / {len(sel)}")
        rep_metrics: dict[str, float] = {}
        for k in top_k_vals:
            score = float(top_k_score(ranks, sel, k))
            rep_metrics[f"eval/top_{k}"]     = score
            all_top_k[k].append(score)
            print(f"  top-{k}: {100 * score:.1f}%")
        for o in offsets:
            rs = float(rank_score(ranks, sel, o))
            rep_metrics[f"eval/rank_score_off{o}"] = rs
            all_rs[o].append(rs)
            print(f"  rank_score (offset={o}): {rs:.4f}")

        wandb.log({f"{k}_rep{rep}": v for k, v in rep_metrics.items()})

        # Save raw model outputs for reproducibility
        os.makedirs(f"data/{wandb.run.id}", exist_ok=True)
        torch.save(
            torch.tensor(probs),
            f"data/{wandb.run.id}/probs_rep{rep}.pt",
        )

    # ---------------------------------------------------------------
    # 6. Summary (averaged over reps)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary (mean ± std over reps)")
    print("=" * 60)
    for k in top_k_vals:
        vals = all_top_k[k]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        wandb.summary[f"eval/top_{k}_mean"] = mean
        wandb.summary[f"eval/top_{k}_std"]  = std
        print(f"  top-{k}: {100 * mean:.1f}% ± {100 * std:.1f}%")
    for o in offsets:
        vals = all_rs[o]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        wandb.summary[f"eval/rank_score_off{o}_mean"] = mean
        wandb.summary[f"eval/rank_score_off{o}_std"]  = std
        print(f"  rank_score (offset={o}): {mean:.4f} ± {std:.4f}")

    wandb.summary["model/n_params"] = n_params
    wandb.summary["model/name"]     = model_name
    wandb.summary["data/name"]      = args.data

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
