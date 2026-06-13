"""
Stage 2a — Optuna hyperparameter optimisation for GNN source detection.

Runs an Optuna study on a validation slice of the TSIR ground-truth simulations,
logs each trial to W&B, persists the study in SQLite by default, and exports a
best-config YAML for a subsequent held-out ``main_train.py`` run.

Usage
-----
::

    python main_optuna.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest
    python main_train.py --cfg results/optuna/<study>/best_config.yml --data toy_holme:latest
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

from hpo import (
    apply_trial_params,
    default_trial_params,
    describe_search_space,
    suggest_hyperparameters,
)


DEFAULT_HPO = {
    "n_trials": 30,
    "timeout": None,
    "metric": "eval/mrr",
    "direction": "maximize",
    "sampler": "tpe",
    "pruner": "hyperband",
    "enqueue_default": True,
    "reps": 1,
    "n_truth": None,
    "truth_start": 0,
    "final_truth_start": None,
    "final_n_truth": None,
    "tune_n_mc": False,
    "trial_epochs": None,
    "trial_patience": None,
    "trial_n_mc": None,
    "study_name": None,
    "storage": None,
    "output_dir": "results/optuna",
    "load_if_exists": True,
    "seed": None,
}


def _extra_wandb_tags_from_env() -> list[str]:
    return [
        tag.strip()
        for tag in os.getenv("WANDB_TAGS", "").split(",")
        if tag.strip()
    ]


@dataclass(frozen=True)
class TruthBudget:
    """Validation and final-evaluation truth windows."""

    hpo_reps: int
    hpo_n_truth: int
    hpo_truth_start: int
    final_reps: int
    final_n_truth: int
    final_truth_start: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg", required=True, help="Base model config YAML")
    p.add_argument("--data", required=True, help="W&B artifact ref or local TSIR run id")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--timeout", type=int, default=None, help="Study timeout in seconds")
    p.add_argument("--study-name", default=None)
    p.add_argument("--storage", default=None, help="Optuna storage URL")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--metric", default=None, help="Metric to optimise, e.g. eval/mrr")
    p.add_argument("--direction", choices=["maximize", "minimize"], default=None)
    p.add_argument("--sampler", choices=["tpe", "random"], default=None)
    p.add_argument("--pruner", choices=["hyperband", "median", "none"], default=None)
    p.add_argument("--enqueue-default", dest="enqueue_default", action="store_true", default=None,
                   help="Evaluate the base config as an explicit first trial so the tuned "
                        "result can never be selected worse than the default (default on).")
    p.add_argument("--no-enqueue-default", dest="enqueue_default", action="store_false",
                   help="Disable enrolling the base config as a protected candidate trial.")
    p.add_argument("--trial-epochs", type=int, default=None,
                   help="Optional epoch cap used only inside Optuna trials")
    p.add_argument("--trial-patience", type=int, default=None,
                   help="Optional early-stopping patience cap used only inside Optuna trials")
    p.add_argument("--trial-n-mc", type=int, default=None,
                   help="Optional train.n_mc cap used only inside Optuna trials")
    p.add_argument("--wandb-project", default="source-detection")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore local trial checkpoints when training trial models.")
    p.add_argument("--dry-run", action="store_true", help="Print study plan only")
    return p.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _load_cfg(path: str, overrides: list[str]) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _apply_overrides(cfg, overrides)
    return cfg


def _apply_overrides(cfg_dict: dict, overrides: list[str]) -> None:
    """Apply ``key.subkey=value`` overrides to a nested config dict in-place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be in key=value format")
        key_path, raw_val = item.split("=", 1)
        keys = key_path.strip().split(".")
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
        for key in keys[:-1]:
            node = node.setdefault(key, {})
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


def _jsonable(value):
    """Convert numpy-like scalars/arrays to JSON-friendly Python values."""
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
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


def _truth_indices_for_rep(
    eval_cfg: dict,
    rep: int,
    n_truth: int,
    n_runs: int,
    reps: int,
):
    """Return the held-out truth-run indices for one repetition."""
    truth_start = int(eval_cfg.get("truth_start", 0))
    if truth_start < 0:
        raise ValueError(f"eval.truth_start must be non-negative, got {truth_start}")
    truth_stop = truth_start + reps * n_truth
    if truth_stop > n_runs:
        raise ValueError(
            f"eval.truth_start + reps * n_truth = {truth_start} + {reps} * "
            f"{n_truth} = {truth_stop} exceeds n_runs={n_runs}."
        )
    start = truth_start + rep * n_truth
    return np.arange(start, start + n_truth)


def _hpo_settings(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    hpo_cfg = {**DEFAULT_HPO, **(cfg.get("hpo") or {})}
    for key, attr in (
        ("n_trials", "n_trials"),
        ("timeout", "timeout"),
        ("study_name", "study_name"),
        ("storage", "storage"),
        ("output_dir", "output_dir"),
        ("metric", "metric"),
        ("direction", "direction"),
        ("sampler", "sampler"),
        ("pruner", "pruner"),
        ("enqueue_default", "enqueue_default"),
        ("trial_epochs", "trial_epochs"),
        ("trial_patience", "trial_patience"),
        ("trial_n_mc", "trial_n_mc"),
    ):
        value = getattr(args, attr)
        if value is not None:
            hpo_cfg[key] = value
    if hpo_cfg["seed"] is None:
        hpo_cfg["seed"] = int(cfg.get("train", {}).get("seed", 42))
    return hpo_cfg


def resolve_truth_budget(
    data_n_runs: int,
    eval_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    hpo_cfg: dict[str, Any],
) -> TruthBudget:
    """Choose disjoint truth windows for tuning and final reporting."""
    hpo_reps = int(hpo_cfg.get("reps") or 1)
    final_reps = int(train_cfg.get("reps", 1))
    hpo_truth_start = int(hpo_cfg.get("truth_start", eval_cfg.get("truth_start", 0)))
    if hpo_truth_start < 0:
        raise ValueError("hpo.truth_start must be non-negative")

    requested_eval_n_truth = int(eval_cfg["n_truth"])
    remaining = data_n_runs - hpo_truth_start
    if remaining <= hpo_reps:
        raise ValueError(
            f"Not enough ground-truth runs for HPO: n_runs={data_n_runs}, "
            f"hpo.truth_start={hpo_truth_start}."
        )

    if hpo_cfg.get("n_truth") is None:
        divisor = max(hpo_reps + final_reps, 2)
        hpo_n_truth = min(requested_eval_n_truth, max(1, remaining // divisor))
    else:
        hpo_n_truth = int(hpo_cfg["n_truth"])

    hpo_stop = hpo_truth_start + hpo_reps * hpo_n_truth
    if hpo_stop > data_n_runs:
        raise ValueError(
            f"HPO validation window ends at {hpo_stop}, but artifact has only "
            f"{data_n_runs} ground-truth runs."
        )

    if hpo_cfg.get("final_truth_start") is None:
        final_truth_start = hpo_stop
    else:
        final_truth_start = int(hpo_cfg["final_truth_start"])
    if final_truth_start < hpo_stop:
        raise ValueError(
            "hpo.final_truth_start overlaps the HPO validation window. "
            f"Use at least {hpo_stop}."
        )

    final_remaining = data_n_runs - final_truth_start
    if final_remaining <= 0:
        raise ValueError(
            f"No held-out truth runs remain after final_truth_start={final_truth_start}."
        )
    if hpo_cfg.get("final_n_truth") is None:
        final_n_truth = min(requested_eval_n_truth, max(1, final_remaining // final_reps))
    else:
        final_n_truth = int(hpo_cfg["final_n_truth"])

    final_stop = final_truth_start + final_reps * final_n_truth
    if final_stop > data_n_runs:
        raise ValueError(
            f"Final evaluation window ends at {final_stop}, but artifact has only "
            f"{data_n_runs} ground-truth runs."
        )

    return TruthBudget(
        hpo_reps=hpo_reps,
        hpo_n_truth=hpo_n_truth,
        hpo_truth_start=hpo_truth_start,
        final_reps=final_reps,
        final_n_truth=final_n_truth,
        final_truth_start=final_truth_start,
    )


def _make_sampler(optuna, name: str, seed: int, n_trials: int):
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "tpe":
        n_startup_trials = max(1, min(10, max(1, n_trials) // 3))
        try:
            return optuna.samplers.TPESampler(
                seed=seed,
                multivariate=True,
                group=True,
                n_startup_trials=n_startup_trials,
            )
        except TypeError:
            return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
    raise ValueError(f"Unknown sampler: {name}")


def _make_pruner(optuna, name: str, max_epochs: int):
    if name == "none":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=max(5, min(50, max_epochs // 10)),
        )
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=max(5, min(20, max_epochs // 20)),
            max_resource=max_epochs,
            reduction_factor=3,
        )
    raise ValueError(f"Unknown pruner: {name}")


def _positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    value_int = int(value)
    if value_int <= 0:
        raise ValueError(f"hpo.{name} must be positive when set, got {value}")
    return value_int


def apply_trial_budget(trial_cfg: dict[str, Any], hpo_cfg: dict[str, Any]) -> None:
    """Apply short-budget settings to Optuna trials without changing final configs."""
    train_cfg = trial_cfg.setdefault("train", {})
    trial_epochs = _positive_int(hpo_cfg.get("trial_epochs"), "trial_epochs")
    trial_patience = _positive_int(hpo_cfg.get("trial_patience"), "trial_patience")
    trial_n_mc = _positive_int(hpo_cfg.get("trial_n_mc"), "trial_n_mc")

    if trial_epochs is not None:
        current = int(train_cfg.get("epochs", trial_epochs))
        train_cfg["epochs"] = min(current, trial_epochs)
    if trial_patience is not None:
        current = int(train_cfg.get("patience", trial_patience))
        train_cfg["patience"] = min(current, trial_patience)
    if trial_n_mc is not None:
        current = int(train_cfg.get("n_mc", trial_n_mc))
        train_cfg["n_mc"] = min(current, trial_n_mc)


def _objective_from_metrics(metrics: dict[str, float], metric: str) -> float:
    if metric in metrics:
        value = metrics[metric]
    else:
        mean_key = f"{metric}_mean"
        if mean_key not in metrics:
            raise KeyError(
                f"Objective metric '{metric}' not found. Available: {sorted(metrics)}"
            )
        value = metrics[mean_key]
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Objective metric '{metric}' is non-finite: {value}")
    return value


def _is_resource_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "exceeded" in msg
        or "out of memory" in msg
        or "cuda out of memory" in msg
        or "resource" in msg
    )


def _aggregate_metrics(rep_metric_lists: dict[str, list[float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, vals in sorted(rep_metric_lists.items()):
        out[f"{key}_mean"] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out


def _remaining_trials_for_target(trials, terminal_states: set, target_trials: int) -> int:
    """Return how many additional trials are needed to reach a target total."""
    finished_trials = sum(1 for trial in trials if trial.state in terminal_states)
    return max(0, int(target_trials) - finished_trials)


def run_trial(
    *,
    trial_cfg: dict[str, Any],
    data_name: str,
    H: nx.Graph,
    H_static: nx.Graph,
    data,
    model_name: str,
    truth_budget: TruthBudget,
    optuna_trial,
    direction: str,
    run_dir: Path,
    checkpoint_root: Path | None = None,
    checkpoint_fresh: bool = False,
) -> dict[str, Any]:
    """Train/evaluate one Optuna trial and return aggregate metrics."""
    import torch
    import wandb

    from eval import compute_all_metrics
    from gnn import get_model_spec
    from training import SIRDataset, Trainer
    from training.checkpointing import atomic_json_dump, checkpoint_timestamp

    train_cfg = trial_cfg["train"]
    eval_cfg = trial_cfg["eval"]
    model_cfg = trial_cfg[model_name]

    if train_cfg["n_mc"] > data.mc_runs:
        raise ValueError(
            f"n_mc={train_cfg['n_mc']} requested but artifact has {data.mc_runs} MC runs."
        )

    spec = get_model_spec(model_name)
    graph_data = spec.builder_fn(H, **_builder_kwargs(model_name, model_cfg))
    graph_data["n_nodes"] = data.n_nodes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(train_cfg.get("seed", 42)))
    torch.manual_seed(int(train_cfg.get("seed", 42)))
    rep_metric_lists: dict[str, list[float]] = {}
    n_params = 0

    for rep in range(truth_budget.hpo_reps):
        n_mc = int(train_cfg["n_mc"])
        select = rng.choice(data.mc_runs, n_mc, replace=False)
        dataset = SIRDataset(
            data.mc_S[:, select, :],
            data.mc_I[:, select, :],
            data.mc_R[:, select, :],
        )

        torch.manual_seed(int(train_cfg.get("seed", 42)) + rep)
        model = spec.build_fn(model_cfg, data.n_nodes, graph_data)
        n_params = sum(p.numel() for p in model.parameters())
        trainer = Trainer(model, spec.forward_fn, graph_data, device)
        rep_checkpoint_dir = checkpoint_root / f"rep{rep}" if checkpoint_root is not None else None
        trial_metadata = {
            "model": model_name,
            "data": data_name,
            "trial": int(optuna_trial.number),
            "rep": int(rep),
            "cfg": trial_cfg,
            "n_nodes": int(data.n_nodes),
            "graph_builder_kwargs": _builder_kwargs(model_name, model_cfg),
            "selected_mc_indices": np.asarray(select, dtype=np.int64).tolist(),
            "truth_budget": truth_budget.__dict__,
        }
        train_losses, val_losses = trainer.fit(
            dataset=dataset,
            batch_size=int(train_cfg["batch_size"]),
            epochs=int(train_cfg["epochs"]),
            patience=int(train_cfg["patience"]),
            lr=float(train_cfg["lr"]),
            weight_decay=float(train_cfg["weight_decay"]),
            test_size=float(train_cfg["test_size"]),
            seed=int(train_cfg.get("seed", 42)) + rep,
            wandb_run=wandb.run,
            rep=rep,
            loss_guard=train_cfg.get("loss_guard"),
            optuna_trial=optuna_trial,
            optuna_report_sign=-1.0 if direction == "maximize" else 1.0,
            optuna_step_offset=rep * int(train_cfg["epochs"]),
            checkpoint_dir=rep_checkpoint_dir,
            checkpoint_metadata=trial_metadata,
            checkpoint_enabled=True,
            checkpoint_resume=True,
            checkpoint_fresh=checkpoint_fresh,
            checkpoint_save_every=int(trial_cfg.get("checkpoint", {}).get("save_every", 1)),
            final_model_path=(
                rep_checkpoint_dir / f"final_model_rep{rep}.pt"
                if rep_checkpoint_dir is not None else None
            ),
        )
        _write_loss_history(str(run_dir / f"loss_history_rep{rep}.csv"), train_losses, val_losses)
        if rep_checkpoint_dir is not None:
            _write_loss_history(
                str(rep_checkpoint_dir / f"loss_history_rep{rep}.csv"),
                train_losses,
                val_losses,
            )

        # Smoothed validation NLL — the Sterchi-style model-selection signal:
        # "performance is measured as the average of the last five validation
        # losses". It is computed on the large held-out validation split of the
        # MC training data, so it is far less noisy than the truth-window MRR on
        # a small window (which previously caused TPE to select under-fitting
        # configs). Selection uses this when hpo.metric = "eval/val_nll".
        val_window = int((trial_cfg.get("hpo") or {}).get("val_loss_window", 5))
        if val_losses:
            k = max(1, min(val_window, len(val_losses)))
            rep_metric_lists.setdefault("eval/val_nll", []).append(
                float(np.mean(val_losses[-k:]))
            )

        select_truth = _truth_indices_for_rep(
            eval_cfg,
            rep=rep,
            n_truth=truth_budget.hpo_n_truth,
            n_runs=data.n_runs,
            reps=truth_budget.hpo_reps,
        )
        probs = trainer.predict_from_tensor(
            truth_S=data.truth_S[:, select_truth, :],
            truth_I=data.truth_I[:, select_truth, :],
            truth_R=data.truth_R[:, select_truth, :],
            batch_size=256,
        )
        lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, data.n_nodes)
        truth_S_flat = data.truth_S[:, select_truth, :].reshape(-1, data.n_nodes)
        rep_metrics = compute_all_metrics(
            probs=probs,
            lik_possible=lik_possible,
            truth_S_flat=truth_S_flat,
            eval_cfg=eval_cfg,
            n_nodes=data.n_nodes,
            n_runs=truth_budget.hpo_n_truth,
            H_static=H_static,
        )
        wandb.log({f"hpo/{key}_rep{rep}": value for key, value in rep_metrics.items()})
        _write_json(
            str(run_dir / f"metrics_rep{rep}.json"),
            {"rep": rep, "metrics": rep_metrics},
        )
        if rep_checkpoint_dir is not None:
            _write_json(
                str(rep_checkpoint_dir / f"metrics_rep{rep}.json"),
                {"rep": rep, "metrics": rep_metrics},
            )
            atomic_json_dump(
                {
                    "status": "evaluated",
                    "compatibility_hash": trainer.last_fit_info.get("compatibility_hash"),
                    "updated_at": checkpoint_timestamp(),
                    "model": model_name,
                    "data": data_name,
                    "trial": int(optuna_trial.number),
                    "rep": int(rep),
                    "n_params": int(n_params),
                    "metrics_path": str(rep_checkpoint_dir / f"metrics_rep{rep}.json"),
                    "final_model": str(rep_checkpoint_dir / f"final_model_rep{rep}.pt"),
                },
                rep_checkpoint_dir / "state.json",
            )
        for key, value in rep_metrics.items():
            if key != "eval/n_valid":
                rep_metric_lists.setdefault(key, []).append(float(value))

    metrics = _aggregate_metrics(rep_metric_lists)
    metrics["model/n_params"] = float(n_params)
    return metrics


def _write_trials_csv(study, path: Path) -> None:
    rows = []
    param_keys: set[str] = set()
    attr_keys: set[str] = set()
    for trial in study.trials:
        param_keys.update(trial.params)
        attr_keys.update(k for k in trial.user_attrs if not k.startswith("_"))
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
        }
        for key in sorted(param_keys):
            row[f"param/{key}"] = trial.params.get(key)
        for key in sorted(attr_keys):
            value = trial.user_attrs.get(key)
            row[f"user/{key}"] = json.dumps(_jsonable(value), sort_keys=True)
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_best_outputs(
    *,
    base_cfg: dict[str, Any],
    best_trial,
    truth_budget: TruthBudget,
    hpo_cfg: dict[str, Any],
    output_dir: Path,
    data_name: str,
) -> tuple[Path, Path]:
    applied_params = best_trial.user_attrs.get("applied_params", dict(best_trial.params))
    best_cfg = copy.deepcopy(base_cfg)
    apply_trial_params(best_cfg, applied_params)
    best_cfg["eval"]["truth_start"] = truth_budget.final_truth_start
    best_cfg["eval"]["n_truth"] = truth_budget.final_n_truth
    best_cfg.setdefault("experiment", {})
    best_cfg["experiment"]["hpo_condition"] = "optuna"
    base_variant = best_cfg["experiment"].get("model_variant", base_cfg["model"])
    best_cfg["experiment"]["model_variant"] = f"{base_variant}_optuna"
    best_cfg.setdefault("hpo_result", {})
    best_cfg["hpo_result"].update({
        "best_trial": int(best_trial.number),
        "best_value": float(best_trial.value),
        "metric": hpo_cfg["metric"],
        "direction": hpo_cfg["direction"],
        "validation_truth_start": truth_budget.hpo_truth_start,
        "validation_n_truth": truth_budget.hpo_n_truth,
        "validation_reps": truth_budget.hpo_reps,
        "final_truth_start": truth_budget.final_truth_start,
        "final_n_truth": truth_budget.final_n_truth,
        "data": data_name,
        "params": _jsonable(applied_params),
    })
    best_path = output_dir / "best_config.yml"
    with open(best_path, "w") as f:
        yaml.safe_dump(_jsonable(best_cfg), f, sort_keys=False)

    overrides_path = output_dir / "best_overrides.txt"
    with open(overrides_path, "w") as f:
        for key, value in sorted(applied_params.items()):
            f.write(f"{key}={value}\n")
        f.write(f"eval.truth_start={truth_budget.final_truth_start}\n")
        f.write(f"eval.n_truth={truth_budget.final_n_truth}\n")
    return best_path, overrides_path


def _log_summary_run(
    *,
    project: str,
    study_name: str,
    study,
    hpo_cfg: dict[str, Any],
    best_config_path: Path,
    trials_csv_path: Path,
    best_json_path: Path,
    storage: str,
    output_dir: Path,
) -> None:
    import wandb

    run = wandb.init(
        project=project,
        job_type="optuna_summary",
        group=study_name,
        name=f"{study_name}-summary",
        config={"hpo": hpo_cfg, "storage": storage},
        tags=["optuna", "optuna_summary", *_extra_wandb_tags_from_env()],
    )
    table = wandb.Table(columns=["number", "state", "value", "params"])
    for trial in study.trials:
        table.add_data(trial.number, trial.state.name, trial.value, json.dumps(_jsonable(trial.params)))
    wandb.log({"optuna/trials": table})
    wandb.summary["optuna/best_trial"] = int(study.best_trial.number)
    wandb.summary["optuna/best_value"] = float(study.best_value)
    wandb.summary["optuna/metric"] = hpo_cfg["metric"]
    wandb.summary["optuna/direction"] = hpo_cfg["direction"]

    artifact = wandb.Artifact(_safe_name(f"{study_name}_optuna"), type="optuna-study")
    for path in (best_config_path, trials_csv_path, best_json_path):
        artifact.add_file(str(path))
    db_path = output_dir / "study.db"
    if db_path.exists():
        artifact.add_file(str(db_path))
    run.log_artifact(artifact)
    wandb.finish()


def main() -> None:
    args = parse_args()
    cfg = _load_cfg(args.cfg, args.override)
    model_name = cfg["model"]

    hpo_cfg = _hpo_settings(cfg, args)
    if hpo_cfg["direction"] not in {"maximize", "minimize"}:
        raise ValueError("hpo.direction must be 'maximize' or 'minimize'")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_key = _safe_name(args.data.split(":")[0])
    study_name = hpo_cfg["study_name"] or f"{data_key}_{model_name}_{timestamp}"
    hpo_cfg["study_name"] = study_name
    output_root = Path(hpo_cfg["output_dir"])
    output_dir = output_root / study_name
    storage = hpo_cfg["storage"] or f"sqlite:///{output_dir / 'study.db'}"
    hpo_cfg["storage"] = storage

    if args.dry_run:
        print("Optuna HPO dry run")
        print(f"Study      : {study_name}")
        print(f"Output dir : {output_dir}")
        print(f"Storage    : {storage}")
        print(f"Metric     : {hpo_cfg['metric']} ({hpo_cfg['direction']})")
        print(
            "Trial caps : "
            f"epochs={hpo_cfg.get('trial_epochs')}, "
            f"patience={hpo_cfg.get('trial_patience')}, "
            f"n_mc={hpo_cfg.get('trial_n_mc')}"
        )
        print("Search space:")
        for key, value in describe_search_space(model_name).items():
            print(f"  {key}: {value}")
        return

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Install project dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    import networkx as nx
    import torch
    import wandb

    from gnn import MODEL_REGISTRY
    from setup import load_tsir_data
    from training import LossGuardAbort

    torch.set_num_threads(1)

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Registered: {sorted(MODEL_REGISTRY)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    controller = wandb.init(
        project=args.wandb_project,
        job_type="optuna_controller",
        group=study_name,
        name=f"{study_name}-controller",
        config={"model": model_name, "data_name": args.data, **cfg, "hpo": hpo_cfg},
        tags=["optuna", "optuna_controller", f"model:{model_name}", *_extra_wandb_tags_from_env()],
    )
    H, data = load_tsir_data(args.data)
    controller.summary["data/n_nodes"] = data.n_nodes
    controller.summary["data/n_runs"] = data.n_runs
    controller.summary["data/mc_runs"] = data.mc_runs
    wandb.finish()

    H_static = nx.Graph()
    H_static.add_nodes_from(range(data.n_nodes))
    for u, v in H.edges():
        H_static.add_edge(int(u), int(v))

    truth_budget = resolve_truth_budget(
        data_n_runs=data.n_runs,
        eval_cfg=cfg["eval"],
        train_cfg=cfg["train"],
        hpo_cfg=hpo_cfg,
    )
    manifest = {
        "study_name": study_name,
        "data": args.data,
        "model": model_name,
        "hpo": hpo_cfg,
        "truth_budget": truth_budget.__dict__,
        "search_space": describe_search_space(model_name),
        "base_cfg": cfg,
    }
    _write_json(str(output_dir / "manifest.json"), manifest)

    n_trials = int(hpo_cfg["n_trials"])
    pruner_epochs = int(hpo_cfg.get("trial_epochs") or cfg["train"]["epochs"])
    sampler = _make_sampler(optuna, str(hpo_cfg["sampler"]), int(hpo_cfg["seed"]), n_trials)
    pruner = _make_pruner(optuna, str(hpo_cfg["pruner"]), pruner_epochs)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=str(hpo_cfg["direction"]),
        sampler=sampler,
        pruner=pruner,
        load_if_exists=bool(hpo_cfg.get("load_if_exists", True)),
    )

    # Protect the strong hand-tuned default: enqueue it as an explicit trial so
    # the selected best can never be worse than the default on the validation
    # window. ``skip_if_exists`` keeps this idempotent across resumes.
    if bool(hpo_cfg.get("enqueue_default", True)):
        default_params = default_trial_params(cfg, model_name)
        if default_params:
            study.enqueue_trial(
                default_params,
                user_attrs={"is_default_config": True},
                skip_if_exists=True,
            )
            print(f"Enqueued default-config trial as a protected candidate: {default_params}")

    max_batch_size = data.n_nodes * min(int(cfg["train"]["n_mc"]), data.mc_runs)

    def objective(trial) -> float:
        trial_cfg = copy.deepcopy(cfg)
        params = suggest_hyperparameters(
            trial,
            trial_cfg,
            model_name,
            max_batch_size=max_batch_size,
            tune_n_mc=bool(hpo_cfg.get("tune_n_mc", False)),
            max_n_mc=data.mc_runs,
        )
        apply_trial_params(trial_cfg, params)
        apply_trial_budget(trial_cfg, hpo_cfg)
        trial_cfg["train"]["reps"] = truth_budget.hpo_reps
        trial_cfg["train"]["seed"] = int(cfg["train"].get("seed", 42)) + trial.number * 1009
        trial_cfg["eval"]["n_truth"] = truth_budget.hpo_n_truth
        trial_cfg["eval"]["truth_start"] = truth_budget.hpo_truth_start
        trial.set_user_attr("applied_params", _jsonable(params))

        run = wandb.init(
            project=args.wandb_project,
            job_type="optuna_trial",
            group=study_name,
            name=f"{study_name}-trial-{trial.number:04d}",
            config={
                "model": model_name,
                "data_name": args.data,
                **trial_cfg,
                "optuna": {
                    "trial_number": trial.number,
                    "study_name": study_name,
                    "metric": hpo_cfg["metric"],
                    "direction": hpo_cfg["direction"],
                    "params": _jsonable(params),
                    "trial_budget": {
                        "epochs": trial_cfg["train"].get("epochs"),
                        "patience": trial_cfg["train"].get("patience"),
                        "n_mc": trial_cfg["train"].get("n_mc"),
                    },
                },
            },
            tags=["optuna", "optuna_trial", f"model:{model_name}", *_extra_wandb_tags_from_env()],
        )
        run_dir = Path("data") / wandb.run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            metrics = run_trial(
                trial_cfg=trial_cfg,
                data_name=args.data,
                H=H,
                H_static=H_static,
                data=data,
                model_name=model_name,
                truth_budget=truth_budget,
                optuna_trial=trial,
                direction=str(hpo_cfg["direction"]),
                run_dir=run_dir,
                checkpoint_root=output_dir / "checkpoints" / f"trial_{trial.number:04d}",
                checkpoint_fresh=bool(args.fresh),
            )
            objective_value = _objective_from_metrics(metrics, str(hpo_cfg["metric"]))
            wandb.log({"optuna/objective": objective_value, **{f"hpo/{k}": v for k, v in metrics.items()}})
            wandb.summary["optuna/objective"] = objective_value
            wandb.summary["optuna/status"] = "complete"
            wandb.summary["run/status"] = "success"
            _write_json(
                str(run_dir / "optuna_trial_summary.json"),
                {
                    "trial": trial.number,
                    "objective": objective_value,
                    "metrics": metrics,
                    "params": params,
                },
            )
            trial.set_user_attr("metrics", _jsonable(metrics))
            trial.set_user_attr("wandb_run_id", wandb.run.id)
            return objective_value
        except LossGuardAbort as exc:
            trial.set_user_attr("pruned_reason", exc.reason)
            wandb.summary["optuna/status"] = "pruned"
            wandb.summary["run/status"] = "loss_guard_aborted"
            raise optuna.TrialPruned(str(exc)) from exc
        except optuna.TrialPruned:
            wandb.summary["optuna/status"] = "pruned"
            wandb.summary["run/status"] = "pruned"
            raise
        except (RuntimeError, ValueError) as exc:
            if _is_resource_error(exc):
                trial.set_user_attr("pruned_reason", str(exc))
                wandb.summary["optuna/status"] = "resource_pruned"
                wandb.summary["run/status"] = "resource_pruned"
                raise optuna.TrialPruned(str(exc)) from exc
            wandb.summary["optuna/status"] = "failed"
            wandb.summary["run/status"] = "failed"
            raise
        finally:
            wandb.finish()

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
        optuna.trial.TrialState.FAIL,
    }
    finished_trials = sum(1 for trial in study.trials if trial.state in terminal_states)
    remaining_trials = _remaining_trials_for_target(study.trials, terminal_states, n_trials)
    if remaining_trials == 0:
        print(
            f"Optuna study already has {finished_trials} finished trials; "
            f"target is {n_trials}, so no new trials are scheduled."
        )
    else:
        if finished_trials:
            print(
                f"Resuming Optuna study with {finished_trials}/{n_trials} "
                f"finished trials; scheduling {remaining_trials} more."
            )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=hpo_cfg.get("timeout"),
            gc_after_trial=True,
        )

    completed = [
        trial for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        raise RuntimeError("Optuna study finished without any complete trials.")

    trials_csv = output_dir / "trials.csv"
    _write_trials_csv(study, trials_csv)
    best_config, overrides_path = _write_best_outputs(
        base_cfg=cfg,
        best_trial=study.best_trial,
        truth_budget=truth_budget,
        hpo_cfg=hpo_cfg,
        output_dir=output_dir,
        data_name=args.data,
    )
    best_json = output_dir / "best_trial.json"
    _write_json(
        str(best_json),
        {
            "number": int(study.best_trial.number),
            "value": float(study.best_value),
            "params": study.best_trial.user_attrs.get("applied_params", study.best_trial.params),
            "metrics": study.best_trial.user_attrs.get("metrics", {}),
            "best_config": str(best_config),
            "best_overrides": str(overrides_path),
        },
    )
    _log_summary_run(
        project=args.wandb_project,
        study_name=study_name,
        study=study,
        hpo_cfg=hpo_cfg,
        best_config_path=best_config,
        trials_csv_path=trials_csv,
        best_json_path=best_json,
        storage=storage,
        output_dir=output_dir,
    )

    print("\nOptuna study complete.")
    print(f"Best value : {study.best_value:.6g}")
    print(f"Best trial : {study.best_trial.number}")
    print(f"Best config: {best_config}")
    print(f"Trials CSV : {trials_csv}")


if __name__ == "__main__":
    main()
