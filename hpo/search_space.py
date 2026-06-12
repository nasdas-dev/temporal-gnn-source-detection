"""Search spaces for Optuna studies.

The project compares several model families with very different graph
representations.  This module keeps the shared optimiser/training knobs
consistent while exposing only model-specific parameters that are meaningful
for each architecture.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol


class TrialLike(Protocol):
    """Small subset of the Optuna ``Trial`` API used by this module."""

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        ...

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
    ) -> float:
        ...

    def suggest_int(self, name: str, low: int, high: int) -> int:
        ...


TRAIN_BATCH_CHOICES = {
    "static_gnn": [32, 64, 128, 256],
    "static_mlp": [32, 64, 128, 256],
    "backtracking": [32, 64, 128],
    "temporal_gnn": [8, 16, 32, 64],
    "dag_gnn": [8, 16, 32, 64],
    "dbgnn": [4, 8, 16, 32, 64],
}


def _set_nested(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for key in parts[:-1]:
        node = node.setdefault(key, {})
    node[parts[-1]] = value


def apply_trial_params(cfg: dict[str, Any], params: Mapping[str, Any]) -> None:
    """Apply ``{"section.key": value}`` parameters to a nested config dict."""
    for key, value in params.items():
        _set_nested(cfg, key, value)


def _filtered_choices(choices: list[int], max_value: int | None) -> list[int]:
    if max_value is None:
        return choices
    filtered = [choice for choice in choices if choice <= max_value]
    return filtered or [min(choices)]


def _suggest_training_params(
    trial: TrialLike,
    model_name: str,
    *,
    max_batch_size: int | None,
    tune_n_mc: bool,
    max_n_mc: int | None,
) -> dict[str, Any]:
    batch_choices = _filtered_choices(
        TRAIN_BATCH_CHOICES.get(model_name, [32, 64, 128]),
        max_batch_size,
    )
    params: dict[str, Any] = {
        "train.lr": trial.suggest_float("train.lr", 1e-4, 5e-3, log=True),
        "train.weight_decay": trial.suggest_float(
            "train.weight_decay", 1e-6, 1e-2, log=True
        ),
        "train.batch_size": trial.suggest_categorical(
            "train.batch_size", batch_choices
        ),
        "train.test_size": trial.suggest_categorical(
            "train.test_size", [0.20, 0.25, 0.30]
        ),
        "train.patience": trial.suggest_categorical(
            "train.patience", [10, 20, 30]
        ),
    }
    if tune_n_mc and max_n_mc is not None:
        n_mc_choices = _filtered_choices([50, 100, 200, 300, 500, 750, 1000], max_n_mc)
        params["train.n_mc"] = trial.suggest_categorical("train.n_mc", n_mc_choices)
    return params


def _suggest_static_gnn(trial: TrialLike) -> dict[str, Any]:
    num_pre = trial.suggest_int("static_gnn.num_preprocess_layers", 0, 2)
    params: dict[str, Any] = {
        "static_gnn.num_preprocess_layers": num_pre,
        "static_gnn.num_postprocess_layers": trial.suggest_int(
            "static_gnn.num_postprocess_layers", 0, 2
        ),
        "static_gnn.num_conv_layers": trial.suggest_int(
            "static_gnn.num_conv_layers", 2, 5
        ),
        "static_gnn.aggr": trial.suggest_categorical(
            "static_gnn.aggr", ["sum", "mean", "max"]
        ),
        "static_gnn.hidden_channels": trial.suggest_categorical(
            "static_gnn.hidden_channels", [16, 32, 64, 128]
        ),
        "static_gnn.dropout_rate": trial.suggest_float(
            "static_gnn.dropout_rate", 0.0, 0.50
        ),
        "static_gnn.batch_norm": trial.suggest_categorical(
            "static_gnn.batch_norm", [True, False]
        ),
        "static_gnn.skip": trial.suggest_categorical(
            "static_gnn.skip", [True, False]
        ),
        "static_gnn.use_edge_weights": trial.suggest_categorical(
            "static_gnn.use_edge_weights", [False, True]
        ),
    }
    if num_pre > 0:
        params["static_gnn.embed_dim_preprocess"] = trial.suggest_categorical(
            "static_gnn.embed_dim_preprocess", [16, 32, 64, 128]
        )
    return params


def _suggest_static_mlp(trial: TrialLike) -> dict[str, Any]:
    num_pre = trial.suggest_int("static_mlp.num_preprocess_layers", 0, 2)
    params: dict[str, Any] = {
        "static_mlp.num_preprocess_layers": num_pre,
        "static_mlp.num_postprocess_layers": trial.suggest_int(
            "static_mlp.num_postprocess_layers", 0, 2
        ),
        "static_mlp.num_hidden_layers": trial.suggest_int(
            "static_mlp.num_hidden_layers", 1, 5
        ),
        "static_mlp.hidden_channels": trial.suggest_categorical(
            "static_mlp.hidden_channels", [16, 32, 64, 128]
        ),
        "static_mlp.dropout_rate": trial.suggest_float(
            "static_mlp.dropout_rate", 0.0, 0.50
        ),
        "static_mlp.batch_norm": trial.suggest_categorical(
            "static_mlp.batch_norm", [True, False]
        ),
        "static_mlp.skip": trial.suggest_categorical(
            "static_mlp.skip", [True, False]
        ),
    }
    if num_pre > 0:
        params["static_mlp.embed_dim_preprocess"] = trial.suggest_categorical(
            "static_mlp.embed_dim_preprocess", [16, 32, 64, 128]
        )
    return params


def _suggest_backtracking(trial: TrialLike) -> dict[str, Any]:
    return {
        "backtracking.hidden_dim": trial.suggest_categorical(
            "backtracking.hidden_dim", [16, 32, 64, 128]
        ),
        "backtracking.num_layers": trial.suggest_int(
            "backtracking.num_layers", 2, 8
        ),
    }


def _suggest_temporal_gnn(trial: TrialLike) -> dict[str, Any]:
    return {
        "temporal_gnn.hidden_channels": trial.suggest_categorical(
            "temporal_gnn.hidden_channels", [16, 32, 64, 128]
        ),
        "temporal_gnn.group_by_time": trial.suggest_categorical(
            "temporal_gnn.group_by_time", [1, 2, 4, 6, 8, 12, 24, 48]
        ),
    }


def _suggest_dag_gnn(trial: TrialLike) -> dict[str, Any]:
    return {
        "dag_gnn.hidden_channels": trial.suggest_categorical(
            "dag_gnn.hidden_channels", [16, 32, 64, 128]
        ),
        "dag_gnn.num_conv_layers": trial.suggest_int(
            "dag_gnn.num_conv_layers", 1, 4
        ),
        "dag_gnn.dropout_rate": trial.suggest_float("dag_gnn.dropout_rate", 0.0, 0.50),
        "dag_gnn.agg": trial.suggest_categorical("dag_gnn.agg", ["mean", "sum"]),
        "dag_gnn.delta_t": trial.suggest_categorical(
            "dag_gnn.delta_t", [None, 2, 4, 8, 12, 24, 48]
        ),
    }


def _suggest_dbgnn(trial: TrialLike) -> dict[str, Any]:
    return {
        "dbgnn.hidden_channels": trial.suggest_categorical(
            "dbgnn.hidden_channels", [32, 64, 128]
        ),
        "dbgnn.num_conv_layers": trial.suggest_int("dbgnn.num_conv_layers", 1, 4),
        "dbgnn.dropout_rate": trial.suggest_float("dbgnn.dropout_rate", 0.0, 0.50),
        "dbgnn.bipartite_agg": trial.suggest_categorical(
            "dbgnn.bipartite_agg", ["sum", "mean"]
        ),
        "dbgnn.order": trial.suggest_categorical("dbgnn.order", [2, 3]),
        "dbgnn.delta": trial.suggest_categorical(
            "dbgnn.delta", [None, 4, 8, 12, 24, 48]
        ),
        "dbgnn.time_bin_size": trial.suggest_categorical(
            "dbgnn.time_bin_size", [1, 2, 4, 8]
        ),
        "dbgnn.max_temporal_states": 2_000_000,
        "dbgnn.max_db_nodes": 500_000,
        "dbgnn.max_db_edges": 2_000_000,
    }


MODEL_SUGGESTERS = {
    "static_gnn": _suggest_static_gnn,
    "static_mlp": _suggest_static_mlp,
    "backtracking": _suggest_backtracking,
    "temporal_gnn": _suggest_temporal_gnn,
    "dag_gnn": _suggest_dag_gnn,
    "dbgnn": _suggest_dbgnn,
}


def suggest_hyperparameters(
    trial: TrialLike,
    cfg: Mapping[str, Any],
    model_name: str,
    *,
    max_batch_size: int | None = None,
    tune_n_mc: bool = False,
    max_n_mc: int | None = None,
) -> dict[str, Any]:
    """Return a flat dict of suggested parameters for one Optuna trial."""
    if model_name not in MODEL_SUGGESTERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {sorted(MODEL_SUGGESTERS)}"
        )
    params = _suggest_training_params(
        trial,
        model_name,
        max_batch_size=max_batch_size,
        tune_n_mc=tune_n_mc,
        max_n_mc=max_n_mc,
    )
    params.update(MODEL_SUGGESTERS[model_name](trial))
    locked = set((cfg.get("hpo") or {}).get("locked_params", []))
    for key in locked:
        params.pop(key, None)
    return params


_MISSING = object()


def _get_dotted(cfg: Mapping[str, Any], dotted_key: str) -> Any:
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _snap_categorical(value: Any, choices: list[Any]) -> Any:
    """Return ``value`` if it is a valid choice, else the nearest numeric choice."""
    for choice in choices:
        # Exact match (and exact type for bools, since True == 1 in Python).
        if value is choice or (value == choice and isinstance(value, bool) == isinstance(choice, bool)):
            return choice
    numeric = [c for c in choices if isinstance(c, (int, float)) and not isinstance(c, bool)]
    if numeric and isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(numeric, key=lambda c: abs(c - value))
    return _MISSING


def default_trial_params(cfg: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """Extract the base config's values for every tunable parameter.

    The returned dict is suitable for ``optuna.Study.enqueue_trial`` so that the
    strong hand-tuned default configuration is evaluated as an explicit trial and
    therefore protected as a candidate: the selected best trial can never score
    worse (on the validation window) than the default. Values are snapped into
    each parameter's search distribution — clipped for int/float ranges, matched
    or rounded to the nearest numeric choice for categoricals — so the enqueued
    trial is always valid. Parameters absent from the config or locked via
    ``hpo.locked_params`` are skipped (Optuna samples them normally).
    """
    spec = describe_search_space(model_name)
    locked = set((cfg.get("hpo") or {}).get("locked_params", []))
    params: dict[str, Any] = {}
    for key, dist in spec.items():
        if key in locked:
            continue
        value = _get_dotted(cfg, key)
        if value is _MISSING or value is None:
            continue
        if isinstance(dist, list):
            snapped = _snap_categorical(value, dist)
            if snapped is not _MISSING:
                params[key] = snapped
            continue
        match = re.fullmatch(
            r"(int|loguniform|uniform)\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]",
            str(dist),
        )
        if not match:
            continue
        kind, low, high = match.group(1), float(match.group(2)), float(match.group(3))
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        numeric = max(low, min(high, numeric))
        params[key] = int(round(numeric)) if kind == "int" else numeric
    return params


def describe_search_space(model_name: str) -> dict[str, Any]:
    """Human-readable search-space summary for dry-runs and manifests."""
    if model_name not in MODEL_SUGGESTERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported: {sorted(MODEL_SUGGESTERS)}"
        )
    general = {
        "train.lr": "loguniform[1e-4, 5e-3]",
        "train.weight_decay": "loguniform[1e-6, 1e-2]",
        "train.batch_size": TRAIN_BATCH_CHOICES.get(model_name, [32, 64, 128]),
        "train.test_size": [0.20, 0.25, 0.30],
        "train.patience": [10, 20, 30],
    }
    model_spaces = {
        "static_gnn": {
            "static_gnn.num_preprocess_layers": "int[0, 2]",
            "static_gnn.embed_dim_preprocess": [16, 32, 64, 128],
            "static_gnn.num_postprocess_layers": "int[0, 2]",
            "static_gnn.num_conv_layers": "int[2, 5]",
            "static_gnn.aggr": ["sum", "mean", "max"],
            "static_gnn.hidden_channels": [16, 32, 64, 128],
            "static_gnn.dropout_rate": "uniform[0.0, 0.5]",
            "static_gnn.batch_norm": [True, False],
            "static_gnn.skip": [True, False],
            "static_gnn.use_edge_weights": [False, True],
        },
        "static_mlp": {
            "static_mlp.num_preprocess_layers": "int[0, 2]",
            "static_mlp.embed_dim_preprocess": [16, 32, 64, 128],
            "static_mlp.num_postprocess_layers": "int[0, 2]",
            "static_mlp.num_hidden_layers": "int[1, 5]",
            "static_mlp.hidden_channels": [16, 32, 64, 128],
            "static_mlp.dropout_rate": "uniform[0.0, 0.5]",
            "static_mlp.batch_norm": [True, False],
            "static_mlp.skip": [True, False],
        },
        "backtracking": {
            "backtracking.hidden_dim": [16, 32, 64, 128],
            "backtracking.num_layers": "int[2, 8]",
        },
        "temporal_gnn": {
            "temporal_gnn.hidden_channels": [16, 32, 64, 128],
            "temporal_gnn.group_by_time": [1, 2, 4, 6, 8, 12, 24, 48],
        },
        "dag_gnn": {
            "dag_gnn.hidden_channels": [16, 32, 64, 128],
            "dag_gnn.num_conv_layers": "int[1, 4]",
            "dag_gnn.dropout_rate": "uniform[0.0, 0.5]",
            "dag_gnn.agg": ["mean", "sum"],
            "dag_gnn.delta_t": [None, 2, 4, 8, 12, 24, 48],
        },
        "dbgnn": {
            "dbgnn.hidden_channels": [32, 64, 128],
            "dbgnn.num_conv_layers": "int[1, 4]",
            "dbgnn.dropout_rate": "uniform[0.0, 0.5]",
            "dbgnn.bipartite_agg": ["sum", "mean"],
            "dbgnn.order": [2, 3],
            "dbgnn.delta": [None, 4, 8, 12, 24, 48],
            "dbgnn.time_bin_size": [1, 2, 4, 8],
        },
    }
    return {**general, **model_spaces[model_name]}
