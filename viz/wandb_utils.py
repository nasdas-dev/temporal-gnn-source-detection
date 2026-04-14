"""
Shared W&B query utilities for visualisation scripts.

All viz scripts that need to pull W&B run data should use these helpers
instead of writing their own API calls, avoiding boilerplate duplication.
"""

from __future__ import annotations

from typing import Any


def fetch_runs_for_artifact(
    artifact_prefix: str,
    project: str = "source-detection",
    entity: str | None = None,
    states: tuple[str, ...] = ("finished",),
) -> list[dict[str, Any]]:
    """Return summary metrics and configs for all W&B runs that used an artifact.

    Parameters
    ----------
    artifact_prefix:
        Dataset name prefix, e.g. ``"france_office"`` or ``"toy_holme"``.
        Matches runs where ``config.data_name`` starts with this string.
    project:
        W&B project name.
    entity:
        W&B entity (username or team).  Uses the API default if ``None``.
    states:
        Only return runs in these states (default: ``"finished"``).

    Returns
    -------
    list of dicts, each with keys:
    - ``model``      — model key (e.g. ``"backtracking"``, ``"uniform"``)
    - ``run_id``     — W&B run ID
    - ``run_name``   — W&B display name
    - ``config``     — full config dict
    - ``summary``    — full summary dict (eval metrics live here)
    - ``data_name``  — artifact name from config
    """
    import wandb
    api = wandb.Api()
    prefix_full = f"{entity}/{project}" if entity else project
    all_runs = api.runs(prefix_full, filters={"state": {"$in": list(states)}})

    results = []
    artifact_key = artifact_prefix.split(":")[0]

    for run in all_runs:
        cfg = dict(run.config)
        data_name = cfg.get("data_name", "")
        if not data_name.startswith(artifact_key):
            continue

        model = cfg.get("model")
        if model is None:
            continue

        results.append({
            "model":     model,
            "run_id":    run.id,
            "run_name":  run.name,
            "config":    cfg,
            "summary":   dict(run.summary),
            "data_name": data_name,
        })

    return results


def fetch_run_history(
    run_path: str,
    samples: int = 5000,
) -> tuple[Any, dict, dict]:
    """Fetch training history, summary, and config for a single W&B run.

    Parameters
    ----------
    run_path:
        Full W&B run path, e.g. ``"entity/project/run_id"``.
    samples:
        Maximum number of history rows to fetch.

    Returns
    -------
    history : pd.DataFrame
        Per-step logged values.
    summary : dict
        Final summary values (eval metrics etc.).
    config : dict
        Run configuration.
    """
    import wandb
    api = wandb.Api()
    run = api.run(run_path)
    return run.history(samples=samples), dict(run.summary), dict(run.config)


def best_run_per_model(
    runs: list[dict[str, Any]],
    metric: str = "eval/mrr_mean",
    higher_is_better: bool = True,
) -> dict[str, dict[str, Any]]:
    """Reduce a list of runs to the single best run per model key.

    Parameters
    ----------
    runs:
        Output of ``fetch_runs_for_artifact``.
    metric:
        Summary key used to rank runs of the same model.
    higher_is_better:
        Whether a higher value of *metric* is better.

    Returns
    -------
    dict mapping model key → best run dict.
    """
    best: dict[str, dict] = {}
    for r in runs:
        model = r["model"]
        val = r["summary"].get(metric, float("-inf") if higher_is_better else float("inf"))
        if model not in best:
            best[model] = r
        else:
            prev = best[model]["summary"].get(
                metric, float("-inf") if higher_is_better else float("inf")
            )
            if higher_is_better and val > prev:
                best[model] = r
            elif not higher_is_better and val < prev:
                best[model] = r
    return best
