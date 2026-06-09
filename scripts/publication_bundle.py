"""Build a publication-ready ``result/`` bundle for experiment runs."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


COPY_SUFFIXES = {".csv", ".json", ".tex", ".pdf", ".png", ".md", ".npz"}
RUN_ASSET_PREFIXES = (
    "metrics",
    "baseline_metrics",
    "loss_history",
    "eval_arrays",
    "reduction_report",
    "network_provenance",
)


def read_status_rows(status_path: Path) -> list[dict[str, str]]:
    """Read status rows if the file exists."""
    if not status_path.exists():
        return []
    with open(status_path, newline="") as f:
        return list(csv.DictReader(f))


def sync_publication_result(
    run_dir: Path,
    status_rows: list[dict[str, str]] | None = None,
    experiment_name: str = "",
    data_root: Path = Path("data"),
) -> Path:
    """Mirror publication-facing outputs into ``run_dir/result``.

    The bundle is intentionally a mirror of generated artifacts rather than a
    replacement for the run directory. It can be refreshed after each completed
    stage and at the end of the full run.
    """
    run_dir = Path(run_dir)
    result_dir = run_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)

    if status_rows is None:
        status_rows = read_status_rows(run_dir / "status.csv")

    _copy_top_level_files(run_dir, result_dir)
    _copy_tree(run_dir / "tables", result_dir / "tables")
    _copy_tree(run_dir / "figures", result_dir / "figures" / "global")
    _copy_scenario_figures(run_dir, result_dir)
    _copy_run_assets(result_dir, status_rows, data_root)

    manifest = _build_manifest(result_dir, status_rows, experiment_name)
    _write_json(result_dir / "latex_inputs.json", manifest)
    _write_readme(result_dir, manifest, experiment_name)
    return result_dir


def _copy_top_level_files(run_dir: Path, result_dir: Path) -> None:
    targets = [
        "manifest.json",
        "status.csv",
        "metrics_long.csv",
        "metrics_summary.csv",
        "run_matrix_summary.csv",
    ]
    for name in targets:
        src = run_dir / name
        if src.exists():
            _copy_file(src, result_dir / name)


def _copy_scenario_figures(run_dir: Path, result_dir: Path) -> None:
    for fig_dir in run_dir.glob("*/*/figures"):
        if not fig_dir.is_dir():
            continue
        network = fig_dir.parent.parent.name
        r0_label = fig_dir.parent.name
        _copy_tree(fig_dir, result_dir / "figures" / "by_scenario" / network / r0_label)


def _copy_run_assets(
    result_dir: Path,
    status_rows: list[dict[str, str]],
    data_root: Path,
) -> None:
    for row in status_rows:
        if row.get("status") != "success":
            continue
        run_id = row.get("run_id", "")
        if not run_id or run_id == "dryrun00":
            continue
        src_dir = data_root / run_id
        if not src_dir.is_dir():
            continue
        network = _clean_part(row.get("network", "unknown_network"))
        r0_label = _clean_part(row.get("r0_label", "unknown_r0"))
        method = "tsir" if row.get("stage") == "tsir" else _clean_part(row.get("model") or row.get("variant") or "baselines")
        dest_dir = result_dir / "runs" / network / r0_label / method
        for src in src_dir.iterdir():
            if not src.is_file():
                continue
            if src.suffix not in COPY_SUFFIXES:
                continue
            if not src.name.startswith(RUN_ASSET_PREFIXES):
                continue
            _copy_file(src, dest_dir / src.name)


def _copy_tree(src_dir: Path, dest_dir: Path) -> None:
    if not src_dir.is_dir():
        return
    for src in src_dir.rglob("*"):
        if not src.is_file() or src.suffix not in COPY_SUFFIXES:
            continue
        rel = src.relative_to(src_dir)
        _copy_file(src, dest_dir / rel)


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _build_manifest(
    result_dir: Path,
    status_rows: list[dict[str, str]],
    experiment_name: str,
) -> dict[str, Any]:
    files = [_rel(result_dir, path) for path in sorted(result_dir.rglob("*")) if path.is_file()]
    categories = {
        "tables": [path for path in files if path.startswith("tables/") and Path(path).suffix in {".csv", ".tex"}],
        "global_figures": [path for path in files if path.startswith("figures/global/") and Path(path).suffix in {".pdf", ".png"}],
        "scenario_figures": [path for path in files if path.startswith("figures/by_scenario/") and Path(path).suffix in {".pdf", ".png"}],
        "metrics": [path for path in files if path in {"metrics_long.csv", "metrics_summary.csv", "run_matrix_summary.csv"}],
        "run_assets": [path for path in files if path.startswith("runs/")],
        "network_provenance": [
            path
            for path in files
            if path.endswith("network_provenance.json") or path.endswith("reduction_report.json")
        ],
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": experiment_name,
        "result_dir": str(result_dir),
        "status_counts": _status_counts(status_rows),
        "files": files,
        "categories": categories,
    }


def _status_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('stage', '')}:{row.get('status', '')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _write_readme(result_dir: Path, manifest: dict[str, Any], experiment_name: str) -> None:
    categories = manifest["categories"]
    lines = [
        "# Publication Result Bundle",
        "",
        f"Experiment: `{experiment_name or 'unknown'}`",
        f"Generated: `{manifest['generated_at']}`",
        "",
        "This directory mirrors the figures, tables, metrics, and lightweight run artifacts needed for a paper results section.",
        "A LaTeX generation script can read `latex_inputs.json` to discover all available assets.",
        "",
        "## Contents",
        "",
        f"- Tables: {len(categories['tables'])}",
        f"- Global figures: {len(categories['global_figures'])}",
        f"- Scenario figures: {len(categories['scenario_figures'])}",
        f"- Metric CSV files: {len(categories['metrics'])}",
        f"- Run-level assets: {len(categories['run_assets'])}",
        f"- Network provenance reports: {len(categories['network_provenance'])}",
        "",
        "## Suggested LaTeX Inputs",
        "",
        "- Use `tables/*.tex` for booktabs-ready tables.",
        "- Use `figures/global/*.pdf` for overview figures.",
        "- Use `figures/by_scenario/<network>/<r0_label>/*.pdf` for scenario-level panels.",
        "- Use `metrics_summary.csv` or `run_matrix_summary.csv` for generated result text.",
        "- Use `runs/<network>/<r0_label>/tsir/network_provenance.json` to cite reduction, calibration, and observed R0 settings.",
        "",
    ]
    (result_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _clean_part(value: str) -> str:
    value = str(value or "unknown")
    return value.replace("/", "_").replace(" ", "_")
