"""
DBGNN higher-order experiment runner.

Runs DBGNN on all calibrated thesis networks for the requested R0 values and
De Bruijn orders. The default matrix is:

    networks = lyon_ward, malawi, france_office, students, biasca, olten
    R0       = 1.0, 1.1, 1.5, 2.5
    k        = 2, 3, 4, 5, 10

Large networks are sampled before TSIR using an activity-preserving snowball
sampler. The sampling budget defaults to the full students network's
``nodes * edges`` cost divided by 72, so students is reduced by about 72x under
that cost proxy and any larger scenario is brought to the same budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import yaml

from run_all_experiments import (
    BETAS,
    LOSS_GUARD,
    MUS,
    NETWORKS,
    R0_VALUES,
    extract_run_id,
    normalize_r0_labels,
    read_network_meta,
    scenario,
)


DEFAULT_R0 = ["1.0", "1.1", "1.5", "2.5"]
DEFAULT_ORDERS = [2, 3, 4, 5, 10]
WANDB_PROJECT = "source-detection"


@dataclass(frozen=True)
class Preset:
    n_runs: int
    mc_runs: int
    n_mc: int
    reps: int
    n_truth: int


PRESETS = {
    "balanced": Preset(n_runs=500, mc_runs=300, n_mc=300, reps=1, n_truth=500),
    "max_quality": Preset(n_runs=1000, mc_runs=500, n_mc=500, reps=1, n_truth=1000),
    "fast": Preset(n_runs=120, mc_runs=80, n_mc=80, reps=1, n_truth=40),
}


@dataclass(frozen=True)
class NetworkStats:
    n_nodes: int
    n_edges: int
    n_contacts: int
    t_max: int
    directed: bool

    @property
    def node_edge_cost(self) -> int:
        return self.n_nodes * max(self.n_edges, 1)


STATUS_FIELDS = [
    "network",
    "r0_label",
    "stage",
    "variant",
    "order",
    "status",
    "run_id",
    "artifact",
    "returncode",
    "message",
    "log_path",
]
TERMINAL_STATUSES = {"success", "loss_guard_aborted", "skipped"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=sorted(PRESETS), default="balanced")
    p.add_argument("--networks", nargs="+", default=["all"], help="Network names, or all")
    p.add_argument("--r0", nargs="+", default=DEFAULT_R0, help="R0 labels/numbers, e.g. 1.0 1.1 r0_15")
    p.add_argument("--orders", nargs="+", type=int, default=DEFAULT_ORDERS, help="DBGNN orders k")
    p.add_argument("--output", default="results/dbgnn_higher_order", help="Root results directory")
    p.add_argument("--run-name", default=None, help="Run directory name. Defaults to timestamp.")
    p.add_argument("--resume", action="store_true", help="Resume an existing run directory")
    p.add_argument("--force", action="store_true", help="Rerun stages even when a terminal status exists")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    p.add_argument("--save-probs", action="store_true", help="Save probs_rep*.pt tensors from main_train.py")
    p.add_argument("--skip-tsir", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--no-sampling", action="store_true", help="Disable large-network sampling")
    p.add_argument("--sample-reference", default="students", help="Network used to derive the sampling budget")
    p.add_argument("--sample-budget-factor", type=float, default=72.0, help="Reference node*edge cost divisor")
    p.add_argument("--min-sample-nodes", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_networks(raw: list[str]) -> list[str]:
    if "all" in raw:
        return list(NETWORKS)
    return raw


def _directed_from_meta(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() in {"yes", "true", "1"}
    return bool(value)


def read_full_network_stats(network: str) -> NetworkStats:
    meta = read_network_meta(network)
    t_max = int(meta["t_max"])
    directed = _directed_from_meta(meta.get("directed", False))
    graph = nx.DiGraph() if directed else nx.Graph()
    csv_path = Path("nwk") / f"{network}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing network CSV: {csv_path}")

    with open(csv_path) as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) != 3:
                continue
            u, v, t = map(int, fields)
            if t > t_max or u == v:
                continue
            if graph.has_edge(u, v):
                if t not in graph.edges[u, v]["times"]:
                    graph.edges[u, v]["times"].append(t)
            else:
                graph.add_edge(u, v, times=[t])

    if graph.number_of_nodes() == 0:
        return NetworkStats(0, 0, 0, t_max, directed)
    components = nx.weakly_connected_components(graph) if directed else nx.connected_components(graph)
    largest = max(components, key=len)
    graph = graph.subgraph(largest).copy()
    contacts = sum(len(d.get("times", [])) for _, _, d in graph.edges(data=True))
    return NetworkStats(graph.number_of_nodes(), graph.number_of_edges(), contacts, t_max, directed)


def compute_sample_budget(stats: dict[str, NetworkStats], reference: str, factor: float) -> int:
    if reference not in stats:
        stats[reference] = read_full_network_stats(reference)
    if factor <= 0:
        raise ValueError("--sample-budget-factor must be positive")
    return max(1, int(stats[reference].node_edge_cost / factor))


def sampling_cfg_for_network(
    network: str,
    stats: dict[str, NetworkStats],
    budget: int,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.no_sampling:
        return None
    st = stats[network]
    if st.node_edge_cost <= budget:
        return None
    return {
        "method": "activity_snowball",
        "max_node_edge_cost": int(budget),
        "cost_metric": "node_edge",
        "seed": int(args.seed),
        "min_nodes": int(args.min_sample_nodes),
        "reference_network": args.sample_reference,
        "reference_budget_factor": float(args.sample_budget_factor),
        "original_node_edge_cost": int(st.node_edge_cost),
    }


def artifact_name(network: str, r0_label: str) -> str:
    return f"dbgnn_higher_order_{network}_{r0_label}"


def variant_name(order: int) -> str:
    return f"dbgnn_k{order}"


def build_tsir_config(
    network: str,
    r0_label: str,
    preset: Preset,
    sample_cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    meta = read_network_meta(network)
    sc = scenario(network, r0_label)
    nwk_cfg: dict[str, Any] = {
        "type": "empirical",
        "name": network,
        "t_max": meta["t_max"],
        "directed": meta["directed"],
    }
    if sample_cfg is not None:
        nwk_cfg["sample"] = sample_cfg
    return {
        "nwk": nwk_cfg,
        "sir": {
            "beta": sc["beta"],
            "mu": sc["mu"],
            "start_t": 0,
            "end_t": meta["t_max"],
            "n_runs": preset.n_runs,
            "mc_runs": preset.mc_runs,
        },
        "experiment": {
            "name": "dbgnn_higher_order",
            "network": network,
            "r0_label": r0_label,
            **sc,
        },
    }


def _template_path(network: str) -> Path:
    direct = Path("exp") / network / "dbgnn.yml"
    if direct.exists():
        return direct
    fallback = Path("exp/france_office/dbgnn.yml")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No DBGNN config template found for {network}")


def build_dbgnn_config(
    network: str,
    r0_label: str,
    order: int,
    preset: Preset,
    save_probs: bool,
) -> dict[str, Any]:
    if order < 2:
        raise ValueError(f"DBGNN order must be >= 2, got {order}")
    with open(_template_path(network)) as f:
        cfg = yaml.safe_load(f)

    cfg["model"] = "dbgnn"
    cfg["eval"] = {
        **cfg.get("eval", {}),
        "min_outbreak": 1,
        "top_k": [1, 3, 5, 10],
        "credible_p": [0.80, 0.90],
        "inverse_rank_offset": [0],
        "n_truth": preset.n_truth,
    }
    cfg["train"] = {
        **cfg.get("train", {}),
        "n_mc": preset.n_mc,
        "reps": preset.reps,
        "loss_guard": LOSS_GUARD,
    }
    cfg.setdefault("output", {})["save_probs"] = save_probs

    db_cfg = cfg.setdefault("dbgnn", {})
    db_cfg["order"] = int(order)
    db_cfg["delta"] = db_cfg.get("delta", 24)
    db_cfg["bipartite_agg"] = db_cfg.get("bipartite_agg", "sum")
    db_cfg["directed"] = read_network_meta(network)["directed"]

    cfg["experiment"] = {
        "name": "dbgnn_higher_order",
        "network": network,
        "r0_label": r0_label,
        "variant": variant_name(order),
        "dbgnn_order": int(order),
        **scenario(network, r0_label),
    }
    if preset.n_runs < preset.reps * preset.n_truth:
        raise ValueError(
            f"Invalid preset: n_runs={preset.n_runs} < reps*n_truth={preset.reps * preset.n_truth}"
        )
    return cfg


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def resolve_run_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.resume and args.run_name is None:
        candidates = sorted(p for p in root.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No previous run directories found under {root}")
        return candidates[-1]
    return root / (args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))


def write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    networks: list[str],
    r0_labels: list[str],
    orders: list[int],
    stats: dict[str, NetworkStats],
    sample_budget: int,
) -> None:
    sample_policies = {
        network: sampling_cfg_for_network(network, stats, sample_budget, args)
        for network in networks
    }
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "dbgnn_higher_order",
        "preset": args.preset,
        "preset_values": PRESETS[args.preset].__dict__,
        "networks": networks,
        "r0_labels": r0_labels,
        "orders": orders,
        "betas": {n: BETAS[n] for n in networks if n in BETAS},
        "mus": {n: MUS[n] for n in networks if n in MUS},
        "sample_budget_node_edge": sample_budget,
        "sample_reference": args.sample_reference,
        "sample_budget_factor": args.sample_budget_factor,
        "sample_policies": sample_policies,
        "network_stats": {n: stats[n].__dict__ for n in sorted(stats)},
        "wandb_project": WANDB_PROJECT,
    }
    with open(run_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def read_status(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_status(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in STATUS_FIELDS})


def update_status(path: Path, row: dict[str, Any]) -> None:
    rows = read_status(path)
    key = (
        row.get("network", ""),
        row.get("r0_label", ""),
        row.get("stage", ""),
        row.get("variant", ""),
        str(row.get("order", "")),
    )
    normalized = {field: str(row.get(field, "")) for field in STATUS_FIELDS}
    kept = [
        r for r in rows
        if (
            r.get("network", ""),
            r.get("r0_label", ""),
            r.get("stage", ""),
            r.get("variant", ""),
            r.get("order", ""),
        ) != key
    ]
    kept.append(normalized)
    write_status(path, kept)


def should_skip(
    status_path: Path,
    args: argparse.Namespace,
    network: str,
    r0_label: str,
    stage: str,
    variant: str = "",
    order: int | str = "",
) -> bool:
    if args.force or not args.resume:
        return False
    for row in read_status(status_path):
        if (
            row.get("network") == network
            and row.get("r0_label") == r0_label
            and row.get("stage") == stage
            and row.get("variant", "") == variant
            and row.get("order", "") == str(order)
            and row.get("status") in TERMINAL_STATUSES
        ):
            return True
    return False


def run_command(cmd: list[str], log_path: Path, dry_run: bool = False) -> tuple[int, str]:
    label = " ".join(cmd)
    if dry_run:
        print(f"  [DRY] {label}")
        return 0, ""

    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with open(log_path, "a") as log_fh:
        log_fh.write(f"\n$ {label}\n")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            captured.append(line)
        proc.wait()
    return proc.returncode, "".join(captured)


def stage_tsir(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    sample_cfg: dict[str, Any] | None,
) -> str:
    artifact = artifact_name(network, r0_label)
    if args.skip_tsir:
        update_status(
            status_path,
            {"network": network, "r0_label": r0_label, "stage": "tsir", "status": "skipped", "artifact": artifact},
        )
        return artifact
    if should_skip(status_path, args, network, r0_label, "tsir"):
        return artifact

    cfg_path = run_dir / "configs" / network / r0_label / "tsir.yml"
    write_yaml(cfg_path, build_tsir_config(network, r0_label, PRESETS[args.preset], sample_cfg))
    log_path = run_dir / network / r0_label / "logs" / "tsir.log"
    cmd = [sys.executable, "main_tsir.py", "--cfg", str(cfg_path), "--data", artifact]
    rc, stdout = run_command(cmd, log_path, args.dry_run)
    status = "success" if rc == 0 else "failed"
    update_status(
        status_path,
        {
            "network": network,
            "r0_label": r0_label,
            "stage": "tsir",
            "status": status,
            "artifact": artifact,
            "returncode": rc,
            "message": "dry_run" if args.dry_run else status,
            "log_path": log_path,
        },
    )
    if rc != 0 and not args.dry_run:
        raise RuntimeError(f"TSIR failed for {network}/{r0_label}; see {log_path}")
    return artifact


def stage_train(
    args: argparse.Namespace,
    run_dir: Path,
    status_path: Path,
    network: str,
    r0_label: str,
    order: int,
    artifact: str,
) -> str | None:
    variant = variant_name(order)
    if args.skip_train:
        update_status(
            status_path,
            {
                "network": network,
                "r0_label": r0_label,
                "stage": "train",
                "variant": variant,
                "order": order,
                "status": "skipped",
                "artifact": artifact,
            },
        )
        return None
    if should_skip(status_path, args, network, r0_label, "train", variant, order):
        rows = read_status(status_path)
        return next(
            (
                r.get("run_id")
                for r in rows
                if r.get("network") == network
                and r.get("r0_label") == r0_label
                and r.get("stage") == "train"
                and r.get("variant") == variant
                and r.get("order") == str(order)
            ),
            None,
        )

    cfg_path = run_dir / "configs" / network / r0_label / f"{variant}.yml"
    write_yaml(cfg_path, build_dbgnn_config(network, r0_label, order, PRESETS[args.preset], args.save_probs))
    log_path = run_dir / network / r0_label / "logs" / f"train_{variant}.log"
    cmd = [sys.executable, "main_train.py", "--cfg", str(cfg_path), "--data", f"{artifact}:latest"]
    if args.save_probs:
        cmd.append("--save-probs")
    rc, stdout = run_command(cmd, log_path, args.dry_run)
    run_id = extract_run_id(stdout) or ("dryrun00" if args.dry_run else "")
    status = "success" if rc == 0 else "loss_guard_aborted" if rc == 88 or "LOSS_GUARD_ABORT" in stdout else "failed"
    update_status(
        status_path,
        {
            "network": network,
            "r0_label": r0_label,
            "stage": "train",
            "variant": variant,
            "order": order,
            "status": status,
            "run_id": run_id,
            "artifact": artifact,
            "returncode": rc,
            "message": status,
            "log_path": log_path,
        },
    )
    return run_id if status == "success" else None


def write_summary_csv(run_dir: Path, status_rows: list[dict[str, str]]) -> None:
    out = run_dir / "run_matrix_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["network", "r0_label", "variant", "order", "status", "run_id", "artifact", "log_path"],
        )
        writer.writeheader()
        for row in status_rows:
            if row.get("stage") == "train":
                writer.writerow({
                    "network": row.get("network", ""),
                    "r0_label": row.get("r0_label", ""),
                    "variant": row.get("variant", ""),
                    "order": row.get("order", ""),
                    "status": row.get("status", ""),
                    "run_id": row.get("run_id", ""),
                    "artifact": row.get("artifact", ""),
                    "log_path": row.get("log_path", ""),
                })


def main() -> None:
    args = parse_args()
    networks = resolve_networks(args.networks)
    r0_labels = normalize_r0_labels(args.r0)
    orders = list(dict.fromkeys(args.orders))
    preset = PRESETS[args.preset]

    for network in networks:
        if network not in BETAS:
            raise ValueError(f"No calibrated beta matrix configured for network '{network}'")
    for order in orders:
        if order < 2:
            raise ValueError(f"DBGNN order must be >= 2, got {order}")

    stats = {network: read_full_network_stats(network) for network in networks}
    sample_budget = compute_sample_budget(stats, args.sample_reference, args.sample_budget_factor)
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.csv"
    write_manifest(run_dir, args, networks, r0_labels, orders, stats, sample_budget)

    print("=" * 72)
    print("DBGNN Higher-Order Experiment Runner")
    print("=" * 72)
    print(f"Run dir  : {run_dir}")
    print(f"Preset   : {args.preset} ({preset})")
    print(f"Networks : {', '.join(networks)}")
    print(f"R0       : {', '.join(r0_labels)}")
    print(f"Orders   : {', '.join(str(k) for k in orders)}")
    if args.no_sampling:
        print("Sampling : disabled")
    else:
        print(
            "Sampling : activity_snowball for networks above "
            f"node*edge budget {sample_budget:,} "
            f"({args.sample_reference}/{args.sample_budget_factor:g})"
        )
    if args.dry_run:
        print("DRY RUN: commands will be printed, not executed")

    for network in networks:
        sample_cfg = sampling_cfg_for_network(network, stats, sample_budget, args)
        if sample_cfg is None:
            print(f"\n### {network}: no sampling (cost={stats[network].node_edge_cost:,})")
        else:
            reduction = stats[network].node_edge_cost / sample_budget
            print(
                f"\n### {network}: sampled to budget {sample_budget:,} "
                f"(expected node*edge reduction >= {reduction:.2f}x)"
            )

        for r0_label in r0_labels:
            sc = scenario(network, r0_label)
            print(f"\n--- {network} / {r0_label}  R0={sc['r0']} beta={sc['beta']} mu={sc['mu']}")
            try:
                artifact = stage_tsir(args, run_dir, status_path, network, r0_label, sample_cfg)
            except Exception as exc:
                print(f"  FATAL TSIR stage failed, skipping scenario: {exc}")
                continue

            for order in orders:
                print(f"  DBGNN k={order}")
                stage_train(args, run_dir, status_path, network, r0_label, order, artifact)

    status_rows = read_status(status_path)
    write_summary_csv(run_dir, status_rows)
    print("\nDone.")
    print(f"Status : {status_path}")
    print(f"Summary: {run_dir / 'run_matrix_summary.csv'}")
    print(f"Results: {run_dir}")


if __name__ == "__main__":
    main()
