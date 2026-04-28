"""
Stage 1 (static GNN) — continuous-time SIR simulation pipeline.

Uses the paper's continuous-time SIR C binary (Sterchi et al. 2025 / Petter
Holme 2018) to generate training and test simulations on the *static projection*
of any network.  Results are logged as a versioned W&B artifact that is fully
compatible with downstream main_train.py and main_eval.py.

Use this instead of main_tsir.py whenever you train a static_gnn model.

Usage
-----
::

    python main_static_sir.py \\
        --cfg exp/karate_static/static_sir.yml \\
        --data karate_static_sir

Downstream usage::

    python main_train.py \\
        --cfg exp/karate_static/static_gnn.yml \\
        --data karate_static_sir:latest

Config format (YAML)
--------------------
::

    nwk:
      name: karate_static       # matches nwk/<name>.csv and nwk/<name>.yml
      directed: false

    sir:
      beta: 1.3                 # infection rate
      nu:   1.0                 # recovery rate
      T:    0.85                # observation time
      n_runs:   5000            # test simulations per source node
      mc_runs:  500             # training simulations per source node
      train_seed: 12345         # optional; random if omitted

    project: source-detection   # optional wandb project name
"""

from __future__ import annotations

import argparse
import os
import platform
import random

import networkx as nx
import wandb
import yaml

from setup.read_network import read_networkx
from sir.static_sir import run_static_sir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cfg",  required=True,
                   help="Path to static_sir YAML config")
    p.add_argument("--data", required=True,
                   help="W&B artifact name, e.g. karate_static_sir")
    return p.parse_args()


def _load_static_graph(nwk_cfg: dict) -> tuple[nx.Graph, int]:
    """Load the network CSV and return the static projection + n_nodes."""
    name     = nwk_cfg["name"]
    directed = nwk_cfg.get("directed", False)

    # Resolve t_max from config or from the network's own yml
    yml_path = f"nwk/{name}.yml"
    if os.path.exists(yml_path):
        with open(yml_path) as f:
            import yaml as _yaml
            nwk_meta = _yaml.safe_load(f)
        t_max = nwk_meta.get("time_steps", nwk_meta.get("t_max"))
    else:
        t_max = nwk_cfg.get("time_steps", nwk_cfg.get("t_max"))

    if t_max is None:
        raise ValueError(f"Cannot determine t_max for network '{name}'")

    # Suppress read_networkx progress output
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        G_temporal = read_networkx(f"nwk/{name}.csv", t_max=t_max, directed=directed)

    # Static projection: union of all edges across time, relabelled 0..N-1
    G_raw = nx.Graph()
    G_raw.add_nodes_from(G_temporal.nodes())
    G_raw.add_edges_from(G_temporal.edges())

    # Relabel to 0-indexed integers (required by sir binary)
    mapping = {node: i for i, node in enumerate(sorted(G_raw.nodes()))}
    G_static = nx.relabel_nodes(G_raw, mapping)
    n_nodes  = G_static.number_of_nodes()

    return G_static, n_nodes


def main() -> None:
    args = parse_args()

    with open(args.cfg) as f:
        raw = yaml.safe_load(f)

    os_tag = platform.system().lower()
    run = wandb.init(
        project  = raw.get("project", "source-detection"),
        config   = raw,
        job_type = "static-sir",
        tags     = ["job:static-sir", f"os:{os_tag}"],
        settings = wandb.Settings(silent=False),
    )
    local_folder = f"data/{run.id}"
    os.makedirs(local_folder, exist_ok=True)

    print(f"\nW&B run : {run.url}")
    print(f"Data dir: {local_folder}\n")

    # ── Network ──────────────────────────────────────────────────────
    print("=" * 60)
    print("Network (static projection)")
    print("=" * 60)
    G_static, n_nodes = _load_static_graph(raw["nwk"])
    n_edges = G_static.number_of_edges()
    print(f"  Nodes : {n_nodes}")
    print(f"  Edges : {n_edges}  (static projection, undirected unique pairs)")

    wandb.summary["n_nodes"] = n_nodes
    wandb.summary["n_edges"] = n_edges
    wandb.summary["network"] = raw["nwk"]["name"]

    # ── SIR parameters ──────────────────────────────────────────────
    sir_cfg    = raw["sir"]
    beta       = float(sir_cfg["beta"])
    nu         = float(sir_cfg["nu"])
    T          = float(sir_cfg["T"])
    n_runs     = int(sir_cfg["n_runs"])
    mc_runs    = int(sir_cfg["mc_runs"])
    train_seed = int(sir_cfg.get("train_seed", random.getrandbits(62)))

    print("\n" + "=" * 60)
    print(f"Continuous-time SIR  β={beta}, ν={nu}, T={T}")
    print(f"  train_seed : {train_seed}")
    print("=" * 60)

    stats = run_static_sir(
        G            = G_static,
        beta         = beta,
        nu           = nu,
        T            = T,
        n_runs       = n_runs,
        mc_runs      = mc_runs,
        train_seed   = train_seed,
        local_folder = local_folder,
    )
    for k, v in stats.items():
        wandb.summary[k] = v

    # ── Log artifact ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Logging artifact '{args.data}'")
    print("=" * 60)
    artifact = wandb.Artifact(
        name        = args.data,
        type        = "static-sir-data",
        description = (
            f"Continuous-time SIR on '{raw['nwk']['name']}' static projection "
            f"(β={beta}, ν={nu}, T={T}, "
            f"n_runs={n_runs}, mc_runs={mc_runs})"
        ),
        metadata = {
            "network":    raw["nwk"]["name"],
            "n_nodes":    n_nodes,
            "n_edges":    n_edges,
            "beta":       beta,
            "nu":         nu,
            "T":          T,
            "n_runs":     n_runs,
            "mc_runs":    mc_runs,
            "train_seed": train_seed,
        },
    )
    artifact.add_dir(local_folder)
    wandb.log_artifact(artifact)
    print(f"Artifact logged.  Reference with: --data {args.data}:latest")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
