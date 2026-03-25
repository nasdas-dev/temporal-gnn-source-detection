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
``monte_carlo.txt``            — Monte Carlo SIR log
``maximal_outbreak.txt``       — maximal outbreak SIR log
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import wandb

from setup import setup_tsir_run
from setup.read_network import load_network, make_array_from_networkx
from tsir.read_run import (
    make_c_readable_from_networkx,
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

    print(f"  Nodes    : {n_nodes}")
    print(f"  Edges    : {n_edges}  (undirected pairs)")
    print(f"  Contacts : {n_contacts}  (total timestamped events)")
    print(f"  t_max    : {cfg.nwk.t_max}")

    wandb.summary["n_nodes"]   = n_nodes
    wandb.summary["n_edges"]   = n_edges
    wandb.summary["n_contacts"] = n_contacts
    wandb.summary["t_max"]     = cfg.nwk.t_max
    wandb.summary["network"]   = cfg.nwk.name

    # Persist the graph so downstream runs can load it without re-reading CSV
    with open(f"{local_folder}/network.gpickle", "wb") as f:
        pickle.dump(H, f)

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
        },
    )
    artifact.add_dir(local_folder)
    wandb.log_artifact(artifact)
    print(f"Artifact logged.  Reference downstream runs with: --data {args.data}:latest")

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    main()
