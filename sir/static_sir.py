"""
Wrapper for the paper's continuous-time SIR C binary (Petter Holme 2018,
adapted by Sterchi et al. 2025).

Compiles gnn/static_source_detection_gnn/sir/sir from source if missing,
then generates training (MC) and test (truth) simulation files in our standard
TSIRData layout so downstream main_train.py / main_eval.py are unchanged.

Binary interface::

    sir <graph.csv> <beta/nu> <T/nu> <sampled_T> <sims_per_seed> <seed> <outdir>

Output per call (in outdir):
    states.bin — int8, shape (sims_per_seed * n_nodes, n_nodes); 0=S,1=I,2=R
    labels.bin — uint32, shape (sims_per_seed * n_nodes,); source node index
"""

from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np

# Fixed seed for reproducible test simulations (matches paper's RANDOM_SEED)
_TEST_SEED = 4253219522064423221

_SIR_SRC_DIR = (
    Path(__file__).resolve().parent.parent
    / "gnn" / "static_source_detection_gnn" / "sir"
)
_SIR_BINARY = _SIR_SRC_DIR / "sir"


def build_binary() -> Path:
    """Compile the SIR C binary from source if not already built."""
    if _SIR_BINARY.exists():
        return _SIR_BINARY
    print("  Compiling sir binary…", end=" ", flush=True)
    (_SIR_SRC_DIR / "o").mkdir(exist_ok=True)
    result = subprocess.run(
        ["make", "-C", str(_SIR_SRC_DIR)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to compile sir binary:\n{result.stderr}"
        )
    print("done")
    return _SIR_BINARY


def export_graph_csv(G: nx.Graph, path: Path) -> None:
    """Write edges as 'u v weight' (one per line) for the SIR binary."""
    with open(path, "w") as f:
        for u, v, d in G.edges(data=True):
            w = d.get("weight", 1)
            f.write(f"{u} {v} {w}\n")


def _run_binary(
    graph_csv: Path,
    beta: float,
    nu: float,
    T: float,
    sims_per_seed: int,
    seed: int,
    outdir: Path,
) -> float:
    """Run sir binary, return average outbreak size (fraction of nodes)."""
    binary = build_binary()
    outdir.mkdir(parents=True, exist_ok=True)
    out = subprocess.check_output([
        str(binary),
        str(graph_csv),
        str(beta / nu),
        str(T / nu),
        "0",               # sampled_T = false
        str(sims_per_seed),
        str(seed),
        str(outdir),
    ])
    first_line = out.decode().split("\n")[0]
    return float(first_line.split(":")[1].strip())


def _load_and_convert(
    sir_outdir: Path,
    n_nodes: int,
    sims_per_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load states.bin → (truth_S, truth_I, truth_R) in our int8 format.

    Paper states: 0=S, 1=I, 2=R.
    Our format: 1 where that state holds, 0 otherwise.
    Shape: (sims_per_seed * n_nodes, n_nodes).
    """
    states = np.fromfile(
        sir_outdir / "states.bin", dtype=np.int8
    ).reshape(n_nodes * sims_per_seed, n_nodes)
    return (
        (states == 0).astype(np.int8),
        (states == 1).astype(np.int8),
        (states == 2).astype(np.int8),
    )


def run_static_sir(
    G: nx.Graph,
    beta: float,
    nu: float,
    T: float,
    n_runs: int,
    mc_runs: int,
    train_seed: int,
    local_folder: str | Path,
) -> dict[str, float]:
    """Generate static SIR simulations and write TSIRData-compatible files.

    Runs the continuous-time SIR binary twice: once for training (MC) data
    with `train_seed`, once for test (truth) data with a fixed held-out seed.
    Nodes in G must be 0-indexed integers (relabel before calling if needed).

    Written files (all in `local_folder`):
        ground_truth_{S,I,R}.bin  — shape (n_nodes * n_runs,  n_nodes) int8
        monte_carlo_{S,I,R}.bin   — shape (n_nodes * mc_runs, n_nodes) int8
        possible_sources.bin      — shape (n_nodes, n_runs, n_nodes)   int8
        maximal_outbreak_S.bin    — shape (n_nodes, n_nodes)           int8  (zeros)
        network.gpickle           — static NetworkX graph

    Parameters
    ----------
    G            : static networkx Graph with 0-indexed integer nodes
    beta         : infection rate
    nu           : recovery rate (1.0 in the paper)
    T            : observation time
    n_runs       : test simulations per source node
    mc_runs      : training simulations per source node
    train_seed   : RNG seed for MC (training) simulations
    local_folder : output directory

    Returns
    -------
    dict with ``avg_outbreak_size_train`` and ``avg_outbreak_size_test``
    """
    local_folder = Path(local_folder)
    local_folder.mkdir(parents=True, exist_ok=True)
    n_nodes = G.number_of_nodes()

    graph_csv = local_folder / "graph.csv"
    export_graph_csv(G, graph_csv)

    with open(local_folder / "network.gpickle", "wb") as f:
        pickle.dump(G, f)

    # ── MC (training) simulations ────────────────────────────────
    print(f"  MC simulations  ({mc_runs} × {n_nodes} sources)…", end=" ", flush=True)
    mc_dir = local_folder / "_sir_mc"
    avg_mc = _run_binary(graph_csv, beta, nu, T, mc_runs, train_seed, mc_dir)
    mc_S, mc_I, mc_R = _load_and_convert(mc_dir, n_nodes, mc_runs)
    mc_S.tofile(local_folder / "monte_carlo_S.bin")
    mc_I.tofile(local_folder / "monte_carlo_I.bin")
    mc_R.tofile(local_folder / "monte_carlo_R.bin")
    print(f"done  (avg outbreak: {avg_mc:.3f})")

    # ── Truth (test) simulations ─────────────────────────────────
    print(f"  Truth simulations  ({n_runs} × {n_nodes} sources)…", end=" ", flush=True)
    test_dir = local_folder / "_sir_test"
    avg_test = _run_binary(graph_csv, beta, nu, T, n_runs, _TEST_SEED, test_dir)
    truth_S, truth_I, truth_R = _load_and_convert(test_dir, n_nodes, n_runs)
    truth_S.tofile(local_folder / "ground_truth_S.bin")
    truth_I.tofile(local_folder / "ground_truth_I.bin")
    truth_R.tofile(local_folder / "ground_truth_R.bin")
    print(f"done  (avg outbreak: {avg_test:.3f})")

    # ── Possible-sources mask ─────────────────────────────────────
    truth_S_3d = truth_S.reshape(n_nodes, n_runs, n_nodes)
    possible   = (1 - truth_S_3d).astype(np.int8)
    possible.tofile(local_folder / "possible_sources.bin")
    print(f"  Possible-sources mask  (avg {possible.mean(axis=(1,2)).mean():.3f} feasible/run)")

    # ── Maximal-outbreak stub ─────────────────────────────────────
    # For a connected graph every node is reachable → all zeros (no node stays S)
    np.zeros((n_nodes, n_nodes), dtype=np.int8).tofile(
        local_folder / "maximal_outbreak_S.bin"
    )

    return {
        "avg_outbreak_size_train": avg_mc,
        "avg_outbreak_size_test":  avg_test,
    }
