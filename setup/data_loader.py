from dataclasses import dataclass
import pickle
import numpy as np
import wandb


@dataclass
class TSIRData:
    config: dict
    n_nodes: int
    n_runs: int
    mc_runs: int
    mc_S: np.ndarray
    mc_I: np.ndarray
    mc_R: np.ndarray
    truth_S: np.ndarray
    truth_I: np.ndarray
    truth_R: np.ndarray
    maximal_outbreak: np.ndarray
    possible: np.ndarray
    lik_possible: np.ndarray


def _load_from_run_id(tsir_run_id: str, config: dict) -> "tuple":
    """Load TSIRData directly from a local data/<run_id>/ directory."""
    with open(f"data/{tsir_run_id}/network.gpickle", "rb") as f:
        H = pickle.load(f)
    # infer n_nodes from network
    n_nodes = H.number_of_nodes()
    mc_S, mc_I, mc_R = (np.fromfile(f"data/{tsir_run_id}/monte_carlo_{s}.bin", dtype=np.int8)
                        .reshape(n_nodes, -1, n_nodes) for s in "SIR")
    mc_runs = mc_S.shape[1]
    truth_S, truth_I, truth_R = (np.fromfile(f"data/{tsir_run_id}/ground_truth_{s}.bin", dtype=np.int8)
                                 .reshape(n_nodes, -1, n_nodes) for s in "SIR")
    n_runs = truth_S.shape[1]
    maximal_outbreak = 1 - np.fromfile(f"data/{tsir_run_id}/maximal_outbreak_S.bin", dtype=np.int8).reshape(n_nodes, n_nodes)
    possible = np.fromfile(f"data/{tsir_run_id}/possible_sources.bin", dtype=np.int8).reshape(n_nodes, n_runs, n_nodes)
    lik_possible = np.where(possible == 1, 0, np.inf)
    return H, TSIRData(config, n_nodes, n_runs, mc_runs, mc_S, mc_I, mc_R, truth_S, truth_I, truth_R, maximal_outbreak, possible, lik_possible)


def load_tsir_data(data_name: str):
    """Load TSIR data by wandb artifact reference or bare run ID.

    Parameters
    ----------
    data_name:
        Either a wandb artifact reference (e.g. ``"toy_holme:latest"``) or
        a bare wandb run ID (e.g. ``"l5s6bop3"``).  When a bare run ID is
        supplied the wandb artifact resolution is skipped entirely, which is
        useful when the wandb API is slow or unavailable.
    """
    import os
    # If data_name looks like a bare run ID (no colon, directory exists) skip wandb API.
    bare_id = data_name.split(":")[0]
    if ":" not in data_name and os.path.isdir(f"data/{data_name}"):
        print(f"  [data_loader] Using local run directory data/{data_name} directly.")
        wandb.run.tags += (data_name,)
        return _load_from_run_id(data_name, {})

    wandb.run.tags += (bare_id,)
    artifact = wandb.run.use_artifact(data_name)
    tsir_run_id = artifact.logged_by().id
    tsir_run = wandb.Api().run(f"{artifact.entity}/{artifact.project}/{tsir_run_id}")
    return _load_from_run_id(tsir_run_id, tsir_run.config)
