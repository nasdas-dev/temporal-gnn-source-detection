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


def load_tsir_data(data_name):
    wandb.run.tags += (data_name.split(":")[0],)
    artifact = wandb.run.use_artifact(data_name)
    tsir_run_id = artifact.logged_by().id
    tsir_run = wandb.Api().run(f"{artifact.entity}/{artifact.project}/{tsir_run_id}")

    n_nodes = tsir_run.summary["n_nodes"]
    with open(f"data/{tsir_run_id}/network.gpickle", "rb") as f:
        H = pickle.load(f)
    mc_S, mc_I, mc_R = (np.fromfile(f"data/{tsir_run_id}/monte_carlo_{state}.bin", dtype=np.int8).
                        reshape(n_nodes, -1, n_nodes) for state in "SIR")
    mc_runs = mc_S.shape[1]
    truth_S, truth_I, truth_R = (np.fromfile(f"data/{tsir_run_id}/ground_truth_{state}.bin", dtype=np.int8).
                                 reshape(n_nodes, -1, n_nodes) for state in "SIR")
    n_runs = truth_S.shape[1]
    maximal_outbreak = (1 - np.fromfile(f"data/{tsir_run_id}/maximal_outbreak_S.bin", dtype=np.int8).
                        reshape(n_nodes, n_nodes))
    possible = (np.fromfile(f"data/{tsir_run_id}/possible_sources.bin", dtype=np.int8).
                reshape(n_nodes, n_runs, n_nodes))
    lik_possible = np.where(possible == 1, 0, np.inf)  # for correcting log-likelihoods later

    return H, TSIRData(tsir_run.config, n_nodes, n_runs, mc_runs, mc_S, mc_I, mc_R, truth_S, truth_I, truth_R, maximal_outbreak, possible, lik_possible)
