import pathlib
import numpy as np
import argparse
import pickle
import wandb
import os
from setup import setup_tsir_run, load_network
from tsir import make_c_readable_from_networkx, sir_ground_truth, sir_maximal_outbreak, sir_monte_carlo
from utils import matmul


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg # e.g., cfg_path = "exp/exp_1_vary_n/barabasi_albert/tsir.yml"
    data_name = args.data # e.g., data_path = "exp_1_vary_n.barabasi_albert"

    # read config and read network
    cfg = setup_tsir_run(cfg_path)
    local_folder = f"data/{wandb.run.id}"
    os.makedirs(local_folder, exist_ok=True)
    H = load_network(cfg, label_attribute = "old_id")
    with open(f"{local_folder}/network.gpickle", "wb") as f:
        pickle.dump(H, f) # save network for visualization later

    # run sir (ground truth, maximal outbreak, monte carlo simulations)
    H_cread = make_c_readable_from_networkx(H, t_max=cfg.nwk.t_max, directed=cfg.nwk.directed)
    n_nodes = len(list(H.nodes()))
    wandb.summary["n_nodes"] = n_nodes
    wandb.summary["n_edges"] = len(list(H.edges()))
    truth_S, truth_I, truth_R = sir_ground_truth(cfg, H_cread, n_nodes, local_folder)
    maximal_outbreak = sir_maximal_outbreak(cfg, H_cread, n_nodes, local_folder)
    mc_S, mc_I, mc_R = sir_monte_carlo(cfg, H_cread, n_nodes, local_folder)

    # for certain configurations of in infected nodes, only some source nodes are possible: filter those out
    print("Precompute those nodes that are candidate sources.")
    print(" --- Compare ground truth to maximal outbreak to exclude some nodes as potential sources.", end=" ")
    outside = matmul((1 - truth_S).astype(np.float32), (1 - maximal_outbreak.T).astype(np.float32))  # where there infections outside maximal outbreak?
    possible = (outside == 0) * (1 - truth_S) # infections cannot be outside maximal outbreak and source must be in infected subgraph
    possible.tofile(f"{local_folder}/possible_sources.bin")

    # create a wandb artifact to track versions of generated sir data
    artifact = wandb.Artifact(data_name, type="dataset")
    wandb.run.tags += (artifact.name,)
    artifact.add_reference(f"file:{pathlib.Path(local_folder).resolve()}")
    wandb.run.log_artifact(artifact)
    wandb.finish()

