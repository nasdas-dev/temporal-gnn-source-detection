import numpy as np
import torch
import yaml
from setup import Config
import wandb
from setup import setup_methods_run, load_tsir_data
from eval import log_likelihood, compute_ranks, top_k_score, rank_score
from sm import jaccard_similarity, soft_margin_numpy

if __name__ == "__main__":
    # read data and eval config
    cfg = setup_methods_run(job_type="sm")
    with open(wandb.config.eval_name) as f:
        config_data = yaml.safe_load(f)
    wandb.config.update(config_data, allow_val_change=True)
    cfg = Config(config_data)

    # load data
    H, data = load_tsir_data(wandb.config.data_name)

    # make sanity checks
    n = wandb.config.n_reps["n"]
    if n > data.mc_runs:
        raise ValueError(f"The dataset does not contain enough Monte Carlo simulations: {data.mc_runs} available, but {n} requested.")
    n_truth = wandb.config.n_reps["n_truth"]
    if n_truth > data.n_runs:
        raise ValueError(f"The dataset does not contain enough Ground Truth simulations: {data.n_runs} available, but {n_truth} requested.")

    # monte carlo mean field method
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Monte Carlo Soft Margin method with device={device}.")
    all_top_k = {a: [] for a in wandb.config.a}
    all_inv_rank = {a: [] for a in wandb.config.a}
    for i in range(wandb.config.n_reps["reps"]):
        print(f" - For repetition {i+1}/{wandb.config.n_reps["reps"]}")
        select = np.random.choice(data.mc_runs, n, replace=False)
        mc_S = data.mc_S[:, select, :]
        select_truth = np.arange(i * n_truth, (i+1) * n_truth) % data.n_runs
        truth_S = data.truth_S[:, select_truth, :].reshape(-1, data.n_nodes)
        possible = data.possible[:, select_truth, :].reshape(-1, data.n_nodes)
        sel = (1 - truth_S).sum(axis=1) >= cfg.eval.min_outbreak  # if needed, only consider sufficiently large outbreaks
        jaccard = jaccard_similarity(mc_S, truth_S, data.n_nodes, device)
        for a in wandb.config.a:
            sm_probs = soft_margin_numpy(jaccard=jaccard, a=a)
            sm_probs = sm_probs * possible  # set impossible sources to zero probability
            # sm_probs = sm_probs / np.sum(sm_probs, axis=1, keepdims=True) # re-normalize
            sm_ranks = compute_ranks(sm_probs, n_nodes=data.n_nodes, n_runs=n_truth)
            all_top_k[a].append([top_k_score(sm_ranks, sel, k) for k in cfg.eval.top_k])
            all_inv_rank[a].append([rank_score(sm_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset])

    # log results
    for a in wandb.config.a:
        wandb.summary[f"soft_margin_a={a}_n={n}_top_k_score"] = all_top_k[a]
        wandb.summary[f"soft_margin_a={a}_n={n}_inverse_rank"] = all_inv_rank[a]
    wandb.finish()
