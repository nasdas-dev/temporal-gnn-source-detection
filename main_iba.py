import argparse

import numpy as np
import yaml
import wandb
from setup import setup_methods_run, make_array_from_networkx, Config, load_tsir_data
from iba import iba, make_c_readable_from_nparray
from eval import log_likelihood, compute_ranks, uniform_probabilities, top_k_score, rank_score, sampled_rank


if __name__ == "__main__":
    # read data and eval config
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg  # e.g., cfg_path = "exp/exp_1_vary_n/barabasi_albert/eval.yml"
    data_name = args.data  # e.g., data_name = "exp_1_vary_n.barabasi_albert:latest"

    # read config
    cfg = setup_methods_run(job_type="iba")
    with open(cfg_path) as f:
        config_data = yaml.safe_load(f)
    wandb.config.update(config_data, allow_val_change=True)
    cfg = Config(config_data)

    # load data
    H, data = load_tsir_data(data_name)
    truth_S = data.truth_S.reshape(-1, data.n_nodes)
    truth_I = data.truth_I.reshape(-1, data.n_nodes)
    truth_R = data.truth_R.reshape(-1, data.n_nodes)
    lik_possible = data.lik_possible.reshape(-1, data.n_nodes)

    # make sanity checks
    sel = (1 - truth_S).sum(axis=1) >= cfg.eval.min_outbreak  # if needed, only consider sufficiently large outbreaks
    if cfg.eval.min_outbreak > 1:
        print(f"For evaluation, only outbreaks with at least {cfg.eval.min_outbreak} infected nodes are considered.")
        print(f" --- That is a total percentage of {100 * np.mean(sel):.1f}% of all outbreaks")

    # individual-based-approximation (IBA) method
    H_array = make_array_from_networkx(H)
    H_cedges = make_c_readable_from_nparray(H_array, end_t=data.config["sir"]["end_t"], n_nodes=data.n_nodes)
    iba_log_S, iba_log_I, iba_log_R = iba(data.config, H_cedges, data.n_nodes)
    iba_log_lik = log_likelihood(truth_S.astype(float), truth_I.astype(float), truth_R.astype(float),
                                 iba_log_S, iba_log_I, iba_log_R, weights=1.0)
    iba_log_lik = iba_log_lik - lik_possible # set impossible sources to -inf
    iba_ranks = compute_ranks(iba_log_lik, n_nodes=data.n_nodes, n_runs=data.n_runs)
    wandb.summary[f"individual_based_top_k_score"] = [top_k_score(iba_ranks, sel, k) for k in cfg.eval.top_k]
    wandb.summary[f"individual_based_inverse_rank"] = [rank_score(iba_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset]

    # finish
    wandb.finish()

