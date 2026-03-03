import numpy as np
import yaml
from setup import Config
import wandb
from setup import setup_methods_run, load_tsir_data
from mc import monte_carlo, monte_carlo_exclude
from eval import log_likelihood, compute_ranks, top_k_score, rank_score


if __name__ == "__main__":
    # read data and eval config
    setup_methods_run(job_type="mc")
    with open(wandb.config.eval_name) as f:
        config_data = yaml.safe_load(f)
    wandb.config.update(config_data, allow_val_change=True)
    cfg = Config(config_data)

    # load data
    H, data = load_tsir_data(wandb.config.data_name)
    truth_S = data.truth_S.reshape(-1, data.n_nodes)
    truth_I = data.truth_I.reshape(-1, data.n_nodes)
    truth_R = data.truth_R.reshape(-1, data.n_nodes)
    lik_possible = data.lik_possible.reshape(-1, data.n_nodes)

    # make sanity checks
    n = wandb.config.n_reps["n"]
    if n > data.mc_runs:
        raise ValueError(f"The dataset does not contain enough Monte Carlo simulations: {data.mc_runs} available, but {n} requested.")
    sel = (1 - truth_S).sum(axis=1) >= cfg.eval.min_outbreak # if needed, only consider sufficiently large outbreaks
    if cfg.eval.min_outbreak > 1:
        print(f"For evaluation, only outbreaks with at least {cfg.eval.min_outbreak} infected nodes are considered.")
        print(f" --- That is a total percentage of {100 * np.mean(sel):.1f}% of all outbreaks")

    # monte carlo mean field method
    print("Compute Monte Carlo Mean Field method.")
    all_top_k = {exclude: [] for exclude in wandb.config.exclude}
    all_inv_rank = {exclude: [] for exclude in wandb.config.exclude}
    for i in range(wandb.config.n_reps["reps"]):
        print(f" - For repetition {i+1}/{wandb.config.n_reps["reps"]}")
        select = np.random.choice(data.mc_runs, n, replace=False)
        mc_S, mc_I, mc_R = data.mc_S[:, select, :], data.mc_I[:, select, :], data.mc_R[:, select, :]
        for exclude in wandb.config.exclude:
            mc_log_S, mc_log_I, mc_log_R, correct = monte_carlo_exclude(
                mc_S, mc_I, mc_R, n, data.n_nodes, data.maximal_outbreak, exclude=exclude
            )
            mc_log_lik = log_likelihood(truth_S.astype(float), truth_I.astype(float), truth_R.astype(float),
                                        mc_log_S, mc_log_I, mc_log_R, weights=wandb.config.weights)
            mc_log_lik = mc_log_lik - lik_possible + correct # set impossible to -inf and use correct factor
            mc_ranks = compute_ranks(mc_log_lik, n_nodes=data.n_nodes, n_runs=data.n_runs)
            all_top_k[exclude].append([top_k_score(mc_ranks, sel, k) for k in cfg.eval.top_k])
            all_inv_rank[exclude].append([rank_score(mc_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset])

    # log results
    for exclude in wandb.config.exclude:
        wandb.summary[f"monte_carlo_exclude={exclude}_n={n}_top_k_score"] = np.mean(all_top_k[exclude], axis=0).tolist()
        wandb.summary[f"monte_carlo_exclude={exclude}_n={n}_inverse_rank"] = np.mean(all_inv_rank[exclude], axis=0).tolist()
    wandb.finish()
