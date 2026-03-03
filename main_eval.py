import argparse
import numpy as np
import wandb
import networkx as nx
from setup import setup_eval_run, load_tsir_data
from eval import compute_ranks, uniform_probabilities, top_k_score, rank_score, sampled_rank


if __name__ == "__main__":
    # read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    cfg_path = args.cfg # e.g., cfg_path = "exp/exp_1_vary_n/barabasi_albert/eval.yml"
    data_name = args.data # e.g., data_name = "exp_1_vary_n.barabasi_albert:latest"

    # read data and config
    cfg = setup_eval_run(cfg_path, job_type="eval")
    H, data = load_tsir_data(data_name)
    truth_S = data.truth_S.reshape(-1, data.n_nodes)
    possible = data.possible.reshape(-1, data.n_nodes)
    lik_possible = data.lik_possible.reshape(-1, data.n_nodes)

    # if needed, only consider sufficiently large outbreaks
    sel = (1 - truth_S).sum(axis=1) >= cfg.eval.min_outbreak
    if cfg.eval.min_outbreak > 1:
        print(f"For evaluation, only outbreaks with at least {cfg.eval.min_outbreak} infected nodes are considered.")
        print(f" --- That is a total percentage of {100 * np.mean(sel):.1f}% of all outbreaks")

    # uniform baseline
    print("Compute Uniform baseline.")
    uniform_probs = uniform_probabilities(possible)
    uniform_ranks = compute_ranks(uniform_probs, n_nodes=data.n_nodes, n_runs=data.n_runs)
    wandb.summary[f"baseline_uniform_top_k_score"] = [top_k_score(uniform_ranks, sel, k) for k in cfg.eval.top_k]
    wandb.summary[f"baseline_uniform_inverse_rank"] = [rank_score(uniform_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset]

    # random baseline
    print("Compute Random baseline.")
    random_ranks = sampled_rank(possible)
    wandb.summary[f"baseline_random_top_k_score"] = [top_k_score(random_ranks, sel, k) for k in cfg.eval.top_k]
    wandb.summary[f"baseline_random_inverse_rank"] = [rank_score(random_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset]

    # highest degree baseline
    print("Compute Highest Degree baseline.")
    degree_arr = np.array([H.degree(v) for v in range(data.n_nodes)])
    degree_matrix = possible * degree_arr
    degree_ranks = compute_ranks(degree_matrix, n_nodes=data.n_nodes, n_runs=data.n_runs)
    wandb.summary[f"baseline_degree_top_k_score"] = [top_k_score(degree_ranks, sel, k) for k in cfg.eval.top_k]
    wandb.summary[f"baseline_degree_inverse_rank"] = [rank_score(degree_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset]

    # jordan center baseline
    print("Compute Jordan Center baseline.")
    dists = dict(nx.all_pairs_shortest_path_length(H))
    shortest_path_lengths_per_node = [np.array([dists[i][j] for j in range(data.n_nodes)]) for i in range(data.n_nodes)]
    print(" --- All shortest paths lengths on static network computed.")
    # for each node, look at its shortest path length to all I or R (not S) of a certain outbreak and take its maximum
    eccentricity = np.column_stack([((1 - truth_S) * arr).max(axis=1) for arr in shortest_path_lengths_per_node])
    jordan_ranks = compute_ranks(-eccentricity - lik_possible, n_nodes=data.n_nodes, n_runs=data.n_runs)
    wandb.summary[f"baseline_jordan_top_k_score"] = [top_k_score(jordan_ranks, sel, k) for k in cfg.eval.top_k]
    wandb.summary[f"baseline_jordan_inverse_rank"] = [rank_score(jordan_ranks, sel, offset) for offset in cfg.eval.inverse_rank_offset]

    # finish
    wandb.finish()
