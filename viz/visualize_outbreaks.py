import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_snapshot(H, run_id, which, seed, save_to, source=0, run=0):
    n_nodes = len(H.nodes())
    n_runs = np.fromfile(f"data/{run_id}/{which}_S.bin", dtype=np.int8).size // n_nodes**2
    truth_S, truth_I, truth_R = (np.fromfile(f"data/{run_id}/ground_truth_{state}.bin", dtype=np.int8).
                                 reshape(n_runs * n_nodes, n_nodes) for state in "SIR")
    row_index = source * n_runs + run # choose one snapshot corresponding to source and run
    snapshot_S, snapshot_I, snapshot_R = truth_S[row_index], truth_I[row_index], truth_R[row_index]
    node_colors = np.full(n_nodes, 'skyblue', dtype=object)
    node_colors[snapshot_R.astype(bool)] = 'orange'
    node_colors[snapshot_I.astype(bool)] = 'red'

    pos = nx.spring_layout(H, seed=seed)
    plt.figure(figsize=(8, 5))
    nx.draw_networkx_nodes(H, pos, nodelist=sorted(H.nodes()), node_size=40, node_color=node_colors)
    nx.draw_networkx_edges(H, pos, alpha=0.3)
    plt.savefig(save_to, dpi=200)
    plt.close()


def outbreak_size_histogram(H, run_id, which, save_to):
    n_nodes = len(H.nodes())
    mc_S = np.fromfile(f"data/{run_id}/{which}_S.bin", dtype=np.int8).reshape(-1, n_nodes)
    outbreak_sizes = np.sum(1-mc_S, axis=1)
    plt.hist(outbreak_sizes, bins=range(0, n_nodes + 2), edgecolor='black')
    plt.xlabel("Outbreak size")
    plt.ylabel("Frequency")
    plt.title(f"Outbreak size distribution")
    plt.savefig(save_to)
    plt.close()

