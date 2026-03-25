import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION (Based on tsir.yml) ---
N_NODES = 15
MC_RUNS = 2000000
DATA_DIR = "/Users/dariush/Developer/Masterarbeit/source-detection-main/data/yo3mrtdb"  # Path to your binary files
GPICKLE_PATH = f"{DATA_DIR}/network.gpickle"


def load_mc_data(state_type):
    """Loads Monte Carlo binaries: Shape (N_nodes, MC_runs, N_nodes)"""
    file_path = f"{DATA_DIR}/monte_carlo_{state_type}.bin"
    # C code uses int8_t (1 byte per node)
    data = np.fromfile(file_path, dtype=np.int8)
    return data.reshape(N_NODES, MC_RUNS, N_NODES)


def visualize_evolution_grid(source_node=0):
    # 1. Load the Network and layout
    with open(GPICKLE_PATH, "rb") as f:
        G = pickle.load(f)
    pos = nx.spring_layout(G, seed=42)

    # 2. Load Binaries
    mc_i = load_mc_data("I")
    mc_r = load_mc_data("R")
    mc_s = load_mc_data("S")

    # 3. Simulate "Time Steps" by sampling batches of runs
    # Because final states are independent, sampling batches allows us to see
    # how the distribution of the pandemic stabilizes over time.
    n_snapshots = 4
    batch_size = MC_RUNS // n_snapshots

    fig, axes = plt.subplots(1, n_snapshots, figsize=(24, 6))

    for i in range(n_snapshots):
        ax = axes[i]

        # Calculate probabilities for this batch
        end_idx = (i + 1) * batch_size
        prob_i = np.mean(mc_i[source_node, :end_idx], axis=0)
        prob_r = np.mean(mc_r[source_node, :end_idx], axis=0)

        # Color nodes: Gradient between S (Green), I (Red), R (Blue)
        # We blend colors based on the probability of being in each state
        node_colors = []
        for n in range(N_NODES):
            # RGB blending: R=Infected, G=Susceptible, B=Recovered
            color = (prob_i[n], 1.0 - (prob_i[n] + prob_r[n]), prob_r[n])
            node_colors.append(color)

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color="gray")
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=800, edgecolors="black")
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight="bold")

        ax.set_title(f"Evolution Stage {i + 1}\n(Sample Size: {end_idx})")
        ax.axis("off")

    plt.suptitle(f"Pandemic Reach Evolution (Source: Node {source_node})", fontsize=22)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_evolution_grid(source_node=0)