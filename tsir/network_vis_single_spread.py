import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION (Based on your tsir.yml) ---
DATA_DIR = "/Users/dariush/Developer/Masterarbeit/source-detection-main/data/yo3mrtdb"  # Path to your binary files
GPICKLE_PATH = f"{DATA_DIR}/network.gpickle"
N_NODES = 15
MC_RUNS = 2000000
def load_mc_data(state_type):
    """Loads Monte Carlo binaries: Shape (N_nodes, MC_runs, N_nodes)"""
    file_path = f"{DATA_DIR}/monte_carlo_{state_type}.bin"
    # C code uses int8_t (1 byte)
    data = np.fromfile(file_path, dtype=np.int8)
    # Reshape as defined in your Python wrapper
    return data.reshape(N_NODES, MC_RUNS, N_NODES)


def visualize_temporal_spread():
    # 1. Load the Network
    with open(GPICKLE_PATH, "rb") as f:
        G = pickle.load(f)
    pos = nx.spring_layout(G, seed=42)

    # 2. Load Monte Carlo I and R binaries
    # These show the state of all nodes if node 'i' was the source
    mc_i = load_mc_data("I")
    mc_r = load_mc_data("R")

    # 3. Choose a Source Node to "Watch"
    source_to_watch = 0

    # Calculate Probabilities for this source across all runs
    # This represents the statistical 'reach' of the pandemic [cite: 132, 133]
    prob_infected = np.mean(mc_i[source_to_watch], axis=0)
    prob_recovered = np.mean(mc_r[source_to_watch], axis=0)

    # 4. Plotting the "Probability Evolution"
    # We use node alpha and size to represent the 'strength' of the spread
    plt.figure(figsize=(12, 10))

    # Draw Background Edges
    nx.draw_networkx_edges(G, pos, alpha=0.1, edge_color="gray")

    # Draw Nodes with a color map representing Infection Likelihood
    # Red = High Probability of Infection, Blue = High Probability of Recovery
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_color=prob_infected,
        cmap=plt.cm.Reds,
        node_size=[500 + (p * 1000) for p in prob_infected],
        edgecolors="black",
        linewidths=1.5
    )

    # Label nodes
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")

    # Add Colorbar for "Infection Wave"
    plt.colorbar(nodes, label="Probability of being Infectious at T=20")

    plt.title(f"Statistical Spread Pattern (Source: Node {source_to_watch})\n"
              f"Based on {MC_RUNS} runs from Petter Holme's SIR model")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    visualize_temporal_spread()