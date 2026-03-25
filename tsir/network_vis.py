import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION (Matching your Python Wrapper) ---
DATA_DIR = "/Users/dariush/Developer/Masterarbeit/source-detection-main/data/yo3mrtdb"
N_NODES = 15
MC_RUNS = 2000000  # This should match cfg.sir.mc_runs in your config


def load_mc_binary(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path): return None

    # Per C code: fwrite(buf, sizeof(int8_t), g.n, fI);
    data = np.fromfile(path, dtype=np.int8)

    # Reshape based on your Python wrapper logic: (n_nodes, mc_runs, n_nodes)
    try:
        return data.reshape(N_NODES, -1, N_NODES)
    except ValueError:
        print(f"Size mismatch in {filename}. Adjust N_NODES or check file.")
        return None


def visualize_spread_by_source():
    # 1. Load Topology
    with open(os.path.join(DATA_DIR, "network.gpickle"), "rb") as f:
        G = pickle.load(f)
    pos = nx.spring_layout(G, seed=42)

    # 2. Load Binaries
    mc_i = load_mc_binary("monte_carlo_I.bin")
    if mc_i is None: return

    # 3. Plotting the "Evolution" of probabilities across different sources
    # We pick 3 nodes to see how starting at different points changes the spread
    sample_sources = [0, N_NODES // 2, N_NODES - 1]
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for idx, src_node in enumerate(sample_sources):
        ax = axes[idx]

        # Calculate probability: Average of all MC runs starting from this source
        # mc_i[src_node] has shape (mc_runs, n_nodes)
        prob_map = np.mean(mc_i[src_node], axis=0)

        # Draw
        nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=prob_map,
                                       cmap=plt.cm.YlOrRd, node_size=700, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_color='black')

        ax.set_title(f"Infection Probabilities\nIf Source is Node {src_node}")
        ax.axis('off')

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=axes.tolist(), shrink=0.8, label="Probability of Infection at stop_t")

    plt.suptitle("Statistical Spread Map (Based on Petter Holme C Binaries)", fontsize=20)
    plt.savefig("monte_carlo_spread.png")
    print("Success! Created 'monte_carlo_spread.png'")
    plt.show()


if __name__ == "__main__":
    visualize_spread_by_source()