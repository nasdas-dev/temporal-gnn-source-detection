"""
Network visualisation with per-node probability overlays.

Adapted from gnn/static_source_detection_gnn/propnetscore/plotting.py.
"""

from typing import Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_task(
    ax: plt.Axes,
    G: nx.Graph,
    node_probs: np.ndarray,
    true_node: int,
    labels: Optional[str] = "id",
    colorbar: bool = True,
    contrast: float = 1.0,
    prob_color: str = "Blues",
    true_color: str = "red",
    node_size: int = 500,
) -> None:
    """Draw a network with nodes coloured by predicted source probability.

    Each node is shaded according to its predicted probability.  The true
    source node is highlighted with a thick coloured border so it stands out
    from the rest.

    Adapted from ``propnetscore/plotting.py`` in the static-source-detection
    subpackage.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on.
    G : nx.Graph
        NetworkX graph to visualise.
    node_probs : np.ndarray, shape (n_nodes,)
        Predicted probability distribution over nodes.  Length must equal
        ``G.number_of_nodes()``.
    true_node : int
        Index of the true source node; receives a highlighted border.
    labels : {"id", "prob", None}
        Which labels to draw on each node.  ``"id"`` shows node indices,
        ``"prob"`` shows rounded probability values, ``None`` draws no labels.
    colorbar : bool
        Whether to attach a colour bar to the axes.
    contrast : float
        Exponent applied to the probabilities before mapping to colours
        (``transformed = p ** (1 / contrast)``).  Values > 1 increase
        contrast; values < 1 decrease it.
    prob_color : str
        Name of a matplotlib colour map used to colour the nodes.
    true_color : str
        Colour of the border drawn around the true source node.
    node_size : int
        Size of each node marker (passed to ``nx.draw_networkx_nodes``).

    Raises
    ------
    ValueError
        If ``len(node_probs) != G.number_of_nodes()``.
    """
    if len(node_probs) != G.number_of_nodes():
        raise ValueError(
            f"len(node_probs)={len(node_probs)} must match "
            f"G.number_of_nodes()={G.number_of_nodes()}"
        )

    # Apply contrast transformation and normalise to [0, 1]
    transformed_probs = node_probs ** (1.0 / contrast)
    max_transformed = float(np.max(transformed_probs))
    normalized_probs = (
        transformed_probs / max_transformed if max_transformed > 0 else transformed_probs
    )

    # Map normalised probabilities to colours
    cmap = plt.get_cmap(prob_color)
    colors = [cmap(float(p)) for p in normalized_probs]

    # Compute spring layout
    pos = nx.spring_layout(G, seed=42)
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, edgecolors="black",
        linewidths=1, node_size=node_size, ax=ax,
    )
    # Highlight the true source node with a thick border
    nx.draw_networkx_nodes(
        G, pos, nodelist=[true_node],
        node_color=[colors[true_node]],
        edgecolors=true_color, linewidths=4, node_size=node_size, ax=ax,
    )

    # Draw node labels
    if labels == "id":
        nx.draw_networkx_labels(G, pos, ax=ax)
    elif labels == "prob":
        prob_labels = {
            i: f"{p:.2f}".rstrip("0").rstrip(".")
            for i, p in enumerate(node_probs)
        }
        font_size = max(6, np.sqrt(node_size) / 8)
        nx.draw_networkx_labels(G, pos, labels=prob_labels, font_size=font_size, ax=ax)

    # Optional colour bar
    if colorbar:
        fig = ax.figure
        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vmin=0, vmax=max_transformed),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Probability")

    ax.set_axis_off()
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
