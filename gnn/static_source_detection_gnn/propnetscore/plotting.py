import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_task(
    ax,
    G: nx.Graph,
    node_probs: np.ndarray,
    true_node: int,
    labels: str | None = "id",  # "id", "prob", or None
    colorbar: bool = True,
    contrast: float = 1.0,
    prob_color: str = 'Blues',
    true_color: str = 'red',
    scores: list = [],
    score_pos: str = "bottom",  # "top" or "bottom"
    node_size: int = 500
):
    if len(node_probs) != G.number_of_nodes():
        raise ValueError("Length of node_probs must match number of nodes in G")

    # Apply contrast transformation
    transformed_probs = node_probs ** (1 / contrast)
    normalized_probs = transformed_probs / max(transformed_probs) if max(transformed_probs) > 0 else transformed_probs

    # Colormap setup
    cmap = plt.get_cmap(prob_color)
    colors = [cmap(p) for p in normalized_probs]

    # Compute layout
    pos = nx.spring_layout(G, seed=42)
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    # Draw network
    nx.draw_networkx_edges(G, pos, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, edgecolors='black',
                           linewidths=1, node_size=node_size, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[true_node], node_color=[colors[true_node]],
                           edgecolors=true_color, linewidths=4, node_size=node_size, ax=ax)

    # Draw labels
    if labels == "id":
        nx.draw_networkx_labels(G, pos, ax=ax)
    elif labels == "prob":
        # Format probabilities with up to 2 decimals, but suppress trailing zeros
        prob_labels = {i: f"{p:.2f}".rstrip('0').rstrip('.') for i, p in enumerate(node_probs)}
        # Compute a font size that fits inside the node circles
        font_size = max(6, np.sqrt(node_size) / 8)
        nx.draw_networkx_labels(G, pos, labels=prob_labels, font_size=font_size, ax=ax)

    fig = ax.figure
    cbar = None
    if colorbar:
        sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=max(transformed_probs)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Probability')

    # Compute and display scores
    if scores:
        import propnetscore.node_selection as ns
        task = ns.NodeSelectionTask(nx.to_numpy_array(G, weight=None), node_probs, true_node)
        score_texts = []

        for entry in scores:
            # Handle both "resistance" and {"diffusion": {"alpha": 0.1}}
            if isinstance(entry, str):
                name, kwargs = entry, {}
            elif isinstance(entry, dict):
                if len(entry) != 1:
                    raise ValueError(f"Invalid score entry: {entry}")
                name, kwargs = next(iter(entry.items()))
            else:
                raise TypeError(f"Unsupported score format: {entry}")

            # Build label with unnamed parameter values
            if kwargs:
                param_str = "(" + ", ".join(str(v) for v in kwargs.values()) + ")"
            else:
                param_str = ""

            method_name = f"{name}_score"
            if hasattr(task, method_name):
                try:
                    value = getattr(task, method_name)(**kwargs)
                    score_texts.append((f"{name}{param_str}", f"{value:.3g}"))
                except Exception as e:
                    score_texts.append((f"{name}{param_str}", f"(error: {e})"))
            else:
                score_texts.append((f"{name}{param_str}", "(not found)"))

        # Compute proper column alignment
        name_width = max(len(name) for name, _ in score_texts)
        value_width = max(len(val) for _, val in score_texts)
        legend_text = "\n".join(
            f"{name.ljust(name_width)} : {val.rjust(value_width)}"
            for name, val in score_texts
        )

        # Compute y position based on user choice
        if score_pos.lower() == "top":
            y_pos = y_max + (y_max - y_min) * 0.15
            va = "bottom"
        else:
            y_pos = y_min - (y_max - y_min) * 0.15
            va = "top"

        ax.text(
            (x_min + x_max) / 2,
            y_pos,
            legend_text,
            fontsize=10,
            va=va,
            ha='center',
            fontfamily='monospace'
        )

    ax.set_axis_off()
    ax.set_xlim(x_min - 0.1, x_max + 0.1)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
