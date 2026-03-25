import os
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns


def plot_vary_n(result, k_values, n_values, a_values, exclude_values, range, linewidth, fontsize, title, n_nodes, save_to):
    result_long = result.melt(id_vars="k", var_name="method", value_name="score")
    result_long["n"] = result_long["method"].str.extract(r"n=(\d+)").astype(float)
    result_long["exclude"] = result_long["method"].str.extract(r"exclude=(\d+)").astype(float)
    result_long["a"] = result_long["method"].str.extract(r"a=(\d*\.?\d+)").astype(float)
    result_long["method"] = result_long["method"].str.replace(r"_n=.*", "", regex=True)

    result_long = result_long[result_long["n"].isin(n_values) | result_long["n"].isna()]
    result_long = result_long[result_long["a"].isin(a_values) | result_long["a"].isna()]
    result_long = result_long[result_long["exclude"].isin(exclude_values) | result_long["exclude"].isna()]

    for k in k_values:
        df_plot = result_long[(result_long["k"] == k)]
        df_plot.loc[:, "method"] = (
            df_plot["method"]
            .replace({"static_gnn": "Graph Neural Network"})
            .str.replace("monte_carlo_", "Monte Carlo Mean Field ")
            .str.replace("soft_margin_", "Soft Margin ")
            .str.replace("individual_based", "Individual Based Approx.")
            .str.replace("baseline_jordan", "Baseline: Jordan Centrality")
            .str.replace("baseline_random", "Baseline: Random Ranks")
            .str.replace("baseline_uniform", "Baseline: Uniform")
            .str.replace("baseline_degree", "Baseline: Degree Centrality")
        )

        palette = {"Graph Neural Network": "purple"}
        mc_colors = sns.color_palette("Oranges", n_colors=len(exclude_values))[::-1]
        for val, col in zip(exclude_values, mc_colors):
            if val == 0:
                palette[f"Monte Carlo Mean Field"] = col
                df_plot.loc[:, "method"] = (df_plot["method"].str.replace(" exclude=0", ""))
            else:
                palette[f"Monte Carlo Mean Field exclude={val}"] = col
        sm_colors = sns.color_palette("Blues", n_colors=len(a_values))
        for val, col in zip(a_values, sm_colors):
            palette[f"Soft Margin a={val}"] = col

        plt.figure(figsize=(8,5))
        plt.rcParams.update({'font.size': fontsize})
        if n_nodes is not None:
            plt.axvline(x=2 ** n_nodes, linestyle="--", color="grey", linewidth=linewidth*0.6)

        ax = sns.lineplot(data=df_plot[~df_plot["n"].isna()], x="n", y="score", hue="method", marker="", palette=palette, hue_order=sorted(palette.keys()),
                     linewidth=linewidth)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))  # if your data is 0–1
        plt.xscale("log")
        n_values = sorted(df_plot["n"].dropna().unique().tolist())
        ticks = [n for n in n_values if n == 10 ** int(np.log10(n))]
        labels = [f"$10^{int(np.log10(n))}$" for n in ticks]
        plt.xticks(ticks, labels)

        y = df_plot.loc[df_plot["method"] == "Individual Based Approx.", "score"].values[0]
        #plt.axhline(y, linestyle="-", color="green", label="Individual Based Approx.", linewidth=linewidth)
        plt.plot([df_plot["n"].min(), df_plot["n"].max()], [y, y], linestyle="-", color="green",
                 label="Individual Based Approx.", linewidth=linewidth)
        #id_max_baseline = df_plot[df_plot["method"].str.startswith(("Baseline", "baseline"))]["score"].idxmax()
        #y = df_plot.loc[df_plot["method"] == "Baseline: Random Ranks", "score"].values[0]
        #plt.axhline(y, linestyle="-", color="black", label="Baseline: Random Ranks")
        plt.ylim(bottom=df_plot["score"].max() - range, top=df_plot["score"].max() + 0.01)

        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("")
        plt.ylabel(f"Top-{k} Score")
        plt.legend(loc="lower right")
        plt.title(title, fontsize=fontsize)
        os.makedirs(save_to, exist_ok=True)
        plt.savefig(f"{save_to}/top_k={k}.png", dpi=200, bbox_inches="tight")
        plt.savefig(f"{save_to}/top_k={k}.svg", dpi=200, bbox_inches="tight")
        plt.gca().get_legend().remove()
        plt.savefig(f"{save_to}/top_k={k}_no_legend.png", dpi=200, bbox_inches="tight")
        plt.savefig(f"{save_to}/top_k={k}_no_legend.svg", dpi=200, bbox_inches="tight")
        plt.close()

