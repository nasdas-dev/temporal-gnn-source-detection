#!/usr/bin/env python3
"""
Backtracking Network (BN) on the Holme toy temporal network.

Trains and evaluates the BN from Ru et al. (AAAI 2023) on the small
temporal contact network in ``toy_example_holme.csv``, logging all
hyperparameters, training curves, and evaluation metrics to W&B.

Pipeline
--------
1. Load ``toy_example_holme.csv`` (u v t triples) into a NetworkX graph.
2. Run C-based SIR simulator to generate ground-truth infection snapshots.
3. Build BN inputs: aggregated edge_index + binary activation-pattern edge_attr.
4. Train the BacktrackingNetwork with batched cross-entropy + early stopping.
5. Evaluate on all (source, run) pairs; log rich metrics and tables to W&B.

Usage (run from project root)::

    python toy-example/run_bn_toy.py [--hidden_dim 32] [--num_layers 3] ...

All hyperparameters have sensible defaults and can be overridden via CLI flags.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import wandb
from sklearn.model_selection import train_test_split

# Setup
# --------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tsir.read_run import make_c_readable_from_networkx, run as tsir_run
from gnn.backtracking_network import BacktrackingNetwork
from eval.ranks import compute_expected_ranks
from eval.scores import top_k_score, rank_score

# --------------------------------------------------------------
# Data Directory
# --------------------------------------------------------------
DATA_DIR = os.path.join(ROOT, "toy-example", "data")


# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

@dataclass
class Config:
    # Network
    csv_path:       str   = "toy-example/toy_example_holme.csv"
    directed:       bool  = False
    # SIR simulation
    beta:           float = 0.30   # per-contact infection probability
    mu:             float = 0.20   # per-timestep recovery probability
    n_runs:         int   = 500    # ground-truth SIR runs per source node
    # BN architecture
    hidden_dim:     int   = 32     # hidden embedding size D
    num_layers:     int   = 3      # number of BN convolutional layers L
    # Optimiser
    lr:             float = 1e-3
    weight_decay:   float = 1e-4
    # Training loop
    batch_size:     int   = 64
    epochs:         int   = 300
    early_stop:     int   = 30     # patience (epochs without val improvement)
    test_size:      float = 0.20   # fraction of samples held out for validation
    # Reproducibility
    seed:           int   = 42
    # W&B
    wandb_project:  str   = "source-detection-bn-toy"
    wandb_entity:   str   = ""     # leave empty for default entity

    def as_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


def parse_config() -> Config:
    """Build Config from CLI flags (all optional; defaults from the dataclass)."""
    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_path",      type=str,   default=cfg.csv_path)
    p.add_argument("--directed",      action="store_true", default=cfg.directed)
    p.add_argument("--beta",          type=float, default=cfg.beta)
    p.add_argument("--mu",            type=float, default=cfg.mu)
    p.add_argument("--n_runs",        type=int,   default=cfg.n_runs)
    p.add_argument("--hidden_dim",    type=int,   default=cfg.hidden_dim)
    p.add_argument("--num_layers",    type=int,   default=cfg.num_layers)
    p.add_argument("--lr",            type=float, default=cfg.lr)
    p.add_argument("--weight_decay",  type=float, default=cfg.weight_decay)
    p.add_argument("--batch_size",    type=int,   default=cfg.batch_size)
    p.add_argument("--epochs",        type=int,   default=cfg.epochs)
    p.add_argument("--early_stop",    type=int,   default=cfg.early_stop)
    p.add_argument("--test_size",     type=float, default=cfg.test_size)
    p.add_argument("--seed",          type=int,   default=cfg.seed)
    p.add_argument("--wandb_project", type=str,   default=cfg.wandb_project)
    p.add_argument("--wandb_entity",  type=str,   default=cfg.wandb_entity)
    args = p.parse_args()
    return Config(**vars(args))


# --------------------------------------------------------------
# Step 1 - Load temporal network
# --------------------------------------------------------------

def load_temporal_network(csv_path: str, directed: bool) -> nx.Graph:
    """Parse a whitespace-separated edge-list (u v t) into a NetworkX graph.

    Node labels and time stamps are both re-indexed to start at 0.
    Each edge carries a ``'times'`` attribute - the sorted list of active
    time steps.

    Parameters
    ----------
    csv_path:
        Path to the CSV file (rows: ``u v t``).
    directed:
        Build a ``DiGraph`` when *True*, a ``Graph`` otherwise.

    Returns
    -------
    G : nx.Graph | nx.DiGraph
        Graph with ``graph['n_nodes']``, ``graph['t_max']``, and per-edge
        ``'times'`` attributes.
    """
    edges_raw: list[tuple[int, int, int]] = []
    with open(csv_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v, t = map(int, line.split())
            edges_raw.append((u, v, t))

    # Re-index nodes to 0-based
    node_ids = sorted({n for u, v, _ in edges_raw for n in (u, v)})
    node_map = {n: i for i, n in enumerate(node_ids)}

    # Re-index times to 0-based
    t_min = min(t for _, _, t in edges_raw)

    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(range(len(node_ids)))

    for u_raw, v_raw, t_raw in edges_raw:
        u = node_map[u_raw]
        v = node_map[v_raw]
        t = t_raw - t_min
        if G.has_edge(u, v):
            G[u][v]["times"].append(t)
        else:
            G.add_edge(u, v, times=[t])

    # Deduplicate and sort activation times per edge
    for _, _, data in G.edges(data=True):
        data["times"] = sorted(set(data["times"]))

    G.graph["n_nodes"]  = len(node_ids)
    G.graph["t_max"]    = max(t for _, _, t in edges_raw) - t_min
    G.graph["directed"] = directed

    return G


# --------------------------------------------------------------
# Step 2 - Build BN graph inputs
# --------------------------------------------------------------

def build_bn_inputs(
    G: nx.Graph,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert the temporal graph into tensors for the BacktrackingNetwork.

    Returns
    -------
    edge_index : LongTensor [2, E]
        COO representation of the aggregated static network G̃_a.
        Undirected graphs are expanded to both directions so that every
        physical contact contributes a message in each direction.
    edge_attr : FloatTensor [E, T]
        Binary activation pattern.  ``edge_attr[e, t] = 1`` iff edge *e*
        was active at time step *t*.
    T : int
        Total number of time steps (= t_max + 1).
    """
    t_max    = G.graph["t_max"]
    directed = G.graph["directed"]
    T        = t_max + 1          # time steps indexed 0 .. t_max

    src_list:  list[int]          = []
    dst_list:  list[int]          = []
    attr_list: list[torch.Tensor] = []

    for u, v, data in G.edges(data=True):
        act = torch.zeros(T)
        for t in data["times"]:
            act[t] = 1.0

        # Forward edge (u → v)
        src_list.append(u);  dst_list.append(v);  attr_list.append(act)

        if not directed:
            # Reverse edge (v → u) - same activation pattern
            src_list.append(v);  dst_list.append(u);  attr_list.append(act.clone())

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.stack(attr_list, dim=0)        # [E, T]

    return edge_index, edge_attr, T

# --------------------------------------------------------------
# Step 3 - Build training tensors from SIR output
# --------------------------------------------------------------

def build_training_data(
    truth_S: np.ndarray,  # [N, N_RUNS, N]  int8 - 1 if susceptible
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    n_nodes: int,
    n_runs:  int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack SIR binary arrays into one-hot node-feature tensors.

    Returns
    -------
    X : FloatTensor [N, N_RUNS, N, 3]
        One-hot SIR state for every node in every simulation sample.
        Encoding: susceptible=[1,0,0], infectious=[0,1,0], recovered=[0,0,1].
    y : LongTensor [N * N_RUNS]
        True source node index for every sample.
        ``y[s * n_runs + r] = s``.
    """
    X = torch.tensor(
        np.stack([truth_S, truth_I, truth_R], axis=-1),
        dtype=torch.float32,
    )                                                    # [N, N_RUNS, N, 3]
    y = torch.tensor(
        np.repeat(np.arange(n_nodes), n_runs),
        dtype=torch.long,
    )                                                    # [N * N_RUNS]
    return X, y


# --------------------------------------------------------------
# Step 4 - Training loop
# --------------------------------------------------------------

def train(
    model:      BacktrackingNetwork,
    X:          torch.Tensor,   # [N, N_RUNS, N, 3]
    y:          torch.Tensor,   # [N * N_RUNS]
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
    cfg:        Config,
    device:     torch.device,
) -> tuple[list[float], list[float]]:
    """Train the BN, logging per-epoch losses to W&B.

    Uses stratified train/val split (stratified by source node) and Adam with
    optional early stopping.

    Returns
    -------
    train_losses, val_losses : list of per-epoch average NLL.
    """
    n_nodes, n_runs = X.shape[0], X.shape[1]
    idx_pairs = [(s, r) for s in range(n_nodes) for r in range(n_runs)]
    labels    = [s for s, _ in idx_pairs]

    train_idx, val_idx = train_test_split(
        idx_pairs,
        test_size    = cfg.test_size,
        random_state = cfg.seed,
        stratify     = labels,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    edge_index = edge_index.to(device)
    edge_attr  = edge_attr.to(device)

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val  = float("inf")
    patience  = 0

    for epoch in range(cfg.epochs):
        # Training
        model.train()
        random.shuffle(train_idx)
        epoch_loss = 0.0

        for b in range(0, len(train_idx), cfg.batch_size):
            batch = train_idx[b : b + cfg.batch_size]

            # x_b: [B, N, 3]  -  final-state one-hot for each sample in batch
            x_b = torch.stack([X[s, r] for s, r in batch]).to(device)
            y_b = torch.tensor(
                [y[s * n_runs + r].item() for s, r in batch], device=device
            )

            optimizer.zero_grad()
            log_probs = model(x_b, edge_index, edge_attr)   # [B, N]
            loss = F.nll_loss(log_probs, y_b, reduction="sum")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        
        with torch.no_grad():
            x_v = torch.stack([X[s, r] for s, r in val_idx]).to(device)
            y_v = torch.tensor(
                [y[s * n_runs + r].item() for s, r in val_idx], device=device
            )
            val_log  = model(x_v, edge_index, edge_attr)
            val_loss = F.nll_loss(val_log, y_v, reduction="sum").item()

        tl = epoch_loss / len(train_idx)
        vl = val_loss   / len(val_idx)
        train_losses.append(tl)
        val_losses.append(vl)

        wandb.log({"epoch": epoch + 1, "train/loss": tl, "val/loss": vl})

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  [{epoch + 1:>4}/{cfg.epochs}]  "
                  f"train={tl:.4f}  val={vl:.4f}")

        # Early stopping
        if vl < best_val:
            best_val = vl
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop:
                print(f"  Early stopping at epoch {epoch + 1}  "
                      f"(best val={best_val:.4f})")
                break

    return train_losses, val_losses


# --------------------------------------------------------------
# Step 5 - Evaluation
# --------------------------------------------------------------


def evaluate(
    model:      BacktrackingNetwork,
    X:          torch.Tensor,   # [N, N_RUNS, N, 3]
    edge_index: torch.Tensor,
    edge_attr:  torch.Tensor,
    sel:        np.ndarray,     # [N * N_RUNS] bool - valid (non-trivial) outbreaks
    n_nodes:    int,
    n_runs:     int,
    device:     torch.device,
) -> dict[str, float]:
    """Run inference on all samples and compute ranking metrics.

    Logs metrics to W&B summary and logs three rich tables/plots:
    - ``results/per_source_table`` - top-1 and rank score per source node
    - ``results/summary_table``    - BN vs random baseline side-by-side
    - ``results/rank_histogram``   - distribution of true-source ranks

    Returns
    -------
    metrics : dict mapping metric name → float value
    """
    model.eval()
    n_total = n_nodes * n_runs
    probs   = np.zeros((n_total, n_nodes), dtype=np.float32)

    with torch.no_grad():
        for s in range(n_nodes):
            x_s   = X[s].to(device)                          # [N_RUNS, N, 3]
            log_p = model(x_s, edge_index.to(device), edge_attr.to(device))
            probs[s * n_runs : (s + 1) * n_runs] = (
                log_p.exp().cpu().numpy()
            )

    ranks = compute_expected_ranks(probs, n_nodes=n_nodes, n_runs=n_runs)
    top1  = top_k_score(ranks, sel, k=1)
    top3  = top_k_score(ranks, sel, k=min(3, n_nodes))
    rs    = rank_score(ranks, sel, offset=0)
    mr    = float(ranks[sel].mean())

    metrics = {
        "eval/top1":        top1,
        "eval/top3":        top3,
        "eval/rank_score":  rs,
        "eval/mean_rank":   mr,
        "eval/n_valid":     int(sel.sum()),
        "eval/n_total":     n_total,
    }
    for k, v in metrics.items():
        wandb.summary[k] = float(v)

    #### WANDB
    per_source_rows = []
    for s in range(n_nodes):
        s_sel   = sel[s * n_runs : (s + 1) * n_runs]
        s_ranks = ranks[s * n_runs : (s + 1) * n_runs]
        if not s_sel.any():
            continue
        per_source_rows.append([
            s,
            int(s_sel.sum()),
            round(float(top_k_score(s_ranks, s_sel, k=1)),             4),
            round(float(top_k_score(s_ranks, s_sel, k=min(3,n_nodes))),4),
            round(float(rank_score(s_ranks, s_sel, offset=0)),          4),
            round(float(s_ranks[s_sel].mean()),                          4),
        ])

    wandb.log({
        "results/per_source_table": wandb.Table(
            columns=["source_node", "n_valid",
                     "top1", f"top{min(3,n_nodes)}", "rank_score", "mean_rank"],
            data=per_source_rows,
        )
    })

    #### WANDB BN vs random baseline
    rand_top1 = 1.0 / n_nodes
    rand_topk = min(3, n_nodes) / n_nodes
    rand_rs   = 1.0 / n_nodes
    rand_mr   = (n_nodes + 1) / 2

    wandb.log({
        "results/summary_table": wandb.Table(
            columns=["method", "top1", f"top{min(3,n_nodes)}",
                     "rank_score", "mean_rank"],
            data=[
                ["Random", round(rand_top1, 4), round(rand_topk, 4),
                 round(rand_rs, 4),  round(rand_mr, 4)],
                ["BN",     round(top1, 4),       round(top3, 4),
                 round(rs, 4),       round(mr, 4)],
            ],
        )
    })

    #### WANDB histogram
    wandb.log({
        "results/rank_histogram": wandb.Histogram(
            ranks[sel].tolist(), num_bins=n_nodes
        )
    })

    return metrics


# --------------------------------------------------------------
# Main
# --------------------------------------------------------------

def main() -> None:
    cfg = parse_config()

    # All subprocess calls in tsir use paths relative to the project root.
    os.chdir(ROOT)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Seed everything for reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------
    # Wandb setup
    # --------------------------------------------------------------

    run = wandb.init(
        project = cfg.wandb_project,
        entity  = cfg.wandb_entity or None,
        config  = cfg.as_dict(),
        tags    = ["backtracking-network", "toy-example", "holme"],
    )
    print(f"\nW&B run: {run.url}\n")

    # 1. Load temporal network

    print("=" * 60)
    print("1. Temporal network")
    print("=" * 60)
    csv_path = os.path.join(ROOT, cfg.csv_path)
    G        = load_temporal_network(csv_path, directed=cfg.directed)
    n_nodes  = G.graph["n_nodes"]
    t_max    = G.graph["t_max"]

    print(f"   Source file : {cfg.csv_path}")
    print(f"   Nodes       : {n_nodes}")
    print(f"   Unique edges: {G.number_of_edges()}  (undirected graph)")
    print(f"   Time range  : 0 .. {t_max}  (T = {t_max + 1} steps)")
    print(f"   Contacts    : {sum(len(d['times']) for _,_,d in G.edges(data=True))}")

    wandb.summary["network/n_nodes"]         = n_nodes
    wandb.summary["network/n_edges"]         = G.number_of_edges()
    wandb.summary["network/t_max"]           = t_max
    wandb.summary["network/directed"]        = cfg.directed

    # Log edge-list as a W&B Table for reference
    wandb.log({
        "network/edge_table": wandb.Table(
            columns=["u", "v", "n_contacts", "active_at_times"],
            data=[
                [u, v, len(d["times"]), str(d["times"])]
                for u, v, d in G.edges(data=True)
            ],
        )
    })

    # --------------------------------------------------------------
    # 2. SIR simulations
    print("\n" + "=" * 60)
    print("2. SIR simulations")
    print("=" * 60)
    H_cread  = make_c_readable_from_networkx(G, t_max=t_max, directed=cfg.directed)
    seed_gt  = random.getrandbits(64)
    wandb.summary["sir/seed"] = seed_gt

    print(f"   Running {cfg.n_runs} runs × {n_nodes} sources "
          f"({n_nodes * cfg.n_runs} total)  …", end=" ", flush=True)
    R0, avg_os, sd, _ = tsir_run(
        H_cread,
        beta    = cfg.beta,
        mu      = cfg.mu,
        start_t = 0,
        end_t   = t_max,
        n       = cfg.n_runs,
        seed    = seed_gt,
        path    = f"{DATA_DIR}/ground_truth_{{}}.bin",
        log     = f"{DATA_DIR}/ground_truth.txt",
    )
    print(f"R0={R0:.2f},  avg outbreak size = {100 * avg_os / n_nodes:.1f}%")

    wandb.summary["sir/R0"]               = R0
    wandb.summary["sir/avg_outbreak_pct"] = avg_os / n_nodes

    # Load binary results - shape [N, N_RUNS, N] (source, run, node)
    truth_S, truth_I, truth_R = (
        np.fromfile(f"{DATA_DIR}/ground_truth_{s}.bin", dtype=np.int8)
          .reshape(n_nodes, cfg.n_runs, n_nodes)
        for s in "SIR"
    )

    # Selection mask: only evaluate on non-trivial outbreaks (≥2 I+R nodes)
    truth_S_flat = truth_S.reshape(-1, n_nodes)          # [N*N_RUNS, N]
    sel = (1 - truth_S_flat).sum(axis=1) >= 2
    print(f"   Valid outbreaks (≥2 I+R nodes): {sel.sum()} / {len(sel)}")

    wandb.summary["eval/n_total"]         = int(len(sel))
    wandb.summary["eval/n_valid"]         = int(sel.sum())

    # 3. Build BN graph inputs
    print("\n" + "=" * 60)
    print("3. BN graph inputs")
    print("=" * 60)
    edge_index, edge_attr, T = build_bn_inputs(G)
    print(f"   edge_index : {tuple(edge_index.shape)}  "
          f"({edge_index.shape[1]} directed edges)")
    print(f"   edge_attr  : {tuple(edge_attr.shape)}  "
          f"(binary activation, T={T} steps)")

    wandb.summary["bn_input/n_directed_edges"] = edge_index.shape[1]
    wandb.summary["bn_input/T"]                = T

    # 4. Training data
    print("\n" + "=" * 60)
    print("4. Training data")
    print("=" * 60)
    X, y = build_training_data(truth_S, truth_I, truth_R, n_nodes, cfg.n_runs)
    print(f"   X : {tuple(X.shape)}   (sources × runs × nodes × 3)")
    print(f"   y : {tuple(y.shape)}   ({y.unique().numel()} unique source classes)")

    # 5. BacktrackingNetwork
    print("\n" + "=" * 60)
    print("5. BacktrackingNetwork")
    print("=" * 60)
    model = BacktrackingNetwork(
        node_feat_dim = 3,
        edge_feat_dim = T,
        hidden_dim    = cfg.hidden_dim,
        num_layers    = cfg.num_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   hidden_dim={cfg.hidden_dim}  |  num_layers={cfg.num_layers}  "
          f"|  parameters={n_params:,}")
    print(f"   Device: {device}")

    wandb.summary["model/n_params"]   = n_params
    wandb.summary["model/device"]     = str(device)
    wandb.watch(model, log="gradients", log_freq=50)

    # 6. Training
    print("\n" + "=" * 60)
    print("6. Training")
    print("=" * 60)
    train_losses, val_losses = train(
        model, X, y, edge_index, edge_attr, cfg, device
    )
    wandb.summary["train/epochs_run"]    = len(train_losses)
    wandb.summary["train/best_val_loss"] = min(val_losses)

    # 7. Eval
    print("\n" + "=" * 60)
    print("7. Evaluation")
    print("=" * 60)
    metrics = evaluate(
        model, X, edge_index, edge_attr, sel, n_nodes, cfg.n_runs, device
    )

    ### PRINT SUMMARY
    k = min(3, n_nodes)
    rand_top1 = 1.0 / n_nodes
    rand_topk = k / n_nodes
    rand_rs   = 1.0 / n_nodes
    rand_mr   = (n_nodes + 1) / 2

    print(f"\n{'Method':<12}  {'Top-1':>7}  {'Top-' + str(k):>7}  "
          f"{'Rank score':>10}  {'Mean rank':>9}")
    print("-" * 52)
    print(f"{'Random':<12}  {100*rand_top1:>6.1f}%  {100*rand_topk:>6.1f}%  "
          f"{rand_rs:>10.3f}  {rand_mr:>9.2f}")
    print(f"{'BN':<12}  {100*metrics['eval/top1']:>6.1f}%  "
          f"{100*metrics['eval/top3']:>6.1f}%  "
          f"{metrics['eval/rank_score']:>10.3f}  "
          f"{metrics['eval/mean_rank']:>9.2f}")
    print(f"\nValid outbreaks: {metrics['eval/n_valid']} / {metrics['eval/n_total']}")
    print(f"W&B run: {run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()
