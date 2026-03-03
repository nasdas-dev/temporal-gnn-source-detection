import os
import yaml
import wandb
import torch
import pickle
import numpy as np
import networkx as nx
from eval import compute_ranks, top_k_score, rank_score
from gnn import StaticGNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F
from setup import setup_methods_run, load_tsir_data

if __name__ == "__main__":
    setup_methods_run(job_type="gnn")
    with open(f"{wandb.config.config_folder}/eval.yml") as f:
        eval_config = yaml.safe_load(f)
    wandb.config.update(eval_config)
    with open(f"{wandb.config.config_folder}/gnn.yml") as f:
        gnn_config = yaml.safe_load(f)
    wandb.config.update(gnn_config)

    # load data
    H, data = load_tsir_data(wandb.config.data_name)

    # make sanity checks
    n = wandb.config.n_reps["n"]
    if n > data.mc_runs:
        raise ValueError(
            f"The dataset does not contain enough Monte Carlo simulations: {data.mc_runs} available, but {n} requested.")

    # device and graph
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.backends.cudnn.benchmark = True
    G = nx.Graph() # edge_index needs correct node ordering
    G.add_nodes_from([v for v in sorted(H.nodes())])
    G.add_edges_from(H.edges(data=True))
    edge_index, weights = from_networkx(G).edge_index, None
    edge_index = edge_index.long()
    E = edge_index.size(1)

    # training data
    all_top_k = []
    all_inv_rank = []
    reps = wandb.config.n_reps["reps"]
    for i in range(reps):
        print(f" - For repetition {i+1}/{wandb.config.n_reps["reps"]}")
        select = np.random.choice(data.mc_runs, n, replace=False)
        mc_S = data.mc_S[:, select, :].reshape(n * data.n_nodes, data.n_nodes)
        mc_I = data.mc_I[:, select, :].reshape(n * data.n_nodes, data.n_nodes)
        mc_R = data.mc_R[:, select, :].reshape(n * data.n_nodes, data.n_nodes)
        TOTAL_SAMPLES = data.n_nodes * n
        indices_np = np.arange(TOTAL_SAMPLES, dtype=np.int64)
        y_np = np.repeat(np.arange(data.n_nodes, dtype=np.int64), n)

        # train test split
        tr_idx_np, va_idx_np = train_test_split(
            indices_np,
            test_size=wandb.config.gnn["test_size"],
            stratify=y_np,
            random_state=42
        )
        train_indices = torch.from_numpy(tr_idx_np).long()
        valid_indices = torch.from_numpy(va_idx_np).long()

        # train test loaders
        train_loader = DataLoader(
            train_indices,
            batch_size=wandb.config.gnn["batch_size"],
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        valid_loader = DataLoader(
            valid_indices,
            batch_size=wandb.config.gnn["batch_size"],
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )

        # prepare data
        X = torch.stack([torch.from_numpy(mc_S).float(), torch.from_numpy(mc_I).float(), torch.from_numpy(mc_R).float()], dim=-1).to(device)
        y = torch.from_numpy(y_np).long().to(device)
        num_node_features = 3

        # model
        model = StaticGNN(
            num_preprocess_layers=wandb.config.gnn["num_preprocess_layers"],
            embed_dim_preprocess=wandb.config.gnn["embed_dim_preprocess"],
            num_postprocess_layers=wandb.config.gnn["num_postprocess_layers"],
            num_conv_layers=wandb.config.gnn["num_conv_layers"],
            aggr="add",
            num_node_features=num_node_features,
            hidden_channels=wandb.config.gnn["hidden_channels"],
            num_classes=data.n_nodes,
            dropout_rate=wandb.config.gnn["dropout_rate"],
            batch_norm=wandb.config.gnn["batch_norm"],
            skip=wandb.config.gnn["skip"]
        ).to(device)

        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=wandb.config.gnn["lr"],
            weight_decay=wandb.config.gnn["weight_decay"]
        )

        # --------------------------
        # VECTORIZED EDGE BATCHING
        # --------------------------
        def make_batched_edge_index(B):
            # Offsets = [0, n_nodes, 2*n_nodes, ..., (B-1)*n_nodes]
            offsets = torch.arange(
                B, dtype=torch.long, device=edge_index.device
            ) * data.n_nodes
            offsets = offsets.repeat_interleave(E)

            be = edge_index.repeat(1, B) + offsets.unsqueeze(0)
            return be.to(device, non_blocking=True)

        # --------------------------
        # TRAIN + TEST FUNCTIONS
        # --------------------------
        def train_epoch(loader):
            model.train()
            for idx in loader:
                idx = idx.to(device, non_blocking=True)
                B = idx.size(0)

                # Build x — X stays on CPU
                x = X[idx].reshape(B * data.n_nodes, 3)
                x = x.to(device, non_blocking=True)

                # Batched edge index
                batched_edge_index = make_batched_edge_index(B)

                # Batch vector
                batch_vec = torch.arange(B, dtype=torch.long).repeat_interleave(data.n_nodes)
                batch_vec = batch_vec.to(device)

                y_batch = y[idx]

                optimizer.zero_grad()
                out = model(x, batched_edge_index, None, batch_vec)
                loss = F.nll_loss(out, y_batch)
                loss.backward()
                optimizer.step()

        def eval_epoch(loader):
            model.eval()
            correct = 0
            total_loss = 0

            with torch.no_grad():
                for idx in loader:
                    idx = idx.to(device, non_blocking=True)
                    B = idx.size(0)

                    x = X[idx].reshape(B * data.n_nodes, 3)
                    x = x.to(device, non_blocking=True)

                    batched_edge_index = make_batched_edge_index(B)

                    batch_vec = torch.arange(B, dtype=torch.long).repeat_interleave(data.n_nodes)
                    batch_vec = batch_vec.to(device)

                    y_batch = y[idx]

                    out = model(x, batched_edge_index, None, batch_vec)
                    pred = out.argmax(dim=1)

                    correct += (pred == y_batch).sum().item()
                    total_loss += F.nll_loss(out, y_batch, reduction='sum').item()

            N = len(loader.dataset)
            return correct / N, total_loss / N

        # --------------------------
        # TRAINING LOOP
        # --------------------------

        best_valid_loss = float('inf')
        best_model_state = None
        early_stop_counter = 0


        for epoch in range(1, wandb.config.gnn["n_epochs"] + 1):
            print(f"\r -- Epoch {epoch}/{wandb.config.gnn["n_epochs"]}...", end=' ', flush=True)

            train_epoch(train_loader)
            train_acc, train_loss = eval_epoch(train_loader)
            val_acc, val_loss = eval_epoch(valid_loader)

            wandb.log({f"train_loss_rep_{i+1}": train_loss, f"val_loss_rep_{i+1}": val_loss})

            if val_loss < best_valid_loss:
                early_stop_counter = 0
                best_valid_loss = val_loss
                best_model_state = model.state_dict()
            else:
                early_stop_counter += 1

            if early_stop_counter >= wandb.config.gnn["patience"]:
                print(f"Early stopping at epoch {epoch}.", end=" ")
                break


        # --------------------------
        # INFERENCE
        # --------------------------

        print("\n --- Inference on ground truth.")
        n_truth = wandb.config.n_reps["n_truth"]
        select_truth = np.arange(i * n_truth, (i + 1) * n_truth) % data.n_runs
        truth_S = data.truth_S[:, select_truth, :].reshape(-1, data.n_nodes)
        truth_I = data.truth_I[:, select_truth, :].reshape(-1, data.n_nodes)
        truth_R = data.truth_R[:, select_truth, :].reshape(-1, data.n_nodes)
        Xtruth = torch.stack([torch.from_numpy(truth_S).float(), torch.from_numpy(truth_I).float(), torch.from_numpy(truth_R).float()], dim=-1)
        lik_possible = data.lik_possible[:, select_truth, :].reshape(-1, data.n_nodes)
        sel = (1 - truth_S).sum(axis=1) >= wandb.config.eval["min_outbreak"]

        # TODO: should be possible to vectorize! iteration through each row not needed
        results = torch.zeros((n_truth*data.n_nodes, data.n_nodes), dtype=torch.float32)
        model_cpu = model.cpu()
        model_cpu.eval()
        with torch.no_grad():
            for idx in range(n_truth * data.n_nodes):
                states_current = Xtruth[idx, :, :]
                # batch of zeros: each node belongs to the same graph
                batch = torch.zeros(data.n_nodes, dtype=torch.int64)
                # forward pass
                out = model(states_current, edge_index, weights, batch)
                # move to CPU and store
                results[idx] = out
        os.makedirs(f"data/{wandb.run.id}", exist_ok=True)
        torch.save(results, f'data/{wandb.run.id}/model_outputs_rep={i+1}.pt')

        # compute scores
        gnn_log_lik = results.cpu().numpy() - lik_possible
        gnn_ranks = compute_ranks(gnn_log_lik, n_nodes=data.n_nodes, n_runs=n_truth)
        all_top_k.append([top_k_score(gnn_ranks, sel, k) for k in wandb.config.eval["top_k"]])
        all_inv_rank.append([rank_score(gnn_ranks, sel, offset) for offset in wandb.config.eval["inverse_rank_offset"]])

    # log and finish
    wandb.summary[f"static_gnn_n={n}_top_k_score"] = all_top_k
    wandb.summary[f"static_gnn_n={n}_inverse_rank"] = all_inv_rank
    wandb.finish()
