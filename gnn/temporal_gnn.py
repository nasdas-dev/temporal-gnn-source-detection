import numpy as np
import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


# TODO: try attention-based convs like GATConv
# TODO: try more constrained layer like GCNConv (Kipf and Welling, fixed normalization, fewer parameters)

class TemporalGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_snapshots):
        super().__init__()
        self.lin_pre = torch.nn.Linear(in_channels, hidden_channels)
        self.convs = torch.nn.ModuleList([SAGEConv(hidden_channels, hidden_channels)
                                         for _ in range(num_snapshots)])
        self.lin_post = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_indeces, edge_attr=None):
        # TODO: add dropout
        x = F.relu(self.lin_pre(x))
        count = 0
        for t in reversed(edge_indeces.keys()):
            x = F.relu(self.convs[count](x, edge_indeces[t], edge_attr))
            count += 1
        x = self.lin_post(x)
        return F.log_softmax(x, dim=1)


def temporal_edge_indeces(H_array, start_t, end_t, group_by_time, directed=False):
    edge_indeces = {}
    H = H_array.copy()
    H = H[H[:, 2] >= start_t]
    H = H[H[:, 2] <= end_t]
    min_t = H[:, 2].min()
    H[:, 2] = (H[:, 2] - min_t) // group_by_time
    row = []
    current_t = start_t
    for u, v, t in H:
        if t > current_t:
            if len(row) > 0:
                edge_indeces[current_t] = torch.tensor(row, dtype=torch.long).T
                row = []
            current_t = int(t)
        row.append((u, v))
        if not directed:
            row.append((v, u))
    if len(row) > 0: # append last one as well
        edge_indeces[current_t] = torch.tensor(row, dtype=torch.long).T
    return edge_indeces


def temporal_gnn(cfg, H_array, n_nodes, folder, mc_S, mc_I, mc_R, truth_S, truth_I, truth_R):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTemporal GNN with device={device}")
    start = time.time()

    # setup
    X = torch.tensor(np.stack([mc_S, mc_I, mc_R], axis=-1), dtype=torch.float).to(device)
    y = torch.tensor(np.repeat(np.arange(n_nodes), cfg.sir.mc_runs)).to(device)
    # TODO: group_by_time is a very clumsy way to do it, refine that
    edge_indeces = temporal_edge_indeces(H_array, start_t=cfg.sir.start_t, end_t=cfg.sir.end_t,
                                         group_by_time=cfg.mthd.tgnn.group_by_time, directed=cfg.nwk.directed)
    edge_indeces = {t: edge_indeces[t].to(device) for t in edge_indeces.keys()}

    # model and optimizer
    model = TemporalGNN(in_channels=3, hidden_channels=cfg.gnn.hidden_channels, out_channels=1, num_snapshots=len(edge_indeces))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.gnn.lr, weight_decay=cfg.gnn.weight_decay)
    torch.manual_seed(cfg.gnn.cpu_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.gnn.gpu_seed)

    # create DataLoaders
    idx_pairs = [(i, j) for i in range(n_nodes) for j in range(cfg.sir.mc_runs)]
    train_idx, test_idx = train_test_split(idx_pairs, test_size=cfg.gnn.test_size, stratify=y.cpu().numpy(), random_state=cfg.gnn.cpu_seed)
    train_indices = TensorDataset(torch.tensor(train_idx, dtype=torch.long))
    test_indices = TensorDataset(torch.tensor(test_idx, dtype=torch.long))
    train_loader = DataLoader(train_indices, batch_size=cfg.gnn.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_indices, batch_size=cfg.gnn.batch_size, shuffle=False, num_workers=4)
    # TODO: test data shouldn't have the same batch size as training data
    # TODO: num_workers should be config parameters as well

    # training loop
    train_losses = []
    test_losses = []
    min_test_loss = float('inf')
    early_stopping_counter = 0
    for epoch in range(cfg.gnn.epochs):

        # --- Training ---
        model.train()
        train_loss = 0
        for (batch_idx_pairs,) in train_loader:
            batch_idx_pairs = batch_idx_pairs.to(device)
            i_s = batch_idx_pairs[:, 0] # which source nodes
            j_s = batch_idx_pairs[:, 1] # which monte carlo runs
            x_batch = X[i_s, j_s]  # shape: [batch_size, n_nodes, 3]
            y_batch = y[i_s * cfg.sir.mc_runs + j_s]  # shape: [batch_size]
            optimizer.zero_grad()
            out = model(x_batch, edge_indeces, None)
            loss = F.nll_loss(out.squeeze(2), y_batch, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for (batch_idx_pairs,) in test_loader:
                batch_idx_pairs = batch_idx_pairs.to(device)
                i_s = batch_idx_pairs[:, 0]
                j_s = batch_idx_pairs[:, 1]
                x_test_batch = X[i_s, j_s]
                y_test_batch = y[i_s * cfg.sir.mc_runs + j_s]
                test_out = model(x_test_batch, edge_indeces, None)  # edge_attr=None
                test_loss += F.nll_loss(test_out.squeeze(2), y_test_batch, reduction='sum').item()

        # --- Logging and Early Stopping ---
        avg_train_loss = train_loss / len(train_idx)
        avg_test_loss = test_loss / len(test_idx)
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f" --- Epoch {epoch + 1:03d}/{cfg.gnn.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f}")
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= cfg.gnn.early_stop:
                print(f" --- Early stopping at epoch {epoch + 1} ---")
                break

    print(f' --- Training done in {time.time() - start:.2f} seconds')

    # final prediction
    print(f" --- Start predicting GNN...", end="\r")
    start = time.time()

    model.eval()
    truth = torch.tensor(np.stack([truth_S, truth_I, truth_R], axis=-1), dtype=torch.float).to(device)
    with torch.no_grad():
        preds = model(truth, edge_indeces, None)  # edge_attr=None
    preds = torch.exp(preds)
    source_probs = preds.squeeze(2).detach().cpu().numpy()

    print(f" --- Start predicting GNN... Done in {time.time() - start:.1f} seconds.")
    return source_probs