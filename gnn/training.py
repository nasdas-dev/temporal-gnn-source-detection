import torch.nn.functional as F
import random
import time
import torch
from sklearn.model_selection import train_test_split


def train_batches(n_nodes, n_runs, X, y, model, optimizer, edge_index, edge_attr,
                  batch_size, epochs, test_size, early_stop):
    idx_pairs = [(i, j) for i in range(n_nodes) for j in range(n_runs)]
    train_idx, test_idx = train_test_split(idx_pairs, test_size=test_size)
    train_losses = []
    test_losses = []
    min_test_loss = float('inf')
    early_stopping_counter = 0
    start = time.time()

    for epoch in range(epochs):
        # batch training
        random.shuffle(train_idx)
        train_loss = 0
        for b in range(0, len(train_idx), batch_size):
            batch = train_idx[b:b + batch_size]
            x_batch = torch.stack([X[i, j] for i, j in batch])
            y_batch = torch.tensor([y[i * n_runs + j] for i, j in batch])
            optimizer.zero_grad()
            out = model(x_batch, edge_index, edge_attr)
            loss = F.nll_loss(out.squeeze(-1), y_batch, reduction='sum')
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # validation loss
        with torch.no_grad():
            x_test = torch.stack([X[i, j] for i, j in test_idx])
            y_test = torch.tensor([y[i * n_runs + j] for i, j in test_idx])
            test_out = model(x_test, edge_index, edge_attr)
            test_loss = F.nll_loss(test_out.squeeze(-1), y_test, reduction='sum').item()
        train_losses.append(train_loss / len(train_idx))
        test_losses.append(test_loss / len(test_idx))
        print(f" --- Epoch {epoch + 1}/{epochs} | train_loss={train_losses[-1]:.3f} | test_loss={test_losses[-1]:.3f}")

        # early stopping
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stop:
                print(f" --- Early stopping at epoch {epoch + 1}")
                break

    print(f' --- Training done in {time.time() - start:.2f} seconds')
    return train_losses, test_losses

