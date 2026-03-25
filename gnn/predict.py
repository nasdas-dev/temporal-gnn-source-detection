import torch
import time


def predict_batches(n_nodes, n_runs, model, truth, edge_index, edge_attr):
    start = time.time()
    print(f" --- Start predicting GNN...", end="\r")
    preds = model(truth, edge_index, edge_attr)
    preds = preds.squeeze(2)
    print(f" --- Start predicting GNN... Done in {time.time() - start:.1f} seconds.")
    return torch.exp(preds)

