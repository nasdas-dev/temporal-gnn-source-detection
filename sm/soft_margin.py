import numpy as np
import torch
import time


def soft_margin_numpy(jaccard, a):
    soft_marg = np.exp(-(jaccard - 1)**2 / a**2)
    probs = np.mean(soft_marg, axis=1)
    return probs.T

def soft_margin_torch(jaccard, a):
    soft_marg = torch.exp(-((jaccard - 1) ** 2) / a ** 2)
    probs = soft_marg.mean(dim=1)
    return probs.T.cpu().numpy()


def jaccard_similarity(mc_S, truth_S, n_nodes, device):
    if device == torch.device("cuda"):
        jaccard = jaccard_similarity_torch(mc_S, truth_S, n_nodes, device=device)
    else:
        jaccard = jaccard_similarity_numpy(mc_S, truth_S, n_nodes)
    return jaccard

def jaccard_similarity_numpy(mc_S, truth_S, n_nodes):
    # convert, otherwise matrix multiplication produces integer overflow against int8: -128 to 127
    simulation = mc_S.astype(np.int32)
    truth = truth_S.astype(np.int32)
    jaccard_nom = np.dot(1 - simulation, 1 - truth.T)
    jaccard_denom = n_nodes - np.dot(simulation, truth.T)
    jaccard = jaccard_nom / jaccard_denom
    return jaccard

def jaccard_similarity_torch(mc_S, truth_S, n_nodes, device, vram_max_gb=2):
    simulation = torch.tensor(mc_S, dtype=torch.float32, device=device)
    truth = torch.tensor(truth_S, dtype=torch.float32, device=device)

    # approximated size of matrix product: simulation size in GB, but multiplied with n_runs (truth_S.shape[0] = n_runs*n_nodes)
    size = simulation.element_size() * simulation.numel() / 1e9 * truth_S.shape[0] / n_nodes
    nr_batches = int(np.ceil(size / vram_max_gb)) # computing number of batches based on size

    jaccard = np.empty((n_nodes, simulation.shape[1], truth_S.shape[0]), dtype=np.float32)
    batch_size = int(np.ceil(simulation.shape[1] / nr_batches))
    for i in range(0, nr_batches):
        print(f"\r -- Processing batch {i + 1}/{nr_batches}...", end=' ', flush=True)
        sim_batch = simulation[:, (i * batch_size):((i+1) * batch_size), :]
        nom = ((1 - sim_batch) @ (1 - truth).T).cpu().numpy()
        denom = (n_nodes - (sim_batch @ truth.T)).cpu().numpy()
        jaccard[:, (i * batch_size):((i+1) * batch_size), :] = nom / denom
    print() # newline
    return jaccard
