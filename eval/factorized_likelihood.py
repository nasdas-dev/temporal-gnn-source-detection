import numpy as np
import torch
import time
from scipy.special import logsumexp
from utils import matmul


def log_likelihood(truth_S, truth_I, truth_R, log_S, log_I, log_R, weights=1.0):
    """Compute the log-likelihood of the observed states given the log-probabilities.
    If weights=1.0, it is just the normal product of probabilities, and 1/n_nodes is geometric mean of probabilities."""
    log_S, log_I, log_R = np.fmax(log_S * weights, -1e300), np.fmax(log_I * weights, -1e300), np.fmax(log_R * weights, -1e300)
    log_lik = matmul(truth_S, log_S.T, log=False) + matmul(truth_I, log_I.T, log=False) + matmul(truth_R, log_R.T, log=False)
    #log_lik = np.dot(truth_S, log_S.T) + np.dot(truth_I, log_I.T) + np.dot(truth_R, log_R.T)
    return log_lik


def log_likelihood_torch(truth_S, truth_I, truth_R, log_S, log_I, log_R, weights=1.0):
    truth_S, truth_I, truth_R = (torch.tensor(x, dtype=torch.float64) for x in (truth_S, truth_I, truth_R))
    log_S, log_I, log_R = (torch.clamp(torch.tensor(x, dtype=torch.float64) * weights, min=-1e300) for x in (log_S, log_I, log_R))
    log_lik = torch.matmul(truth_S, log_S.T) + torch.matmul(truth_I, log_I.T) + torch.matmul(truth_R, log_R.T)
    return log_lik


def source_probabilities(log_lik):
    """Compute the source probabilities given the log-likelihood."""
    log_ns = logsumexp(log_lik, axis=1, keepdims=True) # log of normalizing sum: logsumexp trick for numerical stability
    source_probs = np.exp(log_lik - log_ns)
    return source_probs

