import time
import numpy as np

def monte_carlo(mc_S, mc_I, mc_R, mc_runs, n_nodes, maximal_outbreak):
    """Compute the log-probabilities from Monte Carlo simulations with add-one smoothing. Add-one smoothing makes sure
    that no real possibility has a probability of exactly zero."""
    mc_log_S, mc_log_I, mc_log_R = (np.log((mc.sum(axis=1) + add_one) / (mc_runs + 1))  # add-one smoothing (warning due to np.log(0))
                                    for mc, add_one in zip((mc_S, mc_I, mc_R), (1 - np.eye(n_nodes), maximal_outbreak, maximal_outbreak)))
    return mc_log_S, mc_log_I, mc_log_R


def monte_carlo_exclude(mc_S, mc_I, mc_R, mc_runs, n_nodes, maximal_outbreak, exclude):
    """Almost the same as the function above, but exclude those runs that did not outbreak (outbreak size <= exclude)."""
    mask = (np.sum(1 - mc_S, axis=2) > exclude)[:, :, None]
    den = mask.sum(axis=1)
    mc_log_S = np.log(((mc_S * mask).sum(axis=1) + (1 - np.eye(n_nodes))) / (den + 1))
    mc_log_I = np.log(((mc_I * mask).sum(axis=1) + maximal_outbreak) / (den + 1))
    mc_log_R = np.log(((mc_R * mask).sum(axis=1) + maximal_outbreak) / (den + 1))
    return mc_log_S, mc_log_I, mc_log_R, np.log(den / mc_runs).reshape(1, n_nodes) # last return is the correct factor
    # correct factor is the log of the probability that outbreak size exceeds exclude_cutoff