from typing import Optional
import numpy as np
import networkx as nx


def average_rank(possible):
    average_rank = (np.sum(possible, axis=1) + 1) / 2
    return average_rank

def sampled_rank(possible):
    row_sums = np.sum(possible, axis=1)
    return np.array([np.random.randint(1, s + 1) for s in row_sums])

def uniform_probabilities(possible):
    return possible / np.sum(possible, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Internal helper: discrete-time SIR on a static graph
# ---------------------------------------------------------------------------

def _sir_simulation(
    adj: np.ndarray,
    source: int,
    beta: float,
    mu: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one discrete-time SIR simulation on the static graph.

    Parameters
    ----------
    adj : np.ndarray, shape (n_nodes, n_nodes)
        Binary adjacency matrix.
    source : int
        Index of the seed node.
    beta : float
        Per-edge transmission probability per step.
    mu : float
        Recovery probability per step.
    n_steps : int
        Number of time steps to simulate.

    Returns
    -------
    S, I, R : np.ndarray, shape (n_nodes,), dtype int8
        Terminal SIR states (1 = in that compartment, 0 = not).
    """
    n = adj.shape[0]
    state = np.zeros(n, dtype=np.int8)  # 0=S, 1=I, 2=R
    state[source] = 1

    for _ in range(n_steps):
        new_infected = np.zeros(n, dtype=bool)
        infected_mask = state == 1
        if not infected_mask.any():
            break
        # Each susceptible node i gets infected if any infected neighbour
        # transmits with probability beta
        susceptible_mask = state == 0
        for i in np.where(susceptible_mask)[0]:
            n_infected_neighbours = int(adj[infected_mask, i].sum())
            if n_infected_neighbours > 0:
                p_escape = (1.0 - beta) ** n_infected_neighbours
                if np.random.random() > p_escape:
                    new_infected[i] = True
        # Recovery
        recovered_now = infected_mask & (np.random.random(n) < mu)
        state[new_infected] = 1
        state[recovered_now] = 2

    S = (state == 0).astype(np.int8)
    I = (state == 1).astype(np.int8)
    R = (state == 2).astype(np.int8)
    return S, I, R


# Ported from gnn/static_source_detection_gnn/sourcedet/benchmarks.py
def soft_margin(
    H_static: nx.Graph,
    truth_S: np.ndarray,
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    possible: np.ndarray,
    beta: float = 0.1,
    mu: float = 0.05,
    n_steps: int = 50,
    n_mc: int = 100,
    a_sq_candidates: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soft-margin estimator for epidemic source detection (Antulov-Fantulin et al.).

    For each candidate source node, Monte Carlo SIR simulations are run on
    ``H_static`` and the resulting infection masks are compared to the
    observed infection mask via Jaccard similarity.  The Gaussian soft-margin
    kernel (Eq. 3 of Antulov-Fantulin et al.) converts Jaccard similarities
    to likelihoods, with the convergence-based selection of the bandwidth
    parameter ``a^2`` as described in Section 5 of their SI.

    Parameters
    ----------
    H_static : nx.Graph
        Static (undirected) projection of the temporal network.
    truth_S : np.ndarray, shape (n_nodes,), dtype int8
        Observed susceptible indicator vector (1 = susceptible, 0 = not).
    truth_I : np.ndarray, shape (n_nodes,), dtype int8
        Observed infected indicator vector.
    truth_R : np.ndarray, shape (n_nodes,), dtype int8
        Observed recovered indicator vector.
    possible : np.ndarray, shape (n_nodes,), dtype int8
        Binary mask of candidate source nodes.
    beta : float
        Per-edge transmission probability per step used for MC simulations.
    mu : float
        Recovery probability per step used for MC simulations.
    n_steps : int
        Number of simulation steps per MC run.
    n_mc : int
        Number of MC simulations per candidate source.
    a_sq_candidates : np.ndarray or None
        1-D array of bandwidth candidates ``a^2``.  Defaults to a log-spaced
        grid over [0.01, 10.0] with 30 values.

    Returns
    -------
    probs : np.ndarray, shape (n_nodes,)
        Normalised probability distribution over nodes (zero for impossible
        sources).
    """
    n_nodes = H_static.number_of_nodes()
    nodes = sorted(H_static.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    adj = nx.to_numpy_array(H_static, nodelist=nodes, weight=None)

    if a_sq_candidates is None:
        a_sq_candidates = np.logspace(-2, 1, 30)

    # Observed infection mask: a node is "infected" if not susceptible
    obs_mask = (truth_S == 0).astype(np.float64)  # shape (n_nodes,)
    obs_mask_sum = obs_mask.sum()

    candidate_indices = [i for i in range(n_nodes) if possible[i] == 1]
    if not candidate_indices:
        return np.ones(n_nodes, dtype=np.float32) / n_nodes

    # For each candidate, run n_mc simulations and collect infection masks.
    # sim_masks[s] has shape (n_mc, n_nodes): 1 if node is not susceptible
    sim_masks: dict[int, np.ndarray] = {}
    for s_idx in candidate_indices:
        masks = np.zeros((n_mc, n_nodes), dtype=np.float64)
        for r in range(n_mc):
            S_r, I_r, R_r = _sir_simulation(adj, s_idx, beta, mu, n_steps)
            masks[r] = (S_r == 0).astype(np.float64)
        sim_masks[s_idx] = masks

    # Jaccard similarities: for each a^2 candidate, compute likelihood per source
    # Following the subpackage logic: for source s,
    #   inter_r = masks_s[r] · obs_mask
    #   union_r = masks_s[r].sum() + obs_mask_sum - inter_r
    #   jaccard_r = inter_r / union_r
    #   out_s_r(a^2) = exp(-((jaccard_r - 1)^2) / a^2)
    #   likelihood_s(a^2) = mean over r of out_s_r(a^2)

    n_a = len(a_sq_candidates)
    # output_n[a_idx, s_idx_in_candidates] = likelihood from n_mc sims
    n_cand = len(candidate_indices)
    output_n = np.zeros((n_a, n_cand), dtype=np.float64)

    for ci, s_idx in enumerate(candidate_indices):
        masks = sim_masks[s_idx]  # (n_mc, n_nodes)
        inter = masks @ obs_mask  # (n_mc,)
        sim_sums = masks.sum(axis=1)  # (n_mc,)
        union = sim_sums + obs_mask_sum - inter
        # Avoid division by zero
        union = np.where(union > 0, union, 1.0)
        jaccard = inter / union  # (n_mc,)
        # (n_a, n_mc): Gaussian soft-margin kernel for each a^2
        out = np.exp(-((jaccard - 1.0) ** 2)[None, :] / a_sq_candidates[:, None])
        output_n[:, ci] = out.mean(axis=1)

    # Convergence check: resample 2*n_mc scenarios with replacement
    # and find the last a^2 that still leads to convergence (|p_n - p_2n| <= 0.05)
    rng_indices = np.random.randint(0, n_mc, size=2 * n_mc)
    output_2n = np.zeros((n_a, n_cand), dtype=np.float64)
    for ci, s_idx in enumerate(candidate_indices):
        masks = sim_masks[s_idx]
        masks_2n = masks[rng_indices]
        inter = masks_2n @ obs_mask
        sim_sums = masks_2n.sum(axis=1)
        union = sim_sums + obs_mask_sum - inter
        union = np.where(union > 0, union, 1.0)
        jaccard = inter / union
        out = np.exp(-((jaccard - 1.0) ** 2)[None, :] / a_sq_candidates[:, None])
        output_2n[:, ci] = out.mean(axis=1)

    # Normalise rows to probability distributions
    n_sum = output_n.sum(axis=1, keepdims=True)
    n_sum = np.where(n_sum > 0, n_sum, 1.0)
    prob_n = output_n / n_sum

    n2_sum = output_2n.sum(axis=1, keepdims=True)
    n2_sum = np.where(n2_sum > 0, n2_sum, 1.0)
    prob_2n = output_2n / n2_sum

    # Keep only a^2 values where all candidates have non-zero likelihood
    valid_a = np.where((output_n > 0).all(axis=1))[0]
    if len(valid_a) == 0:
        # Fallback: just use the last a^2 regardless
        chosen_a = n_a - 1
    else:
        # MAP node for each a^2 based on prob_2n
        map_nodes = np.argmax(prob_2n, axis=1)  # (n_a,)
        a1 = prob_n[np.arange(n_a), map_nodes]
        a2 = prob_2n[np.arange(n_a), map_nodes]
        converged = np.abs(a1 - a2) <= 0.05
        converged_valid = np.where(converged[valid_a])[0]
        if len(converged_valid) == 0:
            chosen_a = int(valid_a[-1])
        else:
            chosen_a = int(valid_a[converged_valid[-1]])

    # Build final probability vector over all n_nodes
    log_likelihoods_cand = np.log(output_n[chosen_a] + 1e-300)  # (n_cand,)

    probs = np.full(n_nodes, -np.inf, dtype=np.float64)
    for ci, s_idx in enumerate(candidate_indices):
        probs[s_idx] = log_likelihoods_cand[ci]

    # Convert log-likelihoods to probabilities via softmax over candidates
    finite_mask = np.isfinite(probs)
    if finite_mask.any():
        log_max = probs[finite_mask].max()
        exp_probs = np.zeros(n_nodes, dtype=np.float64)
        exp_probs[finite_mask] = np.exp(probs[finite_mask] - log_max)
        total = exp_probs.sum()
        if total > 0:
            return (exp_probs / total).astype(np.float32)

    # Fallback: uniform over possible
    u = possible.astype(np.float64)
    s = u.sum()
    return (u / s if s > 0 else np.ones(n_nodes) / n_nodes).astype(np.float32)


# Ported from gnn/static_source_detection_gnn/sourcedet/benchmarks.py
def mcs_mean_field(
    H_static: nx.Graph,
    truth_S: np.ndarray,
    truth_I: np.ndarray,
    truth_R: np.ndarray,
    possible: np.ndarray,
    beta: float = 0.1,
    mu: float = 0.05,
    n_steps: int = 50,
    n_mc: int = 100,
) -> np.ndarray:
    """Monte Carlo simulation mean-field baseline (Sterchi et al.).

    For each candidate source node, ``n_mc`` SIR simulations are run on
    ``H_static`` to estimate the marginal probability that each node ends up
    in each SIR compartment.  The log-likelihood of the observed node states
    under this mean-field approximation is then summed across nodes to produce
    a score for each candidate source.

    Parameters
    ----------
    H_static : nx.Graph
        Static (undirected) projection of the temporal network.
    truth_S : np.ndarray, shape (n_nodes,), dtype int8
        Observed susceptible indicator vector (1 = susceptible, 0 = not).
    truth_I : np.ndarray, shape (n_nodes,), dtype int8
        Observed infected indicator vector.
    truth_R : np.ndarray, shape (n_nodes,), dtype int8
        Observed recovered indicator vector.
    possible : np.ndarray, shape (n_nodes,), dtype int8
        Binary mask of candidate source nodes.
    beta : float
        Per-edge transmission probability per step used for MC simulations.
    mu : float
        Recovery probability per step used for MC simulations.
    n_steps : int
        Number of simulation steps per MC run.
    n_mc : int
        Number of MC simulations per candidate source.

    Returns
    -------
    probs : np.ndarray, shape (n_nodes,)
        Normalised probability distribution over nodes (zero for impossible
        sources).
    """
    n_nodes = H_static.number_of_nodes()
    nodes = sorted(H_static.nodes())

    adj = nx.to_numpy_array(H_static, nodelist=nodes, weight=None)

    candidate_indices = [i for i in range(n_nodes) if possible[i] == 1]
    if not candidate_indices:
        return np.ones(n_nodes, dtype=np.float32) / n_nodes

    # Observed state as integer: 0=S, 1=I, 2=R per node
    obs_state = np.zeros(n_nodes, dtype=int)
    obs_state[truth_I == 1] = 1
    obs_state[truth_R == 1] = 2

    log_likelihoods = np.full(n_nodes, -np.inf, dtype=np.float64)

    for s_idx in candidate_indices:
        # Accumulate state counts: node_state_count[node, state] over n_mc runs
        state_counts = np.zeros((n_nodes, 3), dtype=np.float64)
        for _ in range(n_mc):
            S_r, I_r, R_r = _sir_simulation(adj, s_idx, beta, mu, n_steps)
            state_counts[:, 0] += S_r
            state_counts[:, 1] += I_r
            state_counts[:, 2] += R_r

        # Mean-field probabilities: P(node v in state k | source = s)
        # node_state_prob[v, k] = fraction of runs where v ended in state k
        node_state_prob = state_counts / n_mc  # (n_nodes, 3)

        # Following the subpackage logic:
        # For the observed state vector, select the probability of each node's
        # observed state, then sum log-probabilities across nodes.
        # node_state_prob[v, obs_state[v]] gives the model probability for
        # the observed state of node v.
        p_obs = node_state_prob[np.arange(n_nodes), obs_state]  # (n_nodes,)
        # Clip to avoid log(0)
        p_obs = np.clip(p_obs, 1e-12, 1.0)
        log_likelihoods[s_idx] = np.sum(np.log(p_obs))

    # Convert log-likelihoods to probabilities via softmax over candidates
    finite_mask = np.isfinite(log_likelihoods)
    if finite_mask.any():
        log_max = log_likelihoods[finite_mask].max()
        exp_ll = np.zeros(n_nodes, dtype=np.float64)
        exp_ll[finite_mask] = np.exp(log_likelihoods[finite_mask] - log_max)
        total = exp_ll.sum()
        if total > 0:
            return (exp_ll / total).astype(np.float32)

    # Fallback: uniform over possible
    u = possible.astype(np.float64)
    s = u.sum()
    return (u / s if s > 0 else np.ones(n_nodes) / n_nodes).astype(np.float32)

