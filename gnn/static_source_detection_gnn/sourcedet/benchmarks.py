# --------------------------------------------------------------------------
# Benchmarks (or baselines) for GNNs
# --------------------------------------------------------------------------

import torch
import numpy as np
import networkx as nx

def jordan_center(G, infected):
    # G: NetworkX graph
    # infected: list of infected nodes
    # ======================================================
    # Initialize a torch tensor with neg. infinity values.
    # This makes sure the Jordan center method gives the same output as GNN.
    out = torch.ones(1, G.number_of_nodes()) * -float('inf')
    # Get infected subgraph.
    subg = G.subgraph(infected)
    # Get shortest path lengths (spl) among nodes in infected subgraph.
    spl = dict(nx.all_pairs_shortest_path_length(subg))
    # Get max. shortest path length for each node.
    max_spl = {key: max(value.values()) for key, value in spl.items()}
    # Insert the negative max. shortest path length for all nodes in the
    # infected subgraph. The neg. max. spl will make sure that argmax gives
    # back the node with the smallest max. spl.
    out[:,list(max_spl.keys())] = torch.FloatTensor(list(max_spl.values())) * (-1.0)
    # Return out.
    return out

def degree_center(G, infected):
    # G: NetworkX graph
    # infected: list of infected nodes
    # ======================================================
    # Initialize a torch tensor with neg. infinity values.
    # This makes sure the degree centrality method gives the same output as GNN.
    out = torch.ones(1, G.number_of_nodes()) * -float('inf')
    # Get infected subgraph.
    subg = G.subgraph(infected)
    # Get shortest path lengths (spl) among nodes in infected subgraph.
    degrees = dict(subg.degree())
    # Insert the degree values in out.
    out[:,list(degrees.keys())] = torch.FloatTensor(list(degrees.values()))
    # Return out.
    return out

def betweenness_center(G, infected):
    # G: NetworkX graph
    # infected: list of infected nodes
    # ======================================================
    # Initialize a torch tensor with neg. infinity values.
    # This makes sure the betweenness centrality method gives the same output as GNN.
    out = torch.ones(1, G.number_of_nodes()) * -float('inf')
    # Get infected subgraph.
    subg = G.subgraph(infected)
    # Get shortest path lengths (spl) among nodes in infected subgraph.
    btw_centralities = dict(nx.betweenness_centrality(subg))
    # Insert the degree values in out.
    out[:,list(btw_centralities.keys())] = torch.FloatTensor(list(btw_centralities.values()))
    # Return out.
    return out

def closeness_center(G, infected):
    # G: NetworkX graph
    # infected: list of infected nodes
    # ======================================================
    # Initialize a torch tensor with neg. infinity values.
    # This makes sure the closeness centrality method gives the same output as GNN.
    out = torch.ones(1, G.number_of_nodes()) * -float('inf')
    # Get infected subgraph.
    subg = G.subgraph(infected)
    # Get shortest path lengths (spl) among nodes in infected subgraph.
    cln_centralities = dict(nx.closeness_centrality(subg))
    # Insert the degree values in out.
    out[:,list(cln_centralities.keys())] = torch.FloatTensor(list(cln_centralities.values()))
    # Return out.
    return out

def soft_margin(X_mask, X_mask_sum, states_onehot, a_sq):
    # X_mask: tensor with training data that is already transformed as follows: 
    # X.select(2, 0).eq(0).to(torch.int32)
    # As this is the same for every outbreak, taking this out of the function leads to a massive speedup.
    # X_mask_sum: similarly, taking the computation "m1_sum = mask1.sum(dim=1)" out of the function leads to a speedup.
    # states_onehot: states of nodes one-hot encoded (test outbreak)
    # a_sq: convergence parameter a-squared (= a^2), it expects a 1d tensor (even if it is just one parameter value)
    # ======================================================
    # First store number of training examples and number of nodes.
    n_sim, n_nodes = X_mask.shape
    # Here we compute the number of training examples per seed node.
    n_per_node = int(n_sim / n_nodes)
    # Here we get a tensor containing integers (0/1) that indicate infected and not infected nodes.
    mask2 = states_onehot.select(1, 0).eq(0).to(torch.int32)
    # The intersection is conveniently the inner product of the matrix and the vector.
    inter = X_mask @ mask2
    # For the union, we first compute the sum of the two masks row-wise.
    m2_sum = mask2.sum()
    # The union is then just the sum of the two sums minus the intersection.
    union = X_mask_sum + m2_sum - inter
    # Jaccard index.
    out = inter / union
    # The following few lines implement equation (3) in Antulov-Fantulin et al.
    # paper. The part "/ a_sq[:, None]" makes sure that one can
    # compute equation (3) for several a^2 values in one go.
    out = torch.exp(-((out - 1).square())[None, :] / a_sq[:, None])
    # Now we reshape out so that the last dimension batches the simulations per source node.
    out = out.view(out.shape[0], n_nodes, n_per_node)
    # This allows us to conveniently sum over that last dimension to compute the likelihoods.
    output_n = out.sum(dim=2) / n_per_node
    # In order to check convergence, we now sample 2n scenarios per source node by sampling
    # with replacement. This is in line with Antulov-Fantulin et al.'s proposed procedure
    # in Section 5 of the SI. For this, we first create the sampled indices.
    indices = torch.randint(0, n_per_node, (2 * n_per_node,), device=out.device)
    # Now we sample the outbreaks and again compute the likelihood.
    output_2n = out[:, :, indices].sum(dim=2) / (2 * n_per_node)
    # Keep only rows for which all nodes have still non-zero likelihoods.
    mask = (output_n > 0).all(dim=1)
    output_n = output_n[mask]
    output_2n = output_2n[mask]
    # To check convergence, we need the normalized probability distributions.
    # First, we compute row sums.
    row_sums = output_2n.sum(dim=1, keepdim=True)
    # Normalize
    output_2n_prob = output_2n / (row_sums + 1e-8)
    # Same for output_n
    row_sums = output_n.sum(dim=1, keepdim=True)
    output_n_prob = output_n / (row_sums + 1e-8)
    # Get the MAP nodes for each a^2 value in output_2n.
    map_nodes = torch.argmax(output_2n_prob, dim=1)
    # Get the probabilities of MAP nodes.
    a1 = output_n_prob.gather(1, map_nodes.unsqueeze(1)).squeeze(1)
    a2 = output_2n_prob.gather(1, map_nodes.unsqueeze(1)).squeeze(1)
    # Find which a^2 values lead to convergence.
    idxs = torch.nonzero(torch.abs(a1 - a2) <= 0.05, as_tuple=False).squeeze()
    # Choose last index that still leads to convergence.
    last_idx = idxs[-1].item() if idxs.numel() > 1 else 0
    # Return log of the likelihood for last a^2 values that converges.
    return torch.log(output_n[last_idx,...].unsqueeze(0))

def mcs_mean_field(node_state_prob, states_onehot):
    # node_state_prob: tensor with training data
    # states_onehot: states of nodes one-hot encoded
    # ======================================================
    # First store number of nodes.
    n_nodes, _, _ = node_state_prob.shape
    # This creates a mask to read out the node state prob. of the
    # observed states.
    mask = states_onehot.reshape((1, n_nodes, 3)).expand(n_nodes, n_nodes, 3)
    # Now we select according to the mask and reshape properly.
    out = torch.masked_select(node_state_prob, (mask == 1)).reshape((n_nodes, n_nodes))
    # Finally, we sum up the log-prob across columns to get
    # the log-likelihoods for all seed nodes.
    out = torch.log(out).sum(dim=1)
    # Return out but unsqueeze such that it is a row vector.
    return out.unsqueeze(0)

def rozenshtein(G, infected):
    # G: NetworkX graph
    # infected: list of infected nodes
    # ======================================================
    # Return out.
    #return out
    raise NotImplementedError
