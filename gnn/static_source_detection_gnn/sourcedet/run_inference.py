# Import libraries
import os
import sys
import yaml
import time
import torch
import numpy as np
import networkx as nx
from sys import argv
from pathlib import Path
from subprocess import check_output
from random import getrandbits, choice

from .nwk import import_nwk, create_nwk
from .model import GNN
from .evaluate import update, print_results, top_k
from .benchmarks import jordan_center, degree_center, betweenness_center, closeness_center, soft_margin, mcs_mean_field
from .utils import one_hot, log_sum_exp

from scipy.sparse.linalg import eigsh
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F

# Number of test simulations per source node.
SIM_PER_SEED = 100

# This makes sure that we always use the same test simulation runs.
# At least for the same BETA and T values.
RANDOM_SEED = 4253219522064423221

# =========================================================
# INFERENCE ROUTINE ---------------------------------------
# =========================================================

if __name__ == "__main__":

    # Check that number of arguments is correct.
	if len(argv) != 2:
		print('usage: python3 run_inference.py [data directory]')
		exit()

	# Path to data directory.
	dirname = 'data/' + argv[1]

	# Create a subdirectory for inference results.
	os.makedirs(dirname + '/inference')

	# Load parameters from YAML file.
	with open(dirname + "/config.yaml", "r") as fp:
		p = yaml.safe_load(fp)

	# Select device.
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f"Using device: {device}")
	# Allows inbuilt cudnn auto-tuner to find best config for hardware.
	torch.backends.cudnn.benchmark = True

	# =========================================================
	# INPUT GRAPH ---------------------------------------------
	# =========================================================

	# Load edgelist into numpy array and then convert the array to list of tuples.
	edges = np.genfromtxt(dirname + '/graph.csv', delimiter=' ', dtype=np.int64)
	edges = list(map(tuple, edges))

	# Get number of nodes.
	n_nodes = p["GRAPH_SIZE"]

	# Create empty graph.
	G = nx.Graph()

	# Add nodes according to n_nodes.
	# Why so complicated? If any node is an isolate, it won't be in
	# edgelist that we import below.
	G.add_nodes_from([v for v in range(n_nodes)])

	# Add edges to graph.
	G.add_edges_from([(e[0], e[1]) for e in edges])

	# Adjacency info in the format that PyG model expects.
	edge_index = from_networkx(G).edge_index

	# Import edge weights (or set them to None).
	weights = torch.load(dirname + "/edge_weights.pt") if p["EDGE_WEIGHTS"] else None

	# =========================================================
	# TEST DATA -----------------------------------------------
	# =========================================================

	# Simulate outbreaks (call to C code in /sir):
	# This simulates a continuous-time SIR model with exponentially distributed
	# times until infection (per link) and exponentially distributed time until
	# recovery. The recovery rate is by default set to 1 and we only need to
	# provide the infection rate beta. For a different recovery rate nu, we can
	# just provide a modified beta (beta = intended beta / nu) and T
	# (T = intended T / nu). For example, with beta=2 and nu=0.5, we provide
	# the C code the value beta = 2/0.5 = 4.
	# IMPORTANT: we purposefully do NOT take the SIM_PER_SEED out of the dict p!
	out = check_output(['sir/sir', dirname + '/graph.csv', str(p["BETA"] / p["NU"]), str(p["T"] / p["NU"]), str(0), str(SIM_PER_SEED), str(RANDOM_SEED), dirname + '/inference'])

	# Decode the C output (it's now only the avg. outbreaks size and R0).
	a = out.decode().split('\n')

	# Print avg. outbreak size and R0 to console.
	# The average outbreak size is given in the first list element.
	# The average R0 is given in the second list element.
	# The last list element is always an empty string ('').
	print('============================')
	print(a[0])
	print(a[1])

	# Load the binary files with states and labels.
	X = np.fromfile(dirname + "/inference/states.bin", dtype=np.int8).reshape(SIM_PER_SEED * n_nodes, n_nodes)
	y = np.fromfile(dirname + "/inference/labels.bin", dtype=np.int32)

	# Convert states to a torch tensor.
	X = torch.from_numpy(X).long()

	# One-hot encode node states.
	X = torch.nn.functional.one_hot(X, num_classes=3).float()

	# Truth values (sources) in torch format.
	y = torch.from_numpy(y).long()

	# Count number of simulation runs with only one infected node.
	print(f'Number of sim. with only one inf. node: {(X[:,:,0].sum(dim=1) == n_nodes-1).sum()}')

	# Move tensors to GPU (if available), otherwise they are on CPU.
	X = X.to(device)
	y = y.to(device)
	edge_index = edge_index.to(device)
	if weights is not None:
		weights = weights.to(device)

	# =========================================================
	# LOAD TRAINING DATA --------------------------------------
	# =========================================================

	# Load the binary files with states and labels.
	# They are needed for SME and MCMF below.
	Xtrain = np.fromfile(dirname + "/states.bin", dtype=np.int8).reshape(p["SIM_PER_SEED"] * n_nodes, n_nodes)

	# Convert states to a torch tensor.
	Xtrain = torch.from_numpy(Xtrain).long()

	# One-hot encode node states.
	Xtrain = torch.nn.functional.one_hot(Xtrain, num_classes=3).float()

	# =========================================================
	# FEATURE AUGMENTATION ------------------------------------
	# =========================================================

	# Compute normalized graph Laplacian.
	# L_norm = nx.normalized_laplacian_matrix(G)

	# Get the first k eigenvectors of norm. graph Laplacian.
	# _, eigenvectors = eigsh(L_norm, k=5, which='SM')

	# Compute centrality measures.
	bc = nx.betweenness_centrality(G)
	cc = nx.closeness_centrality(G)

	# Compute local clustering coefficients.
	lc = nx.clustering(G)

	# Organize new features in nested list.
	features = [[G.degree[n] for n in sorted(G.nodes)], [bc[n] for n in sorted(G.nodes)], [cc[n] for n in sorted(G.nodes)], [lc[n] for n in sorted(G.nodes)]]

	# Add eigenvectors to features list.
	# features.extend(eigenvectors.T.tolist())

	# Count number of features so that we can "soft-code" the
	# number of input features when we initialize the model.
	num_feat = len(features)

	# Reshape new features.
	features = list(zip(*features))

	features = torch.tensor(features).to(device)

	# =========================================================
	# MODEL SETUP ---------------------------------------------
	# =========================================================

	# Conditionally set the number of input features
	# (depending on whether we do feature augmentation or not).
	num_node_features = (3 + num_feat) if p["FEATURE_AUGMENTATION"] else 3

	# Initialize the model architecture.
	# NOTE: number of features is hardcoded for now.
	model = GNN(
		num_preprocess_layers=p["PREPROCESSING_LAYERS"],
		embed_dim_preprocess=p["EMBED_DIM_PREPROCESS"],
		num_postprocess_layers=p["POSTPROCESSING_LAYERS"],
		num_conv_layers=p["LAYERS"],
		aggr=p["AGGREGATION"],
		num_node_features=num_node_features,
		hidden_channels=p["HIDDEN_CHANNELS"],
		num_classes=n_nodes,
		dropout_rate=p["DROPOUT_RATE"],
		batch_norm=p["BATCH_NORMALIZATION"],
		skip=p["SKIP"]
	).to(device)

	# Load trained model weights from file.
	model.load_state_dict(torch.load(dirname + '/model_weights.pth', weights_only=True))

	# Turn-off training mode for all layers where it is relevant.
	model.eval()

	# =========================================================
	# INFERENCE -----------------------------------------------
	# =========================================================

	# Number of test examples.
	# NOTE: we purposefully do NOT take the SIM_PER_SEED out of the dict p.
	n = n_nodes * SIM_PER_SEED

	# Tensor of zeros to which we will add probabilities of ensemble.
	ensemble = torch.zeros(n, n_nodes)

	# =========================================================
	# GNN -----------------------------------------------------

	# Measure start time.
	start_time = time.perf_counter()

	# Dictionary to store results in.
	res_gnn = dict.fromkeys(["top-1","top-3","top-5","aed","arr","css","sru1","sru2"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_GNN.txt', 'w') as resfile:
		# Iterate over data in test dataset to compute model predictions.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# Augment with additional features if setup is like that.
			if p["FEATURE_AUGMENTATION"]:
				states_current = torch.cat((states_current, features), dim=1)
			# We need to get indices of susceptible nodes.
			susceptible = (states_current[:,0] == 1.0)
			# Move to device (GPU or CPU).
			states_current = states_current.to(device)
			batch = torch.zeros(n_nodes, dtype=torch.int64).to(device)
			# Forward pass to compute model predictions.
			out = model(states_current, edge_index, weights, batch)
			# Add output to ensemble predictions.
			ensemble[idx,:] += log_sum_exp(out.detach().clone().cpu())[0]
			# Output the (log-softmax) probability distribution.
			resfile.write(', '.join(map(str, (i for i in out.flatten().tolist()))) + '\n')
			# Set susceptible nodes to neg. infinity such that they
			# are not considered as sources. We do not currently do
			# that since GNN learns on its own that S nodes cannot be the source.
			# out[:,susceptible] = -float('inf')
			# Update evaluation measures in dict.
			update(y[idx], G, out.cpu(), res_gnn, resfile)

	# How much time has elapsed?
	elapsed = time.perf_counter() - start_time

	# Write to file.
	with open(f"{dirname}/inference/times.txt", "w") as f:
		f.write(f"Average (GNN): {elapsed / n}\n")

	# Divide all values in results dict by size of test dataset (= averaging).
	res_gnn = {key: value/n for key, value in res_gnn.items()}

	# Print results to console.
	print('============================')
	print('GNN RESULTS')
	print_results(res_gnn)
	print(f"Average time per inference run: {elapsed / n:.6f} seconds")

	# =========================================================
	# RANDOM --------------------------------------------------

	# Dictionary to store results in.
	res_random = dict.fromkeys(["top-1","top-3","top-5","aed","arr"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_RANDOM.txt', 'w') as resfile:
		# Iterate over data in test dataset and apply random baseline.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of infection subgraph, such that we can randomly sample from it.
			inf_rec = (states_current[:,0] == 0.0).argwhere().flatten().tolist()
			# Initialize tensor of zeros.
			out = torch.zeros(1, n_nodes)
			# Set probability of a randomly chosen node of the inf. subgraph to 1.
			out[:,choice(inf_rec)] = 1.0
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_random, resfile, distr=False)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_random = {key: value/n for key, value in res_random.items()}

	# Print results to console.
	print('============================')
	print('RANDOM RESULTS')
	print_results(res_random, distr=False)

	# =========================================================
	# JORDAN CENTER -------------------------------------------

	# Dictionary to store results in.
	res_jordan = dict.fromkeys(["top-1","top-3","top-5","aed","arr"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_JORDAN.txt', 'w') as resfile:
		# Iterate over data in test dataset and apply Jordan center method.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of infection subgraph for Jordan center method.
			inf_rec = (states_current[:,0] == 0.0).argwhere().flatten().tolist()
			# Apply Jordan center method (Zhu & Ying).
			# With 'inf_rec.argwhere().flatten()' we provide the infected subgraph.
			out = jordan_center(G, inf_rec)
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_jordan, resfile, distr=False)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_jordan = {key: value/n for key, value in res_jordan.items()}

	# Print results to console.
	print('============================')
	print('JORDAN CENTER RESULTS')
	print_results(res_jordan, distr=False)

	# =========================================================
	# DEGREE CENTRALITY ---------------------------------------

	# Dictionary to store results in.
	res_degree = dict.fromkeys(["top-1","top-3","top-5","aed","arr"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_DEGREE.txt', 'w') as resfile:
		# Iterate over data in test dataset and apply Jordan center method.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of infection subgraph for this method.
			inf_rec = (states_current[:,0] == 0.0).argwhere().flatten().tolist()
			# Compute simple degree centrality values.
			out = degree_center(G, inf_rec)
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_degree, resfile, distr=False)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_degree = {key: value/n for key, value in res_degree.items()}

	# Print results to console.
	print('============================')
	print('DEGREE CENTRALITY RESULTS')
	print_results(res_degree, distr=False)

	# =========================================================
	# BETWEENNESS CENTRALITY ----------------------------------

	# Dictionary to store results in.
	res_btw = dict.fromkeys(["top-1","top-3","top-5","aed","arr"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_BETWEENNESS.txt', 'w') as resfile:
		# Iterate over data in test dataset and apply Jordan center method.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of infection subgraph for this method.
			inf_rec = (states_current[:,0] == 0.0).argwhere().flatten().tolist()
			# Compute simple degree centrality values.
			out = betweenness_center(G, inf_rec)
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_btw, resfile, distr=False)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_btw = {key: value/n for key, value in res_btw.items()}

	# Print results to console.
	print('============================')
	print('BETWEENNESS CENTRALITY RESULTS')
	print_results(res_btw, distr=False)

	# =========================================================
	# CLOSENESS CENTRALITY ------------------------------------

	# Dictionary to store results in.
	res_cln = dict.fromkeys(["top-1","top-3","top-5","aed","arr"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_CLOSENESS.txt', 'w') as resfile:
		# Iterate over data in test dataset and apply Jordan center method.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of infection subgraph for this method.
			inf_rec = (states_current[:,0] == 0.0).argwhere().flatten().tolist()
			# Compute simple degree centrality values.
			out = closeness_center(G, inf_rec)
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_cln, resfile, distr=False)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_cln = {key: value/n for key, value in res_cln.items()}

	# Print results to console.
	print('============================')
	print('CLOSENESS CENTRALITY RESULTS')
	print_results(res_cln, distr=False)

	# =========================================================
	# SOFT MARGIN ESTIMATOR (ANTULOV-FANTULIN ET AL.) ---------

	# Measure start time.
	start_time = time.perf_counter()

	# To speed up computations of SME, we already transform training data
	# into a 2D tensor that encodes with (0/1) which nodes are infected and
	# which are not. torch.select slices along the selected dimension (2)
	# at the given index (0), which is the Susceptible slice. eq(0) sets all
	# elements to True which are 0 (thus all non-susceptible nodes).
	Xtrain_mask = Xtrain.select(2, 0).eq(0).to(torch.int32)

	# Similarly, to speed up the procedure, we compute the rowsums.
	# This returns the number of infected and recovered nodes per training outbreak.
	Xtrain_mask_sum = Xtrain_mask.sum(dim=1)

	# Possible hyperparamter values a^2.
	a_sq = torch.tensor([(0.5**i)**2 for i in range(1, 16)])

	# Dictionary to store results in.
	res_sme = dict.fromkeys(["top-1","top-3","top-5","aed","arr","css","sru1","sru2"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_SME.txt', 'w') as resfile:
		# Iterate over data in test dataset to compute model predictions.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of susceptible nodes.
			susceptible = (states_current[:,0] == 1.0)
			# Run the soft margin estimator.
			out = soft_margin(Xtrain_mask, Xtrain_mask_sum, states_current.to('cpu'), a_sq)
			# Add output to ensemble predictions.
			ensemble[idx,:] += log_sum_exp(out)[0]
			# Output the log-likelihoods
			resfile.write(', '.join(map(str, (i for i in out.flatten().tolist()))) + '\n')
			# Set susceptible nodes to neg. infinity such that they
			# are not considered as sources.
			out[:,susceptible] = -float('inf')
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_sme, resfile)

	# How much time has elapsed?
	elapsed = time.perf_counter() - start_time

	# Write to file.
	with open(f"{dirname}/inference/times.txt", "a") as f:
		f.write(f"Average (SME): {elapsed / n}\n")

	# Divide all values in results dict by size of test dataset (= averaging).
	res_sme = {key: value/n for key, value in res_sme.items()}

	# Print results to console.
	print('============================')
	print('SOFT MARGIN RESULTS')
	print_results(res_sme)
	print(f"Average time per inference run: {elapsed / n:.6f} seconds")

	# =========================================================
	# SIMULATION-BASED NODE PROBABILITIES (STERCHI ET AL.) ----

	# Measure start time.
	start_time = time.perf_counter()

	# First store number of training examples and number of nodes.
	n_sim, n_nodes, _ = Xtrain.shape

	# Here we compute the number of training examples per seed node.
	n_per_node = int(n_sim / n_nodes)

	# Sum up counts per source node.
	node_state_probs = Xtrain.view(n_nodes, n_per_node, n_nodes, 3).sum(dim=1)
	# Add-1 smoothing.
	# This is necessary for small samples as some nodes may never be observed in one
	# of the states which would lead to a node state probability of 0.
	node_state_probs += 1
	# Subtract the 1 from diagonal of "susceptible" slice.
	# Why? Cause the source nodes should have zero prob. of being susceptible.
	node_state_probs[:, :, 0] -= torch.eye(n_nodes, device=node_state_probs.device)
	# Normalize along last dimension to get probabilities.
	node_state_probs = node_state_probs / node_state_probs.sum(dim=2, keepdim=True)

	# Dictionary to store results in.
	res_mcs = dict.fromkeys(["top-1","top-3","top-5","aed","arr","css","sru1","sru2"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_MCS.txt', 'w') as resfile:
		# Iterate over data in test dataset to compute model predictions.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of susceptible nodes.
			susceptible = (states_current[:,0] == 1.0)
			# Run the soft margin estimator.
			out = mcs_mean_field(node_state_probs, states_current.to('cpu'))
			# Add output to ensemble predictions.
			ensemble[idx,:] += log_sum_exp(out)[0]
			# Output the log-likelihoods
			resfile.write(', '.join(map(str, (i for i in out.flatten().tolist()))) + '\n')
			# Set susceptible nodes to neg. infinity such that they
			# are not considered as sources.
			out[:,susceptible] = -float('inf')
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_mcs, resfile)

	# How much time has elapsed?
	elapsed = time.perf_counter() - start_time

	# Write to file.
	with open(f"{dirname}/inference/times.txt", "a") as f:
		f.write(f"Average (MCMF): {elapsed / n}\n")

	# Divide all values in results dict by size of test dataset (= averaging).
	res_mcs = {key: value/n for key, value in res_mcs.items()}

	# Print results to console.
	print('============================')
	print('MCS MEAN-FIELD RESULTS')
	print_results(res_mcs)
	print(f"Average time per inference run: {elapsed / n:.6f} seconds")

	# =========================================================
	# ENSEMBLE ------------------------------------------------

	# Dictionary to store results in.
	res_ens = dict.fromkeys(["top-1","top-3","top-5","aed","arr","css","sru1","sru2"], 0.0)

	# Open txt file for results.
	with open(dirname + '/inference/results_ENS.txt', 'w') as resfile:
		# Iterate over data in test dataset to compute model predictions.
		for idx in range(n):
			# Get the one-hot encoded states for the current test data point.
			states_current = X[idx,:,:]
			# We need to get indices of susceptible nodes.
			susceptible = (states_current[:,0] == 1.0)
			# Get ensemble predictions for current scenario.
			out = (ensemble[idx,:] / 3.0).unsqueeze(0)
			# Set susceptible nodes to neg. infinity such that they
			# are not considered as sources.
			out[:,susceptible] = -float('inf')
			# Update evaluation measures in dict.
			update(y[idx], G, out, res_ens, resfile)

	# Divide all values in results dict by size of test dataset (= averaging).
	res_ens = {key: value/n for key, value in res_ens.items()}

	# Print results to console.
	print('============================')
	print('ENSEMBLE RESULTS')
	print_results(res_ens)
