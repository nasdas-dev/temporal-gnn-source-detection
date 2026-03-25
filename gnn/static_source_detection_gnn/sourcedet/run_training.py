# Import libraries
import os
import sys
import yaml
import time
import argparse
import json
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sys import argv
from pathlib import Path
from subprocess import check_output
from random import getrandbits, randint

from .nwk import import_nwk, create_nwk
from .model import GNN, ResistanceLoss
from .evaluate import top_k, scoring_rules
from .utils import one_hot, EarlyStopper, plot_train_valid_curves

from scipy.sparse.linalg import eigsh
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from torch.utils.data import DataLoader
from torch_geometric.utils.convert import from_networkx
import torch.nn.functional as F

# =========================================================
# TRAINING ROUTINE ----------------------------------------
# =========================================================

if __name__ == "__main__":

	# Setup argument parser.
	parser = argparse.ArgumentParser()

	# First argument: network config file
	parser.add_argument('--nwk', type=str, required=True, help='config file')
	# Second argument: JSON dict with parameters (both GNN architecture and epidemic)
	parser.add_argument('--params', type=json.loads, required=True, help='input parameters')
	# Third argument: seed to reproduce same training data for given epi. parameters/network
	parser.add_argument('--seed', type=str,	required=True, help='seed')

	# Parse the arguments.
	args = parser.parse_args()

	# Load graph config info from YAML file.
	# This seems like a bit overkill but was initially
	# created to account for synthetic graphs (ER, etc.).
	with open(args.nwk, "r") as fp:
		nwk_args = yaml.safe_load(fp)

	# Create a subdirectory for results and network file based on Linux timestamp.
	dirname = 'data/results_' + str(int(time.time())) + str(randint(1000, 9999))
	os.makedirs(dirname)

	# Select device.
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print(f'Using device: {device}')
	# Allows inbuilt cudnn auto-tuner to find best config for hardware.
	torch.backends.cudnn.benchmark = True

	# =========================================================
	# INPUT GRAPH ---------------------------------------------
	# =========================================================

	# Create network or load empirical network.
	G = create_nwk(nwk_args)

	# Compute graph statistics.
	n_nodes, n_edges = G.number_of_nodes(), G.number_of_edges()
	avg_deg = np.mean([v for k, v in G.degree()])
	g_diame = nx.diameter(G.subgraph(max(nx.connected_components(G), key=len)).copy())

	# Add to dict with parameters.
	# This helps inference routine know what to do.
	args.params["GRAPH"] = nwk_args["graph"]
	args.params["EDGE_WEIGHTS"] = nwk_args["weights"]
	args.params["GRAPH_SIZE"] = n_nodes

	# Export YAML file with import configuration parameters
	# so that it can be imported during the inference phase.
	with open(dirname + '/config.yaml', 'w') as outfile:
		yaml.dump(args.params, outfile)

	# Print stats to console.
	print('============================')
	print(f'Number of nodes: {n_nodes}')
	print(f'Number of edges: {n_edges}')
	print(f'Average degree: {avg_deg}')
	print(f'Diameter (LCC): {g_diame}')

	# We add the edge weights to CSV as well.
	# If a graph has no weights, then the weight will be set to 1 by default.
	with open(dirname + "/graph.csv", "w") as f:
		for u, v, d in G.edges(data=True):
			weight = d.get("weight", 1)
			f.write(f"{u} {v} {weight}\n")

	# Transform adjacency info and edge weights into the format that PyG model expects.
	if nwk_args["weights"]:
		edge_index, weights, _ = from_networkx(G, group_edge_attrs="all")
		edge_index, weights = edge_index[1], weights[1]
		# We save the weights
		torch.save(weights, dirname + "/edge_weights.pt")
	else:
		# If there are no edge weights, we set them to None
		edge_index, weights = from_networkx(G).edge_index, None

	# Here we store the second dimension of edge_index
	# (which is twice the number of edges). This is used
	# below to create the batched_edge_index tensor efficiently.
	E = edge_index.size(1)

	# =========================================================
	# TRAINING DATA -------------------------------------------
	# =========================================================

	# Measure start time.
	start_time = time.perf_counter()

	# Simulate outbreaks (call to C code in /sir):
	# This simulates a continuous-time SIR model with exponentially distributed
	# times until infection (per link) and exponentially distributed time until
	# recovery. The recovery rate is by default set to 1 and we only need to 
	# provide the infection rate beta. For a different recovery rate nu, we can
	# just provide a modified beta (beta = intended beta / nu) and T 
	# (T = intended T / nu). For example, with beta=2 and nu=0.5, we provide 
	# the C code the value beta = 2/0.5 = 4.
	out = check_output(['sir/sir', dirname + "/graph.csv", str(args.params["BETA"] / args.params["NU"]), str(args.params["T"] / args.params["NU"]), str(int(args.params["SAMPLED_T"])), str(args.params["SIM_PER_SEED"]), args.seed, dirname])

	# How much time has elapsed?
	elapsed = time.perf_counter() - start_time

	# Print to console.
	print(f"Average time per sim.: {elapsed / (n_nodes * args.params["SIM_PER_SEED"]):.6f} seconds")

	# Write time stats to file.
	with open(f"{dirname}/times.txt", "w") as f:
		f.write(f"Average: {elapsed / (n_nodes * args.params["SIM_PER_SEED"])}\n")

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
	X = np.fromfile(dirname + "/states.bin", dtype=np.int8).reshape(args.params["SIM_PER_SEED"] * n_nodes, n_nodes)
	y = np.fromfile(dirname + "/labels.bin", dtype=np.int32)

	# Convert states to a torch tensor.
	X = torch.from_numpy(X).long()

	# One-hot encode node states.
	X = torch.nn.functional.one_hot(X, num_classes=3).float()

	# Truth values (sources) in torch format.
	y = torch.from_numpy(y).long()

	# Count number of simulation runs with only one infected node.
	print(f'Number of sim. with only one inf. node: {(X[:,:,0].sum(dim=1) == n_nodes-1).sum()}')

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

	# =========================================================
	# TRAIN-VALID SPLIT AND MINI-BATCHES ----------------------
	# =========================================================

	# Tensor of indices that are used in train_test_split ([0, ..., len(dataset)]).
	indices = np.arange(n_nodes * args.params["SIM_PER_SEED"], dtype=np.int64)
	# Sample the indices for stratified train-valid-test split.
	train_indices, valid_indices = train_test_split(indices, test_size=3/10, stratify=y.numpy(), random_state=42)

	# Transform to torch format.
	train_indices = torch.from_numpy(train_indices).long()
	valid_indices = torch.from_numpy(valid_indices).long()

	print('============================')
	print(f'Training examples: {len(train_indices)}')
	print(f'Validation examples: {len(valid_indices)}')

	# Make pin_memory depend on whether we are on GPU or not.
	# cuda.is_available returns True only if we are on GPU.
	# This helps to avoid the warning message.
	pin_mem = torch.cuda.is_available()

	# Create the mini-batches.
	# Why do we create mini-batches for validation set too?
	# Because we can then use the test() function for both the training and the validation set.
	train_loader = DataLoader(train_indices, batch_size=args.params["BATCH_SIZE"], shuffle=True, pin_memory=pin_mem, num_workers=4)
	valid_loader = DataLoader(valid_indices, batch_size=args.params["BATCH_SIZE"], shuffle=False, pin_memory=pin_mem, num_workers=4)

	# Move X and y to device.
	X = X.to(device)
	y = y.to(device)

	# =========================================================
	# MODEL SETUP ---------------------------------------------
	# =========================================================

	# Conditionally set the number of input features
	# (depending on whether we do feature augmentation or not).
	num_node_features = (3 + num_feat) if args.params["FEATURE_AUGMENTATION"] else 3

	# Initialize the model architecture.
	model = GNN(
		num_preprocess_layers=args.params["PREPROCESSING_LAYERS"],
		embed_dim_preprocess=args.params["EMBED_DIM_PREPROCESS"],
		num_postprocess_layers=args.params["POSTPROCESSING_LAYERS"],
		num_conv_layers=args.params["LAYERS"],
		aggr=args.params["AGGREGATION"],
		num_node_features=num_node_features,
		hidden_channels=args.params["HIDDEN_CHANNELS"],
		num_classes=n_nodes,
		dropout_rate=args.params["DROPOUT_RATE"],
		batch_norm=args.params["BATCH_NORMALIZATION"],
		skip=args.params["SKIP"]
	).to(device)

	# Print model to console.
	print('============================')
	print(model)

	# Specify optimizer and learning rate.
	optimizer = torch.optim.Adam(model.parameters(), lr=args.params["LEARNING_RATE"], weight_decay=5e-4)

	# Setup custom loss.
	# criterion = ResistanceLoss(torch.from_numpy(nx.to_numpy_array(G, weight=None)))

	# Function to efficiently make a edge_index object that represents B x graph.
	def make_batched_edge_index(B):
		offsets = torch.arange(B, dtype=torch.long, device=edge_index.device) * n_nodes
		offsets = offsets.repeat_interleave(E)
		be = edge_index.repeat(1, B) + offsets.unsqueeze(0)
		return be.to(device, non_blocking=True)

	# =========================================================
	# TRAINING FUNCTIONS --------------------------------------
	# =========================================================

	# Training function.
	def train(loader, weights, augment_feat=None):
		# Turn-on training mode.
		model.train()
		# Empty list to store times.
		batch_times = []
		# Iterate over the training batches.
		for idx in loader:
			# Measure start time.
			start_time = time.perf_counter()
			# Get batch indices and add them to device.
			idx = idx.to(device, non_blocking=True)
			# Get the batch size (for the last one it may be smaller than what we set).
			B = idx.size(0)
			# Batched edge_index.
			batched_edge_index = make_batched_edge_index(B)
			# Stack node feature matrices along the first dimension.
			x = X[idx].reshape(B * n_nodes, 3)
			# Feature augmentation if they have been provided.
			if augment_feat is not None:
				# Replicate the features according to batch size (B).
				new_feat = torch.tensor(augment_feat * B).to(device)
				# Add features as new columns.
				x = torch.cat((x, new_feat), dim=1)
			# Add x to device.
			x = x.to(device, non_blocking=True)
			# Batched edge weights.
			if weights is not None:
				batched_weights = torch.cat([weights] * B)
			# If no edge weights are used, we set them to None.
			else:
				batched_weights = None
			# Batch vector assigning each node to specific training example, add to device.
			batch = torch.arange(B, dtype=torch.long).repeat_interleave(n_nodes).to(device)
			# Add weights to device if they exist.
			if batched_weights is not None:
				batched_weights = batched_weights.to(device)
			# Clear gradients.
			optimizer.zero_grad()
			# Perform a single forward pass.
			out = model(x, batched_edge_index, batched_weights, batch)
			# Compute the loss.
			loss = F.nll_loss(out, y[idx])
			# loss = criterion(out, y[idx], reduction='sum')
			# Backward pass.
			loss.backward()
			# Gradient descent step.
			optimizer.step()
			# Store time it took for current batch.
			batch_times.append(time.perf_counter() - start_time)
		# Return the batch times.
		return batch_times

	# Test function.
	def test(loader, weights, augment_feat=None):
		# Turn-off training mode for all layers where it is relevant.
		model.eval()
		# Initialize number of correct predictions and loss to 0.
		correct, loss = 0, 0
		# Evaluating the model with torch.no_grad() ensures that
		# no gradients are computed during test mode.
		with torch.no_grad():
			# Iterate over batches.
			for idx in loader:
				# Get batch indices and add them to device.
				idx = idx.to(device, non_blocking=True)
				# Get the batch size (for the last one it may be smaller than what we set).
				B = idx.size(0)
				# Batched edge_index.
				batched_edge_index = make_batched_edge_index(B)
				# Stack node feature matrices along the first dimension.
				x = X[idx].reshape(B * n_nodes, 3)
				# Feature augmentation if they have been provided.
				if augment_feat is not None:
					# Replicate the features according to batch size (B).
					new_feat = torch.tensor(augment_feat * B).to(device)
					# Add features as new columns.
					x = torch.cat((x, new_feat), dim=1)
				# Add x to device.
				x = x.to(device, non_blocking=True)
				# Batched edge weights.
				if weights is not None:
					batched_weights = torch.cat([weights] * B)
				# If no edge weights are used, we set them to None.
				else:
					batched_weights = None
				# Batch vector assigning each node to specific training example, add to device.
				batch = torch.arange(B, dtype=torch.long).repeat_interleave(n_nodes).to(device)
				# Add weights to device if they exist.
				if batched_weights is not None:
					batched_weights = batched_weights.to(device)
				# Forward pass to compute model predictions.
				out = model(x, batched_edge_index, batched_weights, batch)
				# Get the output class with highest probability.
				pred = out.argmax(dim=1)
				# Add number of correct predictions in current batch to 'correct'.
				correct += (pred == y[idx]).sum().item()
				# Add up losses.
				loss += F.nll_loss(out, y[idx], reduction='sum').item()
				# loss += criterion(out, y[idx], reduction='sum')
		# Size of validation set.
		Nvalid = len(loader.dataset)
		# Return accuracy and loss.
		return (float(correct) / Nvalid, float(loss) / Nvalid)

	# =========================================================
	# TRAINING LOOP -------------------------------------------
	# =========================================================

	# Initialize early stopping.
	early_stopper = EarlyStopper(patience=args.params["PATIENCE"], min_delta=0)

	# Emtpy lists for results.
	list_train_res = []
	list_valid_res = []

	# Empty list to store batch times.
	all_batch_times = []

	# Initialize two variables so we can restore model with best validation loss.
	best_valid_loss = float('inf')
	best_model_state = None

	print('============================')

	# Actual training (iteration over epochs).
	for epoch in range(1, args.params["NEPOCHS"] + 1):
		# Train depending on whether features are augmented or not.
		if args.params["FEATURE_AUGMENTATION"]:
			# Train model.
			batch_times = train(train_loader, weights, features)
			# Compute accuracy on train and test.
			train_res = test(train_loader, weights, features)
			valid_res = test(valid_loader, weights, features)
		else:
			# Train model.
			batch_times = train(train_loader, weights)
			# Compute accuracy on train and test.
			train_res = test(train_loader, weights)
			valid_res = test(valid_loader, weights)
		# Store batch times in list.
		all_batch_times.extend(batch_times)
		# Append training and validation results to lists for plotting
		list_train_res.append(train_res)
		list_valid_res.append(valid_res)
		# Print results to console.
		print(f'Epoch: {epoch:03d}, Train Acc: {train_res[0]:.4f}, Train Loss: {train_res[1]:.4f}, Val. Acc: {valid_res[0]:.4f}, Val. Loss: {valid_res[1]:.4f}')
		# If current validation loss is smaller than previously best loss,
		# then save model weights in best_model_state.
		if valid_res[1] < best_valid_loss:
			best_valid_loss = valid_res[1]
			best_model_state = model.state_dict()
		# Checky if we stop training early.
		if early_stopper.early_stop(valid_res[1]):
			break

	print('============================')

	# Plot training curves.
	plot_train_valid_curves(list_train_res, list_valid_res, dirname)

	# Save model weights.
	torch.save(best_model_state, dirname + '/model_weights.pth')

	# Print message to console.
	print(f'Saving model that yielded validation loss of {best_valid_loss:.4f}.')

	# Compute time statistics.
	avg_batch_time, sem_batch_time = np.mean(all_batch_times), np.std(all_batch_times) / np.sqrt(len(all_batch_times))
	print(f"Average batch time: {avg_batch_time:.6f} seconds (SEM: {sem_batch_time: .6f})")

	# Write to file.
	with open(f"{dirname}/times.txt", "a") as f:
		f.write(f"Average: {avg_batch_time}, SEM: {sem_batch_time}\n")
