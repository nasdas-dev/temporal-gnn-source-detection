# --------------------------------------------------------------------------
# Evaluation functions
# --------------------------------------------------------------------------

import torch
import numpy as np
import networkx as nx
import propnetscore.node_selection as ns
from scipy.stats import rankdata
from .utils import log_sum_exp
from .model import ResistanceLoss
from functools import lru_cache

def error_distance(graph, predictions, true_source):
	# Make a copy of the tensor so we can modify it here.
	pred = predictions.detach().clone()
	# This is to find all elements that correspond to the max.
	# If there are multiple nodes having max. value, then
	# we would like to randomly sample the prediction.
	max_elem = torch.nonzero((pred == np.nanmax(pred)).flatten(), as_tuple=True)[0]
	# Sample from max_elem (this gives back sampled index).
	#choice = torch.multinomial(max_elem.float(), 1)
	# Get the prediction.
	#pred = max_elem[choice]
	pred = np.random.choice(max_elem, size=1)
	# Return shortest path length between prediction and true source node.
	# Note that it should always be possible to find a shortest path since
	# we restrict possible sources to nodes in the infected subgraph.
	return nx.shortest_path_length(graph, source=true_source.item(), target=pred.item())

# This function is more complicated than one would expect because
# we want to properly deal with ties. For example, if the top-5 list
# already contains three nodes and now there would be 4 nodes with the
# same probability then we sample two of the 4 nodes uniformely at random
# to fill up the top-5 list.
def top_k(predictions, true_source, k=1):
	# Empty list for top-k node indices.
	indices = []
	# Make a copy of the tensor so we can modify it here.
	pred = predictions.detach().clone()
	# Loop as long as the list has not reached k elements.
	while len(indices) < k:
		# Get the max. value, ignoring missing values.
		m = np.nanmax(pred)
		# Get the indices that correspond to the current max. value.
		idx = torch.nonzero((pred == m).flatten(), as_tuple=True)[0]
		# If all new node indices can be added without surpassing k, then do it.
		if len(idx) <= (k - len(indices)):
			indices.extend(idx.tolist())
		# Otherwise, we randomly sample node indices so as to fill the list up to k.
		else:
			indices.extend(np.random.choice(idx, size=k-len(indices)))
		# Importantly, we now need to set all node indices already chosen to nan.
		pred[:,indices] = float('nan')
	# We return a boolean indicating whether the true source is in top-k or not.
	return True if true_source.item() in indices else False

def reciprocal_rank(predictions, true_source):
	# Make a copy of the tensor so we can apply scipy method on it.
	pred = predictions.detach().clone()
	# OLD: ranks = pred.argsort(descending=True).argsort() + 1
	# Get the ranks for the elements in 'pred'.
	# We need to multiply with (-1.0) because ranks are assigned
	# in ascending order. The 'average' method gives the average rank
	# to all tied elements. Note that if there are e.g. two tied
	# max. elements, they will have rank 1.5, and the next best
	# will have rank 3.
	ranks = rankdata((-1.0) * pred, method='average')
	# Return the reciprocal rank of true source.
	return float(1.0 / ranks[int(true_source.item())])

def credible_set_size(predictions, level=0.9):
	# Make a copy of the tensor.
	pred = predictions.detach().clone()
	# Apply log-sum-exp trick.
	pred_normalized = log_sum_exp(pred)
	# Sort along the first dimension in descending order for cum. sum.
	v = pred_normalized.sort(dim=1, descending=True).values
	# Get a boolean mask that shows from which element on the cum. prob.
	# is larger than the chosen probability level.
	mask = torch.cumsum(v, dim=1) >= level
	# Return the first index that evaluates to True.
	# Add 1 cause indexing starts at 0.
	return mask.max(1, True).indices.item() + 1

def scoring_rules(graph, predictions, true_source, method="brier", kappa=1):
	# Make a copy of the tensor.
	pred = predictions.detach().clone()
	# Apply log-sum-exp trick.
	pred_normalized = log_sum_exp(pred)
	# Get the adjacency_matrix of the graph.
	adjacency_matrix = nx.to_numpy_array(graph, weight=None)
	# Set up the scoring rule.
	task = ns.NodeSelectionTask(adjacency_matrix, pred_normalized.squeeze().numpy(), true_source)
	# Return score based on method chosen.
	match method:
		case "brier":
			return task.brier_score()
		case "log":
			return task.logarithmic_score()
		case "resistance":
			return task.resistance_score()
		case "diffusion":
			return task.diffusion_score(kappa)
		case _:
			raise ValueError(f"Unknown method: {method}")

@lru_cache(maxsize=1)
def resistance_score(graph, predictions, true_source):
	# Make a copy of the tensor.
	pred = predictions.detach().clone()
	# Apply log-sum-exp trick.
	pred_normalized = log_sum_exp(pred)
	# Get adjacency matrix in torch format.
	adj = torch.from_numpy(nx.to_numpy_array(graph, weight=None))
	# Initialize ResistanceLoss (it is cached after first time).
	criterion = ResistanceLoss(adj)
	# Return resistance score.
	return criterion(pred_normalized, true_source, log_softmax=False)

# Function to update all evaluation measures in a dict.
def update(label, graph, predictions, results_dict, resfile, distr=True):
	# Compute performance measures
	t1 = int(top_k(predictions, label, 1))
	t3 = int(top_k(predictions, label, 3))
	t5 = int(top_k(predictions, label, 5))
	ed = error_distance(graph, predictions, label)
	rr = reciprocal_rank(predictions, label)
	# Optional evaluation of credible set size and scoring rule.
	if distr:
		cs = credible_set_size(predictions, level=0.9)
		sr1 = scoring_rules(graph, predictions, label, method="brier")
		#sr2 = resistance_score(graph, predictions, label) # This was a temporary fix. Now, the implementation from propnetscore package is fast.
		sr2 = scoring_rules(graph, predictions, label, method="resistance")
	# Add results to file
	if distr:
		resfile.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(t1, t3, t5, ed, rr, cs, sr1, sr2))
	else:
		resfile.write("{}, {}, {}, {}, {}\n".format(t1, t3, t5, ed, rr))
	# Add measures to dict
	results_dict['top-1'] += t1
	results_dict['top-3'] += t3
	results_dict['top-5'] += t5
	results_dict['aed'] += ed
	results_dict['arr'] += rr
	if distr:
		results_dict['css'] += cs
		results_dict['sru1'] += sr1
		results_dict['sru2'] += sr2

# Function to print results to console.
def print_results(results_dict, distr=True):
	print(f'Top-1 Acc.: {results_dict["top-1"]:.4f}')
	print(f'Top-3 Acc.: {results_dict["top-3"]:.4f}')
	print(f'Top-5 Acc.: {results_dict["top-5"]:.4f}')
	print(f'Avg. error dist.: {results_dict["aed"]:.4f}')
	print(f'Avg. reciprocal rank: {results_dict["arr"]:.4f}')
	# Optionally, print credible set size and scoring rules to console.
	if distr:
		print(f'Avg. credible set size: {results_dict["css"]:.4f}')
		print(f'Avg. scoring rule (Brier): {results_dict["sru1"]:.4f}')
		print(f'Avg. scoring rule (Resistance): {results_dict["sru2"]:.4f}')
