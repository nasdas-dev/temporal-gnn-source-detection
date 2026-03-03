import random
import subprocess
from subprocess import Popen, PIPE
import warnings
import time
import wandb
import numpy as np


def make_c_readable_from_networkx(H, t_max, directed = False):
	"""Assembling the temporal network in the format of the C code. It has the format:
	n_nodes, t_max
	degree of node 0
	neighbor_1 number_of_contacts
	time_1
	time_2
	...
	time_x
	neighbor_k number_of_contacts
	time_1
	time_2
	...
	time_y
	...
	degree of node 1
	...
	Importantly, it is sorted, so it start with node 0, then node 1, etc. Moreover, for each node, we go through
	the neighbors in the order of the latest contact time, e.g. last contact with neighbor_1 later than last contact
	with neighbor_2, etc. It matters for the C code to work properly: "if (t == END) break" in infect()."""

	nwk = str(H.number_of_nodes()) + ' ' + str(t_max) + '\n'
	for i in sorted(H.nodes()): # sorted is super important, it has to be 0,1,2,etc. by convention!
		deg = H.out_degree(i) if directed else H.degree(i)
		nwk += str(deg) + '\n'
		to_sort = []
		for v in H.neighbors(i): # for DiGraph, neighbors and successors are the same
			to_sort.append((max(H.edges[i,v]['times']),v))
		for x, v in sorted(to_sort, reverse=True): # the sorting matters for the C code!
			nwk += str(v) + ' ' + str(len(H.edges[i,v]['times'])) + '\n'
			for t in sorted(H.edges[i,v]['times']):
				nwk += str(t) + '\n'

	return nwk


def run(nwk, beta, mu, start_t, end_t, n, seed, path, log):

	assert(beta > 0)
	if (beta > 1):
		warnings.warn('beta should be <= 1, so set to 1')
		beta = 1.0
	assert(mu >= 0)
	if (mu >= 1):
		warnings.warn('mu should be < 1, so set to 0')
		mu = 0.0
	assert(start_t >= 0)
	assert(end_t > 0)
	assert(end_t > start_t)
	assert(n > 0)

	start = time.time()
	make_tsir = subprocess.run(["make"], cwd="tsir", capture_output=True, text=True)
	p = Popen(['./tsir/tsir', str(beta), str(mu), str(start_t), str(end_t), str(n), str(seed),
			  path.format("S"), path.format("I"), path.format("R"), log],
			  stdout=PIPE, stdin=PIPE, stderr=PIPE)
	o, e = p.communicate(input=bytes(nwk, encoding='utf-8'))
	print(f'Done in {time.time() - start:.2f} seconds')

	info = o.decode().split(' ')
	return float(info[0]), float(info[1]), float(info[2]), float(info[3])

def sir_ground_truth(cfg, H_cread, n_nodes, local_folder):
    print('Running SIR simulation in C for ground truth (see log file)...', end=' ')
    seed = random.getrandbits(64)
    wandb.summary["seed_ground_truth"] = seed
    R0, avg_os, sd, se = run(H_cread, beta=cfg.sir.beta, mu=cfg.sir.mu, start_t=cfg.sir.start_t, end_t=cfg.sir.end_t,
                             n=cfg.sir.n_runs, seed=seed, path=f'{local_folder}/ground_truth_{{}}.bin', log=f'{local_folder}/ground_truth.txt')
    wandb.summary["R0_ground_truth"] = R0
    wandb.summary["avg_outbreak_size_ground_truth"] = avg_os / n_nodes
    print(f' --- R0 estimate: {R0:.2f}; avg. outbreak size: {100 * avg_os / n_nodes:.2f}% '
          f'({avg_os:.2f}, std. deviation: {sd:.2f}, std. error: {se:.2f})')
    truth_S, truth_I, truth_R = (np.fromfile(f"{local_folder}/ground_truth_{state}.bin", dtype=np.int8).
                                 reshape(cfg.sir.n_runs * n_nodes, n_nodes) for state in "SIR")
    return truth_S, truth_I, truth_R

def sir_maximal_outbreak(cfg, H_cread, n_nodes, local_folder):
    print('Running SIR simulation in C for maximal outbreaks (see log file)...', end=' ')
    R0, avg_os, sd, se = run(H_cread, beta=1, mu=0, start_t=cfg.sir.start_t, end_t=cfg.sir.end_t,
                             n=1, seed=0, path=f'{local_folder}/maximal_outbreak_{{}}.bin', log=f'{local_folder}/maximal_outbreak.txt')
    wandb.summary["R0_maximal_outbreak"] = R0
    wandb.summary["avg_outbreak_size_maximal_outbreak"] = avg_os / n_nodes
    print(f' --- R0 estimate: {R0:.2f}; avg. outbreak size: {100 * avg_os / n_nodes:.2f}% '
          f'({avg_os:.2f}, std. deviation: {sd:.2f}, std. error: {se:.2f})')
    maximal_outbreak = np.fromfile(f"{local_folder}/maximal_outbreak_I.bin", dtype=np.int8).reshape(n_nodes, n_nodes)
    return maximal_outbreak

def sir_monte_carlo(cfg, H_cread, n_nodes, local_folder):
    print('Running SIR simulation in C for Monte-Carlo-based methods (see log file)...', end=' ')
    seed = random.getrandbits(64)
    wandb.summary["seed_monte_carlo"] = seed
    R0, avg_os, sd, se = run(H_cread, beta=cfg.sir.beta, mu=cfg.sir.mu, start_t=cfg.sir.start_t, end_t=cfg.sir.end_t,
                             n=cfg.sir.mc_runs, seed=seed, path=f'{local_folder}/monte_carlo_{{}}.bin', log=f'{local_folder}/monte_carlo.txt')
    wandb.summary["R0_monte_carlo"] = R0
    wandb.summary["avg_outbreak_size_monte_carlo"] = avg_os / n_nodes
    print(f' --- R0 estimate: {R0:.2f}; avg. outbreak size: {100 * avg_os / n_nodes:.2f}% '
          f'({avg_os:.2f}, std. deviation: {sd:.2f}, std. error: {se:.2f})')
    mc_S, mc_I, mc_R = (np.fromfile(f"{local_folder}/monte_carlo_{state}.bin", dtype=np.int8).
                        reshape(n_nodes, cfg.sir.mc_runs, n_nodes) for state in "SIR")
    return mc_S, mc_I, mc_R