import os
import subprocess
import warnings
from subprocess import Popen, PIPE
import time
import numpy as np
import wandb


def make_c_readable_from_nparray(H_array, end_t, n_nodes): # H_array must be sorted by time
	"""Assembling the temporal network in the format of the C code (nr_contacts can be 0).:
	n_nodes end_t
	t=0 nr_contacts
	u1 v1
	...
	uk vk
	t=1 nr_contacts
	...
	"""
	nwk = str(n_nodes) + ' ' + str(end_t) + '\n'
	nr_contacts = 0 # at time t
	contacts = '' # at time t
	now = 0
	for u, v, t in H_array:
		while t > now: # while loop only to fill time steps without contacts
			nwk += str(now) + ' ' + str(nr_contacts) + '\n'
			nwk += contacts
			now += 1
			nr_contacts = 0
			contacts = ''
		nr_contacts += 1
		contacts += str(u) + ' ' + str(v) + '\n'
	if t < end_t:
		print(f'Warning: t_max={t} of the network is smaller than end_t={end_t} of the algorithm!')

	nwk += str(now) + ' ' + str(nr_contacts) + '\n' # last time step
	nwk += contacts
	while now < end_t:
		now += 1
		nwk += str(now) + ' ' + str(0) + '\n'
	return nwk


def iba_in_c(nwk, beta, mu, start_t, end_t, directed, path, log):

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

	start = time.time()
	make_iba = subprocess.run(["make"], cwd="iba", capture_output=True, text=True)
	p = Popen(['./iba/iba', str(beta), str(mu), str(start_t), str(end_t), str(int(directed)),
			  path.format("S"), path.format("I"), path.format("R"), log],
			  stdout=PIPE, stdin=PIPE, stderr=PIPE)
	o, e = p.communicate(input=bytes(nwk, encoding='utf-8'))
	print(f'Done in {time.time() - start:.2f} seconds')

	info = o.decode()
	print(info, end='')


def iba(tsir_config, H_cedges, n_nodes):
	print('Running individual-based-approximation (IBA) C code (see log file)...', end=' ')
	os.makedirs(f"data/{wandb.run.id}", exist_ok=True)
	iba_in_c(H_cedges, beta=tsir_config["sir"]["beta"], mu=tsir_config["sir"]["mu"], start_t=tsir_config["sir"]["start_t"], end_t=tsir_config["sir"]["end_t"],
			 directed=tsir_config["nwk"]["directed"], path=f"data/{wandb.run.id}/iba_result_{{}}.bin", log=f'data/{wandb.run.id}/iba_result.txt')
	iba_log_S, iba_log_I, iba_log_R = (np.fromfile(f"data/{wandb.run.id}/iba_result_{state}.bin", dtype=np.float64).
									   reshape(n_nodes, n_nodes) for state in "SIR")
	return iba_log_S, iba_log_I, iba_log_R
