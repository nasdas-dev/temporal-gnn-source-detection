# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for detecting epidemic sources in temporal networks. It combines fast C code (adapted from [Petter Holme's TSIR](https://arxiv.org/abs/2007.14386)) for SIR simulations with Python inference methods: Graph Neural Networks, Individual-Based Approximation (IBA), Monte Carlo mean-field, and Soft Margin.

## Environment Setup

```bash
pip install virtualenv
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Compile the C modules before first use:
```bash
mkdir -p tsir/o && cd tsir && make
mkdir -p iba/o  && cd iba  && make
```

## Typical Workflow

All experiments follow the same pipeline. Step 1 must run before the others.

```bash
# 1. Generate SIR simulations → stored as wandb artifact
python main_tsir.py --cfg exp/exp_1_vary_n/erdos_renyi/tsir.yml --data exp_1_vary_n.erdos_renyi

# 2. Run inference methods (each loads the artifact from step 1 via wandb)
python main_gnn.py   # config embedded in wandb sweep / yml
python main_iba.py
python main_mc.py
python main_sm.py

# 3. Evaluate baselines (uniform, random, degree, Jordan center)
python main_eval.py --cfg exp/exp_1_vary_n/erdos_renyi/eval.yml --data <artifact_reference>

# Visualize results
python viz/plot_vary_n.py
```

## Architecture

### Data flow

```
main_tsir.py
  → loads network (nwk/) via setup/read_network.py
  → runs C SIR (tsir/) producing ground truth + MC simulations
  → saves binary arrays + network pickle to data/<wandb_run_id>/
  → logs as a versioned wandb artifact

inference mains (gnn/iba/mc/sm)
  → load artifact via setup/data_loader.py → TSIRData dataclass
  → run method, output per-node probability scores
  → log metrics to wandb

main_eval.py
  → loads same artifact
  → runs baseline methods (eval/benchmark.py)
  → computes scores via eval/scores.py (rank_score, top_k_score, …)
```

### Key data structures

- `TSIRData` (`setup/data_loader.py`): central dataclass holding all simulation tensors:
  - `truth_S/I/R`: shape `(n_nodes, n_runs, n_nodes)` — ground-truth SIR states per source, run, node
  - `mc_S/I/R`: shape `(n_nodes, mc_runs, n_nodes)` — Monte Carlo simulations
  - `maximal_outbreak`, `possible`: binary masks for filtering candidate sources
- Binary files stored as `int8` flat arrays, loaded with `np.fromfile`

### Modules

| Module | Role |
|--------|------|
| `tsir/` | C implementation of temporal SIR + Python wrapper |
| `iba/` | C Individual-Based Approximation + Python wrapper |
| `mc/` | Monte Carlo mean-field inference (`monte_carlo.py`) |
| `sm/` | Soft Margin via Jaccard similarity (`soft_margin.py`) |
| `gnn/` | PyTorch Geometric models: `StaticGNN` (main) and `TemporalGNN` |
| `eval/` | Scoring metrics and baseline methods |
| `setup/` | Config loading, network reading, wandb run setup, artifact loading |
| `exp/` | YAML configs for each experiment/network combination |
| `viz/` | Plotting and network visualization |

### Configuration system

YAML configs (under `exp/`) are loaded into `Config` objects (`setup/read_config.py`) that support dot-notation access (e.g. `cfg.nwk.name`, `cfg.sir.beta`). Key top-level sections:

- `nwk`: network parameters (type, name, t_max, n, directed, …)
- `sir`: simulation parameters (beta, mu, start_t, end_t, n_runs, mc_runs)
- `gnn` / `eval`: method-specific hyperparameters

### GNN model (`gnn/static_gnn.py`)

`StaticGNN` is a configurable `torch.nn.Module` with optional preprocessing MLP → graph convolutions (GraphConv/GCN/SAGE/GAT/GIN) → optional postprocessing MLP → final linear layer. Supports skip connections (raw node features concatenated before output), batch normalization, PReLU activations, and dropout. Output is log-softmax over nodes.

### wandb integration

Every run (tsir, gnn, iba, mc, sm, eval) is tracked in the `source-detection` wandb project. Data artifacts are version-controlled; downstream runs reference upstream artifact names (e.g. `exp_1_vary_n.erdos_renyi:v0`).

## Gitignored directories (create manually if needed)

```
data/    nwk/    logs/    plots/    playg/
```
