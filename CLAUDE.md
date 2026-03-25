# CLAUDE.md — Source Detection in Temporal Networks

## Project Overview
Research codebase for detecting epidemic sources in temporal networks. Combines fast C code (adapted from Petter Holme's TSIR) for SIR simulations with Python inference methods: Graph Neural Networks.

## ⚡ Agent System
This project uses specialized subagents in `.claude/agents/`. Claude should auto-delegate tasks to the appropriate agent based on task type:

| Task Type | Agent | When to Use |
|-----------|-------|-------------|
| Implementing GNN models from papers | `model-builder` | Any new model architecture, reimplementation, or modification |
| Reviewing code against papers | `paper-reviewer` | After implementation, before committing |
| Writing thesis sections | `thesis-writer` | Drafting, editing, or restructuring thesis text |

### Agent Workflow
For implementing a new model from a paper:
1. **model-builder** agent reads the paper spec in `papers/` and implements in `gnn/`
2. **paper-reviewer** agent compares implementation to paper architecture
3. Fix any issues identified by reviewer
4. Run training pipeline and evaluate
5. **thesis-writer** agent drafts the results/methods section

## Typical Workflow
All experiments follow the same pipeline. Step 1 must run before the others.
```bash
# 1. Generate SIR simulations → stored as wandb artifact
python main_tsir.py --cfg exp/exp_1_vary_n/erdos_renyi/tsir.yml --data exp_1_vary_n.erdos_renyi

# 2. Run inference methods (each loads the artifact from step 1 via wandb)
python main_gnn.py   # config embedded in wandb sweep / yml

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

inference main (gnn)
  → load artifact via setup/data_loader.py → TSIRData dataclass
  → run method, output per-node probability scores
  → log metrics to wandb

run_bn_toy_example.py
  → Implements and runs BacktrackingNetwork.py based on Ru et al.
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
| `gnn/` | PyTorch Geometric models: `StaticGNN`, `TemporalGNN`, `BacktrackingNetwork` |
| `setup/` | Config loading, network reading, wandb run setup, artifact loading |
| `eval/` | Scoring functions: rank_score, top_k_score, etc. |
| `viz/` | Plotting utilities |

### Configuration system
YAML configs (under `exp/`) are loaded into `Config` objects (`setup/read_config.py`) that support dot-notation access (e.g. `cfg.nwk.name`, `cfg.sir.beta`). Key top-level sections:
- `nwk`: network parameters (type, name, t_max, n, directed, …)
- `sir`: simulation parameters (beta, mu, start_t, end_t, n_runs, mc_runs)
- `gnn` / `eval`: method-specific hyperparameters

### GNN model (`gnn/static_gnn.py`)
`StaticGNN` is a configurable `torch.nn.Module` with optional preprocessing MLP → graph convolutions (GraphConv/GCN/SAGE/GAT/GIN) → optional postprocessing MLP → final linear layer. Supports skip connections (raw node features concatenated before output), batch normalization, PReLU activations, and dropout. Output is log-softmax over nodes.

### wandb integration
Every run (tsir, gnn, iba, mc, sm, eval) is tracked in the `source-detection` wandb project. Data artifacts are version-controlled; downstream runs reference upstream artifact names (e.g. `exp_1_vary_n.erdos_renyi:v0`).

## Reference Files
- `papers/` — Paper specifications and architecture notes for each model
- `thesis/` — Thesis draft sections in markdown
- `thesis/literature_notes.md` — Literature review notes and paper summaries
- `thesis/outline.md` — Thesis structure and chapter plan

## Code Style & Conventions
- Python 3.10+, PyTorch 2.x, PyTorch Geometric
- Type hints on all function signatures
- Docstrings in NumPy format
- Config via YAML, never hardcode hyperparameters
- All experiments tracked in wandb
- Tests in `tests/` using pytest
