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
python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme

# 2. Train GNN model (logs MRR, Top-k, Brier, entropy, credible coverage to wandb)
#    Saves eval_arrays_rep{r}.npz alongside probs for lightweight viz loading
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest
#    Override individual config values without editing YAML:
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest \
    --override train.n_mc=100 train.reps=1

# 3. Evaluate baselines (all metrics logged, eval_arrays_{baseline}.npz saved)
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest

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
| `eval/` | Scoring functions + centralised metric computation |
| `viz/` | Plotting utilities |

### Evaluation module (`eval/`)
All metric computation is centralised in `eval/metrics.py`:

- **`compute_all_metrics(probs, lik_possible, truth_S_flat, eval_cfg, n_nodes, n_runs)`**
  Returns a flat `dict[str, float]` with the full metric suite:
  `eval/mrr`, `eval/top_{k}`, `eval/rank_score_off{o}`, `eval/brier`,
  `eval/norm_brier`, `eval/norm_entropy`, `eval/cred_cov_{p_int}`, `eval/n_valid`.
  Called by both `main_train.py` and `main_eval.py`.

- **`per_sample_arrays(probs, lik_possible, truth_S_flat, eval_cfg, n_nodes, n_runs)`**
  Returns `{ranks, outbreak_sizes, sel, true_sources}` as numpy arrays.
  Saved as `data/<run_id>/eval_arrays_rep{r}.npz` (training) or
  `data/<run_id>/eval_arrays_{baseline}.npz` (baselines) for viz scripts.

### Eval config keys (all `exp/*/eval.yml` and model YAMLs)
```yaml
eval:
  min_outbreak: 2           # minimum infected nodes for a valid run
  top_k: [1, 3, 5, 10]     # top-k accuracy thresholds
  inverse_rank_offset: [0]  # rank score offset (0 = pure MRR)
  n_truth: 1000             # ground-truth runs to evaluate per rep
  credible_p: [0.80, 0.90]  # credible set coverage thresholds
```

### Configuration system
YAML configs (under `exp/`) are loaded into `Config` objects (`setup/read_config.py`) that support dot-notation access (e.g. `cfg.nwk.name`, `cfg.sir.beta`). Key top-level sections:
- `nwk`: network parameters (type, name, t_max, n, directed, …)
- `sir`: simulation parameters (beta, mu, start_t, end_t, n_runs, mc_runs)
- `gnn` / `eval`: method-specific hyperparameters

### GNN model (`gnn/static_gnn.py`)
`StaticGNN` is a configurable `torch.nn.Module` with optional preprocessing MLP → graph convolutions (GraphConv/GCN/SAGE/GAT/GIN) → optional postprocessing MLP → final linear layer. Supports skip connections (raw node features concatenated before output), batch normalization, PReLU activations, and dropout. Output is log-softmax over nodes.

### wandb integration
Every run (tsir, gnn, iba, mc, sm, eval) is tracked in the `source-detection` wandb project. Data artifacts are version-controlled; downstream runs reference upstream artifact names (e.g. `exp_1_vary_n.erdos_renyi:v0`).

### Visualisation scripts (`viz/`)
All scripts are parameterised CLI tools.  The `eval_arrays_rep{r}.npz` or
`eval_arrays_{baseline}.npz` files produced by Phases 2/3 must exist locally.

| Script | Purpose | Key args |
|--------|---------|----------|
| `viz/rank_vs_outbreak.py` | Rank vs. outbreak size (scatter + binned mean) | `--run-id --label --output` |
| `viz/topk_vs_outbreak.py` | Top-k accuracy vs. outbreak size | `--run-id --k --output` |
| `viz/training_curves.py` | Train/val NLL loss from W&B | `--run-path --output` |
| `viz/perf_vs_n.py` | Scaling: MRR/Top-k vs. N (needs W&B) | `--artifact-prefix --metric` |
| `viz/training_size_scaling.py` | MRR vs. n_mc training size (needs W&B) | `--artifact --metric` |

All scripts import shared style from `viz/style.py` and W&B helpers from `viz/wandb_utils.py`.

### Table generation (`eval/tables.py`)
```bash
# Network stats table — fully offline
python -m eval.tables network_stats --networks france_office karate_static \
    --output figures/tables/

# Benchmark table — requires finished W&B runs
python -m eval.tables benchmark --data france_office karate_static \
    --output figures/tables/
```

### Training size sweep
```bash
# Run sweep (varies n_mc over 7 values, 3 reps each)
bash run_training_size_sweep.sh --data france_office:latest

# Then visualise
python viz/training_size_scaling.py --artifact france_office
```

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
