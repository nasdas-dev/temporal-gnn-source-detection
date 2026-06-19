# Graph Neural Networks for Epidemic Source Detection on Temporal Contact Networks

Code for my master's thesis on finding the origin of an epidemic from a single
late snapshot of a temporal contact network. Outbreaks are simulated with a fast
temporal-SIR C kernel (after Holme); the inference models are graph neural
networks trained on Monte-Carlo simulations and compared against classical
heuristics.

## Setup

Needs Python 3.10+ and a C compiler (`cc` / `make`) for the SIR kernel.

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
wandb login          # simulations and runs are tracked in the "source-detection" project
```

The C code in `tsir/` is compiled automatically the first time it runs (`make`
is called from the Python wrapper), so there is no separate build step.

## Layout

| Path | What's there |
|------|--------------|
| `main_tsir.py` | simulate SIR outbreaks on a network, save them as a versioned data artifact |
| `main_optuna.py` | Optuna hyperparameter search for one model on one network |
| `main_train.py` | train a GNN and evaluate it (MRR, top-k, Brier, calibration) |
| `main_eval.py` | score the heuristic baselines on the same data |
| `run_all_experiments.py` | H1 suite — every model on every network, paired tuned/untuned |
| `run_coarse_graining_experiment.py` | H2 suite — temporal-resolution (Δt) sweep |
| `gnn/` | the models: StaticGNN, StaticMLP, TemporalGNN, BacktrackingNetwork, DBGNN |
| `tsir/` | temporal-SIR C kernel and its Python wrapper |
| `setup/` | config loading, network reading, graph reduction |
| `eval/` | metrics, scoring, baseline implementations, table generation |
| `viz/` | plotting scripts |
| `exp/<network>/` | per-network YAML configs (`tsir.yml`, one per model, `eval.yml`) |
| `nwk/` | the contact-network datasets (edge lists + metadata) |

Each model outputs a probability distribution over the nodes — the candidate
sources. Data artifacts (`data/`), run outputs (`results/`) and figures
(`figures/`) are generated at run time and are not tracked in git.

## Running one experiment by hand

The pipeline has four stages. Stage 1 must run first because the others read its
data artifact.

```bash
# 1. simulate outbreaks (writes a wandb artifact named "lyon_ward")
python main_tsir.py --cfg exp/lyon_ward/tsir.yml --data lyon_ward

# 2. tune one model (exports results/optuna/<study>/best_config.yml)
python main_optuna.py --cfg exp/lyon_ward/backtracking.yml --data lyon_ward:latest \
    --n-trials 30

# 3. train on the best config, evaluate on a held-out truth window
python main_train.py --cfg results/optuna/<study>/best_config.yml --data lyon_ward:latest

# 4. score the heuristic baselines on the same data
python main_eval.py --cfg exp/lyon_ward/eval.yml --data lyon_ward:latest
```

Individual config values can be overridden without editing the YAML:

```bash
python main_train.py --cfg exp/lyon_ward/backtracking.yml --data lyon_ward:latest \
    --override train.n_mc=100 train.reps=1
```

## H1 — does temporal structure help?

`run_all_experiments.py` runs the full comparison: each model (StaticGNN,
StaticMLP, TemporalGNN, BacktrackingNetwork, DBGNN k2/k3) and the heuristic
baselines on each network at R0 ≈ 2, with paired tuned/untuned finals on a shared
held-out window. Output lands in `results/thesis_final/<run-name>/`.

```bash
# quick smoke test on one network
python run_all_experiments.py --preset fast --networks lyon_ward --models static_gnn

# full paired run on a set of networks (3 seeds, 30-trial HPO)
python run_all_experiments.py \
    --networks lyon_ward malawi france_office \
    --r0 r0_20 --preset tuner --hpo-trials 30
```

## H2 — does the advantage fade as time is coarsened?

`run_coarse_graining_experiment.py` walks the temporal models down a Δt ladder
(native resolution → fully collapsed, i.e. static) and tracks how their lead over
the StaticGNN shrinks, together with the compute cost of each representation.

```bash
# quick check on one network
python run_coarse_graining_experiment.py --preset fast --no-hpo --networks lyon_ward

# full Δt sweep
python run_coarse_graining_experiment.py --networks lyon_ward malawi --preset tuner
```

Pass `--dry-run` to either runner to print the planned commands without running
them.
