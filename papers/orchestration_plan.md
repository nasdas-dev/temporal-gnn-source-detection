# Orchestration Plan — Unified Experiment Pipeline

## The Problem

The current codebase has **fragmented execution paths**:
- `main_gnn.py` only runs StaticGNN, with its own training loop and wandb setup
- `toy-example/run_bn_toy.py` is a standalone script with its own SIR, training, and eval
- `gnn/temporal_gnn.py` has a self-contained `temporal_gnn()` function with its own training loop
- `gnn/training.py` and `gnn/predict.py` are shared helpers, but only used by some models
- Each model has a **different forward signature** and batching strategy
- There is no `main_tsir.py` or `main_eval.py` at the project root
- No experiment config directory (`exp/`) exists

This makes it impossible to fairly compare models on the same data with the same evaluation.

## Design Principles

1. **One command per stage** — `main_tsir.py` generates data, `main_train.py` trains any model, `main_eval.py` runs non-ML baselines
2. **Model registry** — models register by name, selected via `--model static_gnn`
3. **Unified interface** — every model wraps into a common API: `(batch_snapshots, graph_data) → [B, N] log-probs`
4. **Shared training loop** — one `Trainer` class that works for all models
5. **Shared evaluation** — identical metrics computation regardless of method
6. **Config-driven** — YAML files under `exp/` define everything; no hardcoded hyperparameters
7. **W&B lineage** — TSIR → artifacts → training runs → evaluation, all linked

## The Core Challenge: Model Interface Mismatch

Each model currently has a different forward signature:

| Model | Forward signature | Batching |
|-------|------------------|----------|
| `StaticGNN` | `forward(x, edge_index, edge_weights, batch)` | PyG-style: `x=[B*N, F]`, replicated `edge_index`, `batch=[B*N]` |
| `BacktrackingNetwork` | `forward(x, edge_index, edge_attr)` | Internal: `x=[B, N, 3]`, single `edge_index`, `edge_attr=[E, T]` |
| `TemporalGNN` | `forward(x, edge_indeces, edge_attr)` | Internal: `x=[B, N, 3]`, `edge_indeces={t: edge_index}` |
| Future: DBGNN | `forward(x, edge_index, ...)` | TBD |
| Future: DAG-GNN | `forward(x, edge_index, ...)` | TBD |

**Solution**: A `GraphData` container + per-model `prepare_batch()` function. The trainer calls a model-specific prep function that transforms the common data format into what each model needs.

---

## Target Architecture

```
project-thesis-agents/
│
├── main_tsir.py              # Stage 1: SIR simulation → W&B artifact
├── main_train.py             # Stage 2: Train any GNN model
├── main_eval.py              # Stage 3: Non-ML baselines
│
├── exp/                      # Experiment configurations
│   ├── toy_holme/            # Toy example (small temporal network)
│   │   ├── tsir.yml          # SIR parameters + network spec
│   │   ├── static_gnn.yml    # StaticGNN hyperparameters
│   │   ├── backtracking.yml  # BN hyperparameters
│   │   ├── temporal_gnn.yml  # TemporalGNN hyperparameters
│   │   └── eval.yml          # Evaluation settings
│   ├── exp_vary_n/           # Experiment: vary network size
│   │   ├── erdos_renyi/
│   │   │   ├── tsir.yml
│   │   │   ├── static_gnn.yml
│   │   │   ├── ...
│   │   │   └── eval.yml
│   │   └── barabasi_albert/
│   │       └── ...
│   └── exp_vary_beta/        # Experiment: vary infection rate
│       └── ...
│
├── gnn/
│   ├── __init__.py           # Model registry exports
│   ├── registry.py           # MODEL_REGISTRY + factory function
│   ├── static_gnn.py         # StaticGNN (existing, unchanged)
│   ├── backtracking_network.py  # BN (existing, unchanged)
│   ├── temporal_gnn.py       # TemporalGNN (existing, minor refactor)
│   ├── dbgnn.py              # De Bruijn GNN (future)
│   ├── dag_gnn.py            # DAG-GNN (future)
│   └── graph_builder.py      # Build model-specific graph inputs from NetworkX
│
├── training/
│   ├── __init__.py
│   ├── trainer.py            # Unified training loop
│   └── data.py               # SIRDataset + collate functions
│
├── eval/                     # (existing, extended)
│   ├── __init__.py
│   ├── scores.py             # (existing)
│   ├── ranks.py              # (existing)
│   ├── benchmark.py          # (existing, extended with more baselines)
│   └── evaluate.py           # Unified evaluation function
│
├── setup/                    # (existing, unchanged)
├── tsir/                     # (existing, unchanged)
├── viz/                      # (existing, extended)
└── nwk/                      # (existing, unchanged)
```

---

## Stage 1: `main_tsir.py` — Data Generation

**Purpose**: Run SIR simulations on a temporal network, save as W&B artifact.

**Invocation**:
```bash
python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme
```

**YAML config** (`exp/toy_holme/tsir.yml`):
```yaml
nwk:
  type: empirical        # or synthetic
  name: toy_example_holme
  directed: false
  # For empirical: loads nwk/<name>.csv + nwk/<name>.yml
  # For synthetic: additional params (n, p, m, seed)

sir:
  beta: 0.30
  mu: 0.20
  start_t: 0
  end_t: null            # null = use network's t_max
  n_runs: 5000           # ground-truth runs per source
  mc_runs: 500           # Monte Carlo training runs per source
```

**Pipeline**:
1. `setup_tsir_run(cfg_path)` → W&B run + Config object
2. Load network → NetworkX graph `H`
3. `make_c_readable_from_networkx(H)` → C-formatted string
4. `sir_ground_truth(...)` → `truth_{S,I,R}.bin` `[N, n_runs, N]`
5. `sir_monte_carlo(...)` → `mc_{S,I,R}.bin` `[N, mc_runs, N]`
6. `sir_maximal_outbreak(...)` → `maximal_outbreak_{S,I,R}.bin` `[N, N]`
7. Compute `possible_sources.bin`
8. Save `network.gpickle`
9. Log all as W&B artifact named `--data`

**Output**: W&B artifact `toy_holme:v0` containing all binary files + network pickle.

This stage already mostly works via `setup/setup_experiment.py` + `tsir/read_run.py`. We just need a clean `main_tsir.py` entry point.

---

## Stage 2: `main_train.py` — Unified Model Training

**Purpose**: Train any GNN model on a TSIR artifact, evaluate on ground truth, log to W&B.

**Invocation**:
```bash
# Train StaticGNN
python main_train.py --cfg exp/toy_holme/static_gnn.yml --data toy_holme:latest

# Train BacktrackingNetwork
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest

# Train TemporalGNN
python main_train.py --cfg exp/toy_holme/temporal_gnn.yml --data toy_holme:latest
```

**YAML config** (`exp/toy_holme/static_gnn.yml`):
```yaml
model: static_gnn           # Key: selects model from registry

eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]

train:
  n_mc: 500                 # MC samples to use for training (≤ mc_runs)
  n_truth: 1000             # ground-truth samples for evaluation
  reps: 1                   # number of train+eval repetitions
  test_size: 0.30
  batch_size: 128
  epochs: 500
  patience: 5
  lr: 0.001
  weight_decay: 0.0005
  seed: 42

# Model-specific hyperparameters (passed to model constructor)
static_gnn:
  num_preprocess_layers: 0
  embed_dim_preprocess: 16
  num_postprocess_layers: 0
  num_conv_layers: 4
  aggr: sum
  hidden_channels: 64
  dropout_rate: 0.2
  batch_norm: true
  skip: true
  feature_augmentation: false
  use_edge_weights: false
```

**YAML config** (`exp/toy_holme/backtracking.yml`):
```yaml
model: backtracking

eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]

train:
  n_mc: 5000
  n_truth: 1000
  reps: 1
  test_size: 0.20
  batch_size: 128
  epochs: 500
  patience: 30
  lr: 0.001
  weight_decay: 0.0001
  seed: 42

backtracking:
  hidden_dim: 32
  num_layers: 6
```

### Model Registry (`gnn/registry.py`)

```python
MODEL_REGISTRY: dict[str, type] = {}

def register_model(name: str):
    """Decorator to register a model class."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> torch.nn.Module:
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
```

### Graph Builder (`gnn/graph_builder.py`)

Each model needs different graph representations from the same NetworkX temporal graph. This module provides builder functions:

```python
def build_static_graph(H: nx.Graph, use_edge_weights: bool = False) -> dict:
    """Static projection: collapse temporal edges, optional contact-count weights."""
    # Returns: {"edge_index": [2, E], "edge_weight": [E] or None}

def build_temporal_graph(H: nx.Graph, t_max: int) -> dict:
    """Temporal: edge_index + binary activation pattern edge_attr [E, T]."""
    # Returns: {"edge_index": [2, E], "edge_attr": [E, T], "T": int}

def build_temporal_snapshots(H_array: np.ndarray, start_t: int, end_t: int,
                              group_by_time: int, directed: bool) -> dict:
    """Time-sliced snapshots: dict of edge_index per time step."""
    # Returns: {"edge_indeces": {t: edge_index}, "num_snapshots": int}

def build_de_bruijn_graph(H: nx.Graph, ...) -> dict:
    """De Bruijn graph construction."""
    # Future

def build_dag_graph(H: nx.Graph, ...) -> dict:
    """DAG event graph construction."""
    # Future
```

### Model ↔ Graph Type Mapping

Each registered model declares which graph builder it needs:

| Model name | Graph builder | Model constructor args (from config) |
|------------|--------------|--------------------------------------|
| `static_gnn` | `build_static_graph` | `num_preprocess_layers, embed_dim_preprocess, num_postprocess_layers, num_conv_layers, aggr, hidden_channels, dropout_rate, batch_norm, skip` + `num_node_features=3, num_classes=n_nodes` |
| `backtracking` | `build_temporal_graph` | `node_feat_dim=3, edge_feat_dim=T, hidden_dim, num_layers` |
| `temporal_gnn` | `build_temporal_snapshots` | `in_channels=3, hidden_channels, out_channels=1, num_snapshots` |
| `dbgnn` | `build_de_bruijn_graph` | TBD |
| `dag_gnn` | `build_dag_graph` | TBD |

### Batch Preparation

The critical piece: how to transform `(X_batch, graph_data)` into the right format for each model.

```python
# In gnn/registry.py or each model file:

BATCH_MODES = {
    "static_gnn": "pyg",           # PyG batching: replicate graph, flatten nodes
    "backtracking": "internal",     # Model handles batching: x=[B, N, F]
    "temporal_gnn": "internal",     # Model handles batching: x=[B, N, F]
    "dbgnn": "internal",            # TBD
    "dag_gnn": "internal",          # TBD
}
```

**PyG batching** (StaticGNN):
```python
def prepare_pyg_batch(X_samples, edge_index, edge_weight, n_nodes, device):
    B = X_samples.shape[0]
    x = X_samples.reshape(B * n_nodes, -1)                  # [B*N, F]
    E = edge_index.size(1)
    offsets = torch.arange(B, device=device) * n_nodes
    batched_ei = edge_index.repeat(1, B) + offsets.repeat_interleave(E).unsqueeze(0)
    batched_ew = edge_weight.repeat(B) if edge_weight is not None else None
    batch_vec = torch.arange(B, device=device).repeat_interleave(n_nodes)
    return x, batched_ei, batched_ew, batch_vec
```

**Internal batching** (BN, TemporalGNN):
```python
def prepare_internal_batch(X_samples, graph_data, device):
    # X_samples already [B, N, F] — just move to device
    return X_samples.to(device), {k: v.to(device) for k, v in graph_data.items()}
```

### Unified Training Loop (`training/trainer.py`)

```python
class Trainer:
    def __init__(self, model, model_name, graph_data, n_nodes, device, cfg):
        self.model = model
        self.model_name = model_name
        self.graph_data = graph_data
        self.batch_mode = BATCH_MODES[model_name]
        ...

    def train(self, X_train, y_train, X_val, y_val) -> tuple[list, list]:
        """Run full training with early stopping. Returns train/val loss curves."""
        ...

    def predict(self, X_truth) -> np.ndarray:
        """Run inference on ground truth. Returns [N_samples, N] probabilities."""
        ...
```

The trainer calls the right batch-preparation function based on `self.batch_mode`, then calls `self.model(...)` with the correct arguments.

### Unified Forward Call

To avoid if/else chains in the trainer, each model can define a `predict_step` classmethod or we use a simple dispatch:

```python
def forward_step(model, model_name, x_prepared, graph_prepared):
    """Dispatch to the correct forward call."""
    if BATCH_MODES[model_name] == "pyg":
        x, batched_ei, batched_ew, batch_vec = x_prepared
        return model(x, batched_ei, batched_ew, batch_vec)
    elif model_name == "backtracking":
        return model(x_prepared, graph_prepared["edge_index"], graph_prepared["edge_attr"])
    elif model_name == "temporal_gnn":
        return model(x_prepared, graph_prepared["edge_indeces"])
    # ... extensible for future models
```

### `main_train.py` Pipeline

```
1. Parse --cfg and --data
2. wandb.init(project="source-detection", job_type="train", config=...)
3. Load TSIR artifact → TSIRData + NetworkX graph H
4. Build graph inputs: graph_data = graph_builder[model_name](H, ...)
5. For each repetition:
   a. Sample MC runs → X [N*n_mc, N, 3], y [N*n_mc]
   b. Train/val split (stratified by source)
   c. Create model from registry
   d. Create Trainer
   e. trainer.train(X_train, y_train, X_val, y_val)
      → logs per-epoch train/val loss to wandb
   f. trainer.predict(X_truth) → probs [N*n_truth, N]
   g. Compute ranks, top-k, rank_score
   h. Log metrics to wandb
6. Log summary metrics (averaged over reps)
7. wandb.finish()
```

---

## Stage 3: `main_eval.py` — Non-ML Baselines

**Purpose**: Evaluate topology-based and probabilistic baselines on the same TSIR data.

**Invocation**:
```bash
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest
```

**YAML config** (`exp/toy_holme/eval.yml`):
```yaml
eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]
  n_truth: 1000           # how many ground-truth runs to evaluate

baselines:
  - uniform                # Uniform over possible sources
  - random                 # Random draw from possible sources
  - degree                 # Highest degree in infected subgraph
  - closeness              # Highest closeness centrality
  - betweenness            # Highest betweenness centrality
  - jordan_center          # Jordan center of infected subgraph
```

**Pipeline**:
1. Load TSIR artifact → TSIRData + NetworkX graph
2. For each baseline method:
   a. For each (source, run) pair in ground truth:
      - Extract infected subgraph
      - Apply baseline heuristic → ranking or probability distribution
   b. Compute ranks, top-k, rank_score
   c. Log to wandb
3. Log comparison table to wandb

This replaces the need for baselines to be computed inside each model script.

---

## W&B Lineage & Comparison

All runs link back to the same TSIR artifact, enabling direct comparison:

```
TSIR artifact: toy_holme:v0
    ├── train run: static_gnn     (job_type=train, model=static_gnn)
    ├── train run: backtracking   (job_type=train, model=backtracking)
    ├── train run: temporal_gnn   (job_type=train, model=temporal_gnn)
    ├── train run: dbgnn          (job_type=train, model=dbgnn)
    └── eval run: baselines       (job_type=eval)
```

**W&B tags for filtering**:
- `job:tsir`, `job:train`, `job:eval`
- `model:<model_name>` (for train runs)
- `data:<artifact_name>` (links to TSIR data)
- `exp:<experiment_name>` (e.g., `exp:toy_holme`, `exp:vary_n`)

**Standard summary keys** (all methods log the same keys):
```
eval/top_1, eval/top_3, eval/top_5
eval/rank_score
eval/mean_rank
eval/n_valid, eval/n_total
```

This makes W&B table/chart comparisons trivial: group by `model`, filter by `data`.

---

## Config Hierarchy

```yaml
# exp/toy_holme/static_gnn.yml
model: static_gnn              # → selects from MODEL_REGISTRY

eval:                          # → shared evaluation settings
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]

train:                         # → shared training settings
  n_mc: 500
  n_truth: 1000
  reps: 1
  test_size: 0.30
  batch_size: 128
  epochs: 500
  patience: 5
  lr: 0.001
  weight_decay: 0.0005
  seed: 42

static_gnn:                    # → model-specific (key matches model name)
  num_conv_layers: 4
  hidden_channels: 64
  ...
```

The `model` key is the **single point of dispatch**. Everything flows from it:
- Which graph builder to use
- Which model class to instantiate
- Which batch preparation to use
- Which model-specific hyperparameters to read

---

## Migration Path (What Changes)

### Files to CREATE:
| File | Purpose |
|------|---------|
| `main_tsir.py` | Entry point for SIR data generation |
| `main_train.py` | Unified training entry point (replaces `main_gnn.py`) |
| `main_eval.py` | Baseline evaluation entry point |
| `gnn/registry.py` | Model registry + `create_model()` factory |
| `gnn/graph_builder.py` | Graph construction functions per model type |
| `training/__init__.py` | Package init |
| `training/trainer.py` | Unified Trainer class |
| `training/data.py` | SIR snapshot dataset + collate |
| `eval/evaluate.py` | Unified `evaluate()` function |
| `exp/toy_holme/tsir.yml` | Example TSIR config |
| `exp/toy_holme/static_gnn.yml` | Example StaticGNN config |
| `exp/toy_holme/backtracking.yml` | Example BN config |
| `exp/toy_holme/eval.yml` | Example eval config |

### Files to MODIFY:
| File | Change |
|------|--------|
| `gnn/__init__.py` | Add registry imports, register existing models |
| `gnn/static_gnn.py` | Add `@register_model("static_gnn")` decorator |
| `gnn/backtracking_network.py` | Add `@register_model("backtracking")` decorator |
| `gnn/temporal_gnn.py` | Add `@register_model("temporal_gnn")` decorator, extract model class from function |
| `eval/benchmark.py` | Add Jordan center, degree, closeness, betweenness baselines |
| `setup/setup_method.py` | Adapt `setup_methods_run` for unified train runs |

### Files to KEEP (unchanged):
| File | Reason |
|------|--------|
| `setup/read_config.py` | Config class works fine |
| `setup/data_loader.py` | TSIRData + `load_tsir_data` work fine |
| `setup/read_network.py` | Network loading works fine |
| `setup/setup_experiment.py` | TSIR setup works fine |
| `tsir/*` | C code + Python wrapper — no changes needed |
| `eval/scores.py` | Scoring functions — no changes needed |
| `eval/ranks.py` | Rank computation — no changes needed |
| `nwk/*` | Network data — no changes needed |

### Files to DEPRECATE:
| File | Replaced by |
|------|-------------|
| `main_gnn.py` | `main_train.py` (unified) |
| `gnn/training.py` | `training/trainer.py` |
| `gnn/predict.py` | `training/trainer.py` (predict method) |
| `toy-example/run_bn_toy.py` | `main_train.py --cfg exp/toy_holme/backtracking.yml` |

The toy-example directory can remain as documentation/reference but is no longer the execution path.

---

## Complete Execution Example

```bash
# === EXPERIMENT: toy_holme ===

# 1. Generate SIR data (once per network+SIR config)
python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme

# 2. Train all GNN models (each references the same artifact)
python main_train.py --cfg exp/toy_holme/static_gnn.yml --data toy_holme:latest
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest
python main_train.py --cfg exp/toy_holme/temporal_gnn.yml --data toy_holme:latest
# Future:
# python main_train.py --cfg exp/toy_holme/dbgnn.yml --data toy_holme:latest
# python main_train.py --cfg exp/toy_holme/dag_gnn.yml --data toy_holme:latest

# 3. Evaluate non-ML baselines
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest

# 4. Visualize comparison (reads from wandb)
python viz/plot_compare.py --data toy_holme
```

```bash
# === EXPERIMENT SWEEP: vary network size ===

for n in 50 100 200 500; do
    python main_tsir.py --cfg exp/vary_n/er_n${n}/tsir.yml --data vary_n.er_n${n}
    python main_train.py --cfg exp/vary_n/er_n${n}/static_gnn.yml --data vary_n.er_n${n}:latest
    python main_train.py --cfg exp/vary_n/er_n${n}/backtracking.yml --data vary_n.er_n${n}:latest
    python main_eval.py --cfg exp/vary_n/er_n${n}/eval.yml --data vary_n.er_n${n}:latest
done

python viz/plot_vary_n.py --experiment vary_n
```

---

## Implementation Order

### Phase 1: Core Infrastructure (do first)
1. `gnn/registry.py` — model registry
2. `gnn/graph_builder.py` — graph construction per model type
3. `training/trainer.py` — unified training loop
4. `training/data.py` — dataset + collate
5. Register existing models (StaticGNN, BN, TemporalGNN)

### Phase 2: Entry Points
6. `main_tsir.py` — clean entry point (wraps existing setup + tsir code)
7. `main_train.py` — unified training (replaces `main_gnn.py`)
8. `main_eval.py` — baseline evaluation
9. `eval/evaluate.py` — shared evaluation function

### Phase 3: Configs
10. Create `exp/toy_holme/` configs
11. Create `exp/` configs for each planned experiment

### Phase 4: New Models (as thesis progresses)
12. Implement + register DBGNN
13. Implement + register DAG-GNN
14. Each new model only needs: model class + graph builder function + YAML config

### Phase 5: Visualization
15. `viz/plot_compare.py` — side-by-side model comparison from wandb

---

## Key Design Decisions

### Q: Should models inherit from a base class?
**A: No.** A base class would force refactoring existing models. Instead, use the registry + external dispatch pattern. Models keep their existing forward signatures. The `Trainer` class handles the translation layer.

### Q: Should we use PyTorch Lightning?
**A: No.** The training loop is simple enough that Lightning adds complexity without benefit. A 100-line `Trainer` class is sufficient and keeps the codebase self-contained.

### Q: One config file or many per experiment?
**A: One per model + one for TSIR + one for eval baselines.** This keeps configs focused and allows running different model configs independently. The `--data` flag links them.

### Q: What about hyperparameter sweeps?
**A: Use W&B Sweeps.** The YAML config maps directly to `wandb.config`, so `wandb sweep` works out of the box. No custom sweep infrastructure needed.

### Q: Where do model outputs (probabilities) get saved?
**A: Both locally and in wandb.** Save `model_outputs.pt` to `data/<wandb_run_id>/` and optionally log as a wandb artifact for downstream analysis.
