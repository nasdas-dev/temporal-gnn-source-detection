# Orchestration Plan — Sonnet Instructions

> **What this is**: A complete, self-contained instruction set for Claude Sonnet to execute the master thesis project "GNN for Epidemic Source Detection on Temporal Contact Networks." Follow phases in order. Each phase has clear deliverables and verification steps.

---

## Project State Summary

### What exists and works
- `main_tsir.py` — SIR simulation → wandb artifact (fully working)
- `main_train.py` — Unified GNN training for all 3 models (fully working)
- `gnn/registry.py` — Model registry with `ModelSpec` pattern
- `gnn/graph_builder.py` — 3 builders: `build_static_graph`, `build_temporal_activation`, `build_temporal_snapshots`
- `training/trainer.py` — Unified `Trainer` class with `fit()` + `predict_from_tensor()`
- `training/data.py` — `SIRDataset` (PyTorch Dataset)
- `gnn/static_gnn.py` — StaticGNN (baseline, static projection)
- `gnn/backtracking_network.py` — BacktrackingNetwork (Ru et al., AAAI 2023)
- `gnn/temporal_gnn.py` — TemporalGNN (SAGEConv per time-slice)
- `setup/` — Config loading, network reading, wandb setup, artifact loading
- `eval/` — `scores.py`, `ranks.py`, `benchmark.py` (scoring functions)
- `exp/` — YAML configs for toy_holme, karate_static, france_office, lyon_ward, malawi
- `nwk/` — 8 temporal contact networks (CSV + YAML metadata)
- `tsir/` — C SIR simulation code + Python wrapper

### What is missing (your job to build)
1. **`main_eval.py`** — Baseline evaluation entry point (referenced but not implemented)
2. **DBGNN model** — De Bruijn GNN (Qarkaxhija et al., 2022) — only utility in `utils/make_de_bruijn_graph.py`
3. **DAG-GNN model** — DAG convolutions on temporal event graph (Rey et al., 2025)
4. **Experiment sweep configs** — Systematic configs for varying N, beta, network type
5. **Visualization pipeline** — Comparison plots pulling from wandb
6. **Thesis sections** — Methods, results, discussion chapters

### Key architectural contracts (DO NOT BREAK)
- Every model registers via `MODEL_REGISTRY` in `gnn/__init__.py`
- Every model has a `ModelSpec(cls, forward_fn, builder_fn, build_fn)`
- Every `forward_fn` signature: `(model, x_batch [B,N,F], graph_data dict, device) → [B,N] log-probs`
- Every `builder_fn` signature: `(H: nx.Graph, **kwargs) → dict` with key `"n_nodes"`
- Every `build_fn` signature: `(model_cfg dict, n_nodes int, graph_data dict) → nn.Module`
- Output is always `log_softmax` over nodes
- Training uses NLL loss
- All configs are YAML under `exp/<network_name>/<model>.yml`
- All experiments tracked in wandb project `source-detection`

---

## Phase 1: Complete the Evaluation Infrastructure

**Goal**: `main_eval.py` that runs all non-ML baselines on any TSIR artifact, logging to wandb with the exact same metrics as `main_train.py`.

### 1.1 Create `main_eval.py`

```
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest
```

**Read these files first**:
- `main_train.py` (follow the same pattern for wandb init, data loading, metric logging)
- `eval/benchmark.py` (existing baseline functions)
- `eval/scores.py` + `eval/ranks.py` (metric computation)
- `exp/toy_holme/eval.yml` (config format)

**Implementation**:
1. Load TSIR artifact (same as `main_train.py` lines 96-115)
2. For each baseline in `cfg.baselines`:
   - Compute scores/probabilities for all (source, run) ground-truth pairs
   - Compute ranks via `compute_ranks()`
   - Compute `top_k_score()` and `rank_score()` with same params as training eval
   - Log to wandb with keys: `eval/top_{k}`, `eval/rank_score_off{o}`, tagged `model:<baseline_name>`
3. Log wandb summary with same key format as `main_train.py`

**Baselines to support** (from `eval/benchmark.py` + new):
- `uniform` — uniform probability over possible sources
- `random` — random draw from possible sources
- `degree` — highest degree node in infected subgraph
- `closeness` — closeness centrality in infected subgraph
- `betweenness` — betweenness centrality in infected subgraph
- `jordan_center` — Jordan center of infected subgraph

For `degree`/`closeness`/`betweenness`/`jordan_center`: extract the infected subgraph from the ground-truth snapshot, compute the centrality metric on it, and convert to a probability distribution over nodes (highest centrality = highest probability). Use `networkx` centrality functions.

**Eval YAML format** (already exists in `exp/toy_holme/eval.yml`):
```yaml
eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]
  n_truth: 1000

baselines:
  - uniform
  - random
  - degree
  - closeness
  - betweenness
  - jordan_center
```

### 1.2 Verification
```bash
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest
```
- Check wandb: each baseline should have `eval/top_1`, `eval/top_3`, `eval/top_5`, `eval/rank_score_off0`
- `uniform` top-1 should be roughly `1/n_nodes`
- `jordan_center` should outperform `uniform`

---

## Phase 2: Implement DBGNN (De Bruijn Graph Neural Network)

**Goal**: Implement the De Bruijn GNN from Qarkaxhija et al. (2022) and register it in the pipeline.

### 2.1 Read and understand
- Read `utils/make_de_bruijn_graph.py` (existing De Bruijn graph construction utility)
- Read the paper: "De Bruijn goes Neural: Causality-Aware Graph Neural Networks for Time Series Data on Dynamic Graphs" (arXiv:2209.08311)
- Read `papers/` for any existing spec file; if none exists, create one first via `/analyze-paper`
- Read `gnn/backtracking_network.py` as reference for how temporal models integrate

### 2.2 Create `gnn/graph_builder.py` addition: `build_de_bruijn_graph()`

The De Bruijn graph transforms a temporal contact network into a higher-order graph where:
- **Nodes** = length-k walks (sequences of k consecutive contacts)
- **Edges** = overlap between walks (suffix of one = prefix of another)
- Node features = aggregated SIR states along the walk

**Key parameters** (from YAML config):
- `k`: order of the De Bruijn graph (walk length, typically 2 or 3)
- `use_weights`: whether to weight edges by walk frequency

**Return format** (must match contract):
```python
{
    "n_nodes": int,           # Original graph nodes (for output mapping)
    "db_n_nodes": int,        # De Bruijn graph nodes
    "edge_index": Tensor,     # [2, E_db] De Bruijn graph edges
    "walk_node_map": Tensor,  # [db_n_nodes, k] mapping DB nodes → original node sequences
    "k": int,                 # Walk order
}
```

### 2.3 Create `gnn/dbgnn.py`

**Architecture**:
1. **Input projection**: Map original SIR features → De Bruijn node features via `walk_node_map`
   - For each DB node (which represents a walk v1→v2→...→vk), aggregate features of v1...vk
   - Aggregation: concatenation or mean of node features along the walk
2. **GNN layers**: Standard message passing (GCN/GAT/SAGE) on the De Bruijn graph
3. **Readout**: Map De Bruijn node scores back to original nodes
   - Each original node v has multiple DB nodes containing it
   - Aggregate (mean/max) DB node outputs that map back to v
4. **Output**: `log_softmax` over original nodes `[B, N]`

**Forward signature**: `forward(x, edge_index, walk_node_map)` where x is `[B, N, 3]` original SIR states.

### 2.4 Register in pipeline

In `gnn/__init__.py`, add:
```python
from .dbgnn import DBGNN

MODEL_REGISTRY["dbgnn"] = ModelSpec(
    cls=DBGNN,
    forward_fn=dbgnn_forward,      # defined in training/trainer.py
    builder_fn=build_de_bruijn_graph,
    build_fn=build_dbgnn,
)
```

Add `dbgnn_forward()` to `training/trainer.py` following the pattern of `backtracking_forward()`.

Add `build_dbgnn()` function (model constructor from config).

### 2.5 Create config: `exp/toy_holme/dbgnn.yml`
```yaml
model: dbgnn

eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]
  n_truth: 1000

train:
  n_mc: 500
  reps: 3
  test_size: 0.20
  batch_size: 128
  epochs: 500
  patience: 15
  lr: 0.001
  weight_decay: 0.0005
  seed: 42

dbgnn:
  k: 2                    # De Bruijn walk order
  hidden_channels: 64
  num_conv_layers: 4
  conv_type: sage          # gcn, sage, gat
  dropout_rate: 0.2
  readout_agg: mean        # mean, max, sum
```

### 2.6 Verification
```bash
python main_train.py --cfg exp/toy_holme/dbgnn.yml --data toy_holme:latest
```
- Model should train without errors
- Loss should decrease
- Top-1 should be > uniform baseline

---

## Phase 3: Implement DAG-GNN (Directed Acyclic Graph Convolution)

**Goal**: Implement DAG-GNN from Rey et al. (2025) operating on the temporal event graph.

### 3.1 Read and understand
- Read the paper: "Directed Acyclic Graph Convolutional Networks" (arXiv:2506.12218)
- Read about temporal event graphs: Saramäki et al. (2019)
- Read `gnn/backtracking_network.py` for integration pattern

### 3.2 Create `gnn/graph_builder.py` addition: `build_dag_event_graph()`

The temporal event graph (TEG) transforms contacts into a DAG:
- **Nodes** = individual contact events (u, v, t)
- **Directed edges** = causal connections: event (u,v,t1) → event (v,w,t2) where t1 < t2
  - A contact can causally enable a later contact if they share a node and respect time ordering
- This naturally forms a DAG (no cycles due to time ordering)

**Key parameters**:
- `delta_t`: maximum time gap for causal connection (if None, any t1 < t2)
- `include_self_loops`: whether (u,v,t1) → (u,w,t2) counts (same source node)

**Return format**:
```python
{
    "n_nodes": int,              # Original graph nodes
    "dag_n_nodes": int,          # Event graph nodes (= number of contacts)
    "dag_edge_index": Tensor,    # [2, E_dag] directed edges in event DAG
    "event_to_node": Tensor,     # [dag_n_nodes, 2] maps each event to (u, v) in original graph
    "event_times": Tensor,       # [dag_n_nodes] time of each event
    "dag_node_features": None,   # Will be computed from SIR states at forward time
}
```

### 3.3 Create `gnn/dag_gnn.py`

**Architecture** (from Rey et al.):
1. **Event feature construction**: For event (u,v,t), combine SIR features of u and v at observation time
   - Input: `[B, N, 3]` SIR states → project to event features `[B, E_events, F]`
2. **DAG convolution layers**: Message passing that respects DAG topology
   - Process nodes in topological order (guaranteed since DAG)
   - Each layer: aggregate from predecessors only (not successors)
   - The paper defines specific aggregation for DAGs that differs from standard MPNN
3. **Readout**: Aggregate event-level outputs back to original nodes
   - Each original node v is involved in multiple events
   - Aggregate (mean/attention) over all events involving v
4. **Output**: `log_softmax` over original nodes `[B, N]`

**Key implementation detail**: Since the DAG has a topological ordering, convolution can be done efficiently layer-by-layer following this ordering, rather than full matrix multiplication.

### 3.4 Register in pipeline (same pattern as Phase 2)

### 3.5 Create configs for all networks
```
exp/toy_holme/dag_gnn.yml
exp/karate_static/dag_gnn.yml
exp/france_office/dag_gnn.yml
...
```

### 3.6 Verification
```bash
python main_train.py --cfg exp/toy_holme/dag_gnn.yml --data toy_holme:latest
```

---

## Phase 4: Systematic Experiment Configuration

**Goal**: Create a complete set of experiment configs so that every model runs on every network under identical conditions.

### 4.1 Network × Model matrix

Every cell in this matrix needs a YAML config:

| Network | StaticGNN | BacktrackingNet | TemporalGNN | DBGNN | DAG-GNN | Baselines |
|---------|-----------|-----------------|-------------|-------|---------|-----------|
| toy_holme | ✅ | ✅ | ✅ | TODO | TODO | TODO(main_eval) |
| karate_static | ✅ | ✅ | ✅ | TODO | TODO | TODO |
| france_office | ✅ | ✅ | ✅ | TODO | TODO | TODO |
| lyon_ward | TODO | TODO | TODO | TODO | TODO | TODO |
| malawi | TODO | TODO | TODO | TODO | TODO | TODO |
| students | TODO | TODO | TODO | TODO | TODO | TODO |
| escort | TODO | TODO | TODO | TODO | TODO | TODO |
| pig_data | TODO | TODO | TODO | TODO | TODO | TODO |

### 4.2 Create missing TSIR configs

For each network, there should be a `tsir.yml`. Read the network's `nwk/<name>.yml` to get parameters (directed, time_steps, etc.) and create appropriate SIR parameters.

**Guidelines for SIR parameters per network size**:
- Small networks (< 50 nodes): `n_runs=5000, mc_runs=500, beta=0.30, mu=0.20`
- Medium networks (50-200 nodes): `n_runs=2000, mc_runs=300, beta=0.20, mu=0.15`
- Large networks (200+ nodes): `n_runs=1000, mc_runs=200, beta=0.15, mu=0.10`

Adjust beta/mu based on network density — denser networks need lower beta to avoid trivial outbreaks.

### 4.3 Create missing model configs

For each `exp/<network>/`, ensure all 5 model YAMLs + eval.yml exist. Copy from `exp/toy_holme/` as template and adjust:
- `train.n_mc` — scale with network size
- `train.batch_size` — reduce for large networks (memory)
- `train.epochs/patience` — increase for larger networks
- Model-specific params stay the same initially (tune later)

### 4.4 Create experiment sweep configs

Create systematic experiment directories:

```
exp/
├── toy_holme/           # Quick smoke test
├── karate_static/       # Small real network
├── france_office/       # Medium temporal
├── lyon_ward/           # Medium temporal
├── malawi/              # Medium temporal
├── students/            # Large temporal
├── escort/              # Large temporal
├── pig_data/            # Very large temporal
│
├── sweep_vary_beta/     # Hypothesis H2: vary temporal resolution
│   ├── toy_holme_beta010/
│   ├── toy_holme_beta020/
│   ├── toy_holme_beta030/
│   ├── toy_holme_beta050/
│   └── ...
│
└── sweep_vary_observation/  # Hypothesis H3: vary observation time
    ├── france_office_t25/
    ├── france_office_t50/
    ├── france_office_t75/
    ├── france_office_t100/
    └── ...
```

### 4.5 Create `run_experiment.sh`

A master script that runs the full pipeline for one network:

```bash
#!/bin/bash
# Usage: ./run_experiment.sh <network_name>
# Example: ./run_experiment.sh toy_holme

NWK=$1
EXP_DIR="exp/${NWK}"

echo "=== Stage 1: TSIR simulation ==="
python main_tsir.py --cfg ${EXP_DIR}/tsir.yml --data ${NWK}

echo "=== Stage 2: GNN training ==="
for model in static_gnn backtracking temporal_gnn dbgnn dag_gnn; do
    if [ -f "${EXP_DIR}/${model}.yml" ]; then
        echo "--- Training ${model} ---"
        python main_train.py --cfg ${EXP_DIR}/${model}.yml --data ${NWK}:latest
    fi
done

echo "=== Stage 3: Baselines ==="
python main_eval.py --cfg ${EXP_DIR}/eval.yml --data ${NWK}:latest

echo "=== Done: ${NWK} ==="
```

And a full sweep runner:

```bash
#!/bin/bash
# run_all_experiments.sh
for nwk in toy_holme karate_static france_office lyon_ward malawi; do
    ./run_experiment.sh ${nwk}
done
```

### 4.6 Verification
- Every `exp/<network>/` directory has: `tsir.yml`, `static_gnn.yml`, `backtracking.yml`, `temporal_gnn.yml`, `dbgnn.yml`, `dag_gnn.yml`, `eval.yml`
- `run_experiment.sh toy_holme` completes without errors
- Wandb shows all models + baselines for toy_holme with comparable metrics

---

## Phase 5: Visualization Pipeline

**Goal**: Automated comparison plots that pull results from wandb.

### 5.1 Create `viz/plot_compare.py`

Fetches wandb runs for a given data artifact and produces:
1. **Bar chart**: Top-1/3/5 accuracy per model (grouped bars)
2. **Table**: All metrics per model (LaTeX-formatted for thesis)
3. **Box plot**: Rank distribution per model

```bash
python viz/plot_compare.py --data toy_holme --output figures/toy_holme_comparison.pdf
```

### 5.2 Create `viz/plot_sweep.py`

For sweep experiments (vary beta, vary observation time):
1. **Line plot**: Top-1 accuracy vs. beta/observation_time, one line per model
2. **Heatmap**: Model × parameter value → Top-1 accuracy

```bash
python viz/plot_sweep.py --experiment sweep_vary_beta --metric eval/top_1 --output figures/sweep_beta.pdf
```

### 5.3 Create `viz/plot_networks.py`

Network statistics table + visualization:
- Number of nodes, edges, contacts, time steps, density
- For each network: small graph visualization

### 5.4 Verification
- `plot_compare.py` produces a readable PDF with all models compared
- Figures are publication-quality (proper labels, legends, font sizes)

---

## Phase 6: Thesis Writing

**Goal**: Draft all remaining thesis chapters using the thesis-writer agent.

### 6.1 Chapter priority order

1. **Chapter 4: Methodology** — Formalization, data generation, temporal encoding strategies
2. **Chapter 5: Model Architectures** — All 5 models described formally with equations
3. **Chapter 6: Experimental Setup** — Datasets, metrics, hyperparameters (write BEFORE running experiments)
4. **Chapter 6: Results** — After experiments complete, fill in results
5. **Chapter 7: Discussion** — Interpret results, test hypotheses H1-H3
6. **Chapter 1: Introduction** — Refine after results are known
7. **Chapter 2: Foundations** — Can be written any time
8. **Chapter 3: Related Work** — Refine existing draft
9. **Chapter 8: Conclusion** — Write last

### 6.2 For each chapter

Use the `thesis-writer` agent. Always provide:
- The chapter topic and section numbers from `thesis/outline.md`
- Relevant code files to reference
- Any experiment results (wandb links or result files)
- The target length (typically 3-5 pages per section)

Write to `thesis/<chapter_number>_<name>.md`.

### 6.3 Key content requirements

**Chapter 4 must include**:
- Formal problem definition (temporal contact network G(V, E^t), SIR model, observation snapshot)
- How TSIR generates training data (Monte Carlo process)
- The 4 temporal encoding strategies with formal definitions:
  1. Static projection: G_static = (V, E_agg) where (u,v) ∈ E_agg iff ∃t: (u,v,t) ∈ E^t
  2. Edge texture encoding: Each edge carries binary vector a_e ∈ {0,1}^T
  3. Time-sliced snapshots: {G_t = (V, E_t)}_{t=1}^T
  4. De Bruijn higher-order: Walk-based node expansion
  5. DAG event graph: Events as nodes with causal edges

**Chapter 5 must include**:
- For each model: full architecture diagram, layer-by-layer equations, input/output shapes
- Reference to paper + any modifications made for this project
- Complexity analysis (number of parameters as function of N, T, hidden_dim)

**Chapter 6 must include**:
- Dataset table (all 8 networks with statistics)
- Full hyperparameter table per model
- Results table: model × network × metric
- Statistical significance (confidence intervals over repetitions)

---

## Phase 7: Ablation Studies & Hypothesis Testing

**Goal**: Targeted experiments to test the 3 research hypotheses.

### 7.1 H1: Causal Path Integrity

**Experiment**: Compare models on networks with varying temporal structure.
- Take the same static graph topology
- Generate temporal versions with different contact patterns:
  - Random timestamps (no causal structure)
  - Sequential timestamps (strong causal structure)
  - Clustered timestamps (moderate causal structure)
- Prediction: Temporal models (BN, DBGNN, DAG-GNN) should show largest gains on sequential patterns

### 7.2 H2: Granularity Convergence

**Experiment**: Vary the `group_by_time` parameter in TemporalGNN and equivalent binning in other temporal models.
- `group_by_time = 1` → full temporal resolution
- `group_by_time = 2, 5, 10, ...` → coarser bins
- `group_by_time = T_max` → equivalent to static projection
- Prediction: Performance should monotonically decrease toward static baseline as bins increase

### 7.3 H3: Temporal Signal-to-Noise Resilience

**Experiment**: Vary observation time `end_t` in TSIR config.
- Early observation (25% of T): few infected nodes, strong temporal signal
- Mid observation (50% of T): moderate infection, some signal
- Late observation (75-100% of T): many infected, weak temporal signal
- Prediction: Temporal models should degrade more gracefully than static

### 7.4 Create ablation configs

For each hypothesis, create experiment directories under `exp/ablation_H1/`, `exp/ablation_H2/`, `exp/ablation_H3/` with appropriate TSIR and model configs.

---

## Execution Order Summary

```
Phase 1 → main_eval.py (1-2 hours)
Phase 2 → DBGNN implementation (4-8 hours)
Phase 3 → DAG-GNN implementation (4-8 hours)
Phase 4 → All experiment configs (2-3 hours)
Phase 5 → Visualization pipeline (2-3 hours)
Phase 6 → Thesis writing (ongoing, interleaved)
Phase 7 → Ablation studies (after Phase 4 experiments complete)
```

### Critical path
```
Phase 1 ──→ Phase 4 ──→ Run experiments ──→ Phase 5 ──→ Phase 6 (results)
Phase 2 ──→ Phase 4 (needs DBGNN configs)
Phase 3 ──→ Phase 4 (needs DAG-GNN configs)
Phase 6 (methods) can start immediately
Phase 7 depends on Phase 4 + results
```

---

## File Reference Quick-Lookup

| Need to understand... | Read this file |
|---|---|
| Model registration pattern | `gnn/registry.py`, `gnn/__init__.py` |
| How forward dispatch works | `training/trainer.py` (top section) |
| How graph building works | `gnn/graph_builder.py` |
| Existing model (copy pattern) | `gnn/backtracking_network.py` |
| Training loop | `training/trainer.py` → `Trainer.fit()` |
| Data loading | `setup/data_loader.py` → `load_tsir_data()` |
| Config system | `setup/read_config.py` |
| Scoring/evaluation | `eval/scores.py`, `eval/ranks.py` |
| Existing baselines | `eval/benchmark.py` |
| Network loading | `setup/read_network.py` |
| TSIR simulation | `tsir/read_run.py`, `main_tsir.py` |
| Unified training entry | `main_train.py` |
| Experiment configs | `exp/toy_holme/*.yml` (templates) |
| Thesis structure | `thesis/outline.md` |
| Literature context | `thesis/literature_notes.md` |

## Rules for Sonnet

1. **Read before writing** — Always read the existing file before modifying it
2. **Follow existing patterns** — New models MUST follow the `ModelSpec` registration pattern exactly
3. **Don't break the pipeline** — After any change, verify `main_train.py` still works with existing models
4. **Config-driven** — Never hardcode hyperparameters; always read from YAML config
5. **Same metrics everywhere** — All models and baselines must log the exact same wandb keys
6. **Test on toy_holme first** — Always validate on the smallest network before scaling up
7. **One commit per phase** — Keep changes atomic and reviewable
8. **Use the agent system** — Delegate to `model-builder` for new architectures, `paper-reviewer` for verification, `thesis-writer` for writing
9. **Preserve wandb lineage** — Every training/eval run must reference the TSIR artifact it uses
10. **Don't refactor what works** — The existing 3 models + trainer are tested and working. Don't touch them unless necessary for integration.
