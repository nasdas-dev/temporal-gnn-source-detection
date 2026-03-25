# Static Projection Baseline — Implementation Specification

**Paper**: Master Thesis Proposal — "Graph Neural Networks for Epidemic Source Detection on Temporal Contact Networks" (Dario Akhavan Safa, Prof. A. Bernstein, UZH)
**Focus**: Static Projection Baseline (Sub-task 2, bullet 1: "Implement a baseline GNN using a static projection of the temporal network")
**Source code reference**: `gnn/static_source_detection_gnn/` by Nathan Brack / Martin Sterchi (FHNW)

## Architecture Overview

The Static Projection Baseline collapses a temporal contact network G(V, E^t) into a static graph G_static(V, E) by ignoring all temporal information. Each pair of nodes that had *any* contact becomes a single undirected edge. The static graph is then fed into a configurable GNN (the `StaticGNN` class already in `gnn/static_gnn.py`) that takes one-hot SIR snapshot states as node features and outputs a log-softmax probability distribution over nodes for source prediction.

This baseline establishes a lower bound for temporal-aware models: if temporal GNN architectures (BacktrackingNetwork, de Bruijn GNN, etc.) cannot outperform this static baseline, the temporal information is not being exploited effectively.

## Input Format

### Graph Structure
- **Type**: Static undirected graph (the "static projection" of the temporal network)
- **Construction**: From temporal edge list `(u, v, t)`, build static edge set `{(u,v) : exists t s.t. (u,v,t) in E^t}`
- **Edge weights** (optional): Number of temporal contacts between u and v, i.e. `w(u,v) = |{t : (u,v,t) in E^t}|`
- **PyG format**: `edge_index [2, 2*|E|]` (both directions for undirected), optional `edge_weight [2*|E|]`

### Node Features
- **Dimensions**: `[N, 3]` per sample (one-hot SIR state)
- **Encoding**: Susceptible = [1,0,0], Infectious = [0,1,0], Recovered = [0,0,1]
- **Optional feature augmentation**: degree, betweenness, closeness centrality, clustering coefficient (+4 features = 7 total)
- **Source**: Final SIR snapshot from temporal SIR simulation (same ground truth used by all methods)

### Batching Strategy (from existing `static_source_detection_gnn`)
- Replicate `edge_index` B times with node offsets: `batched_edge_index = edge_index.repeat(1, B) + offsets`
- Flatten node features: `x = X[batch_indices].reshape(B * N, num_features)`
- Batch vector: `batch = torch.arange(B).repeat_interleave(N)`
- This creates one large disconnected graph with B components — standard PyG mini-batching

## Layer-by-Layer Architecture

The `StaticGNN` (already implemented in `gnn/static_gnn.py`) has this pipeline:

### Layer Group 1: Preprocessing MLP (optional, 0-2 layers)
- Type: Linear -> BatchNorm (opt.) -> PReLU -> Dropout
- Input dim: `num_node_features` (3 or 7)
- Output dim: `embed_dim_preprocess` (default 16)
- Purpose: Project raw SIR features into a richer embedding space

### Layer Group 2: Graph Convolutions (2-8 layers)
- Type: `GraphConv` from PyG (supports sum/mean aggregation)
- Input dim: `embed_dim_preprocess` (or `num_node_features` if no preprocessing)
- Output dim: `hidden_channels` (default 64)
- Each layer: GraphConv -> BatchNorm (opt.) -> PReLU -> Dropout
- Supports edge weights via the `edge_weight` argument

### Layer Group 3: Postprocessing MLP (optional, 0-2 layers)
- Type: Linear -> BatchNorm (opt.) -> PReLU -> Dropout
- Input/Output dim: `hidden_channels`
- Purpose: Further refine node embeddings before classification

### Layer Group 4: Final Layer
- Input dim: `hidden_channels + num_node_features` (if skip=True) or `hidden_channels`
- Output dim: 1 (per node) -> reshape to `[B, N]` -> log_softmax
- Skip connection: concatenate original node features before final projection

### Output
- Shape: `[B, N]` — log-probability of each node being the source
- Activation: `log_softmax(dim=1)`

## Loss Function

- **Type**: Negative Log-Likelihood (NLL)
- **Formula**: `loss = F.nll_loss(log_probs, y_true)`
- **Note**: The existing code also has `ResistanceLoss` (based on graph resistance distance) as an alternative, but NLL is the primary choice

## Training Details

### From existing `static_source_detection_gnn` defaults
- **Optimizer**: Adam (lr=0.001, weight_decay=5e-4)
- **Batch size**: 128
- **Epochs**: 500 (max)
- **Early stopping**: patience=5, monitoring validation loss
- **Train/val split**: 70/30, stratified by source node
- **Model selection**: Restore best model by validation loss

### Hyperparameters (configurable via CLI/Config)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden embedding size |
| `num_layers` | 4 | Number of graph conv layers |
| `num_preprocess_layers` | 0 | MLP layers before convolution |
| `embed_dim_preprocess` | 16 | Preprocessing embedding dim |
| `num_postprocess_layers` | 0 | MLP layers after convolution |
| `aggr` | "sum" | GraphConv aggregation |
| `dropout` | 0.2 | Dropout rate |
| `batch_norm` | True | Use batch normalization |
| `skip` | True | Skip connections |
| `feature_augmentation` | False | Add centrality features |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 5e-4 | Adam weight decay |
| `batch_size` | 128 | Mini-batch size |
| `epochs` | 500 | Max training epochs |
| `early_stop` | 5 | Early stopping patience |
| `test_size` | 0.30 | Validation fraction |
| `use_edge_weights` | False | Use contact count as edge weight |

### SIR Simulation Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `beta` | 0.30 | Per-contact infection probability |
| `mu` | 0.20 | Per-timestep recovery probability |
| `n_runs` | 5000 | SIR runs per source node |

## Key Equations

### Static Projection
Given temporal network G(V, E^t) with edges (u, v, t):

```
E_static = {(u,v) : exists t such that (u,v,t) in E^t}
w(u,v) = |{t : (u,v,t) in E^t}|     (optional edge weight)
```

### GraphConv Message Passing (per layer l)
```
h_v^(l) = W_1 * h_v^(l-1) + aggr_{u in N(v)} ( w(u,v) * W_2 * h_u^(l-1) )
```
where `aggr` is sum or mean aggregation.

### Classification Output
```
p(source = v | snapshot) = softmax( MLP_final( [h_v^(L) ; x_v] ) )   (if skip)
p(source = v | snapshot) = softmax( MLP_final( h_v^(L) ) )            (no skip)
```

## Integration Notes

### How this connects to the existing project pipeline

1. **SIR Simulation**: Use the same `tsir/` C-based SIR simulator as `run_bn_toy.py`. Call `make_c_readable_from_networkx()` + `tsir_run()` to produce ground truth arrays.

2. **Data format bridge**: The existing `static_source_detection_gnn` uses its own C code (`sir/sir`) with slightly different output format (flat `states.bin` with int8 values -1/0/1, and `labels.bin` with int32). The new script should use the project's TSIR pipeline which outputs `truth_S/I/R` arrays of shape `[N, N_RUNS, N]` in int8 format, then convert to one-hot `[N, N_RUNS, N, 3]` — exactly as `run_bn_toy.py` does.

3. **Model**: Use the existing `StaticGNN` class from `gnn/static_gnn.py` (already imported via `gnn/__init__.py`). No need to copy the model code.

4. **Evaluation**: Use the project's `eval/ranks.py` (`compute_expected_ranks`) and `eval/scores.py` (`top_k_score`, `rank_score`) — same as `run_bn_toy.py`.

5. **W&B logging**: Follow the same pattern as `run_bn_toy.py`:
   - `wandb.init()` with config dict
   - Log per-epoch train/val loss
   - Log evaluation tables (per-source breakdown, summary vs random baseline, rank histogram)
   - Use `wandb.summary` for final metrics

### Static projection construction
The temporal graph loaded by `load_temporal_network()` (from `run_bn_toy.py`) already produces a NetworkX graph where edges carry `'times'` lists. The static projection simply uses this graph's topology, optionally with `len(edge['times'])` as edge weights. Convert to PyG `edge_index` via `torch_geometric.utils.from_networkx()` or manual COO construction.

### Script structure (mirror `run_bn_toy.py`)
```
toy-example/run_static_gnn_toy.py

1. Config dataclass with all hyperparameters + CLI parsing
2. load_temporal_network() — reuse from run_bn_toy.py
3. build_static_projection() — NEW: collapse temporal → static, return edge_index + optional edge_weight
4. build_training_data() — reuse from run_bn_toy.py (one-hot SIR states)
5. train() — adapted from run_bn_toy.py but using StaticGNN forward(x, edge_index, edge_weights, batch)
6. evaluate() — reuse from run_bn_toy.py (compute_expected_ranks + top_k_score + rank_score)
7. main() — orchestrate pipeline with W&B
```

### Key differences from BacktrackingNetwork (`run_bn_toy.py`)

| Aspect | BacktrackingNetwork | Static Projection |
|--------|-------------------|-------------------|
| Graph input | `edge_index` + `edge_attr [E, T]` (temporal) | `edge_index` + optional `edge_weight [E]` (static) |
| Model | `BacktrackingNetwork(node_feat_dim, edge_feat_dim, hidden_dim, num_layers)` | `StaticGNN(num_preprocess_layers, embed_dim_preprocess, num_postprocess_layers, num_conv_layers, aggr, num_node_features, hidden_channels, num_classes, dropout, batch_norm, skip)` |
| Forward call | `model(x_batch, edge_index, edge_attr)` → `[B, N]` | `model(x_flat, batched_edge_index, batched_weights, batch_vec)` → `[B, N]` |
| Batching | BN handles batch dim internally (x has shape [B, N, 3]) | PyG-style: replicate graph B times, flatten all nodes into one large graph |
| Temporal info | Full activation pattern per edge | Discarded (only topology + optional contact count) |

### Batching detail (critical implementation note)

The `StaticGNN.forward()` expects `x` of shape `[B*N, F]` and uses `batch.unique().shape[0]` to infer B for the reshape. The batched edge index must offset node IDs per sample:

```python
def make_batched_edge_index(edge_index, B, n_nodes, E):
    offsets = torch.arange(B) * n_nodes
    offsets = offsets.repeat_interleave(E)
    return edge_index.repeat(1, B) + offsets.unsqueeze(0)
```

## Ambiguities & Decisions Needed

1. **Edge weights**: The original `static_source_detection_gnn` supports optional edge weights. For the static projection, we can either:
   - (a) Use unweighted edges (simplest baseline) — **recommended default**
   - (b) Weight by contact count `w(u,v) = |{t : (u,v,t) in E^t}|`
   - **Decision**: Make this configurable via `use_edge_weights` flag; default to False.

2. **Feature augmentation**: The original code supports adding centrality features (degree, betweenness, closeness, clustering). This adds 4 extra features per node.
   - **Decision**: Make configurable via `feature_augmentation` flag; default to False for fair comparison with BN.

3. **SIR model differences**: The original `static_source_detection_gnn` uses continuous-time SIR (exponential inter-event times), while the project's TSIR uses discrete-time SIR. Since we're using the project's TSIR for consistency across all methods, this is resolved.
   - **Decision**: Use project's TSIR pipeline (same as `run_bn_toy.py`).

4. **Aggregation function**: The original code uses `GraphConv` with configurable aggregation (sum, mean). The paper doesn't specify a preference.
   - **Decision**: Default to "sum" aggregation (matches original code defaults).

5. **Weight decay**: The original uses 5e-4, while `run_bn_toy.py` uses 1e-4.
   - **Decision**: Default to 5e-4 (matches the static GNN literature), but make configurable.

## Recommended Next Steps

1. Run `/implement-model` with this spec to generate `toy-example/run_static_gnn_toy.py`
2. Run paper-reviewer agent to verify the implementation matches the spec
3. Run the toy example and compare with BacktrackingNetwork results
4. thesis-writer agent to draft the methodology section for the static baseline
