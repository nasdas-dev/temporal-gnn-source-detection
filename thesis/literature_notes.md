# Literature Notes — Source Detection in Temporal Networks
---

## Kern-Papers (Core Thesis Pillars)

### Qarkaxhija et al. (2022) — De Bruijn goes Neural
- **Key idea**: Use higher-order De Bruijn graphs to model causal walks and non-Markovian dynamics
- **Method**: DBGNN; iterative line-graph construction to capture temporal-topological patterns
- **Relevance**: Central architecture; the only model explicitly handling causal walks in temporal graphs
- **Status**: Implemented (Adaptation for Source Detection)

### Rey et al. (2025) — Directed Acyclic Graph Convolutional Networks
- **Key idea**: Convolutional learning specifically for DAGs using causal graph filters
- **Method**: DCN / PDCN; spectral-domain representation that bypasses nilpotent adjacency issues
- **Relevance**: Natural architecture for the Temporal Event Graph (TEG) representation
- **Status**: Implemented

### Saramäki et al. (2019) — Weighted Temporal Event Graphs
- **Key idea**: Mapping temporal network structures into static Directed Acyclic Graphs (DAGs)
- **Method**: TEG framework; nodes represent events, edges represent time-respecting paths
- **Relevance**: Essential preprocessing step to enable DAG-GNN operations on temporal data
- **Status**: Implemented (Preprocessing)

### Sterchi, Brack & Hilfiker (2025) — GNNs for Source Detection: Review and Benchmark
- **Key idea**: Critical analysis of existing GNN source detection methods and proposal of a principled architecture
- **Method**: Systematic review and benchmark against classical methods (Jordan Center, SME, etc.)
- **Relevance**: Foundational for the thesis evaluation framework and SOTA definitions
- **Status**: Implemented (Evaluation Framework)

---

## Other Relevant Papers & Suggestions

### Shah & Zaman (2011) — Rumors in a Network
- **Key idea**: Rumor centrality as maximum likelihood estimator for source
- **Method**: Analytical, based on tree-like approximation
- **Limitation**: Assumes tree structure, doesn't handle temporal networks well
- **Relevance**: Our Jordan center baseline is inspired by this

### Pinto et al. (2012) — Locating the Source of Diffusion
- **Key idea**: Source detection with partial observations (sparse observers)
- **Method**: Gaussian approximation of infection times
- **Relevance**: We also handle partial observation scenarios

### Ru et al. (2023) — Source Detection via STGCN
- **Key idea**: Backtracking network that processes temporal contact info via "Edge Textures"
- **Method**: GNN-based, kernel-based temporal aggregation with marginalization over start time $T$
- **Relevance**: State-of-the-Art reference for temporal source detection
- **Status**: Implemented

### Holme (2020) — Fast and General SIR Simulation
- **Key idea**: Efficient temporal SIR simulation on contact networks
- **Method**: C implementation, event-driven simulation
- **Relevance**: Adapted C code used for our `tsir/` simulation engine

### von Pichowski et al. (2024) — Inference of Sequential Patterns for DBGNNs
- **Key idea**: Extension of DBGNN using Hypergeometric Graph Ensembles to find anomalous temporal patterns
- **Method**: HYPA-DBGNN; statistically-informed inductive bias for temporal edge filtering
- **Relevance**: Potential "Enhanced DBGNN" variant for testing
- **Status**: Not started (Suggestion)

### Rossi et al. (2020) — Temporal Graph Networks (TGN)
- **Key idea**: Generic framework for DL on dynamic graphs using per-node memory and attention
- **Method**: TGN; combines GRU/LSTM memory modules with Graph Attention
- **Relevance**: High-performance temporal baseline with memory mechanisms
- **Status**: Not started (Suggestion)

### Dong et al. (2019) — Multiple Rumor Source Detection with GCN
- **Key idea**: First GCN application for source detection using Label Propagation (LPA) for features
- **Method**: Supervised GCN with LPA-based feature preprocessing
- **Relevance**: Classic static GNN baseline for comparison
- **Status**: Evaluated