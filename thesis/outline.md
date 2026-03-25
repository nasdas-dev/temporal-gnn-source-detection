# Thesis Outline: Source Detection in Temporal Contact Networks Using GNNs

## Chapter 1: Introduction
- 1.1 Motivation — why epidemic source detection matters
- 1.2 Problem Statement — formal definition of source detection and notation
- 1.3 Methodology Overview — GNN Architectures for Temporal Source Detection
    - 1.3.1 De Bruijn Graph Neural Networks (DBGNN)
    - 1.3.2 Backtracking Networks (BN) with Edge Textures
    - 1.3.3 DAG-GNN on Temporal Event Graphs (TEG)
- 1.4 Contributions — what this thesis adds to backtracking research
- 1.5 Thesis Structure — roadmap
- 1.6 Research Hypotheses
	- **H1: Causal Path Integrity:** TGNNs provide the highest marginal gain in structured networks where 				
	temporal sequences strictly limit the reachable set of nodes.
	- **H2: Granularity Convergence:** The superiority of temporal architectures is bounded by the temporal
	resolution of the input; accuracy converges to static baselines as time-bins increase.
	- **H3: Temporal Signal-to-Noise Resilience:** TGNNs maintain higher accuracy than static models in
	late-stage outbreaks by pruning the combinatorial explosion of paths in the aggregated contact history.

## Chapter 2: Foundations
- 2.1 Network Science and Graph Theory
    - 2.1.1 Static vs. Temporal Networks
    - 2.1.2 Causality and Causal Walks in Dynamic Graphs
- 2.2 Epidemic Spreading Models
    - 2.2.1 Compartmental Models: SI, SIR, SIS, and SIRS
    - 2.2.2 The SIR Model and Stochastic Transitions
- 2.3 Classical Source Detection Algorithms
    - 2.3.1 Rumor Centrality and Jordan Center
    - 2.3.2 Simulation-based Methods: Soft Margin Estimator (SME)
- 2.4 Graph Neural Networks (GNNs)
    - 2.4.1 Message Passing Paradigm
    - 2.4.2 GNNs for Static Source Detection (GCN, GraphSAGE, GAT)
    - 2.4.3 Transition to Temporal Source Detection

## Chapter 3: Related Work
- 3.1 Analytical Methods — Shah & Zaman, Jordan center
- 3.2 Simulation-Based Methods — Monte Carlo approaches
- 3.3 Machine Learning Methods — GNN-based approaches
- 3.4 Backtracking Approaches — Ru et al., Lokhov et al.
- 3.5 Summary and Research Gap

## Chapter 4: Methodology & Representation
- 4.1 Formalization of the Backtracking Problem
- 4.2 Data Generation via Monte Carlo Simulations (TSIR)
- 4.3 Representation of Temporal Information
    - 4.3.1 Static Projection (Baseline)
    - 4.3.2 Temporal Discretization (Timeslices)
    - 4.3.3 Edge Feature Encoding for Timestamps

## Chapter 5: Model Architectures and Implementation
- 5.1 Evaluated GNN Models
    - 5.1.1 Static Projection (Baseline)
    - 5.1.2 De Bruijn Graph Neural Networks (DBGNN)
    - 5.1.3 Kernel-based GNN / Backtracking Network (BN)
    - 5.1.4 DAG-GNN on Temporal Event Graphs (TEG)
- 5.2 Hyperparameter Optimization and Training Setup

## Chapter 6: Experimental Evaluation
- 6.1 Datasets
    - 6.1.1 Synthetic Graphs (Varying $N$ and $\beta$)
    - 6.1.2 Real-world Contact Networks
- 6.2 Evaluation Metrics
    - 6.2.1 Top-k Accuracy
    - 6.2.2 Average Error Distance and Reciprocal Rank
    - 6.2.3 Resistance Score for Uncertainty Quantification
- 6.3 Results and Analysis
    - 6.3.1 Performance Comparison: GNNs vs. Classical Heuristics
    - 6.3.2 Impact of Network Density and Observation Time $T$
    - 6.3.3 Robustness to Missing Information (Links/Node States)

## Chapter 7: Discussion
- 7.1 Interpretation of Results
- 7.2 Advantages of Temporal Architectures over Static Models
- 7.3 Ablation Studies (e.g., impact of Edge Textures)

## Chapter 8: Conclusion
- 8.1 Summary of Contributions
- 8.2 Limitations
- 8.3 Future Work

## Appendices
- A.1 Implementation Details
- A.2 Extended Results Tables