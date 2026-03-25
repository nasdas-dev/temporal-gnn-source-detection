# Chapter 5: Model Architectures and Implementation

## 5.1 Backtracking Network (Ru et al., 2023)

The Backtracking Network (BN) of Ru et al. \cite{Ru2023} is the first graph neural network method designed specifically for epidemic source detection on temporal contact networks. It is the primary temporal GNN baseline in this thesis and the architecture against which the De Bruijn GNN and the TEG-DAG-GNN are evaluated. We describe its problem formulation, input encoding, architecture, training procedure, and known limitations in full detail.

### 5.1.1 Problem Formulation

The temporal contact network is represented as a sequence of static snapshot graphs $\mathcal{G} = \{G_0, G_1, \ldots, G_T\}$ sharing a common node set $\mathcal{V}$ of size $N$. Each snapshot $G_t = (\mathcal{V}, E_t)$ records which pairs of nodes were in contact at discrete time step $t$. The epidemic spreading process follows the SIR model: at each time step, an infected node $i$ transmits the pathogen to each susceptible neighbour $j \in \mathcal{N}(i)$ with probability $p$, and independently recovers with probability $q$. The resulting process terminates in a final state $S_T$ in which every node is classified as susceptible (S), infectious (I), or recovered (R).

In practice the exact start time of the outbreak is not known. The posited temporal network $\tilde{\mathcal{G}}$ encompasses a window $[t_{\min}, t_{\max}]$ that is estimated to contain the true index event, but the precise start time $t_0 \in [t_{\min}, t_{\max}]$ is treated as a latent variable. The source detection task is then: given the observed final state $S_T$ and the posited network $\tilde{\mathcal{G}}$, infer the initial source node $S_0$ (patient zero). Ru et al. express this as the learned mapping

$$S_0 = \mathbf{BN}(S_T,\ \tilde{\mathcal{G}}).$$

### 5.1.2 Encoding Available Information

Two sources of information are provided to the model as input features.

**Node features — final state $S_T$.** Each node's SIR state at time $T$ is one-hot encoded as a three-dimensional feature vector $C_i \in \{0,1\}^3$, with susceptible nodes represented as $[1, 0, 0]$, infectious nodes as $[0, 1, 0]$, and recovered nodes as $[0, 0, 1]$. The full node feature matrix is $\mathbf{C} \in \{0,1\}^{N \times 3}$.

**Edge features — binary activation patterns (edge textures).** The temporal structure of the contact network is encoded by first constructing the aggregated static graph $\tilde{G}_a = (\mathcal{V}, E_a)$, where $E_a = \bigcup_{t=0}^{T} E_t$ contains every edge that appears in at least one snapshot. Each edge $e \in E_a$ is assigned a binary activation vector $X_e \in \{0,1\}^T$, where entry $t$ equals one if and only if the edge was active in snapshot $G_t$. The full edge feature matrix is $X_a = \{X_e \mid e \in E_a\} \in \{0,1\}^{|E_a| \times T}$. This representation, which Ru et al. call an edge texture, compresses the full temporal contact history into a fixed-length binary descriptor attached to each edge in the aggregated graph. The construction is implemented in the project's `gnn/graph_builder.py` via the `build_temporal_activation` function.

### 5.1.3 Architecture

The BN processes the input through three sequential components: initial feature projections, a stack of $L$ kernel-based convolutional layers, and a final projection to scalar per-node scores.

**Initial projections.** The raw node and edge features are projected into a common $D$-dimensional hidden space. The node projection $p^v : \mathbb{R}^3 \rightarrow \mathbb{R}^D$ and edge projection $p^e : \mathbb{R}^T \rightarrow \mathbb{R}^D$ are each a single linear layer followed by a ReLU activation:

$$h^0_i = p^v(C_i), \qquad g^0_{(i,j)} = p^e\!\left(X_{(i,j)}\right),$$

where $h^0_i \in \mathbb{R}^D$ and $g^0_{(i,j)} \in \mathbb{R}^D$ denote the initial hidden representations of node $i$ and edge $(i,j)$, respectively.

**Kernel-based convolutional layers.** Each of the $L$ layers applies a kernel-based convolutional operator \cite{Gilmer2017}\cite{Simonovsky2017} consisting of two interdependent update equations. The edge update reads the current edge hidden state together with the hidden state of the source node and produces a new edge representation:

$$g^{l+1}_{(i,j)} = f^e\!\left(\left[g^l_{(i,j)},\ h^l_i\right]\right),$$

where $[\cdot, \cdot]$ denotes concatenation and $f^e$ is a fully connected layer with ReLU activation. The node update then aggregates the newly computed incoming edge messages and applies a self-transformation:

$$h^{l+1}_i = \mathrm{ReLU}\!\left(f^v\!\left(h^l_i\right) + \sum_{j \in \mathcal{N}(i)} g^{l+1}_{(j,i)}\right),$$

where $f^v$ is an independent fully connected layer with ReLU activation. The two functions $f^e$ and $f^v$ do not share parameters. The design couples edge and node representations across layers: edge states are informed by the source node's embedding, and node states are informed by all incoming edge messages. This bidirectional coupling allows the network to propagate information about the temporal activation structure of edges into the node embeddings that ultimately determine source scores.

**Final projection and output.** After $L$ convolutional layers, the node hidden state $h^L_i \in \mathbb{R}^D$ is projected to a scalar score $h_i \in \mathbb{R}$ via a final linear layer.

### 5.1.4 Expert Knowledge and the Output Distribution

A key design feature of the BN is the injection of domain knowledge as a hard constraint on the output distribution. Regardless of the learned scores, a node that is susceptible at the final observation time $T$ cannot logically be the epidemic source: the source must have been infected at $t_0 \leq T$, and under SIR dynamics a susceptible node has never been infected. The BN enforces this constraint by setting the output logit of every susceptible node to negative infinity before the softmax:

$$h_s \leftarrow -\infty \quad \forall s \in \mathcal{V} \text{ such that } C_s = [1, 0, 0].$$

In the implementation this is realised as a masked fill with a large negative constant (`float('-inf')` in PyTorch) applied via the boolean susceptibility mask derived from the one-hot node features. The per-node log-probabilities are then computed as

$$\log \hat{p}_i = \log \frac{\exp(h_i)}{\sum_{j=1}^{N} \exp(h_j)},$$

which assigns probability zero to all susceptible nodes regardless of their learned scores, without modifying any network weights.

### 5.1.5 Training Objective

The model is trained by minimising the cross-entropy between the predicted log-probabilities and the one-hot label indicating the true source:

$$\mathcal{L} = -\sum_{i=1}^{N} y_i \log \hat{p}_i,$$

where $y_i = 1$ if node $i$ is the true patient zero and $y_i = 0$ otherwise. Because exactly one node is the source per training sample, this reduces to the standard negative log-likelihood of the true source under the predicted distribution. The loss is minimised using the Adam optimiser \cite{Kingma2015} with a learning rate of $10^{-3}$ and a batch size of 128. Ru et al. report that optimal performance is achieved at $L = 5$ convolutional layers; performance saturates or degrades at higher layer counts, a behaviour consistent with the over-smoothing phenomenon observed in other GNN architectures, whereby repeated neighbourhood aggregation causes node representations to converge toward indistinguishable values as the receptive field expands to encompass most of the network.

### 5.1.6 Training Data Generation

Training samples are generated by Monte Carlo simulation of the SIR spreading process on the temporal network. For each sample, a start time $t_0$ is drawn uniformly at random from the interval $[t_{\min}, t_{\max}]$, and a random patient zero is selected from the set of nodes active at $t_0$. The SIR process is simulated on $\tilde{\mathcal{G}}$ from $t_0$ until a fixed end time $t_{\text{end}}$, producing a final snapshot of SIR states. The pair (final snapshot, true patient zero) constitutes one training instance. This procedure is used for all GNN models in this thesis and is implemented using an adapted version of Holme's fast C-based temporal SIR simulation engine \cite{Holme2020}. The simulation code is wrapped in the project's `tsir/` module, and data is logged as versioned artifacts via Weights and Biases.

### 5.1.7 Robustness to Missing Information

Ru et al. evaluate the BN under two additional scenarios that degrade the available information. In the first, a random fraction of edges is removed from each snapshot, simulating incomplete knowledge of the contact network structure. In the second, only a random fraction of node states is observed in the final snapshot $S_T$, with the remaining nodes assigned an "unknown" state. The BN maintains a Top-5 accuracy of approximately 0.7 even when 80\% of final node states are unobserved, suggesting that the edge texture representation provides a strong structural prior that partially compensates for the absence of direct observational evidence. Performance degrades more sharply when the contact network structure itself is incomplete, which is expected given that the edge texture is the primary carrier of temporal information in the model. This thesis revisits these robustness scenarios in Chapter 6.

### 5.1.8 Critical Assessment

The BN achieves strong results and its design choices are well motivated. However, Sterchi et al. \cite{Sterchi2025} raise an important concern about the evaluation methodology in Ru et al.: the comparison with the SME baseline uses only 200 Monte Carlo simulations per candidate source, which is a strongly sub-optimal configuration for SME and likely yields an underestimate of SME's true accuracy. The advantage of the BN over SME reported by Ru et al. may therefore be overstated. This thesis addresses this limitation by evaluating all baselines, including SME, with a substantially larger number of Monte Carlo simulations, ensuring that the reported comparisons are fair.

At a deeper architectural level, the edge texture representation is a lossy encoding of the temporal contact history: two edges with identical activation counts but different orderings will receive different textures, but two edges whose activation patterns are permutations of each other will be treated as different even if their spreading-relevant properties are identical. The BN therefore encodes temporal ordering implicitly through the binary pattern, but does not explicitly model causal walk structure. This is precisely the limitation that the De Bruijn GNN and the TEG-DAG-GNN are designed to address, and the comparison between all three architectures is the central contribution of this thesis.
