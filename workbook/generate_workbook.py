import os

content = r"""% !TEX program = xelatex
\documentclass[11pt,a4paper,oneside]{report}

\usepackage{fontspec}
\setmainfont{Helvetica Neue}[Scale=0.95]
\setmonofont{JetBrainsMonoNerdFont-Regular.ttf}[
    Path = /Users/dariush/Library/Fonts/,
    BoldFont = JetBrainsMonoNerdFont-Bold.ttf,
    ItalicFont = JetBrainsMonoNerdFont-Italic.ttf,
    BoldItalicFont = JetBrainsMonoNerdFont-BoldItalic.ttf,
    Scale=0.85
]

\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage[hidelinks,colorlinks=true,linkcolor=blue!60!black,urlcolor=blue!60!black]{hyperref}
\usepackage{xcolor}
\usepackage{graphicx}
\usepackage{float}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{parskip}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{mdframed}
\usepackage{caption}

% ---- listings ----
\usepackage{listings}
\definecolor{codebg}{RGB}{248,248,248}
\definecolor{codeframe}{RGB}{200,200,200}
\lstset{
  basicstyle=\ttfamily\footnotesize,
  keywordstyle=\color{blue!70!black}\bfseries,
  commentstyle=\color{gray!70}\itshape,
  stringstyle=\color{orange!80!black},
  numbers=left, numberstyle=\tiny\color{gray},
  numbersep=8pt, breaklines=true,
  showstringspaces=false,
  frame=single, rulecolor=\color{codeframe},
  backgroundcolor=\color{codebg},
  captionpos=b, tabsize=4,
  language=Python,
  xleftmargin=12pt,
}

% ---- tcolorbox ----
\usepackage{tcolorbox}
\tcbuselibrary{theorems,skins,breakable,listings}

\newtcolorbox{conceptbox}[2][]{
  colback=blue!4!white, colframe=blue!55!black,
  fonttitle=\bfseries\sffamily, title={#2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{analogybox}[2][]{
  colback=green!4!white, colframe=green!50!black,
  fonttitle=\bfseries\sffamily, title={Analogy: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{examplebox}[2][]{
  colback=orange!5!white, colframe=orange!70!black,
  fonttitle=\bfseries\sffamily, title={Example: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{keyinsight}[2][]{
  colback=purple!5!white, colframe=purple!60!black,
  fonttitle=\bfseries\sffamily, title={Key Insight: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{codewalk}[2][]{
  colback=gray!4!white, colframe=gray!50,
  fonttitle=\bfseries\sffamily, title={Code Walk: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{paperbox}[2][]{
  colback=red!4!white, colframe=red!55!black,
  fonttitle=\bfseries\sffamily, title={Paper Spotlight: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}
\newtcolorbox{mathbox}[2][]{
  colback=yellow!5!white, colframe=yellow!60!black,
  fonttitle=\bfseries\sffamily, title={Mathematical Formalism: #2}, #1, breakable,
  left=8pt, right=8pt, top=6pt, bottom=6pt, boxrule=0.8pt, arc=4pt
}

% ---- TikZ ----
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes,fit,backgrounds,calc,decorations.pathreplacing}

% ---- headers ----
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\sffamily\leftmark}
\fancyhead[R]{\small\sffamily\thepage}
\renewcommand{\headrulewidth}{0.4pt}

\titleformat{\chapter}[display]
  {\normalfont\huge\bfseries\sffamily}{\chaptertitlename\ \thechapter}{10pt}{\Huge}

\begin{document}

\begin{titlepage}
  \centering
  \vspace*{1.5cm}
  {\Huge\bfseries\sffamily Epidemic Source Detection\\using Graph Neural Networks\\[0.4em]
   \Large A Comprehensive Master's Thesis Workbook}\\[1.5em]
  {\large\itshape An exhaustive deep-dive into the maths, algorithms, and code\\
  behind locating Patient Zero on Temporal Contact Networks}\\[3em]
  {\large Master's Thesis Companion Guide}\\[0.5em]
  {\large\today}\\[4em]
  \begin{mdframed}[backgroundcolor=blue!6,linecolor=blue!40,linewidth=1.5pt,roundcorner=5pt,innerleftmargin=15pt,innerrightmargin=15pt,innertopmargin=15pt,innerbottommargin=15pt]
  \centering
  \textbf{How to use this workbook}\\[0.5em]
  \small
  This workbook is an exhaustive companion to the master's thesis. It teaches the core literature from the ground up. Every mathematical concept is derived, and every algorithm is explained with code. Read it front-to-back to master the entire field of neural epidemic source detection.

  Boxes with coloured borders signal: {\color{blue!60!black}core concepts}, {\color{green!50!black}intuitive analogies},
  {\color{orange!70!black}worked examples}, {\color{purple!60!black}key insights},
  {\color{gray!60}code walkthroughs}, {\color{red!55!black}paper breakdowns}, and {\color{yellow!60!black}mathematical derivations}.
  \end{mdframed}
\end{titlepage}

\tableofcontents
\clearpage

% ==============================================================================
\chapter{Introduction \& Epidemiological Foundations}
% ==============================================================================

\section{The Inverse Problem of Source Detection}

The overarching question of this thesis is: \textbf{Given a final snapshot of an epidemic and a record of the historical contact network, can we accurately identify the very first person who brought the virus into the network?}

In epidemiology, finding Patient Zero is fundamentally an \emph{inverse problem}. We are given the output of a complex, stochastic system and asked to infer its initial conditions.

\begin{mathbox}{Formalizing the Inverse Problem}
Let $G(V, E^t)$ be a temporal contact network, where $V$ is the set of nodes and $E^t$ is the set of temporal edges $(u, v, t)$. Let $\mathcal{M}$ represent the underlying stochastic spreading model (e.g., SIR) governed by infection rate $\beta$ and recovery rate $\mu$.

At time $t=0$, a single source node $v^* \in V$ is infected.
At observation time $t=T$, we observe the state of all nodes, denoted as $E_T = \{X_v(T)\}_{v \in V}$, where $X_v \in \{S, I, R\}$.

Our goal is to compute the posterior probability distribution over all possible source nodes $q \in V$, given the observed snapshot and the network:
\[
P(v^* = q \mid E_T, G, \mathcal{M}) = \frac{P(E_T \mid v^* = q, G, \mathcal{M}) P(v^* = q)}{P(E_T \mid G, \mathcal{M})}
\]
Because the denominator is constant for all candidates, we focus on maximizing the likelihood:
\[
v^*_{\text{MAP}} = \arg\max_{q \in V} P(E_T \mid v^* = q, G, \mathcal{M})
\]
Computing this likelihood exactly is computationally intractable for large networks due to the combinatorially vast number of possible infection pathways. Therefore, we approximate it using deep learning and Monte Carlo simulations.
\end{mathbox}

\section{The SIR Model and Event-Driven Simulation}

The exact mathematical unfolding of the epidemic is governed by the \textbf{SIR Model}.

\begin{center}
\begin{tikzpicture}[
  state/.style={circle, draw, thick, minimum size=1.8cm, font=\bfseries\Large},
  arr/.style={-{Stealth[length=3mm]}, thick, line width=1.5pt}
]
  \node[state, fill=blue!15, text=blue!80!black]   (S) at (0,0) {S};
  \node[state, fill=red!20, text=red!80!black]    (I) at (5,0) {I};
  \node[state, fill=green!20, text=green!80!black]  (R) at (10,0) {R};
  
  \draw[arr] (S) -- node[above, font=\large]{Infection ($\beta$)} (I);
  \draw[arr] (I) -- node[above, font=\large]{Recovery ($\mu$)} (R);
  
  \node[align=center,below=0.4cm of S, font=\small] {Susceptible\\(Healthy, vulnerable)};
  \node[align=center,below=0.4cm of I, font=\small] {Infectious\\(Sick, contagious)};
  \node[align=center,below=0.4cm of R, font=\small] {Recovered\\(Immune, dead-end)};
\end{tikzpicture}
\end{center}

\begin{mathbox}{Continuous-Time Stochastic Transitions}
The progression of the disease is a continuous-time Markov jump process.
\begin{itemize}
    \item \textbf{Infection event:} A susceptible node $v$ transitions to the infectious state $I$ at a rate equal to $\beta \times k_I(v)$, where $\beta$ is the base transmission rate, and $k_I(v)$ is the number of currently infectious neighbors of $v$.
    \item \textbf{Recovery event:} An infectious node transitions to the recovered state $R$ at a constant rate $\mu$. The time a node spends in the infectious compartment is exponentially distributed with mean $1/\mu$.
\end{itemize}
\end{mathbox}

To train our neural networks, we require an immense amount of data (e.g., generating 500 outbreaks for each of the 232 nodes in a network requires 116,000 simulations). We utilize a highly optimized C-based event-driven simulator developed by Petter Holme.

\begin{conceptbox}{Event-Driven Simulation (The Gillespie Algorithm)}
Instead of checking every node at every small time step $dt$, event-driven simulation jumps directly to the time of the next event using a priority queue (Min-Heap).
\begin{enumerate}
    \item When a node becomes Infected, we immediately sample a random \emph{Recovery Time} from an exponential distribution: $t_{rec} \sim \text{Exp}(\mu)$. This recovery event is placed in the queue.
    \item For every neighbor of the infected node, we sample a potential \emph{Infection Time} $t_{inf} \sim \text{Exp}(\beta)$. If $t_{inf} < t_{rec}$ (the infection happens before the source recovers), we schedule the infection event in the queue.
    \item The simulation pops the earliest event from the queue, updates the network state, schedules new events, and repeats.
\end{enumerate}
This reduces the computational complexity dramatically to $\mathcal{O}(N \log N)$ or $\mathcal{O}(N^2 \log N)$ in the worst case.
\end{conceptbox}

% ==============================================================================
\chapter{Paper 1: Weighted Temporal Event Graphs (Saramäki et al.)}
\label{chap:wteg}
% ==============================================================================

\begin{paperbox}{Weighted temporal event graphs}
\textbf{Authors:} Jari Saramäki, Mikko Kivelä, Márton Karsai (2019)\\
\textbf{The Core Problem:} Collapsing temporal networks into static graphs destroys the arrow of time, creating "fake" paths that pathogens or information could never actually travel along.\\
\textbf{The Solution:} Map the temporal network into a \emph{Directed Acyclic Graph (DAG)} where the \textbf{nodes are events} and the \textbf{edges represent valid temporal adjacencies}. This preserves all time-respecting paths without information loss.
\end{paperbox}

\section{The Arrow of Time and Time-Respecting Paths}

In a standard static graph, if A connects to B, and B connects to C, we assume information can flow $A \to B \to C$. In a temporal network, this is only true if the $A \to B$ contact occurred \emph{before} the $B \to C$ contact.

\begin{figure}[H]
\centering
\begin{tikzpicture}[
  node/.style={circle, draw, thick, minimum size=0.9cm, font=\small\bfseries},
  arr/.style={-{Stealth}, thick}
]
% Left: wrong direction
\begin{scope}[xshift=0cm]
  \node[node,fill=red!20]   (A) at (0,2) {A};
  \node[node,fill=gray!20]  (B) at (2,2) {B};
  \node[node,fill=gray!20]  (C) at (4,2) {C};
  \draw[arr,red!60, line width=1.2pt] (A) -- node[above]{\small $t=5$} (B);
  \draw[arr,gray!60, line width=1.2pt] (B) -- node[above]{\small $t=3$} (C);
  \node[below=0.1cm of B, font=\footnotesize, text=red]{B-C happens BEFORE A-B};
  \node[below=0.8cm,font=\normalsize\bfseries,text=red!70] at (2,0.8) {A CANNOT infect C via B};
\end{scope}
% Right: correct direction
\begin{scope}[xshift=8cm]
  \node[node,fill=red!20]   (A2) at (0,2) {A};
  \node[node,fill=gray!20]  (B2) at (2,2) {B};
  \node[node,fill=gray!20]  (C2) at (4,2) {C};
  \draw[arr,green!60!black, line width=1.2pt] (A2) -- node[above]{\small $t=3$} (B2);
  \draw[arr,green!60!black, line width=1.2pt] (B2) -- node[above]{\small $t=5$} (C2);
  \node[below=0.1cm of B2, font=\footnotesize, text=green!60!black]{A-B happens BEFORE B-C};
  \node[below=0.8cm,font=\normalsize\bfseries,text=green!60!black] at (2,0.8) {A CAN infect C via B};
\end{scope}
\end{tikzpicture}
\caption{The concept of chronological ordering. Only the right configuration forms a valid causal path.}
\end{figure}

\begin{mathbox}{Formalizing Temporal Adjacency}
Let $G = (V_G, E_G, T)$ be a temporal network. An event is denoted as $e_1(i, j, t_1)$, meaning an interaction between node $i$ and $j$ at time $t_1$.

\textbf{Definition: Temporal Adjacency.} Two events $e_1(i, j, t_1)$ and $e_2(k, l, t_2)$ are \emph{temporally adjacent}, denoted $e_1 \to e_2$, if they share at least one node ($\{i,j\} \cap \{k,l\} \neq \emptyset$) and they are consecutive in time ($t_2 > t_1$).

\textbf{Definition: $\Delta t$-Adjacency.} Two events are $\Delta t$-adjacent, denoted $e_1 \xrightarrow{\Delta t} e_2$, if they are temporally adjacent and the time difference $\delta t(e_1, e_2) = t_2 - t_1 \le \Delta t$. This represents an upper bound on how long a node can "hold" the infection before passing it on.
\end{mathbox}

\section{Constructing the Weighted Event Graph $D$}

Saramäki et al. propose mapping the temporal network $G$ into a static, weighted, directed acyclic graph (DAG) called $D$. This is a masterstroke because it allows us to use fast static-graph algorithms (like finding weakly connected components) while perfectly preserving temporal rules.

\begin{conceptbox}{The Weighted Event Graph $D = (V_D, L_D, w)$}
\begin{enumerate}
    \item \textbf{Nodes ($V_D$):} Every temporal event $e \in E_G$ becomes a node in $D$.
    \item \textbf{Links ($L_D$):} Directed links are drawn between events that are temporally adjacent ($e_1 \to e_2$).
    \item \textbf{Weights ($w$):} The weight of a link is the time difference $\delta t(e_1, e_2) = t_2 - t_1$.
\end{enumerate}
\end{conceptbox}

\textbf{Path Equivalence Theorem:} A path $P$ is a directed vertex path in the event graph $D$ \textbf{if and only if} $P$ is a time-respecting event path in the original temporal network $G$.

This concept forms the structural foundation for the \textbf{DAG-GNN} architecture used later in our codebase, which performs message passing precisely over these causally valid event paths.

% ==============================================================================
\chapter{Paper 2: The Backtracking Network (Ru et al.)}
\label{chap:bn}
% ==============================================================================

\begin{paperbox}{Inferring Patient Zero on Temporal Networks via Graph Neural Networks}
\textbf{Authors:} Xiaolei Ru, Jack Murdoch Moore, Xin-Ya Zhang, Yeting Zeng, Gang Yan (AAAI 2023)\\
\textbf{The Core Problem:} How can we locate a source using only the final snapshot and the temporal contact network, without knowing the exact times individuals were infected?\\
\textbf{The Solution:} Tailor a Graph Neural Network to establish an inverse statistical association between the final state and the initial state. The network relies on \textbf{Temporal Activation Patterns} on edges, updating edge embeddings before node embeddings.
\end{paperbox}

\section{Encoding the Input}

Rather than building a complex DAG or De Bruijn graph, the Backtracking Network (BN) maintains the standard static graph topology but injects temporal information as dense feature vectors on the edges.

\begin{mathbox}{Encoding Available Information}
1. \textbf{Final State (Nodes):} The state of an individual $i$ is one-hot encoded: $C_i \in \{[1,0,0], [0,1,0], [0,0,1]\}$. These are initially projected to $h_i^0 = \text{ReLU}(\mathbf{W}^v C_i)$.

2. \textbf{Temporal Activation Patterns (Edges):} The total time $T$ is divided into discrete slices. For each edge $e$, we create a binary vector $X_e \in \{0, 1\}^T$ where $X_e[t] = 1$ if the edge was active at time $t$, and $0$ otherwise. These are projected to $g_{(i,j)}^0 = \text{ReLU}(\mathbf{W}^e X_{(i,j)})$.
\end{mathbox}

\section{The Message Passing Logic}

The key innovation of the BN architecture is the \textbf{interdependent update}. In standard GNNs, edges only act as routes. In BN, edges have their own hidden state that evolves layer by layer based on the state of the node they originate from.

\begin{analogybox}{The Telephone Game with Memory}
Imagine a telephone game. Usually, Alice whispers to Bob (standard GNN). In the Backtracking Network, the "wire" connecting Alice to Bob is a smart wire. Before passing the message to Bob, the wire looks at its own history ("Was I even plugged in during the epidemic?") and looks at Alice's current state ("Is Alice actually sick?"). It updates its own state to decide how important this message is, and \emph{then} passes it to Bob.
\end{analogybox}

\begin{mathbox}{Layer Updates (Eq. 3 and 4 in paper)}
For each layer $l = 0, \dots, L-1$:

\textbf{1. Edge Update (Eq. 3):}
\[
g_{(i,j)}^{l+1} = f^e \left( \left[ g_{(i,j)}^l \;\big\|\; h_i^l \right] \right)
\]
The edge embedding $g_{(i,j)}$ is updated by concatenating its previous state with the current state of the \emph{source node} $h_i^l$. Here, $f^e$ is an MLP.

\textbf{2. Node Update (Eq. 4):}
\[
h_i^{l+1} = \text{ReLU} \left( f^v(h_i^l) + \sum_{j \in \mathcal{N}(i)} g_{(j,i)}^{l+1} \right)
\]
Node $i$ updates its state by aggregating the freshly updated edge messages arriving from all its neighbors $j$.
\end{mathbox}

\section{Expert Knowledge Injection}

The paper includes a critical step that drastically improves performance. The GNN learns purely statistical associations. It might learn that node 5 is highly central and give it a high score, even if node 5 is currently Susceptible. But mathematically, \emph{the source can never be susceptible at the end of the epidemic}.

\begin{mathbox}{Eq. 6: Forcing Probabilities}
Let $h_i$ be the scalar output score for node $i$ from the final layer. We apply expert knowledge:
\[
h_s \gets h_s - \infty \quad \text{for all susceptible nodes } s
\]
When passed through the final softmax function, $\exp(-\infty) = 0$, strictly enforcing that $P(\text{source}=s) = 0$.
\end{mathbox}

\section{PyTorch Implementation: The Scatter Add Trick}

In our codebase, how do we efficiently compute the node update $\sum_{j \in \mathcal{N}(i)} g_{(j,i)}^{l+1}$? We use \texttt{torch.scatter\_add\_}.

\begin{codewalk}{BNConvLayer.forward (gnn/backtracking\_network.py)}
\begin{lstlisting}
def forward(self, h, g, edge_index):
    # h: [B, N, D] - Batch size B, N nodes, dimension D
    # g: [B, E, D] - E edges
    src, dst = edge_index[0], edge_index[1]
    B, N, D = h.shape

    # 1. Edge Update: cat(g, h_src)
    h_src = h[:, src, :]                            # Extract source nodes for all edges
    g_new = self.f_e(torch.cat([g, h_src], dim=-1)) # [B, E, D]

    # 2. Node Update: Aggregate messages to destination nodes
    # We expand the 'dst' index tensor to match the 3D shape [B, E, D]
    dst_idx = dst.view(1, -1, 1).expand(B, -1, D)
    
    # Create empty tensor for aggregated messages
    agg = h.new_zeros(B, N, D)
    
    # Scatter add: for every edge 'e', add g_new[:, e, :] to agg[:, dst[e], :]
    agg.scatter_add_(1, dst_idx, g_new)

    h_new = F.relu(self.f_v(h) + agg)
    return h_new, g_new
\end{lstlisting}
\end{codewalk}

% ==============================================================================
\chapter{Paper 3: De Bruijn Graph Neural Networks (Qarkaxhija et al.)}
\label{chap:debruijn}
% ==============================================================================

\begin{paperbox}{De Bruijn goes Neural: Causality-Aware Graph Neural Networks}
\textbf{Authors:} Lisi Qarkaxhija, Vincenzo Perri, Ingo Scholtes (2022)\\
\textbf{The Core Problem:} Standard GNNs run on time-aggregated static networks. This results in \emph{Markovian} (memoryless) message passing. The GNN is completely blind to whether paths actually respected the arrow of time.\\
\textbf{The Solution:} Transform the graph into a higher-order \textbf{De Bruijn Graph}, where nodes represent $k-1$ causal walks and edges represent length $k$ causal walks. Message passing on this new graph is inherently non-Markovian and physically accurate.
\end{paperbox}

\section{The Non-Markovian Nature of Dynamic Graphs}

If we collapse a temporal network into a static graph, message passing simulates a random walk. A message arrives at node $B$, and node $B$ passes it to all neighbors $A, C, D$. This is a First-Order Markov process: the next step depends \emph{only} on the current node ($B$), with no memory of where the message came from.

However, in a dynamic graph, if a message arrived from $A \to B$ at time $t=1$, it can only proceed to $C$ if the $B \to C$ contact happens at $t > 1$. The valid next step depends on the \emph{history} of the path. This requires a Higher-Order (Non-Markovian) model.

\section{The De Bruijn Transformation}

\begin{mathbox}{Definition: $k$-th order De Bruijn graph model}
For a dynamic graph $G^T$, a $k$-th order De Bruijn graph model $G^{(k)} = (V^{(k)}, E^{(k)})$ is defined as:
\begin{itemize}
    \item \textbf{Nodes ($V^{(k)}$):} Each node $u$ is a causal walk of length $k-1$ in the original graph $G^T$.
    \item \textbf{Edges ($E^{(k)}$):} A directed edge exists from $u = (u_0, \dots, u_{k-1})$ to $v = (v_1, \dots, v_k)$ if they overlap exactly by $k-1$ nodes (i.e., $u_i = v_i$ for $i = 1 \dots k-1$) and the union $(u_0, \dots, u_{k-1}, v_k)$ forms a valid causal walk of length $k$.
\end{itemize}
\end{mathbox}

In our implementation, we use $k=2$.
\begin{itemize}
    \item A node in the De Bruijn graph is a length-1 causal walk (a single contact event): $u_{new} = (A \to B, t_1)$.
    \item An edge connects $u_{new}$ to $v_{new} = (B \to C, t_2)$ if $t_2 > t_1$.
\end{itemize}
Message passing on $G^{(k)}$ is strictly confined to valid chronological paths!

\section{The Sentinel Node Trick}

There is a major problem with the De Bruijn approach: The nodes in $G^{(k)}$ are \emph{events}, but our goal is to predict probabilities over the \emph{original people} (the original nodes $V$). 

The paper proposes a complex "Bipartite projection layer" (Equation 2) to map event embeddings back to original nodes. In our codebase, we achieve this more elegantly using \textbf{Sentinel Nodes}.

For every original node $v$, and at every timestep $t$, we add a dummy "sentinel" event $(v \to v, t)$. 
\begin{itemize}
    \item Sentinels initialize their features from the original node's SIR state.
    \item Real events $(u \to v, t)$ initialize their features by concatenating the SIR states of both participants.
    \item Causal edges are drawn to and from sentinels naturally.
    \item \textbf{The Readout:} To get the final embedding for original node $v$, we simply look at the final embedding of its very last sentinel node $(v \to v, T)$.
\end{itemize}

\begin{codewalk}{DBGNN.forward (gnn/dbgnn.py)}
\begin{lstlisting}
# src_nodes and dst_nodes map De Bruijn nodes back to original graph nodes
# sent_mask is a boolean array identifying Sentinel Nodes (src == dst)

# 1. Project sentinel nodes (input dim = 3)
x_sent = x[:, src_nodes[sent_mask], :]
h[:, sent_mask, :] = self.proj_sentinel(x_sent)

# 2. Project real event nodes (input dim = 6, concat of src and dst)
ev_mask = ~sent_mask
x_src_ev = x[:, src_nodes[ev_mask], :]
x_dst_ev = x[:, dst_nodes[ev_mask], :]
h[:, ev_mask, :] = self.proj_event(torch.cat([x_src_ev, x_dst_ev], dim=-1))

# 3. Message passing strictly on the De Bruijn topology
for conv in self.convs:
    h = conv(h, edge_index)

# 4. Readout from final sentinels
# sentinel_end_indices maps original node 'v' to the De Bruijn index of (v,v,T)
h_out = h[:, sentinel_end_indices, :]  
scores = self.out(h_out).squeeze(-1)
\end{lstlisting}
\end{codewalk}

% ==============================================================================
\chapter{Evaluation and the Statistical Baselines}
\label{chap:eval}
% ==============================================================================

\section{The Soft Margin Estimator (SME)}

Proposed by Antulov-Fantulin et al. (2015), SME is our most rigorous probabilistic baseline. For every candidate node $q$, we run $n$ simulations to generate final snapshots $r_{q,i}$. We compare these simulated snapshots to the real observed snapshot $r^*$.

\begin{mathbox}{SME Jaccard Similarity and Likelihood}
The similarity between two snapshots is measured using the Jaccard Index over the set of infected/recovered nodes:
\[
\varphi(r^*, r_{q,i}) = \frac{|r^* \cap r_{q,i}|}{|r^* \cup r_{q,i}|}
\]
Treating this similarity as a random variable, the empirical probability density function is smoothed using a Gaussian kernel $w_a(x) = \exp(-(x-1)^2/a^2)$. The estimated likelihood for source $q$ is:
\[
\hat{P}(r^* \mid q) = \frac{1}{n} \sum_{i=1}^n \exp\left(-\frac{(\varphi(r^*, r_{q,i}) - 1)^2}{a^2}\right)
\]
\end{mathbox}

\section{Monte Carlo Mean Field (MCMF)}

MCMF (Sterchi et al.) is a faster, highly competitive baseline. Instead of comparing whole snapshots, it calculates marginal probabilities for each node independently.
For a candidate source $q$, what is the probability that node $v$ ends up in state $X_v(T)$? We calculate this by simply counting the fraction of simulations starting from $q$ where $v$ ended up in $X_v(T)$.

\begin{mathbox}{MCMF Factorization}
Assuming conditional independence between all nodes given the source $q$, the likelihood of the entire snapshot $E_T$ factorizes:
\[
\hat{P}(E_T \mid q) = \prod_{v \in V} P(X_v(T) = x_v(T) \mid q)
\]
Because multiplying many small probabilities causes numerical underflow, we compute the log-likelihood:
\[
\log \hat{P}(E_T \mid q) = \sum_{v \in V} \log P(X_v(T) = x_v(T) \mid q)
\]
\end{mathbox}

\section{Evaluation Metrics}

Once a model outputs a probability distribution $P$ over all nodes, we rank them. Let the rank of the true source be $R$.

\begin{itemize}
    \item \textbf{Top-K Accuracy:} $\mathbb{I}[R \le K]$. The percentage of test instances where the true source was in the top K guesses.
    \item \textbf{Mean Reciprocal Rank (Rank Score):} $\frac{1}{N} \sum \frac{1}{R_i}$. This metric heavily penalizes severe misses. Ranking the source 2nd gives a score of 0.5. Ranking it 100th gives a score of 0.01.
    \item \textbf{Error Distance:} The shortest path distance in the network between the model's top guess (MAP estimate) and the true source.
\end{itemize}

\end{document}
"""

with open("thesis/workbook.tex", "w") as f:
    f.write(content)

print("Created workbook.tex")
