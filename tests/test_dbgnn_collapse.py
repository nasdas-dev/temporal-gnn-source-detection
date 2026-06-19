"""Regression test for the DBGNN empty-De-Bruijn-graph (collapse) fallback.

When the De Bruijn graph is empty (``n_db == 0``) — which is guaranteed at the
fully time-collapsed Δt in the H2 coarse-graining sweep, since a single time bin
admits no strictly-increasing causal walk — the model must fall back to the
first-order static branch and behave like a static GCN. A regression here would
force ``bipartite_agg`` to zeros, making the output bias-only (uniform over
candidate nodes), which masquerades as "convergence to StaticGNN" for the wrong
reason.
"""

from __future__ import annotations

import networkx as nx
import torch

from gnn.dbgnn import DBGNN
from gnn.graph_builder import build_de_bruijn_graph


def _empty_debruijn_inputs(order: int):
    db_edge_index = torch.zeros(2, 0, dtype=torch.long)
    db_edge_weight = torch.zeros(0)
    db_node_to_original = torch.zeros(0, order, dtype=torch.long)
    db_node_last = torch.zeros(0, dtype=torch.long)
    return db_edge_index, db_edge_weight, db_node_to_original, db_node_last


def test_dbgnn_empty_debruijn_uses_static_branch():
    """With an empty De Bruijn graph, candidate scores must reflect the static
    graph structure (not collapse to a uniform/bias-only distribution)."""
    torch.manual_seed(0)
    N, B = 6, 2
    model = DBGNN(hidden_channels=8, num_conv_layers=2, order=2)
    model.eval()

    # One-hot SIR features [S, I, R]: nodes 0..3 infected (candidate sources),
    # nodes 4..5 susceptible (masked out of the source distribution).
    x = torch.zeros(B, N, 3)
    x[:, :4, 1] = 1.0  # I
    x[:, 4:, 0] = 1.0  # S

    db = _empty_debruijn_inputs(order=2)

    # Non-trivial static graph (path 0-1-2-3) with self-loops, so the candidate
    # nodes have genuinely different neighbourhoods.
    src = torch.tensor([0, 1, 1, 2, 2, 3, 0, 1, 2, 3])
    dst = torch.tensor([1, 0, 2, 1, 3, 2, 0, 1, 2, 3])
    static_edge_index = torch.stack([src, dst])
    static_edge_weight = torch.ones(static_edge_index.shape[1])

    with torch.no_grad():
        logp = model(x, *db, static_edge_index, static_edge_weight)

    assert logp.shape == (B, N)
    # Susceptible nodes are masked to -inf.
    assert torch.isinf(logp[:, 4:]).all()
    # Candidate nodes must NOT be uniform: the static branch must influence the
    # scores. The bug made them identical (bias-only) -> spread == 0.
    cand = logp[:, :4]
    spread = cand.max(dim=1).values - cand.min(dim=1).values
    assert bool((spread > 1e-4).all().item()), (
        f"DBGNN collapse output is ~uniform: {cand.detach().cpu().tolist()}"
    )


def test_dbgnn_empty_debruijn_outputs_valid_logprobs():
    """The collapse fallback must still yield a valid masked log-softmax."""
    torch.manual_seed(1)
    N, B = 5, 1
    model = DBGNN(hidden_channels=8, num_conv_layers=1, order=3)
    model.eval()

    x = torch.zeros(B, N, 3)
    x[:, :3, 1] = 1.0  # I (candidates)
    x[:, 3:, 0] = 1.0  # S

    db = _empty_debruijn_inputs(order=3)
    static_edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 1, 2]])
    static_edge_weight = torch.ones(static_edge_index.shape[1])

    with torch.no_grad():
        logp = model(x, *db, static_edge_index, static_edge_weight)

    cand = logp[0, :3]
    assert bool(torch.isfinite(cand).all().item())
    # log-softmax over the unmasked candidates sums to ~1 in probability space.
    assert bool(torch.allclose(cand.exp().sum(), torch.tensor(1.0), atol=1e-4))


def _single_time_bin_graph() -> nx.Graph:
    """Temporal contact graph where every contact shares one time bin (collapse)."""
    H = nx.Graph()
    H.add_edge(0, 1, times=[0])
    H.add_edge(1, 2, times=[0])
    H.add_edge(2, 3, times=[0])
    return H


def test_de_bruijn_collapse_is_order_invariant():
    """At a fully time-collapsed Δt there are no causal walk completions, so the
    higher-order branch must be dropped (n_db=0) for EVERY order — both k=2 and
    k=3 reduce to the same pure static GCN, with no self-loop-only k=2 channel."""
    H = _single_time_bin_graph()
    for order in (2, 3, 4):
        graph_data = build_de_bruijn_graph(H, order=order)
        assert graph_data["n_db_nodes"] == 0, f"order {order} should collapse to empty DB graph"
        assert graph_data["db_edge_index"].shape[1] == 0
        assert graph_data["db_node_to_original"].shape[0] == 0
        # The static branch is still present and identical across orders.
        assert graph_data["static_edge_index"].shape[1] > 0


def test_de_bruijn_keeps_higher_order_when_causal_walks_exist():
    """Sanity: with genuine increasing-time contacts the higher-order branch is
    retained (the collapse drop must not fire on normal temporal data)."""
    H = nx.Graph()
    H.add_edge(0, 1, times=[0])
    H.add_edge(1, 2, times=[1])  # causal walk (0,1)@0 -> (1,2)@1 exists
    graph_data = build_de_bruijn_graph(H, order=2)
    assert graph_data["n_db_nodes"] > 0
    assert graph_data["db_edge_index"].shape[1] > 0
