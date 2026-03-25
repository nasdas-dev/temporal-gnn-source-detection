"""
Model registry for the source-detection pipeline.

Usage
-----
All models are registered once in ``gnn/__init__.py``.
Everywhere else, look up a model by its string name::

    from gnn import MODEL_REGISTRY
    spec = MODEL_REGISTRY["static_gnn"]
    graph_data = spec.builder_fn(H)
    model     = spec.cls(...)
    log_probs  = spec.forward_fn(model, x_batch, graph_data, device)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type

import torch


@dataclass
class ModelSpec:
    """All information needed to build, call, and feed graph data to a model."""

    cls: Type[torch.nn.Module]
    """The model class (uninstantiated)."""

    forward_fn: Callable
    """
    Signature::

        forward_fn(
            model:      torch.nn.Module,
            x_batch:    torch.Tensor,   # [B, N, F] SIR snapshot batch on CPU
            graph_data: dict,           # output of builder_fn
            device:     torch.device,
        ) -> torch.Tensor               # [B, N] log-probabilities
    """

    builder_fn: Callable
    """
    Signature::

        builder_fn(H: nx.Graph, **kwargs) -> dict

    Returns a dict of pre-computed tensors (edge_index, edge_attr, …) plus
    always the key ``"n_nodes": int``.
    """

    build_fn: Callable
    """
    Signature::

        build_fn(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module

    Instantiates the model from the YAML model-config section, the number of
    nodes in the graph, and the pre-built graph_data dict (which may carry
    model-specific info such as ``T`` for BacktrackingNetwork or
    ``num_snapshots`` for TemporalGNN).
    """


# Central registry: model-name → ModelSpec
MODEL_REGISTRY: dict[str, ModelSpec] = {}


def get_model_spec(name: str) -> ModelSpec:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Registered models: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]
