"""
GNN package — models + registry.

All models are registered here so that ``main_train.py`` can look them up by
name.  To add a new model:

1. Implement ``MyModel`` in ``gnn/my_model.py``.
2. Add a builder function in ``gnn/graph_builder.py``.
3. Add a forward function in ``training/trainer.py``.
4. Add a build function below and register.
"""

import torch

from .static_gnn             import StaticGNN
from .backtracking_network   import BacktrackingNetwork
from .temporal_gnn           import TemporalGNN
from .dbgnn                  import DBGNN
from .dag_gnn                import DAGGNN

from .registry     import MODEL_REGISTRY, ModelSpec, get_model_spec
from .graph_builder import (
    build_static_graph,
    build_temporal_activation,
    build_temporal_snapshots,
    build_de_bruijn_graph,
    build_dag_event_graph,
)
from training.trainer import (
    static_gnn_forward,
    backtracking_forward,
    temporal_gnn_forward,
    dbgnn_forward,
    dag_gnn_forward,
)


# ---------------------------------------------------------------------------
# Model build functions
# Each reads the YAML model-config section + graph_data and returns an
# instantiated (untrained) model.
# ---------------------------------------------------------------------------

def _build_static_gnn(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module:
    return StaticGNN(
        num_preprocess_layers  = model_cfg["num_preprocess_layers"],
        embed_dim_preprocess   = model_cfg["embed_dim_preprocess"],
        num_postprocess_layers = model_cfg["num_postprocess_layers"],
        num_conv_layers        = model_cfg["num_conv_layers"],
        aggr                   = model_cfg["aggr"],
        num_node_features      = 3,
        hidden_channels        = model_cfg["hidden_channels"],
        num_classes            = n_nodes,
        dropout_rate           = model_cfg["dropout_rate"],
        batch_norm             = model_cfg["batch_norm"],
        skip                   = model_cfg["skip"],
    )


def _build_backtracking(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module:
    return BacktrackingNetwork(
        node_feat_dim = 3,
        edge_feat_dim = graph_data["T"],      # T = number of time steps
        hidden_dim    = model_cfg["hidden_dim"],
        num_layers    = model_cfg["num_layers"],
    )


def _build_temporal_gnn(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module:
    return TemporalGNN(
        in_channels      = 3,
        hidden_channels  = model_cfg["hidden_channels"],
        out_channels     = 1,
        num_snapshots    = graph_data["num_snapshots"],
    )


def _build_dbgnn(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module:
    return DBGNN(
        hidden_channels = model_cfg["hidden_channels"],
        num_conv_layers = model_cfg["num_conv_layers"],
        conv_type       = model_cfg.get("conv_type", "sage"),
        dropout_rate    = model_cfg.get("dropout_rate", 0.2),
    )


def _build_dag_gnn(model_cfg: dict, n_nodes: int, graph_data: dict) -> torch.nn.Module:
    return DAGGNN(
        hidden_channels = model_cfg["hidden_channels"],
        num_conv_layers = model_cfg["num_conv_layers"],
        dropout_rate    = model_cfg.get("dropout_rate", 0.2),
        agg             = model_cfg.get("agg", "mean"),
    )


# ---------------------------------------------------------------------------
# Model registry
# The key is the string used in YAML configs (``model: <key>``) and on the
# CLI (``--model <key>``).
# ---------------------------------------------------------------------------

MODEL_REGISTRY["static_gnn"] = ModelSpec(
    cls        = StaticGNN,
    forward_fn = static_gnn_forward,
    builder_fn = build_static_graph,
    build_fn   = _build_static_gnn,
)

MODEL_REGISTRY["backtracking"] = ModelSpec(
    cls        = BacktrackingNetwork,
    forward_fn = backtracking_forward,
    builder_fn = build_temporal_activation,
    build_fn   = _build_backtracking,
)

MODEL_REGISTRY["temporal_gnn"] = ModelSpec(
    cls        = TemporalGNN,
    forward_fn = temporal_gnn_forward,
    builder_fn = build_temporal_snapshots,
    build_fn   = _build_temporal_gnn,
)

MODEL_REGISTRY["dbgnn"] = ModelSpec(
    cls        = DBGNN,
    forward_fn = dbgnn_forward,
    builder_fn = build_de_bruijn_graph,
    build_fn   = _build_dbgnn,
)

MODEL_REGISTRY["dag_gnn"] = ModelSpec(
    cls        = DAGGNN,
    forward_fn = dag_gnn_forward,
    builder_fn = build_dag_event_graph,
    build_fn   = _build_dag_gnn,
)

__all__ = [
    "StaticGNN",
    "BacktrackingNetwork",
    "TemporalGNN",
    "DBGNN",
    "DAGGNN",
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_model_spec",
]
