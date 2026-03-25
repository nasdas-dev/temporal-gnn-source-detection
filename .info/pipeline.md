# Pipeline Guide

## Three-stage execution

```
main_tsir.py  →  main_train.py  →  (future: main_eval.py)
  [data]           [any model]        [baselines]
```

Every stage is standalone. Stage 1 runs once per network+SIR config. Stage 2 runs once per model.

---

## Starting a new experiment run

**Step 1 — generate simulation data**
```bash
python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme
```
Runs the C-based SIR simulator, saves all binary arrays to `data/<wandb_run_id>/`, and publishes a versioned W&B artifact named `toy_holme`.

**Step 2 — train a model**
```bash
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest
```
Loads the artifact from Step 1, builds the graph, trains the model with early stopping, evaluates on ground-truth runs, and logs all metrics to W&B. Repeat for any other model:

```bash
python main_train.py --cfg exp/toy_holme/static_gnn.yml   --data toy_holme:latest
python main_train.py --cfg exp/toy_holme/temporal_gnn.yml --data toy_holme:latest
```

All three runs reference the same artifact — W&B links them for direct comparison.

---

## Integrating a new network

1. Place the temporal edge-list CSV at `nwk/<name>.csv` (format: `u v t` per line)
2. Create `nwk/<name>.yml` with network metadata:
```yaml
directed: no
time_steps: <t_max_from_csv>
time_granularity: ~
```
3. Create `exp/<name>/tsir.yml`:
```yaml
nwk:
  type: empirical
  name: <name>
  t_max: <value>
sir:
  beta: 0.30
  mu: 0.20
  start_t: 0
  end_t: <same as t_max>
  n_runs: 5000
  mc_runs: 500
```
4. Copy model configs from an existing experiment:
```bash
cp exp/karate_static/{static_gnn,backtracking,temporal_gnn,eval}.yml exp/<name>/
```
5. Run:
```bash
python main_tsir.py --cfg exp/<name>/tsir.yml --data <name>
python main_train.py --cfg exp/<name>/backtracking.yml --data <name>:latest
```

---

## Integrating a new model

1. **Implement** the model class in `gnn/<model_name>.py`
2. **Add a graph builder** in `gnn/graph_builder.py` — a function `build_<model_name>(H, **kwargs) -> dict` that converts the NetworkX graph into whatever tensors the model needs
3. **Add a forward function** in `training/trainer.py` — a function `<model_name>_forward(model, x_batch, graph_data, device) -> [B, N] log-probs`
4. **Register** in `gnn/__init__.py`:
```python
from .my_model import MyModel
from .graph_builder import build_my_model
from training.trainer import my_model_forward

def _build_my_model(model_cfg, n_nodes, graph_data):
    return MyModel(hidden_dim=model_cfg["hidden_dim"], ...)

MODEL_REGISTRY["my_model"] = ModelSpec(
    cls        = MyModel,
    forward_fn = my_model_forward,
    builder_fn = build_my_model,
    build_fn   = _build_my_model,
)
```
5. **Create a config** `exp/<name>/my_model.yml`:
```yaml
model: my_model
eval: { min_outbreak: 2, top_k: [1, 3, 5], ... }
train: { n_mc: 500, epochs: 500, ... }
my_model:
  hidden_dim: 64
  ...
```
6. Run — `main_train.py` needs no changes:
```bash
python main_train.py --cfg exp/<name>/my_model.yml --data <name>:latest
```

---

## Summary of all moving parts

| What | Where | Purpose |
|---|---|---|
| Network data | `nwk/<name>.csv` + `nwk/<name>.yml` | Raw temporal edge-list + metadata |
| Experiment configs | `exp/<name>/{tsir,model,eval}.yml` | All hyperparameters, nothing hardcoded |
| SIR entry point | `main_tsir.py` | Generates + uploads data artifact |
| Training entry point | `main_train.py` | Trains any registered model |
| Model code | `gnn/<model>.py` | Pure PyTorch, no pipeline logic |
| Graph builders | `gnn/graph_builder.py` | Network → model-specific tensors |
| Forward dispatch | `training/trainer.py` | Unified training loop |
| Model registry | `gnn/__init__.py` | Single place to wire everything together |
| W&B artifacts | `toy_holme:latest` etc. | Links data → training runs for comparison |
