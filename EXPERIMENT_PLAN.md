# Experiment Plan: toy_holme — 3-Model Comparison

## Goal
Run a clean, reproducible experiment on `toy_holme` comparing:
1. **StaticGNN** — static projection baseline (collapses temporal edges into one graph)
2. **TemporalGNN** — one SAGE layer per time-slice, processed in reverse
3. **BacktrackingNetwork** — Ru et al. kernel-based GNN with edge activation patterns

Then make the pipeline trivially extensible to other datasets (change one config value).

---

## Phase 0: Codebase Cleanup (before running anything)

### 0.1 — Remove `gnn/static_source_detection_gnn/` entirely
**Why:** This is a legacy standalone package with its own SIR simulator, training loop, evaluation, and network files — completely duplicating the main pipeline. The main pipeline's `StaticGNN` (in `gnn/static_gnn.py`) already serves as the static projection baseline and is registered in `MODEL_REGISTRY`. Keeping this subpackage creates confusion about which "static GNN" is being used.

**Action:** `rm -rf gnn/static_source_detection_gnn/`

### 0.2 — Remove `main_gnn.py` if it's dead code
**Why:** The actual training entry point is `main_train.py`. Check if `main_gnn.py` is a stale duplicate or serves a different purpose. If it's unused or superseded by `main_train.py`, delete it.

**Action:** Read `main_gnn.py`. If it duplicates `main_train.py` logic or is not referenced anywhere, delete it.

### 0.3 — Audit `gnn/predict.py`
**Why:** Check if this module is used by the current pipeline or is leftover. The `Trainer.predict_from_tensor()` method handles inference. If `predict.py` is unused, remove it.

**Action:** Read it, grep for imports. Remove if dead.

### 0.4 — Audit `ORCHESTRATION_PLAN.md` and `SETUP_GUIDE.md`
**Why:** These may contain stale information now superseded by this plan and `CLAUDE.md`. If they describe the old pipeline or are outdated, remove or update them.

**Action:** Read both. Remove if stale, or update if they contain useful operational info.

### 0.5 — Clean up `__pycache__` references
**Action:** Already gitignored. Just ensure no `__pycache__` dirs are tracked.

### 0.6 — Verify `.claude/` agents directory
**Why:** The `.claude/agents/` directory wasn't committed. Check if it needs to be (it contains agent definitions referenced in CLAUDE.md).

**Action:** Read `.claude/` contents. If agents are defined there and useful, add to git. If it's just IDE/tool config, keep gitignored.

---

## Phase 1: Verify & Fix Configs

### 1.1 — Verify `exp/toy_holme/tsir.yml`
The current config is:
```yaml
nwk:
  type: empirical
  name: toy_holme
  t_max: 7
sir:
  beta: 0.30
  mu: 0.20
  start_t: 0
  end_t: 7
  n_runs: 5000
  mc_runs: 500
```

**Check:** This looks reasonable for a small network (toy_holme has ~6 nodes, 7 timesteps). The 5000 ground-truth runs and 500 MC runs per source should be sufficient. **No changes needed** unless the network is larger than expected.

### 1.2 — Verify `exp/toy_holme/static_gnn.yml`
**Check these parameters:**
- `train.n_mc: 500` — must match `sir.mc_runs` from tsir.yml (currently matches)
- `train.reps: 3` — good, gives variance estimate
- `eval.n_truth: 1000` — must be ≤ `sir.n_runs` (5000). Fine.
- `eval.min_outbreak: 2` — filters single-node outbreaks. Appropriate.
- Model hyperparams: 4 conv layers, 64 hidden, sum aggregation, skip connections — reasonable baseline.

**No changes needed.**

### 1.3 — Verify `exp/toy_holme/backtracking.yml`
**Read this file and check:**
- That `model: backtracking` is set correctly
- That backtracking-specific params exist: `hidden_dim`, `num_layers`
- That `train.*` and `eval.*` match the static_gnn config (same training regime for fair comparison)

**Critical:** All three model configs MUST share identical `train.*` and `eval.*` sections for a fair comparison. The only difference should be model-specific hyperparameters.

**Action:** If train/eval params differ across configs, normalize them to identical values.

### 1.4 — Verify `exp/toy_holme/temporal_gnn.yml`
Same checks as 1.3. Ensure:
- `model: temporal_gnn`
- temporal_gnn-specific: `hidden_channels`, `num_snapshots` (should auto-derive from network)
- Same train/eval sections

### 1.5 — Verify `exp/toy_holme/eval.yml` (baselines)
Should list at minimum: `uniform`, `random`, `degree`, `jordan_center`. Closeness and betweenness are nice-to-have.

### 1.6 — Create a shared experiment config pattern (optional but recommended)
**Why:** To ensure identical train/eval params across models without manual sync.

**Action:** Consider creating `exp/toy_holme/_shared.yml` with:
```yaml
eval:
  min_outbreak: 2
  top_k: [1, 3, 5]
  inverse_rank_offset: [0]
  n_truth: 1000
train:
  n_mc: 500
  reps: 5        # increase from 3 to 5 for better variance estimate
  test_size: 0.30
  batch_size: 128
  epochs: 500
  patience: 10
  lr: 0.001
  weight_decay: 0.0005
  seed: 42
```

Then each model config uses YAML anchors or the Config system merges shared + model-specific. **However**, if the current Config system doesn't support inheritance, just manually keep them in sync. Don't over-engineer this.

---

## Phase 2: Verify Pipeline Code

### 2.1 — Verify `graph_builder.py` handles toy_holme correctly
The toy_holme network is small (~6 nodes). Check:
- `build_static_graph(H)` — collapses temporal edges. Should work.
- `build_temporal_activation(H)` — builds binary activation pattern [E, T]. Ensure T = t_max+1.
- `build_temporal_snapshots(H)` — builds per-timestep edge_index dict. Check `group_by_time` param.

**Action:** Read `graph_builder.py` and verify these three functions handle the toy_holme network structure correctly (empirical CSV format with timestamps 1-7).

### 2.2 — Verify `training/trainer.py` forward functions
Each model has a different forward signature dispatched by the Trainer:
- `static_gnn_forward(model, X_batch, graph_data, device)` — replicates graph B times
- `backtracking_forward(model, X_batch, graph_data, device)` — passes edge_index + edge_attr
- `temporal_gnn_forward(model, X_batch, graph_data, device)` — passes dict of edge_indices

**Action:** Read the forward functions in `trainer.py`. Ensure they correctly unpack `graph_data` from the builder. This is the most likely place for bugs — the builder outputs a dict, and the forward function must know the exact keys.

### 2.3 — Verify the metric computation in `main_train.py`
After inference, `main_train.py` computes:
1. `Trainer.predict_from_tensor(truth_S, truth_I, truth_R)` → probs
2. `compute_ranks(probs, n_nodes, n_truth)` → ranks
3. `top_k_score(ranks, sel, k)` for each k
4. `rank_score(ranks, sel, offset)` for each offset

**Check:** The `sel` mask (valid outbreaks with ≥ min_outbreak infected) is computed correctly. The `possible` mask from TSIRData should be applied.

### 2.4 — Verify wandb artifact resolution
`load_tsir_data("toy_holme:latest")` must resolve to the correct artifact. After running tsir, the artifact name is set in `main_tsir.py` as the `--data` argument value.

**Check:** That `main_tsir.py --data toy_holme` creates artifact named `toy_holme` (not something else like `exp_toy_holme`).

---

## Phase 3: Run the Experiment

### 3.1 — Generate SIR data
```bash
python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme
```

**Expected output:**
- wandb run in `source-detection` project
- Artifact `toy_holme:v0` containing binary SIR arrays + network pickle
- Should complete in < 5 minutes for a ~6 node network

**Verify:** Check wandb dashboard or local `data/` directory for output files.

### 3.2 — Train & evaluate all three models
Run sequentially (each takes minutes on toy_holme):

```bash
# 1. Static GNN (baseline)
python main_train.py --cfg exp/toy_holme/static_gnn.yml --data toy_holme:latest

# 2. Temporal GNN
python main_train.py --cfg exp/toy_holme/temporal_gnn.yml --data toy_holme:latest

# 3. Backtracking Network
python main_train.py --cfg exp/toy_holme/backtracking.yml --data toy_holme:latest
```

**Expected output per model:**
- wandb run with logged metrics:
  - `mean_top_1`, `mean_top_3` (fraction of times true source in top-k)
  - `mean_inverse_rank` (higher = better)
  - Training curves (train/val NLL per epoch)
- Each run should complete quickly on this small network

### 3.3 — Run baseline heuristics
```bash
python main_eval.py --cfg exp/toy_holme/eval.yml --data toy_holme:latest
```

**Expected output:**
- wandb run with same metrics for uniform, random, degree, closeness, betweenness, jordan_center

### 3.4 — Alternative: Use run_experiment.sh
```bash
./run_experiment.sh toy_holme --models "static_gnn temporal_gnn backtracking"
```
This wraps all three stages. **But verify** the script passes the right `--data` artifact references.

---

## Phase 4: Results Comparison

### 4.1 — Create a comparison visualization
**Action:** Write a script `viz/plot_model_comparison.py` that:
1. Queries wandb API for all runs in the `toy_holme` experiment group
2. Extracts metrics: top-1, top-3, inverse_rank per model (including baselines)
3. Produces a grouped bar chart comparing all methods

**Structure:**
```python
import wandb
import matplotlib.pyplot as plt
import pandas as pd

def fetch_results(project="source-detection", network="toy_holme"):
    """Fetch all runs for a given network from wandb."""
    api = wandb.Api()
    runs = api.runs(project, filters={"config.nwk.name": network})
    results = []
    for run in runs:
        results.append({
            "model": run.config.get("model", run.job_type),
            "top_1": run.summary.get("mean_top_1"),
            "top_3": run.summary.get("mean_top_3"),
            "inv_rank": run.summary.get("mean_inverse_rank"),
        })
    return pd.DataFrame(results)

def plot_comparison(df, network_name):
    """Grouped bar chart of metrics by model."""
    # ... standard matplotlib grouped bar chart
```

### 4.2 — Print a summary table
At minimum, print to stdout:
```
┌─────────────────────┬───────┬───────┬────────────┐
│ Model               │ Top-1 │ Top-3 │ Inv. Rank  │
├─────────────────────┼───────┼───────┼────────────┤
│ Uniform (baseline)  │ 0.167 │ 0.500 │ 0.287      │
│ Jordan Center       │ 0.320 │ 0.640 │ 0.410      │
│ StaticGNN           │ 0.450 │ 0.780 │ 0.520      │
│ TemporalGNN         │ 0.510 │ 0.830 │ 0.580      │
│ BacktrackingNetwork │ 0.560 │ 0.870 │ 0.620      │
└─────────────────────┴───────┴───────┴────────────┘
```
(Numbers are illustrative — real results will vary.)

---

## Phase 5: Extensibility Verification

### 5.1 — Verify dataset switching works
After the toy_holme experiment succeeds, verify that switching to another network requires ONLY:
1. Having the network CSV + YML in `nwk/`
2. Having matching configs in `exp/<network_name>/`
3. Running `./run_experiment.sh <network_name>`

**Test:** Dry-run (just check configs exist) for `karate_static` or `france_office`.

### 5.2 — Document the experiment workflow
Update `CLAUDE.md` or create a brief note in `exp/README.md` describing:
- How to add a new network
- How to add a new model
- How to run a comparison experiment

---

## Execution Order Summary

```
Phase 0: Cleanup
  0.1  rm -rf gnn/static_source_detection_gnn/
  0.2  Audit & possibly remove main_gnn.py
  0.3  Audit & possibly remove gnn/predict.py
  0.4  Audit ORCHESTRATION_PLAN.md, SETUP_GUIDE.md

Phase 1: Config verification
  1.1  Verify tsir.yml
  1.2  Verify static_gnn.yml
  1.3  Verify backtracking.yml — normalize train/eval to match
  1.4  Verify temporal_gnn.yml — normalize train/eval to match
  1.5  Verify eval.yml

Phase 2: Code verification
  2.1  Verify graph_builder.py for all 3 builders
  2.2  Verify trainer.py forward functions
  2.3  Verify metric computation in main_train.py
  2.4  Verify wandb artifact naming

Phase 3: Execution
  3.1  python main_tsir.py --cfg exp/toy_holme/tsir.yml --data toy_holme
  3.2  python main_train.py (× 3 models)
  3.3  python main_eval.py (baselines)

Phase 4: Results
  4.1  Create viz/plot_model_comparison.py
  4.2  Generate comparison table + chart

Phase 5: Extensibility
  5.1  Verify dataset switching
  5.2  Document workflow
```

---

## Risk Areas (where bugs are most likely)

1. **`graph_builder.py` ↔ `trainer.py` interface**: The graph_data dict keys must exactly match what the forward functions expect. Any key mismatch = silent failure or crash.

2. **TemporalGNN `num_snapshots` config**: Must match the actual number of time slices in the network. If `group_by_time` is used, this changes. Verify the config value matches `t_max / group_by_time`.

3. **BacktrackingNetwork `edge_feat_dim`**: Must equal T (number of time slices). If `build_temporal_activation` returns `edge_attr` of shape [E, T], then `edge_feat_dim` in the config must equal T. This is likely auto-derived but verify.

4. **Small network effects**: toy_holme has ~6 nodes. With only 6 possible sources, metrics can be noisy. The `reps` parameter (recommend 5+) helps, but don't over-interpret small differences.

5. **wandb offline mode**: If no internet, set `WANDB_MODE=offline`. The pipeline should still work but artifact resolution changes.

6. **Expert knowledge masking**: Both BacktrackingNetwork and the other models mask susceptible nodes. Verify this works correctly when most nodes are susceptible (small outbreaks in a small network).

---

## What NOT to Do

- Do NOT create a new training loop or evaluation script. The existing `main_train.py` + `Trainer` + `MODEL_REGISTRY` already supports all three models.
- Do NOT modify the C SIR simulator. It works.
- Do NOT create custom data loaders per model. `SIRDataset` + `graph_builder` handle everything.
- Do NOT add wandb sweep configs for this initial experiment. Fixed hyperparams first, sweeps later.
- Do NOT refactor the Config system to support inheritance. Manual sync of train/eval params across 3 YAML files is fine for now.
