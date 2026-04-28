#!/usr/bin/env bash
# Replicate the static-GNN results for the Karate network from Table 5 of:
#   Sterchi et al. (2025) "Review and Benchmarks of Source Detection in Networks"
#
# Target (GNN, Karate, no DA):
#   Top-5 acc : 73.31% (±0.27%)
#   Rec. rank : 0.55   (±0.0016)
#   Error dist: 0.95   (±0.0167)
#   90% CSS   : 9.42   (±0.15)
#   Resistance: 0.2155 (±0.0002)
#
# Architecture (Table 4, Karate column):
#   1 preprocessing layer, embed_dim=16
#   3 GraphConv/sum layers, hidden=16
#   0 postprocessing layers
#   skip=True, BatchNorm, PReLU, dropout=0.1
#
# Training: 500 MC runs/node, 70/30 split, lr=1e-3, wd=5e-4, patience=5
# Test    : 100 scenarios/node = 3,400 total (matches Figure 4 caption)
#
# SIR parameters (exact paper values, continuous-time):
#   β=1.3, ν=1.0, T=0.85 → ~40% infected (from gnn/static_source_detection_gnn/run_several.sh)
#   Uses the paper's own C binary (sir/sir) via main_static_sir.py.
#
# Usage:
#   ./run_paper_replication_karate.sh            # full pipeline
#   ./run_paper_replication_karate.sh --skip-sir  # skip simulation if data exists

set -euo pipefail

NWK="karate_static"
ARTIFACT="karate_static_sir"    # distinct from TSIR artifacts
SKIP_SIR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-sir|--skip-tsir) SKIP_SIR=true ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

echo "============================================================"
echo " Paper replication: Static GNN on Karate (Sterchi et al.)"
echo "============================================================"

# ── Stage 1: Continuous-time SIR simulation ──────────────────────
if [ "${SKIP_SIR}" = false ]; then
  echo ""
  echo "=== Stage 1: Static SIR simulation (paper's C binary) ==="
  python main_static_sir.py \
    --cfg "exp/${NWK}/static_sir.yml" \
    --data "${ARTIFACT}"
else
  echo "Skipping Stage 1 (--skip-sir)"
fi

# ── Stage 2: Static GNN training (3 reps, different seeds) ───────
echo ""
echo "=== Stage 2: Static GNN training ==="
GNN_OUT=$(python main_train.py \
  --cfg "exp/${NWK}/static_gnn.yml" \
  --data "${ARTIFACT}:latest" 2>&1 | tee /dev/stderr)
# Extract run ID from wandb URL (e.g. ".../runs/fp9jy4gl") — more reliable than run name
GNN_RUN=$(echo "${GNN_OUT}" | grep -oP "/runs/\K[a-z0-9]+" | head -1 || true)
echo "  GNN run ID: ${GNN_RUN:-unknown}"

# ── Stage 3: Baselines ───────────────────────────────────────────
echo ""
echo "=== Stage 3: Baselines ==="
EVAL_OUT=$(python main_eval.py \
  --cfg "exp/${NWK}/eval.yml" \
  --data "${ARTIFACT}:latest" \
  --override eval.n_truth=100 2>&1 | tee /dev/stderr)
# Extract run ID from wandb URL — avoids needing name→ID resolution in viz
EVAL_RUN=$(echo "${EVAL_OUT}" | grep -oP "/runs/\K[a-z0-9]+" | head -1 || true)
echo "  Eval run ID: ${EVAL_RUN:-unknown}"

echo ""
echo "============================================================"
echo " Done.  Compare wandb metrics against Table 5 targets above."
echo "============================================================"

# ── Stage 4: Figures and tables ──────────────────────────────────
echo ""
echo "=== Stage 4: Figures and tables ==="
VIZ_ARGS=(--artifact "${ARTIFACT}" --output "figures/karate_replication")
[[ -n "${GNN_RUN}"  ]] && VIZ_ARGS+=(--gnn-run-id  "${GNN_RUN}")
[[ -n "${EVAL_RUN}" ]] && VIZ_ARGS+=(--eval-run-id "${EVAL_RUN}")

python viz_karate_paper.py "${VIZ_ARGS[@]}"

echo ""
echo "Figures written to figures/karate_replication/"
