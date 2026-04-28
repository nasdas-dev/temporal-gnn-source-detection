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
# NOTE on SIR parameters:
#   The paper uses a continuous-time SIR with β=1.300, μ=1.0 (rates) and
#   observation time T=0.85, chosen to produce ~40% infected nodes.
#   Our TSIR C code uses discrete per-contact probabilities (β≤1, μ<1).
#   Calibration with the actual C binary shows that β=0.30, μ=0.20, end_t=4
#   reproduces the ~40% target (empirically ~42% on karate_static).
#   karate_static is a static network replicated across 101 timesteps, so
#   end_t=4 simply sets the observation window — all edge contacts are used.
#
# Usage:
#   ./run_paper_replication_karate.sh            # full pipeline
#   ./run_paper_replication_karate.sh --skip-tsir  # skip simulation if data exists

set -euo pipefail

NWK="karate_static"
ARTIFACT="karate_static"
SKIP_TSIR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tsir) SKIP_TSIR=true ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

echo "============================================================"
echo " Paper replication: Static GNN on Karate (Sterchi et al.)"
echo "============================================================"

# ── Stage 1: SIR simulation ──────────────────────────────────────
if [ "${SKIP_TSIR}" = false ]; then
  echo ""
  echo "=== Stage 1: TSIR simulation ==="
  python main_tsir.py \
    --cfg "exp/${NWK}/tsir.yml" \
    --data "${ARTIFACT}"
else
  echo "Skipping Stage 1 (--skip-tsir)"
fi

# ── Stage 2: Static GNN training (3 reps, different seeds) ───────
echo ""
echo "=== Stage 2: Static GNN training ==="
GNN_OUT=$(python main_train.py \
  --cfg "exp/${NWK}/static_gnn.yml" \
  --data "${ARTIFACT}:latest" 2>&1 | tee /dev/stderr)
GNN_RUN=$(echo "${GNN_OUT}" | grep -oP "Syncing run \K\S+" | head -1 || true)
echo "  GNN run name: ${GNN_RUN:-unknown}"

# ── Stage 3: Baselines ───────────────────────────────────────────
echo ""
echo "=== Stage 3: Baselines ==="
EVAL_OUT=$(python main_eval.py \
  --cfg "exp/${NWK}/eval.yml" \
  --data "${ARTIFACT}:latest" \
  --override eval.n_truth=100 2>&1 | tee /dev/stderr)
EVAL_RUN=$(echo "${EVAL_OUT}" | grep -oP "Syncing run \K\S+" | head -1 || true)
echo "  Eval run name: ${EVAL_RUN:-unknown}"

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
