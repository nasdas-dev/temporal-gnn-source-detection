#!/usr/bin/env bash
# Run the full 3-stage pipeline for one network experiment.
#
# Usage:
#   ./run_experiment.sh <network_name>
#   ./run_experiment.sh toy_holme
#   ./run_experiment.sh france_office
#   ./run_experiment.sh sweep_vary_beta/beta030
#
# The <network_name> is used both as:
#   - The experiment directory: exp/<network_name>/
#   - The W&B artifact name: <network_name> (dots replaced by underscores)
#
# Stage 1 (tsir) is skipped if --skip-tsir is passed, e.g.:
#   ./run_experiment.sh france_office --skip-tsir
#
# Model selection: pass --models to limit which models to train, e.g.:
#   ./run_experiment.sh france_office --models "static_gnn backtracking"

set -euo pipefail

NWK="${1:?Usage: $0 <network_name> [--skip-tsir] [--models 'model1 model2']}"
SKIP_TSIR=false
MODELS="static_gnn backtracking temporal_gnn dbgnn dag_gnn"

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-tsir) SKIP_TSIR=true ;;
    --models)    MODELS="$2"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
  shift
done

EXP_DIR="exp/${NWK}"
ARTIFACT="${NWK//\//.}"   # replace slashes with dots for wandb artifact name

if [ ! -d "${EXP_DIR}" ]; then
  echo "ERROR: Experiment directory '${EXP_DIR}' not found."
  exit 1
fi

echo "============================================================"
echo " Experiment : ${NWK}"
echo " Exp dir    : ${EXP_DIR}"
echo " Artifact   : ${ARTIFACT}"
echo "============================================================"

# Stage 1: TSIR simulation
if [ "${SKIP_TSIR}" = false ]; then
  if [ -f "${EXP_DIR}/tsir.yml" ]; then
    echo ""
    echo "=== Stage 1: TSIR simulation ==="
    python main_tsir.py --cfg "${EXP_DIR}/tsir.yml" --data "${ARTIFACT}"
  else
    echo "WARNING: No tsir.yml found in ${EXP_DIR}, skipping Stage 1."
  fi
else
  echo "Skipping Stage 1 (--skip-tsir)"
fi

# Stage 2: GNN training
echo ""
echo "=== Stage 2: GNN training ==="
for model in ${MODELS}; do
  cfg="${EXP_DIR}/${model}.yml"
  if [ -f "${cfg}" ]; then
    echo "--- ${model} ---"
    python main_train.py --cfg "${cfg}" --data "${ARTIFACT}:latest"
  else
    echo "  (skipping ${model}: no config at ${cfg})"
  fi
done

# Stage 3: Baselines
echo ""
echo "=== Stage 3: Baselines ==="
if [ -f "${EXP_DIR}/eval.yml" ]; then
  python main_eval.py --cfg "${EXP_DIR}/eval.yml" --data "${ARTIFACT}:latest"
else
  echo "WARNING: No eval.yml found in ${EXP_DIR}, skipping baselines."
fi

echo ""
echo "============================================================"
echo " Done: ${NWK}"
echo "============================================================"
