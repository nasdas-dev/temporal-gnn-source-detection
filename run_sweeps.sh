#!/usr/bin/env bash
# Run all hypothesis-testing sweeps.
#
# Usage:
#   ./run_sweeps.sh                         # all sweeps
#   ./run_sweeps.sh --sweep vary_beta       # only beta sweep
#   ./run_sweeps.sh --sweep vary_observation

set -euo pipefail

SWEEP="all"
MODELS="static_gnn backtracking temporal_gnn dbgnn dag_gnn"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)  SWEEP="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p logs

run_sweep() {
  local SWEEP_DIR="$1"
  echo ""
  echo "############################################################"
  echo "# Sweep: ${SWEEP_DIR}"
  echo "############################################################"
  for subdir in exp/${SWEEP_DIR}/*/; do
    local subname
    subname=$(basename "${subdir}")
    local exp_path="${SWEEP_DIR}/${subname}"
    echo ""
    echo "--- ${exp_path} ---"
    LOG="logs/${SWEEP_DIR//\//_}_${subname}.log"
    ./run_experiment.sh "${exp_path}" --models "${MODELS}" 2>&1 | tee "${LOG}"
  done
}

case "${SWEEP}" in
  vary_beta | all)
    run_sweep "sweep_vary_beta"
    ;&
  vary_observation | all)
    run_sweep "sweep_vary_observation"
    ;;
  *)
    echo "Unknown sweep: ${SWEEP}. Options: vary_beta, vary_observation, all"
    exit 1
    ;;
esac

echo ""
echo "All sweeps complete. Logs saved to logs/"
