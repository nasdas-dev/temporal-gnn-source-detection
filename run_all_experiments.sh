#!/usr/bin/env bash
# Run the full pipeline for ALL primary networks.
# Progress is logged to logs/ directory.
#
# Usage:
#   ./run_all_experiments.sh
#   ./run_all_experiments.sh --models "static_gnn backtracking"  # subset of models
#   ./run_all_experiments.sh --networks "toy_holme karate_static"  # subset of networks

set -euo pipefail

MODELS="static_gnn backtracking temporal_gnn dbgnn dag_gnn"
NETWORKS="toy_holme karate_static france_office lyon_ward malawi"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)   MODELS="$2";   shift 2 ;;
    --networks) NETWORKS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

mkdir -p logs

for NWK in ${NETWORKS}; do
  echo ""
  echo "############################################################"
  echo "# Network: ${NWK}"
  echo "############################################################"
  LOG="logs/${NWK//\//_}.log"
  ./run_experiment.sh "${NWK}" --models "${MODELS}" 2>&1 | tee "${LOG}"
done

echo ""
echo "All experiments complete. Logs saved to logs/"
