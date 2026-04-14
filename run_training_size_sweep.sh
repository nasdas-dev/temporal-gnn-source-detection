#!/usr/bin/env bash
# Training Set Size Sweep
#
# Trains a model with varying numbers of MC training samples (n_mc) to
# measure how performance scales with training data size.
#
# Usage:
#   bash run_training_size_sweep.sh [--data ARTIFACT] [--cfg CONFIG_YML] [--model MODEL]
#
# Defaults:
#   --data  france_office:latest
#   --cfg   exp/france_office/training_size_sweep.yml
#
# Each run is a full main_train.py call with --override train.n_mc=N.
# W&B tags the run with "training_size_sweep" so it can be queried by
# viz/training_size_scaling.py.
#
# After all runs complete, visualise with:
#   python viz/training_size_scaling.py --artifact france_office

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DATA="france_office:latest"
CFG="exp/france_office/training_size_sweep.yml"
N_MC_VALUES=(10 25 50 100 200 350 500)

# ---------------------------------------------------------------------------
# Parse CLI arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)  DATA="$2";  shift 2 ;;
        --cfg)   CFG="$2";   shift 2 ;;
        *)       echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "============================================================"
echo "Training Size Sweep"
echo "  Data    : $DATA"
echo "  Config  : $CFG"
echo "  n_mc    : ${N_MC_VALUES[*]}"
echo "============================================================"
echo ""

TOTAL=${#N_MC_VALUES[@]}
IDX=0

for N_MC in "${N_MC_VALUES[@]}"; do
    IDX=$((IDX + 1))
    echo "------------------------------------------------------------"
    echo "  Run $IDX/$TOTAL  |  n_mc = $N_MC"
    echo "------------------------------------------------------------"

    python main_train.py \
        --cfg  "$CFG" \
        --data "$DATA" \
        --override train.n_mc="$N_MC"

    echo ""
done

echo "============================================================"
echo "Sweep complete ($TOTAL runs)."
echo ""
echo "Visualise with:"
echo "  python viz/training_size_scaling.py --artifact ${DATA%%:*}"
echo "============================================================"
