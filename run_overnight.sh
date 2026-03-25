#!/usr/bin/env bash
# Overnight experiment runner
# Runs all 3 networks × 3 parameter sets (beta/mu/end_t)
# Usage: bash run_overnight.sh

set -e  # stop on first error

NETWORKS="lyon_ward malawi france_office"

# Parameter sets: "beta mu end_t tag"
PARAM_SETS=(
    "0.24 0.01 250 b024"
    "0.16 0.01 250 b016"
    "0.11 0.01 250 b011"
)

for DATASET in $NETWORKS; do
    for PARAMS in "${PARAM_SETS[@]}"; do
        read -r BETA MU END_T TAG <<< "$PARAMS"
        ARTIFACT="${DATASET}_${TAG}"
        CFG_TMP="exp/${DATASET}/tsir_${TAG}.yml"

        echo ""
        echo "============================================================"
        echo "  Dataset: $DATASET  |  beta=$BETA  mu=$MU  end_t=$END_T"
        echo "============================================================"

        # --- Step 0: Write a temporary tsir config with patched parameters ---
        cp "exp/${DATASET}/tsir.yml" "$CFG_TMP"
        sed -i "s/^  beta:.*/  beta: ${BETA}/"   "$CFG_TMP"
        sed -i "s/^  mu:.*/  mu: ${MU}/"         "$CFG_TMP"
        sed -i "s/^  end_t:.*/  end_t: ${END_T}/" "$CFG_TMP"

        # --- Step 1: SIR simulations ---
        echo "[1/4] Running TSIR: $ARTIFACT"
        python main_tsir.py --cfg "$CFG_TMP" --data "$ARTIFACT"

        # Extract the run ID from the data/ directory (most recently created)
        RUN_ID=$(ls -t data/ | head -1)
        echo "      Run ID: $RUN_ID"

        # --- Step 2: Train GNN models ---
        echo "[2/4] Training backtracking ($RUN_ID)"
        python main_train.py --cfg "exp/${DATASET}/backtracking.yml"  --data "$RUN_ID"

        echo "[3/4] Training static_gnn ($RUN_ID)"
        python main_train.py --cfg "exp/${DATASET}/static_gnn.yml"    --data "$RUN_ID"

        echo "[3/4] Training temporal_gnn ($RUN_ID)"
        python main_train.py --cfg "exp/${DATASET}/temporal_gnn.yml"  --data "$RUN_ID"

        # --- Step 3: Evaluate baselines ---
        echo "[4/4] Evaluating baselines ($RUN_ID)"
        python main_eval.py  --cfg "exp/${DATASET}/eval.yml"          --data "$RUN_ID"

        echo "  Done: $ARTIFACT (run=$RUN_ID)"
    done
done

echo ""
echo "All experiments finished."
