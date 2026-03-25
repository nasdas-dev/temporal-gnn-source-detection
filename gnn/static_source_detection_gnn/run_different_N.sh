#!/bin/bash

# -------- CONFIGURATION --------
LOG_DIR="run_logs"
NWK="nwk/dolphin.yaml"

mkdir -p "$LOG_DIR"

# -------- GENERATE SEED --------
SEED=$(od -An -N8 -tu8 < /dev/urandom | tr -d ' ')
echo "Generated 64-bit seed: $SEED"

# -------- FIXED PARAMETERS --------
AGG="sum"
BN=true
BS=128
BETA=0.9
DROPOUT=0.1
EMB=16
FA=false
HC=64
LAYERS=6
LR=0.001
EPOCHS=500
NU=1.0
PAT=5
POST=0
PRE=2
SKIP=true
T=2.2
SAMPLED_T=false

# -------- VARIABLE PARAMETER --------
#SIMSs=(50 500 5000 10000)
SIMSs=(50)

# -------- LOOP OVER Ts --------
COUNTER=0

for SIMS in "${SIMSs[@]}"; do
  ((COUNTER++))
  LOGFILE="$LOG_DIR/run_$COUNTER.log"

  PARAMS=$(cat <<EOF
{
  "AGGREGATION": "$AGG",
  "BATCH_NORMALIZATION": $BN,
  "BATCH_SIZE": $BS,
  "BETA": $BETA,
  "DROPOUT_RATE": $DROPOUT,
  "EMBED_DIM_PREPROCESS": $EMB,
  "FEATURE_AUGMENTATION": $FA,
  "HIDDEN_CHANNELS": $HC,
  "LAYERS": $LAYERS,
  "LEARNING_RATE": $LR,
  "NEPOCHS": $EPOCHS,
  "NU": $NU,
  "PATIENCE": $PAT,
  "POSTPROCESSING_LAYERS": $POST,
  "PREPROCESSING_LAYERS": $PRE,
  "SIM_PER_SEED": $SIMS,
  "SKIP": $SKIP,
  "T": $T,
  "SAMPLED_T": $SAMPLED_T
}
EOF
)

  echo "🔁 Run $COUNTER (NSIM=$SIMS)..."
  echo "$PARAMS" > "$LOGFILE.params.json"

  python3 -u -m sourcedet.run_training --nwk "$NWK" --params "$(cat "$LOGFILE.params.json")" --seed "$SEED" 2>&1 | tee "$LOGFILE"
done

# -------- DONE --------
echo "✅ All runs complete. Logs saved in $LOG_DIR"