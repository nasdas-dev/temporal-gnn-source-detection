#!/bin/bash

# -------- CONFIGURATION --------
LOG_DIR="run_logs"
NWK="nwk/karate.yaml"

mkdir -p "$LOG_DIR"

# Define runs
RUNS=(1 2 3)

# -------- PARAMETERS --------
AGG="sum"
BN=true
BS=128
BETA=1.3
DROPOUT=0.1
EMB=16
FAs=(false true)
HC=16
LAYERS=3
LR=0.001
EPOCHS=500
NU=1.0
PAT=5
POST=0
PRE=1
SIMS=500
SKIP=true
T=0.85
SAMPLED_T=false

# -------- LOOP --------

for R in "${RUNS[@]}"; do

  SEED=$(od -An -N8 -tu8 < /dev/urandom | tr -d ' ')
  echo "Generated 64-bit seed: $SEED"

  for FA in "${FAs[@]}"; do
    LOGFILE="$LOG_DIR/run_$R_$FA.log"

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

    echo "🔁 Run $R (FA=$FA)..."
    echo "$PARAMS" > "$LOGFILE.params.json"

    python3 -u -m sourcedet.run_training --nwk "$NWK" --params "$(cat "$LOGFILE.params.json")" --seed "$SEED" 2>&1 | tee "$LOGFILE"
  done 
done

# -------- DONE --------
echo "✅ All runs complete. Logs saved in $LOG_DIR"
