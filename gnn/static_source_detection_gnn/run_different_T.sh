#!/bin/bash

# -------- CONFIGURATION --------
LOG_DIR="run_logs"
NWK="nwk/karate.yaml"

mkdir -p "$LOG_DIR"

# -------- GENERATE SEED --------
SEED=$(od -An -N8 -tu8 < /dev/urandom | tr -d ' ')
echo "Generated 64-bit seed: $SEED"

# -------- FIXED PARAMETERS --------
AGG="sum"
BN=true
BS=128
BETA=1.3
DROPOUT=0.1
EMB=16
FA=false
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
SAMPLED_T=false

# -------- VARIABLE PARAMETER --------
Ts=(0.2125 0.4250 0.6375 0.8500 1.0625 1.2750 1.4875 1.7000 1.9125 2.1250 2.3375 2.5500 2.7625 2.9750 3.1875 3.4000) # Karate
#Ts=(0.085 0.170 0.255 0.340 0.425 0.510 0.595 0.680 0.765 0.850 0.935 1.020 1.105 1.190 1.275 1.360) # Iceland
#Ts=(0.55 1.10 1.65 2.20 2.75 3.30 3.85 4.40 4.95 5.50 6.05 6.60 7.15 7.70 8.25 8.80) # Dolphin
#Ts=(0.875 1.750 2.625 3.500 4.375 5.250 6.125 7.000 7.875 8.750 9.625 10.500 11.375 12.250 13.125 14.000) # Fraternity
#Ts=(0.875 1.750 2.625 3.500 4.375 5.250 6.125 7.000 7.875 8.750 9.625 10.500 11.375 12.250 13.125 14.000) # Workplace
#Ts=(1.875 3.750 5.625 7.500 9.375 11.250 13.125 15.000 16.875 18.750 20.625 22.500 24.375 26.250 28.125 30.000) # High School (2013)
#Ts=(2.625 5.250 7.875 10.500 13.125 15.750 18.375 21.000 23.625 26.250 28.875 31.500 34.125 36.750 39.375 42.000) # High School (2010)

# -------- LOOP OVER Ts --------
COUNTER=0

for T in "${Ts[@]}"; do
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

  echo "🔁 Run $COUNTER (T=$T)..."
  echo "$PARAMS" > "$LOGFILE.params.json"

  python3 -u -m sourcedet.run_training --nwk "$NWK" --params "$(cat "$LOGFILE.params.json")" --seed "$SEED" 2>&1 | tee "$LOGFILE"
done

# -------- DONE --------
echo "✅ All runs complete. Logs saved in $LOG_DIR"