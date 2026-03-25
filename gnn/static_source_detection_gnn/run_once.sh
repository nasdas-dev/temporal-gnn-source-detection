#!/bin/bash

# Run with './run_once.sh'

# -------- GENERAL CONFIGURATION --------
# Network config file
NWK="nwk/karate.yaml"
# Name of directory for log files
LOG_DIR="run_logs"
# Make the directory for log files
mkdir -p "$LOG_DIR"

# -------- GENERATE SEED --------
SEED=$(od -An -N8 -tu8 < /dev/urandom | tr -d ' ')
#SEED=17942923487261980178
echo "Generated 64-bit seed: $SEED"

# -------- SPECIFIC PARAMETERS --------
PARAMS='{
  "AGGREGATION": "sum",
  "BATCH_NORMALIZATION": true,
  "BATCH_SIZE": 128,
  "BETA": 1.3,
  "DROPOUT_RATE": 0.2,
  "EMBED_DIM_PREPROCESS": 16,
  "FEATURE_AUGMENTATION": true,
  "HIDDEN_CHANNELS": 64,
  "LAYERS": 2,
  "LEARNING_RATE": 0.001,
  "NEPOCHS": 500,
  "NU": 1.0,
  "PATIENCE": 5,
  "POSTPROCESSING_LAYERS": 0,
  "PREPROCESSING_LAYERS": 0,
  "SIM_PER_SEED": 100,
  "SKIP": true,
  "T": 0.85,
  "SAMPLED_T": false
}'

# Create the log file
LOGFILE="$LOG_DIR/run_once.log"

# Output to console
echo "Running with parameters:"
echo "$PARAMS"

# Run python script
# Save output in log file but keep printing to console (with 'tee')
python3 -u -m sourcedet.run_training --nwk "$NWK" --params "$PARAMS" --seed "$SEED" 2>&1 | tee "$LOGFILE"
