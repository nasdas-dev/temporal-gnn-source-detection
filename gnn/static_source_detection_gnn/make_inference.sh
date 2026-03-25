#!/bin/bash

# -------- CONFIGURATION --------
PARENT_DIR="Karate_T"
BASE_PATH="data/$PARENT_DIR"

# -------- RUN INFERENCE --------
if [ ! -d "$BASE_PATH" ]; then
  echo "❌ Directory '$BASE_PATH' does not exist."
  exit 1
fi

for SUBDIR in "$BASE_PATH"/*/; do
  if [ -d "$SUBDIR" ]; then
    # Remove the "data/" prefix from the path
    REL_PATH="${SUBDIR#data/}"

    echo "🔍 Running inference on: $REL_PATH"
    python3 -u -m sourcedet.run_inference "$REL_PATH"
  fi
done

echo "✅ Inference complete for all subdirectories in: $PARENT_DIR"
