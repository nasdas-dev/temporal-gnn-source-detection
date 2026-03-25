import json
import subprocess
import optuna
import tempfile
from random import getrandbits

# Run with:
# python3 hyperparameter_tuning.py

# Path to your training script and network YAML file
PYTHON_SCRIPT = "sourcedet.run_training"
NWK = "nwk/karate.yaml"

# How many runs?
NTRIALS = 2

# Seed for reproducibility of training data.
# This makes sure that for the same epidemic parameters, 
# we train all trials on the same training data.
SEED = getrandbits(64)

# Objective function to optimize.
# The function will return the average over the 5 last validation set losses.
def objective(trial):
    # Define the hyperparameter search space.
    # IMPORTANT: we can also fix the hyperparameters, e.g., with "AGGREGATION": "sum".
    params = {
        "AGGREGATION": "sum",
        "BATCH_NORMALIZATION": True,
        "BATCH_SIZE": 128,
        "BETA": 1.3, # Karate: 1.3 | Iceland: 5.1 | Dolphin: 0.9 | Fraternity: 0.073
        "DROPOUT_RATE": trial.suggest_categorical("DROPOUT_RATE", [0.0, 0.1, 0.2, 0.3]),
        "EMBED_DIM_PREPROCESS": trial.suggest_categorical("EMBED_DIM_PREPROCESS", [16, 32, 64]),
        "FEATURE_AUGMENTATION": False,
        "HIDDEN_CHANNELS": trial.suggest_categorical("HIDDEN_CHANNELS", [16, 32, 64]),
        "LAYERS": trial.suggest_categorical("LAYERS", [2, 3, 4, 5, 6, 7, 8]),
        "LEARNING_RATE": 0.001,
        "NEPOCHS": 500,
        "NU": 1.0,
        "PATIENCE": 5,
        "POSTPROCESSING_LAYERS": trial.suggest_categorical("POSTPROCESSING_LAYERS", [0, 1, 2]),
        "PREPROCESSING_LAYERS": trial.suggest_categorical("PREPROCESSING_LAYERS", [0, 1, 2]),
        "SIM_PER_SEED": 500,
        "SKIP": trial.suggest_categorical("SKIP", [True, False]),
        "T": 0.85, # Karate: 0.85 | Iceland: 0.34 | Dolphin: 2.2 | Fraternity: 3.5
        "SAMPLED_T": False
    }

    # Save params temporarily as JSON
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as f:
        json.dump(params, f)
        f.flush()

        # Call the training script
        result = subprocess.run(
            ["python3", "-u", "-m", PYTHON_SCRIPT, "--nwk", NWK, "--params", json.dumps(params), "--seed", str(SEED)],
            capture_output=True,
            text=True
        )

    # Return the standard output.
    print(result.stdout)
    
    # Empty list
    losses = []
    
    # Try to compute the average over the last 5 validation set losses.
    try:
        for line in result.stdout.splitlines():
            if "Val. Loss:" in line:
                try:
                    loss = float(line.split("Val. Loss:")[1].strip())
                    losses.append(loss)
                except ValueError as ve:
                    print(f"⚠️ Could not parse loss from line: {line} — {ve}")
        if losses:
            recent_losses = losses[-5:]  # Take the last 5 losses
            avg_loss = sum(recent_losses) / len(recent_losses)
            return avg_loss
    except Exception as e:
        print(f"⚠️ Failed to extract validation losses: {e}")

    # Penalize if the script fails or doesn't return a metric.
    return float('inf')

# Run the optimization.
if __name__ == "__main__":
    # Initialize the optimizer (top-5 accuracy should be maximized).
    study = optuna.create_study(direction="minimize")
    # Optimize.
    study.optimize(objective, n_trials=NTRIALS)
    # Print the averaged val. set losses of all trials to txt file.
    with open("trial_results.txt", "w") as f:
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                f.write(f"Trial {trial.number}: Value = {trial.value}\n")
    # Print random seed and best combination.
    print(f'Seed: {SEED}')
    print("Best trial:")
    print(study.best_trial)
