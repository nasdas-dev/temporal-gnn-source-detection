---
description: >
  After running experiments, this command analyzes results and drafts
  the corresponding thesis section. Reads wandb logs or result files
  and produces a results/discussion section.
---

# Write Results Section

## Step 1: Gather Experiment Info
Ask the user for:
- Which experiment was run (e.g., `exp_1_vary_n`, `exp_2_vary_beta`)
- Which models were compared
- Where results are stored (wandb run IDs or local files)

## Step 2: Analyze Results
Read the experiment outputs:
1. Check `data/` for result files
2. Read config YAMLs in `exp/` to understand what was tested
3. Look for evaluation metrics (rank_score, top_k_score, etc.)

## Step 3: Draft Results (thesis-writer agent)
Use the **thesis-writer** subagent to:
1. Write a results subsection with:
   - Experimental setup description (networks, parameters, baselines)
   - Results table with specific numbers
   - Key findings and comparison to baselines
   - Discussion of why methods performed as they did
2. Save to `thesis/chapters/results_<experiment_name>.md`

## Step 4: Suggest Visualizations
Based on the results, suggest:
- Which plots to create using `viz/`
- What the figures should show
- Figure captions for the thesis
