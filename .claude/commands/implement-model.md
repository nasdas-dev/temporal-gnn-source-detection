---
description: >
  Implements a GNN model from a paper, reviews the implementation, then drafts
  the thesis methodology section. Runs the full model-builder → paper-reviewer
  → thesis-writer pipeline.
---

# Implement Model from Paper

You are orchestrating the full model implementation pipeline. Follow these steps IN ORDER.

## Step 1: Gather Context
Ask the user for:
- Which paper/model to implement (check `papers/` for existing specs)
- Any specific modifications or constraints

## Step 2: Implementation (model-builder agent)
Use the **model-builder** subagent to:
1. Read the paper specification
2. Implement the model in `gnn/`
3. Create a default config YAML in `exp/`
4. Register in `gnn/__init__.py`

## Step 3: Review (paper-reviewer agent)
Use the **paper-reviewer** subagent to:
1. Compare the new implementation against the paper
2. Produce a structured review report
3. Save the review to `papers/reviews/<model_name>_review.md`

## Step 4: Fix Issues
If the review found ❌ Missing/Incorrect items:
1. Fix each issue in order
2. Re-run the paper-reviewer to verify fixes

## Step 5: Draft Methodology (thesis-writer agent)
Use the **thesis-writer** subagent to:
1. Draft the methodology subsection for this model
2. Save to `thesis/chapters/methodology_<model_name>.md`

## Step 6: Summary
Present a summary to the user:
- Files created/modified
- Review status (PASS / PASS WITH NOTES / FAIL)
- Next steps (training, evaluation)
