---
description: >
  Reads a research paper (PDF or notes) and produces a structured implementation
  specification that the model-builder agent can follow. Use this before
  /implement-model to prepare the paper spec.
---

# Analyze Paper for Implementation

## Step 1: Read the Paper
Ask the user for:
- Path to the paper PDF or notes
- Which specific model/method from the paper to focus on

## Step 2: Extract Architecture Specification
Read the paper and extract into a structured spec:

```markdown
# [Model Name] — Implementation Specification
**Paper**: [Full citation]
**Focus**: [Which method/model from the paper]

## Architecture Overview
[High-level description of what the model does]

## Input Format
- Node features: [what features, dimensions]
- Edge features: [if any]
- Graph structure: [static/temporal, directed/undirected]
- Temporal info: [how time is handled]

## Layer-by-Layer Architecture
### Layer 1: [Name]
- Type: [MLP / GCN / GAT / custom]
- Input dim: [X]
- Output dim: [Y]
- Activation: [ReLU / etc.]
- Notes: [any special behavior]

### Layer 2: [Name]
[... repeat for each layer]

## Loss Function
- Type: [NLL / cross-entropy / custom]
- Formula: [LaTeX if available]

## Training Details (from paper)
- Optimizer: [Adam / SGD / etc.]
- Learning rate: [value or schedule]
- Epochs: [number]
- Batch size: [if applicable]

## Key Equations
[LaTeX equations from the paper that define the model]

## Ambiguities & Decisions Needed
- [List anything the paper doesn't specify clearly]
- [Suggest reasonable defaults for each]

## Integration Notes
- How this connects to TSIRData input format
- Required changes to config system
- Compatibility with existing evaluation pipeline
```

## Step 3: Save Specification
Save to `papers/<model_name>_spec.md`

## Step 4: Recommend Next Steps
Tell the user to run `/implement-model` with this spec.
