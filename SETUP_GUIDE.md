# Multi-Agent Thesis Setup — Complete Guide

## What This Is

A Claude Code subagent system for your master thesis on temporal contact network
source detection using GNNs. Three specialized agents collaborate through Claude Code's
built-in delegation system — no external frameworks, no extra cost beyond your
Claude subscription.

## Directory Structure

```
your-project/
├── CLAUDE.md                          # Enhanced project context (replace your existing one)
├── .claude/
│   ├── agents/
│   │   ├── model-builder.md           # Implements GNN architectures from papers
│   │   ├── paper-reviewer.md          # Verifies code matches paper specs
│   │   └── thesis-writer.md           # Drafts thesis sections
│   └── commands/
│       ├── implement-model.md         # /implement-model — full pipeline
│       ├── write-results.md           # /write-results — analyze & draft
│       └── analyze-paper.md           # /analyze-paper — create impl spec
├── papers/
│   ├── backtracking_network_spec.md   # Example paper spec (template)
│   └── reviews/                       # Agent review reports go here
├── thesis/
│   ├── outline.md                     # Thesis structure
│   ├── literature_notes.md            # Paper summaries for agents
│   └── chapters/                      # Thesis sections drafted by agents
└── [your existing code: gnn/, tsir/, setup/, eval/, viz/, exp/]
```

## Setup Instructions

### Step 1: Install the files
Copy the provided files into your project:

```bash
# From your project root:
cp -r thesis-agent-setup/.claude .claude
cp -r thesis-agent-setup/papers papers        # or merge if you already have one
cp -r thesis-agent-setup/thesis thesis        # or merge if you already have one
cp thesis-agent-setup/CLAUDE.md CLAUDE.md     # MERGE with your existing CLAUDE.md
mkdir -p papers/reviews thesis/chapters
```

**Important**: Don't blindly overwrite your existing CLAUDE.md. Merge the new
sections (especially the Agent System table and Agent Workflow) into your existing file.

### Step 2: Verify agents are loaded
Start a Claude Code session in your project and run:

```
/agents
```

You should see three agents listed:
- model-builder (blue)
- paper-reviewer (orange)
- thesis-writer (green)

### Step 3: Test with a simple task
Try:
```
Use the paper-reviewer agent to review gnn/static_gnn.py against its design intent
```

If it delegates to the orange-colored agent, everything is working.

## How to Use

### Daily Workflow

**Implementing a new model:**
```
/analyze-paper
> [provide paper PDF or describe the model]
```
This creates a spec in `papers/`. Then:
```
/implement-model
> [point to the spec just created]
```
This runs model-builder → paper-reviewer → thesis-writer in sequence.

**After running experiments:**
```
/write-results
> exp_1_vary_n, compared StaticGNN vs BacktrackingNetwork vs baselines
```

**Quick tasks (no slash command needed):**
```
Use the thesis-writer agent to improve the introduction in thesis/chapters/intro.md
```
```
Use the model-builder agent to add dropout to the TemporalGNN
```
```
Use the paper-reviewer agent to check if my GAT implementation uses the right
number of attention heads
```

### When to Invoke Agents Explicitly vs Let Claude Auto-Delegate

Claude will auto-delegate based on the agent descriptions, but you can be explicit:

| Situation | What to Do |
|-----------|-----------|
| Clear-cut task matching one agent | Let Claude auto-delegate |
| You want a specific agent | Say "Use the X agent to..." |
| Complex multi-step task | Use a slash command |
| Quick question, no agent needed | Just ask Claude directly |

### Tips for Best Results

1. **Prepare paper specs before implementation.** The model-builder works dramatically
   better when it has a structured spec in `papers/` rather than a raw PDF.

2. **Review before committing.** Always run paper-reviewer after model-builder.
   It catches subtle issues like wrong aggregation functions or missing skip connections.

3. **Keep literature_notes.md updated.** The thesis-writer references this file.
   Better notes = better thesis drafts.

4. **Use Sonnet for most tasks, Opus for hard architecture decisions.** The agents
   default to Sonnet. For a particularly complex model with ambiguous paper descriptions,
   temporarily switch: `Use the model-builder agent with opus to implement...`

5. **One model at a time.** Don't try to implement three models in one session.
   Context gets diluted. Do one full pipeline per session.

## Adding the Gemini Flash Verification Layer (Free)

For additional verification beyond the paper-reviewer agent, set up Gemini Flash
as an external check. This is optional but recommended for critical implementations.

### Setup
1. Get a free Gemini API key from https://aistudio.google.com/
2. Create a simple verification script:

```python
# scripts/gemini_verify.py
"""
Quick verification of GNN implementation against paper.
Uses Gemini Flash (free tier) as an independent second opinion.
"""
import google.generativeai as genai
import argparse
from pathlib import Path

def verify(code_path: str, spec_path: str):
    genai.configure(api_key="YOUR_KEY")  # or use env var
    model = genai.GenerativeModel("gemini-2.0-flash")

    code = Path(code_path).read_text()
    spec = Path(spec_path).read_text()

    prompt = f"""You are reviewing a PyTorch GNN implementation against its paper specification.

PAPER SPECIFICATION:
{spec}

CODE IMPLEMENTATION:
{code}

Compare the implementation to the paper specification. For each component in the spec:
1. Is it present in the code? (YES/NO/PARTIAL)
2. Does it match the paper's description? (MATCH/MISMATCH/UNCLEAR)
3. If mismatch, what specifically differs?

Be precise and cite specific line numbers or function names.
Output a structured comparison table, then an overall verdict (PASS/FAIL/NEEDS REVIEW).
"""

    response = model.generate_content(prompt)
    print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", required=True, help="Path to implementation file")
    parser.add_argument("--spec", required=True, help="Path to paper spec file")
    args = parser.parse_args()
    verify(args.code, args.spec)
```

### Usage
```bash
python scripts/gemini_verify.py --code gnn/temporal_gnn.py --spec papers/temporal_gnn_spec.md
```

This gives you a free, independent second opinion beyond Claude's paper-reviewer agent.

## Cost Management

### Estimated usage per task (Claude Pro $20/month)

| Task | Approx. Token Cost | Rate Limit Impact |
|------|-------------------|-------------------|
| /analyze-paper | ~5K tokens | Minimal |
| /implement-model (full pipeline) | ~30-50K tokens | Moderate |
| /write-results | ~15-25K tokens | Low-Moderate |
| Quick agent task | ~5-10K tokens | Minimal |

### Staying within Pro limits
- Do 1-2 major implementation sessions per day (not 5)
- Use Sonnet (default) — Opus burns limits 5x faster
- Keep paper specs concise — long specs = more input tokens
- Use the Gemini verification script for routine checks instead of paper-reviewer

### If you hit rate limits
- Switch to direct coding (you know your codebase)
- Queue up review tasks for the next rate limit reset
- Use the thesis-writer for lower-priority writing tasks (lower token cost)

## Customizing the Agents

The agents are just markdown files. Edit them freely:

- **Change the model**: Edit `model: sonnet` → `model: opus` for harder tasks
- **Restrict tools**: Add/remove from the `tools:` list
- **Modify the system prompt**: Edit the markdown content below the frontmatter
- **Add new agents**: Create new `.md` files in `.claude/agents/`

### Agent ideas you might add later:
- **experiment-planner**: Designs experiment configs and suggests hyperparameter sweeps
- **code-debugger**: Specialized in debugging PyTorch/PyG training issues
- **figure-maker**: Creates thesis-quality matplotlib figures from result data
