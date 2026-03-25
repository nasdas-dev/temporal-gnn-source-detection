"""
Quick verification of GNN implementation against paper spec.
Uses Gemini Flash (free tier) as an independent second opinion.

Usage:
    python scripts/gemini_verify.py --code gnn/temporal_gnn.py --spec papers/temporal_gnn_spec.md

Setup:
    pip install google-generativeai
    export GEMINI_API_KEY="your-key-here"
    # Get free key at: https://aistudio.google.com/
"""
import os
import argparse
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    print("Install: pip install google-generativeai")
    print("Get free API key: https://aistudio.google.com/")
    exit(1)


def verify(code_path: str, spec_path: str, verbose: bool = False) -> None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable")
        print("Get free key at: https://aistudio.google.com/")
        exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    code = Path(code_path).read_text()
    spec = Path(spec_path).read_text()

    prompt = f"""You are a meticulous ML research code reviewer. Compare this PyTorch GNN
implementation against its paper specification.

PAPER SPECIFICATION:
{spec}

CODE IMPLEMENTATION:
{code}

Produce a structured review:

## Component Comparison Table
| Paper Component | Present in Code? | Matches Paper? | Notes |
|----------------|-----------------|----------------|-------|
| [component] | YES/NO/PARTIAL | MATCH/MISMATCH/UNCLEAR | [details] |

## Critical Issues (if any)
- [Issue]: [What paper says] vs [What code does]

## Minor Discrepancies (if any)
- [Item]: [Details]

## Overall Verdict
PASS / FAIL / NEEDS REVIEW

## Confidence
[How confident are you? What couldn't you verify from the spec alone?]
"""

    if verbose:
        print(f"Checking {code_path} against {spec_path}...")
        print(f"Code: {len(code)} chars, Spec: {len(spec)} chars")
        print("---")

    response = model.generate_content(prompt)
    print(response.text)


def main():
    parser = argparse.ArgumentParser(
        description="Verify GNN implementation against paper spec using Gemini Flash"
    )
    parser.add_argument(
        "--code", required=True, help="Path to implementation .py file"
    )
    parser.add_argument(
        "--spec", required=True, help="Path to paper spec .md file"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print extra info"
    )
    args = parser.parse_args()

    if not Path(args.code).exists():
        print(f"Code file not found: {args.code}")
        exit(1)
    if not Path(args.spec).exists():
        print(f"Spec file not found: {args.spec}")
        exit(1)

    verify(args.code, args.spec, args.verbose)


if __name__ == "__main__":
    main()
