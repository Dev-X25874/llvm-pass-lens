"""
Step 1: Generate the compiler pass interaction dataset using Claude.
Run this first before train.py.
"""

import anthropic
import json
import sys
from collections import Counter

PROMPT = """Generate 50 compiler pass interaction examples in JSONL format. Each line is one JSON object with exactly these fields: pass_a, pass_b, ir_snippet, label, explanation
Label must be exactly one of: safe / interferes / pass_a_dominates / pass_b_dominates
Rules:
Use real LLVM IR syntax only
Every example must use a unique combination of pass_a and pass_b
Label distribution must be EXACTLY: 12 safe, 13 interferes, 12 pass_a_dominates, 13 pass_b_dominates
For pass_b_dominates: pass_b must run first conceptually and make pass_a redundant or impossible — explanation must explicitly state why pass_b's transformation eliminates pass_a's work
For pass_a_dominates: explanation must explicitly state why pass_a's transformation eliminates pass_b's work
Explanation must reference specific IR instruction names and register names from the ir_snippet
IR snippets must clearly show the asymmetry — the dominant pass's target instructions must be visible
No markdown, no backticks, no extra text
Pass names must be plain lowercase with hyphens like: licm, loop-unroll, gvn, sroa, mem2reg
Stop after exactly 50 complete JSON objects
Output raw JSONL only, one object per line, nothing else"""

REQUIRED_FIELDS = {"pass_a", "pass_b", "ir_snippet", "label", "explanation"}
VALID_LABELS    = {"safe", "interferes", "pass_a_dominates", "pass_b_dominates"}
EXPECTED_DIST   = {"safe": 12, "interferes": 13, "pass_a_dominates": 12, "pass_b_dominates": 13}
OUT_PATH        = "compiler_passes.jsonl"


def validate(examples):
    errors     = []
    seen_pairs = set()
    for i, ex in enumerate(examples):
        missing = REQUIRED_FIELDS - set(ex.keys())
        if missing:
            errors.append(f"Example {i}: missing fields {missing}")
            continue
        if ex["label"] not in VALID_LABELS:
            errors.append(f"Example {i}: invalid label '{ex['label']}'")
        pair = (ex["pass_a"], ex["pass_b"])
        if pair in seen_pairs:
            errors.append(f"Example {i}: duplicate pair {pair}")
        seen_pairs.add(pair)
    dist = Counter(ex.get("label") for ex in examples)
    for label, expected in EXPECTED_DIST.items():
        actual = dist.get(label, 0)
        if actual != expected:
            errors.append(f"Label '{label}': expected {expected}, got {actual}")
    return errors


def main():
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    print("Calling Claude to generate dataset...")
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=8096,
        messages=[{"role": "user", "content": PROMPT}]
    )

    raw      = message.content[0].text.strip()
    examples = []
    for i, line in enumerate(raw.split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            examples.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Parse error line {i}: {e}")

    print(f"Parsed {len(examples)} examples")

    errors = validate(examples)
    if errors:
        print("Validation issues:")
        for e in errors:
            print(f"  - {e}")
        if len(errors) > 5:
            print("Too many errors — aborting. Re-run to retry.")
            sys.exit(1)
        print("Minor issues — saving anyway.")
    else:
        print("All validation checks passed.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    dist = Counter(ex["label"] for ex in examples)
    print(f"Saved {len(examples)} examples to {OUT_PATH}")
    print(f"Label distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
