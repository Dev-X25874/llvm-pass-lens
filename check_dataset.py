"""
check_dataset.py — reports data-quality issues in compiler_passes.jsonl
that need human/opt re-verification. Does NOT auto-fix anything, since
only re-running `opt` (ground_truth_gen.py) can determine which label
is actually correct for a contradictory pair.

Usage:
    python check_dataset.py
"""

import json
from collections import defaultdict, Counter

DATA_PATH = "compiler_passes.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    exs = load_jsonl(DATA_PATH)
    print(f"Total examples: {len(exs)}")

    label_counts = Counter(e["label"] for e in exs)
    print(f"Label distribution: {dict(label_counts)}")

    # Group by unordered pass-pair (same logic as split_data's old
    # pair-only grouping) to find label inconsistencies.
    groups = defaultdict(list)
    for e in exs:
        key = frozenset((e["pass_a"], e["pass_b"]))
        groups[key].append(e)

    inconsistent = {k: v for k, v in groups.items() if len(set(e["label"] for e in v)) > 1}
    print(f"\nDistinct unordered pass-pairs: {len(groups)}")
    print(f"Pass-pairs with CONTRADICTORY labels: {len(inconsistent)}")

    if inconsistent:
        print("\n--- Contradictory pass-pairs (need re-verification via ground_truth_gen.py) ---")
        for key, rows in inconsistent.items():
            print(f"\nPass pair: {set(key)}")
            for r in rows:
                print(f"  {r['pass_a']:<20} -> {r['pass_b']:<20} | label: {r['label']}")

    # IR snippet reuse / leakage risk
    ir_counts = Counter(e["ir_snippet"] for e in exs)
    print(f"\nDistinct IR snippets: {len(ir_counts)} (reused across {len(exs)} examples)")
    print("This low diversity is why train/test IR-snippet leakage was possible")
    print("with the original pass-pair-only split — trainfixed.py's split_data()")
    print("has since been patched to also group by ir_snippet.")

    # Write contradictory rows to a separate file for easy review/regen.
    if inconsistent:
        out_path = "contradictory_label_rows.jsonl"
        with open(out_path, "w") as f:
            for key, rows in inconsistent.items():
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"\nWrote {sum(len(v) for v in inconsistent.values())} contradictory rows -> {out_path}")
        print("Re-verify these with ground_truth_gen.py (requires LLVM 18+ / opt on PATH)")
        print("before trusting them in future training runs.")


if __name__ == "__main__":
    main()