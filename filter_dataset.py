"""
filter_dataset.py — removes examples belonging to contradictory-label
pass-pair groups (identified by check_dataset.py) from the training set.

Does NOT guess which label is correct for a contradictory pair — since
that requires actually re-running `opt`, this script just drops the
whole ambiguous group entirely, leaving only unambiguous, trustworthy
examples for training.

Run check_dataset.py FIRST — this script requires contradictory_label_rows.jsonl
to already exist.

Usage:
    python check_dataset.py       # writes contradictory_label_rows.jsonl
    python filter_dataset.py      # writes compiler_passes_clean.jsonl
"""

import json
from collections import Counter

SOURCE_PATH       = "compiler_passes.jsonl"
CONTRADICTORY_PATH = "contradictory_label_rows.jsonl"
OUTPUT_PATH        = "compiler_passes_clean.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def row_key(ex):
    # Unique enough identity for a row: pass pair (ordered, since
    # direction matters) + the IR snippet.
    return (ex["pass_a"], ex["pass_b"], ex["ir_snippet"])


def main():
    all_ex = load_jsonl(SOURCE_PATH)
    contradictory = load_jsonl(CONTRADICTORY_PATH)

    bad_keys = set(row_key(e) for e in contradictory)

    kept = [e for e in all_ex if row_key(e) not in bad_keys]
    dropped = len(all_ex) - len(kept)

    with open(OUTPUT_PATH, "w") as f:
        for e in kept:
            f.write(json.dumps(e) + "\n")

    label_counts = Counter(e["label"] for e in kept)
    print(f"Original dataset: {len(all_ex)} examples")
    print(f"Dropped (contradictory groups): {dropped} examples")
    print(f"Clean dataset written -> {OUTPUT_PATH} ({len(kept)} examples)")
    print(f"Clean label distribution: {dict(label_counts)}")
    print()
    print("Next step: in trainfixed.py, change:")
    print('    DATA_PATH = "compiler_passes.jsonl"')
    print("to:")
    print('    DATA_PATH = "compiler_passes_clean.jsonl"')
    print("then rerun training.")


if __name__ == "__main__":
    main()