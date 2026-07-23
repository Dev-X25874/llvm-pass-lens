# llvm-pass-lens

Fine-tuning a 120B LLM to classify LLVM compiler pass interactions using real `opt`-verified ground truth.

**Result: 1.7% baseline → 72.9% fine-tuned → +71.2% improvement**

Blog post: https://sayakmondal1.substack.com/p/how-i-fine-tuned-a-120b-parameter

---

## What it does

Given two LLVM passes and an IR snippet, the model classifies their interaction as:
- `safe` — pass order does not affect the output (AB == BA)
- `interferes` — pass order changes the output (AB != BA)

Every label is derived by actually running `opt` twice and comparing normalized outputs. No LLM-generated labels.

---

## Files

| File | What it does |
|---|---|
| `ground_truth_gen.py` | Generates dataset by running real `opt` — zero hallucination |
| `trainfixed.py` | LoRA fine-tuning pipeline via TML Tinker SDK |
| `compiler_passes.jsonl` | 280 verified training examples (140 safe / 140 interferes) |
| `baseline_predictions.jsonl` | Baseline vs fine-tuned predictions on test set |
| `requirements.txt` | Python dependencies |

---

## Setup

To train (no LLVM needed — training only reads the pre-generated dataset):

    pip install -r requirements.txt
    export TINKER_API_KEY="your_key_here"
    python3 trainfixed.py

To regenerate the dataset from scratch (requires LLVM 18+ / `opt` on PATH):

    python3 ground_truth_gen.py --count 200

Generates 100 safe + 100 interferes examples verified by `opt`.

---

## Train/test split methodology

The split is grouped by unordered pass-pair, not by row. Commutativity is
symmetric — `(pass_a, pass_b)` and `(pass_b, pass_a)` always carry the same
label — so a naive row-level shuffle lets mirrored examples leak across the
train/test boundary, letting the model memorize a pair instead of generalizing.
`split_data()` groups on `frozenset({pass_a, pass_b})` before splitting, so
every mirrored pair stays entirely on one side.

---

## Results

| | Accuracy |
|---|---|
| Baseline | 1.7% |
| Fine-tuned | 72.9% |
| Improvement | +71.2% |

**Per-class breakdown (fine-tuned):**

| Label | Accuracy |
|---|---|
| interferes | 26/28 = 92.9% |
| safe | 17/31 = 54.8% |

The model is substantially stronger at detecting `interferes` than confirming
`safe` — current weak point, and the next thing to improve.

---

Built in GitHub Codespaces (dataset generation) + native Windows (training).
LLVM 18.1.3. TML Tinker SDK 0.22.3.