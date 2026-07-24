# llvm-pass-lens

Fine-tuning a 120B LLM to classify LLVM compiler pass interactions using real `opt`-verified ground truth.

## ⚠️ Known issues (found and fixed during review)

- **Fixed:** the original `results_comparison.txt` and `baseline_predictions.jsonl`
  contained hardcoded placeholder text ("always guesses safe"), not real model
  output — no code in this repo ever generated those files. The previously
  reported +71.2% improvement was not a real measurement. Both files have
  been removed.
- **Fixed:** `extract_label()` used naive substring matching, so "unsafe" was
  misread as "safe", and "interfere" (without a trailing s) wasn't recognized
  at all. Both are now handled correctly with word-boundary regex.
- **Fixed:** the original 80-token eval budget was far too small for
  `gpt-oss-120b`, a reasoning model that writes an internal "analysis" pass
  before its final answer — nearly all baseline responses were getting cut
  off mid-thought. Raised to 1000 tokens so responses actually complete.
- **Fixed:** the baseline eval prompt never told the model what label
  vocabulary to use ("safe"/"interferes"), so it reasonably answered with its
  own wording ("No Interaction", "Independent", etc.) and was graded as wrong
  regardless of correct reasoning. The prompt now states the required labels
  explicitly.
- **Fixed:** train/test split originally grouped by pass-pair only, but this
  dataset reuses only ~20 unique IR snippets across many pass-pairs, so the
  same IR code could appear in both train and test. Split now groups by IR
  snippet to guarantee zero code-level leakage.
- **Fixed:** 98 of the original 280 examples (35%) belonged to pass-pairs with
  contradictory labels (e.g. the same pair labeled both `safe` and
  `interferes` across duplicate rows). These were removed
  (`filter_dataset.py` → `compiler_passes_clean.jsonl`, 182 examples
  remaining); they still need re-verification via `ground_truth_gen.py`
  before being reintroduced.
- **Known limitation:** the cleaned dataset's class balance shifted to
  `safe: 110, interferes: 72` (not the original even 140/140), and the held-out
  test set is only 39 examples — a real, verified number, but based on a
  small sample, so treat it as an estimate rather than a precise figure.

## Result (real, verified end-to-end run)

**53.8% baseline → 82.1% fine-tuned → +28.2% improvement**

(measured on a 39-example, leakage-free, contradiction-free held-out test set)

Blog post (postmortem): https://sayakmondal1.substack.com/p/i-fine-tuned-a-120b-model-on-compiler

| Label | Fine-tuned accuracy |
|---|---|
| interferes | 9/10 = 90.0% |
| safe | 23/29 = 79.3% |

Baseline accuracy is measured with the base model told the correct label
vocabulary and given enough tokens to finish its reasoning before answering
— earlier baseline numbers in this repo's history were measured unfairly
(unstated vocabulary, insufficient token budget) and are not representative.

---

## What it does

Given two LLVM passes and an IR snippet, the model classifies their interaction as:

- `safe` — pass order does not affect the output (AB == BA)
- `interferes` — pass order changes the output (AB != BA)

## Files

- `trainfixed.py` — training + eval script (fixed, see Known issues)
- `check_dataset.py` — reports contradictory-label pass-pairs, does not auto-fix
- `filter_dataset.py` — removes contradictory-label rows, writes `compiler_passes_clean.jsonl`
- `compiler_passes.jsonl` — original dataset (280 examples, contains known label
  inconsistencies — see Known issues)
- `compiler_passes_clean.jsonl` — cleaned dataset used for the real training run (182 examples)
- `contradictory_label_rows.jsonl` — the 98 rows removed, pending re-verification
- `ground_truth_gen.py` — generates ground truth labels via real `opt` runs (requires LLVM 18+)
- `baseline_raw_outputs.jsonl` / `fine-tuned_raw_outputs.jsonl` — raw model
  text for every test example, for manual spot-checking instead of trusting
  the parser blindly

## Setup

```bash
pip install -r requirements.txt
export TINKER_API_KEY="your_key_here"
python trainfixed.py
```

Requires an LLVM 18+ `opt` on PATH only if regenerating the dataset via
`ground_truth_gen.py` — training itself only reads the pre-generated
`compiler_passes_clean.jsonl`.
