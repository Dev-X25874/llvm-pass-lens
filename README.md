cat > README.md << 'ENDOFFILE'
# llvm-pass-lens

Fine-tuning a 120B LLM to classify LLVM compiler pass interactions using real `opt`-verified ground truth.

**Result: 5.4% baseline → 75.0% fine-tuned → +69.6% improvement**

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

```bash
pip install -r requirements.txt
export TINKER_API_KEY="your_key_here"
```

LLVM 18+ must be installed (`/usr/bin/opt`).

---

## Generate dataset

```bash
python3 ground_truth_gen.py --count 200
```

Generates 100 safe + 100 interferes examples verified by `opt`.

---

## Train

```bash
python3 trainfixed.py
```

Trains `openai/gpt-oss-120b` with LoRA rank 16 for 5 epochs. Prints baseline vs fine-tuned accuracy at the end.

---

## Results

| | Accuracy |
|---|---|
| Baseline | 5.4% |
| Fine-tuned | 75.0% |
| Improvement | +69.6% |

---

*Built in GitHub Codespaces. LLVM 18.1.3. TML Tinker SDK 0.22.3.*
ENDOFFILE
