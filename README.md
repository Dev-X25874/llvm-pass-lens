# compiler-pass-finetune

Fine-tuning a 20B LLM on compiler pass interaction data using Tinker.

---

## Files

| File | What it does |
|---|---|
| `generate_dataset.py` | Calls Claude to generate the 50-example JSONL dataset |
| `list_models.py` | Lists all models in your Tinker account so you can find the 20B model ID |
| `train.py` | Runs the LoRA fine-tune on Tinker's GPUs |
| `requirements.txt` | Python dependencies |

---

## Setup

### Step 1 — Install dependencies

```
pip install -r requirements.txt
```

### Step 2 — Set your API keys

**Windows (Command Prompt):**
```
set TINKER_API_KEY=your_tinker_key_here
set ANTHROPIC_API_KEY=your_anthropic_key_here
```

**Windows (PowerShell):**
```
$env:TINKER_API_KEY="your_tinker_key_here"
$env:ANTHROPIC_API_KEY="your_anthropic_key_here"
```

### Step 3 — Find your 20B model ID

```
python list_models.py
```

Copy the `ID` of whichever 20B model you want. Open `train.py` and paste it into:

```python
MODEL_ID = "paste_your_model_id_here"
```

---

## Running

### Step 4 — Generate the dataset

```
python generate_dataset.py
```

This creates `compiler_passes.jsonl` with 50 examples. Check the output — it should say "All validation checks passed."

### Step 5 — Run the small test first

Before the real 20B run, confirm the pipeline works by temporarily changing `MODEL_ID` in `train.py` to a small model (e.g. `Qwen/Qwen3-8B` or whatever small model appears in your list). Run:

```
python train.py
```

If it completes without errors, the pipeline is confirmed.

### Step 6 — Real run with your 20B model

Change `MODEL_ID` back to the 20B model ID and run again:

```
python train.py
```

Tinker handles the GPUs. Your script runs locally on CPU. Training logs and checkpoints are visible at `tinker-console.thinkingmachines.ai` → Training Runs.

---

## What to expect

- Loss should drop across epochs
- A checkpoint is saved every 10 optimizer steps
- At the end, the script runs a quick inference test and prints model output vs expected output
- The final weights are saved to Tinker — visible in the console under Checkpoints

---

## Scaling up

When you're ready for a bigger dataset:

1. Edit the prompt in `generate_dataset.py` to ask for more examples (500, 1000, etc.)
2. Or run `generate_dataset.py` multiple times and concatenate the JSONL files
3. Deduplicate on `(pass_a, pass_b)` pairs
4. Re-run `train.py` — same code, same model, more data
