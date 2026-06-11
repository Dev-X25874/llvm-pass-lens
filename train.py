"""
Step 2: Fine-tune a 20B model on the compiler pass dataset using Tinker.

Before running:
  1. pip install tinker anthropic
  2. set TINKER_API_KEY=your_key
  3. set ANTHROPIC_API_KEY=your_key
  4. Run generate_dataset.py first to create compiler_passes.jsonl
  5. Fill in MODEL_ID below (run list_models.py to find your 20B model)
"""

import asyncio
import json
import os
import math
import tinker
from tinker import types

# ── Config — only thing you edit ──────────────────────────────────────────────
MODEL_ID      = "YOUR_20B_MODEL_ID"   # get this from: python list_models.py
DATA_PATH     = "compiler_passes.jsonl"
LORA_RANK     = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 3
BATCH_SIZE    = 4    # examples per forward_backward call
SAVE_EVERY    = 10   # checkpoint every N optimizer steps
# ──────────────────────────────────────────────────────────────────────────────


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_prompt(ex):
    return (
        f"Analyze the interaction between these two LLVM compiler passes.\n\n"
        f"Pass A: {ex['pass_a']}\n"
        f"Pass B: {ex['pass_b']}\n\n"
        f"IR Snippet:\n{ex['ir_snippet']}\n\n"
        f"### Response\n"
    )


def format_completion(ex):
    return f"Label: {ex['label']}\nExplanation: {ex['explanation']}"


def make_datum(ex, tokenizer):
    """
    Build one Tinker Datum.

    Sequence layout:   [ prompt_tokens  |  completion_tokens ]
    Loss weights:      [      0  ...    |       1  ...       ]
    Target tokens:     full sequence shifted right by 1 (next-token prediction)
    """
    prompt_tokens     = tokenizer.encode(format_prompt(ex))
    completion_tokens = tokenizer.encode(format_completion(ex))
    full_tokens       = prompt_tokens + completion_tokens

    eos_id        = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    target_tokens = full_tokens[1:] + [eos_id]
    weights       = [0.0] * len(prompt_tokens) + [1.0] * len(completion_tokens)

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=full_tokens),
        loss_fn_inputs=dict(
            target_tokens=types.ModelInput.from_ints(tokens=target_tokens),
            weights=weights,
        )
    )


def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


async def train():
    # ── Load data ──────────────────────────────────────────────────────────────
    examples = load_jsonl(DATA_PATH)
    print(f"Loaded {len(examples)} examples from {DATA_PATH}")

    # ── Connect to Tinker ──────────────────────────────────────────────────────
    service_client = tinker.ServiceClient()   # reads TINKER_API_KEY from env

    print(f"Creating LoRA training client — model: {MODEL_ID}, rank: {LORA_RANK}")
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_ID,
        rank=LORA_RANK,
    )

    tokenizer = training_client.get_tokenizer()
    print(f"Tokenizer ready: {type(tokenizer).__name__}")

    # ── Tokenize all examples once ─────────────────────────────────────────────
    print("Tokenizing dataset...")
    data = [make_datum(ex, tokenizer) for ex in examples]
    print(f"Dataset tokenized: {len(data)} Datum objects")

    # ── Training loop ──────────────────────────────────────────────────────────
    adam_params  = types.AdamParams(learning_rate=LEARNING_RATE, weight_decay=0.01)
    global_step  = 0
    total_batches = math.ceil(len(data) / BATCH_SIZE) * NUM_EPOCHS

    print(f"\nStarting training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
    print(f"Total optimizer steps: {total_batches}\n")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in batch(data, BATCH_SIZE):
            # Compute gradients — returns immediately with a future
            fwdbwd_future = training_client.forward_backward(batch_data, "cross_entropy")

            # Kick off optimizer step — Tinker pipelines this automatically
            optim_future = training_client.optim_step(adam_params)

            # Wait for both to complete
            fwdbwd_result = fwdbwd_future.result()
            optim_future.result()

            epoch_loss  += fwdbwd_result.loss
            num_batches += 1
            global_step += 1

            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step}/{total_batches} | Loss: {fwdbwd_result.loss:.4f}")

            # Checkpoint
            if global_step % SAVE_EVERY == 0:
                ckpt_name = f"checkpoint-step-{global_step}"
                print(f"  Saving checkpoint: {ckpt_name}")
                training_client.save_state(ckpt_name).result()
                print(f"  Checkpoint saved.")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"\nEpoch {epoch+1} complete — avg loss: {avg_loss:.4f}\n")

    # ── Save final weights and get sampler ─────────────────────────────────────
    print("Saving final weights...")
    sampling_client = training_client.save_weights_and_get_sampling_client()
    print("Training complete. Weights saved.")

    # ── Quick sanity-check sample ──────────────────────────────────────────────
    print("\nRunning a quick inference test...")
    test_ex = examples[0]
    prompt_text   = format_prompt(test_ex)
    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_input  = types.ModelInput.from_ints(tokens=prompt_tokens)
    sample_params = types.SamplingParams(max_tokens=150, temperature=0.0)

    result = sampling_client.sample(prompt_input, 1, sample_params).result()
    generated = tokenizer.decode(result.sequences[0].tokens)

    print(f"\n--- Test Input ---\nPass A: {test_ex['pass_a']}\nPass B: {test_ex['pass_b']}")
    print(f"\n--- Model Output ---\n{generated}")
    print(f"\n--- Expected ---\n{format_completion(test_ex)}")


if __name__ == "__main__":
    asyncio.run(train())
