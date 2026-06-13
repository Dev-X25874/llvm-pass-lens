"""
train.py — fine-tune openai/gpt-oss-120b on compiler pass data.
Fixed: parallel eval (was sequential, causing 20+ min baseline eval).
"""

import asyncio, json, math, random
import tinker
from tinker import types

MODEL_ID      = "openai/gpt-oss-120b"
DATA_PATH = "compiler_passes.jsonl"
LORA_RANK     = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS    = 5
BATCH_SIZE    = 8
SAVE_EVERY    = 10
TRAIN_SPLIT   = 0.8
RANDOM_SEED   = 42
EVAL_WORKERS  = 16  # parallel eval calls


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def split_data(examples, ratio, seed):
    random.seed(seed)
    s = examples[:]
    random.shuffle(s)
    n = int(len(s) * ratio)
    return s[:n], s[n:]


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
    pt  = tokenizer.encode(format_prompt(ex))
    ct  = tokenizer.encode(format_completion(ex))
    ft  = pt + ct
    eos = tokenizer.eos_token_id or 0
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=ft),
        loss_fn_inputs=dict(
            
            target_tokens=ft[1:] + [eos],
            weights=[0.0]*len(pt) + [1.0]*len(ct),
        )
    )


def extract_label(text):
    text = text.lower()
    for l in ["pass_a_dominates", "pass_b_dominates", "interferes", "safe"]:
        if l in text:
            return l
    return None


def batch(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i+size]


async def eval_one(sem, sampling_client, tokenizer, ex, params):
    async with sem:
        messages = [
            {"role": "system", "content": "You analyze LLVM compiler pass interactions. Respond with Label and Explanation only."},
            {"role": "user",   "content": format_prompt(ex)},
        ]
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(text)
        inp    = types.ModelInput.from_ints(tokens=tokens)

        loop   = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: sampling_client.sample(inp, 1, params).result())

        raw = tokenizer.decode(result.sequences[0].tokens).strip()
        if "<|channel|>final<|message|>" in raw:
            raw = raw.split("<|channel|>final<|message|>")[-1]
            if "<|return|>" in raw:
                raw = raw.split("<|return|>")[0]

        predicted = extract_label(raw)
        return predicted == ex["label"], ex["label"], predicted


async def eval_model(sampling_client, tokenizer, test_examples, label="Model"):
    params  = types.SamplingParams(max_tokens=80, temperature=0.0)
    sem     = asyncio.Semaphore(EVAL_WORKERS)
    tasks   = [eval_one(sem, sampling_client, tokenizer, ex, params) for ex in test_examples]

    print(f"Running {label} eval on {len(test_examples)} examples ({EVAL_WORKERS} parallel)...")
    results = await asyncio.gather(*tasks)

    correct = sum(1 for r in results if r[0])
    acc     = correct / len(results) * 100
    print(f"{label} Accuracy: {correct}/{len(results)} = {acc:.1f}%")

    from collections import defaultdict
    per_label = defaultdict(lambda: [0,0])
    for ok, exp, pred in results:
        per_label[exp][1] += 1
        if ok: per_label[exp][0] += 1
    for lbl, (c, t) in sorted(per_label.items()):
        print(f"  {lbl:<22} {c}/{t} = {c/t*100:.1f}%")
    return acc


async def train():
    all_ex = load_jsonl(DATA_PATH)
    train_ex, test_ex = split_data(all_ex, TRAIN_SPLIT, RANDOM_SEED)
    print(f"Dataset: {len(all_ex)} total | {len(train_ex)} train | {len(test_ex)} test")

    service_client = tinker.ServiceClient()

    print(f"Creating LoRA training client — {MODEL_ID}, rank={LORA_RANK}")
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_ID, rank=LORA_RANK,
    )
    tokenizer = training_client.get_tokenizer()
    print("Tokenizer ready.")

    # Baseline eval — parallel
    print("\n" + "="*50)
    print("BASELINE EVAL")
    print("="*50)
    base_sc  = service_client.create_sampling_client(base_model=MODEL_ID)
    base_acc = await eval_model(base_sc, tokenizer, test_ex, "Baseline")

    # Tokenize
    print("\nTokenizing training data...")
    train_data  = [make_datum(ex, tokenizer) for ex in train_ex]
    adam_params = types.AdamParams(learning_rate=LEARNING_RATE, weight_decay=0.01)
    total_steps = math.ceil(len(train_data) / BATCH_SIZE) * NUM_EPOCHS
    global_step = 0

    print(f"\n{'='*50}")
    print(f"TRAINING: {NUM_EPOCHS} epochs | batch={BATCH_SIZE} | {total_steps} steps")
    print("="*50)

    loop = asyncio.get_event_loop()
    for epoch in range(NUM_EPOCHS):
        for b in batch(train_data, BATCH_SIZE):
            await loop.run_in_executor(None, lambda b=b: training_client.forward_backward(b, "cross_entropy").result())
            await loop.run_in_executor(None, lambda: training_client.optim_step(adam_params).result())
            global_step += 1
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Step {global_step}/{total_steps}", flush=True)

            if global_step % SAVE_EVERY == 0:
                ckpt = f"checkpoint-step-{global_step}"
                await loop.run_in_executor(None, lambda c=ckpt: training_client.save_state(c).result())
                print(f"  Saved: {ckpt}")

        print(f"Epoch {epoch+1} complete.")

    print("Saving final weights...")
    ft_sc = await loop.run_in_executor(None, training_client.save_weights_and_get_sampling_client)
    print("Weights saved.")

    print("\n" + "="*50)
    print("FINE-TUNED EVAL")
    print("="*50)
    ft_acc = await eval_model(ft_sc, tokenizer, test_ex, "Fine-tuned")

    imp = ft_acc - base_acc
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"  Baseline accuracy:   {base_acc:.1f}%")
    print(f"  Fine-tuned accuracy: {ft_acc:.1f}%")
    print(f"  Improvement:         +{imp:.1f}%")
    print("="*50)


if __name__ == "__main__":
    asyncio.run(train())