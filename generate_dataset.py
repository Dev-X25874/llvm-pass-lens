"""
generate_dataset.py — ground-truth compiler pass dataset via TML gpt-oss-120b.

Usage:
    export TINKER_API_KEY="tml-..."
    python3 generate_dataset.py --label pass_b_dominates --count 300
    python3 generate_dataset.py --label interferes --count 300
    python3 generate_dataset.py --label safe --count 200
    python3 generate_dataset.py --small   # test run, 10 examples
"""

import json, sys, re, argparse, random, time
from collections import Counter
import tinker
from tinker import types

OUT_PATH = "compiler_passes.jsonl"

PASSES = [
    "sroa", "mem2reg", "gvn", "licm", "dce", "adce", "instcombine",
    "simplifycfg", "reassociate", "loop-unroll", "loop-vectorize",
    "loop-rotate", "loop-unswitch", "indvars", "loop-strength-reduce",
    "inline", "always-inline", "ipsccp", "globalopt", "globaldce",
    "tailcallelim", "jump-threading", "consthoist", "memcpyopt",
    "lower-switch", "sccp", "deadargelim",
]

LABEL_DEFS = """
EXACT DEFINITIONS — memorise before generating:

pass_a_dominates:
  Pass A runs first and PHYSICALLY REMOVES the IR construct that pass B needs.
  After A, pass B finds NOTHING to do — it is a complete no-op.
  WRONG if explanation says B "still works", "still has work", "still benefits", "independently".

pass_b_dominates:
  Pass B runs first and PHYSICALLY REMOVES the IR construct that pass A needs.
  After B, pass A finds NOTHING to do — it is a complete no-op.
  WRONG if explanation says A "still works", "still has work", "still benefits", "independently".

safe:
  Both passes transform COMPLETELY DISJOINT IR. Both do full useful work regardless of order.
  Do NOT label safe if one pass helps the other.

interferes:
  Both passes target the SAME IR construct via INCOMPATIBLE mechanisms,
  OR one pass produces IR that breaks the other's preconditions.
"""

SYSTEM = "You output only raw JSONL. No markdown, no backticks, no explanation. One JSON object per line."

def make_prompt(pass_a, pass_b, label, n):
    instr = {
        "pass_b_dominates": (
            f"Generate {n} examples with label=pass_b_dominates.\n"
            f"IR must contain the construct {pass_b} consumes.\n"
            f"After {pass_b} runs, {pass_a} must find NOTHING to do."
        ),
        "pass_a_dominates": (
            f"Generate {n} examples with label=pass_a_dominates.\n"
            f"IR must contain the construct {pass_a} consumes.\n"
            f"After {pass_a} runs, {pass_b} must find NOTHING to do."
        ),
        "safe": (
            f"Generate {n} examples with label=safe.\n"
            f"IR must contain constructs that BOTH {pass_a} AND {pass_b} each transform independently."
        ),
        "interferes": (
            f"Generate {n} examples with label=interferes.\n"
            f"IR must show both passes competing for the SAME construct via incompatible mechanisms."
        ),
    }[label]

    return f"""{LABEL_DEFS}
IR RULES:
- All %vars defined before use. No bare identifiers like 'x' — always '%x'.
- Every basic block needs a terminator (ret, br, switch).
- Use opaque ptr style: ptr not i32*.
- 8-20 lines per snippet.

Pass A: {pass_a}
Pass B: {pass_b}

{instr}

Output ONLY a JSON object per line, fields: pass_a, pass_b, ir_snippet, label, explanation.
pass_a and pass_b must be lowercase with hyphens exactly as given above.
explanation: 1-2 sentences naming specific IR instructions and registers."""


# ── validators ────────────────────────────────────────────────────────────────
REQUIRED = {"pass_a", "pass_b", "ir_snippet", "label", "explanation"}
VALID_LABELS = {"safe", "interferes", "pass_a_dominates", "pass_b_dominates"}

KNOWN_ATTR = {
    'nuw','nsw','exact','inbounds','volatile','atomic','align',
    'true','false','null','undef','poison','zeroinitializer',
    'noundef','readonly','writeonly','noalias','acquire','release',
    'acq_rel','seq_cst','monotonic','unordered','zeroext','signext',
}
TYPE_RE = r'(?:i8|i16|i32|i64|i128|f32|f64|half|bfloat|ptr|void)'
TYPE_OR_ATTR = re.compile(
    rf'^({TYPE_RE}|label|i\d+|float|double|half|ptr|void'
    r'|nuw|nsw|exact|inbounds|volatile|atomic|align|noundef'
    r'|true|false|null|undef|poison|zeroinitializer'
    r'|acquire|release|acq_rel|seq_cst|monotonic|unordered)$'
)

def validate_ir(s):
    if not s or len(s.strip()) < 20: return False, "too short"
    if not re.search(r'\bdefine\b|\bdeclare\b', s): return False, "no define"
    if not re.search(r'\bret\b|\bbr\b', s): return False, "no terminator"
    clean = re.sub(r'<\d+\s+x\s+\S+>', '<vec>', s)
    for m in re.finditer(rf'(?:{TYPE_RE})[ \t]+([A-Za-z_][A-Za-z0-9_.]*)', clean):
        t = m.group(1)
        if t not in KNOWN_ATTR: return False, f"bare id '{t}' after type"
    for m in re.finditer(r',\s*([A-Za-z_][A-Za-z0-9_.]*)\b', clean):
        t = m.group(1)
        if not TYPE_OR_ATTR.match(t): return False, f"bare id '{t}' after comma"
    # SSA def-use
    defined = set()
    for m in re.finditer(r'\(([^)]*)\)', s):
        for a in re.finditer(r'%[\w.]+', m.group(1)): defined.add(a.group(0))
    for m in re.finditer(r'(%[\w.]+)\s*=', s): defined.add(m.group(1))
    for m in re.finditer(r'^([\w.]+):\s*$', s, re.MULTILINE): defined.add('%'+m.group(1))
    phi_labels = {m.group(1) for m in re.finditer(r'\[[^\]]*,\s*(%[\w.]+)\]', s)}
    br_labels  = {m.group(1) for m in re.finditer(r'label\s+(%[\w.]+)', s)}
    for m in re.finditer(r'(%[\w.]+)', s):
        name = m.group(1)
        if re.match(r'\s*=', s[m.end():m.end()+3]): continue
        if name in phi_labels or name in br_labels: continue
        if name not in defined: return False, f"undefined '{name}'"
    return True, "ok"

def validate_example(ex, target_label):
    miss = REQUIRED - set(ex.keys())
    if miss: return False, f"missing {miss}"
    if ex["label"] != target_label: return False, f"wrong label: {ex['label']}"
    ok, r = validate_ir(ex["ir_snippet"])
    if not ok: return False, f"IR: {r}"
    expl = ex["explanation"].lower()
    pa = ex["pass_a"].lower().replace('-','').replace('_','')
    pb = ex["pass_b"].lower().replace('-','').replace('_','')
    en = expl.replace('-','').replace('_','')
    if pa not in en: return False, "explanation missing pass_a"
    if pb not in en: return False, "explanation missing pass_b"
    if target_label in ("pass_a_dominates","pass_b_dominates"):
        bad = ['still works','still has work','still finds','independently',
               'still benefits','unaffected','coexist','cooperate','both passes']
        for b in bad:
            if b in expl: return False, f"dominates but says '{b}'"
        good = ['eliminat','nothing left','no work','no loop','no remaining',
                'finds nothing','has nothing','no redundan','no dead','no alloca',
                'no induction','no call','removed','leaves','no longer','already']
        if not any(g in expl for g in good): return False, "no elimination language"
    return True, "ok"


# ── TML helpers ───────────────────────────────────────────────────────────────
def extract_final(raw):
    marker = "<|channel|>final<|message|>"
    end    = "<|return|>"
    if marker in raw:
        text = raw.split(marker)[-1]
        if end in text: text = text.split(end)[0]
        return text.strip()
    return raw.strip()

def parse_jsonl(text):
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or not line.startswith("{"): continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict): out.append(obj)
        except: pass
    return out

def normalize_pass(name):
    return name.lower().strip().replace(' ', '-').replace('_', '-')

def load_existing_hashes():
    hashes = set()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    k = f"{obj.get('pass_a')}|{obj.get('pass_b')}|{obj.get('ir_snippet','')[:100]}"
                    hashes.add(hash(k))
                except: pass
    except FileNotFoundError: pass
    return hashes


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="pass_b_dominates",
                        choices=["pass_b_dominates","pass_a_dominates","safe","interferes"])
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    if args.small: args.count = 10

    print(f"Target: {args.count} × {args.label} → {OUT_PATH}")

    service_client  = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model="openai/gpt-oss-120b")
    tokenizer       = sampling_client.get_tokenizer()
    print("Connected to gpt-oss-120b")

    existing = load_existing_hashes()
    print(f"Loaded {len(existing)} existing for dedup")

    pairs = [(a,b) for a in PASSES for b in PASSES if a != b]
    random.shuffle(pairs)

    accepted = 0
    rejected = 0
    pair_idx = 0

    while accepted < args.count:
        if pair_idx >= len(pairs):
            random.shuffle(pairs)
            pair_idx = 0

        pass_a, pass_b = pairs[pair_idx]
        pair_idx += 1

        n = min(5, args.count - accepted)
        prompt_text = make_prompt(pass_a, pass_b, args.label, n)
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": prompt_text},
        ]
        text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(text)
        inp    = types.ModelInput.from_ints(tokens=tokens)
        params = types.SamplingParams(max_tokens=2000, temperature=0.7)

        try:
            result = sampling_client.sample(inp, 1, params).result()
            raw    = tokenizer.decode(result.sequences[0].tokens).strip()
        except Exception as e:
            print(f"  [ERROR] {pass_a}/{pass_b}: {e}")
            continue

        final    = extract_final(raw)
        examples = parse_jsonl(final)

        for ex in examples:
            ex["pass_a"] = normalize_pass(pass_a)
            ex["pass_b"] = normalize_pass(pass_b)
            ex["label"]  = args.label

            ok, reason = validate_example(ex, args.label)
            if not ok:
                rejected += 1
                if args.small: print(f"  [REJECT] {pass_a}/{pass_b}: {reason}")
                continue

            k = f"{ex['pass_a']}|{ex['pass_b']}|{ex['ir_snippet'][:100]}"
            h = hash(k)
            if h in existing:
                continue
            existing.add(h)

            with open(OUT_PATH, "a") as f:
                f.write(json.dumps(ex) + "\n")
            accepted += 1
            print(f"  [+{accepted}] {pass_a}/{pass_b}", flush=True)

            if accepted >= args.count:
                break

    print(f"\nDone. accepted={accepted} rejected={rejected}")
    rate = accepted/(accepted+rejected+1)*100
    print(f"Acceptance rate: {rate:.1f}%")

if __name__ == "__main__":
    main()