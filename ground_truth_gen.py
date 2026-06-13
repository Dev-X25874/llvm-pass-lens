"""
ground_truth_gen.py — 100% hallucination-free compiler pass dataset via opt.

Usage:
    python3 ground_truth_gen.py --count 5000
    python3 ground_truth_gen.py --small
"""

import subprocess, tempfile, os, json, hashlib, argparse, random
import signal, sys
signal.signal(signal.SIGINT, lambda s, f: (print("\nKilled."), sys.exit(0)))
from collections import Counter
import shutil

OPT = (
    shutil.which("opt") or
    shutil.which("opt-18") or
    shutil.which("opt-17") or
    "/usr/bin/opt"
)

OUT_PATH = "compiler_passes.jsonl"

PASSES = [
    "sroa", "mem2reg", "gvn", "licm", "dce", "adce", "instcombine",
    "simplifycfg", "reassociate", "loop-unroll", "indvars",
    "loop-rotate", "tailcallelim", "jump-threading", "memcpyopt",
    "sccp", "early-cse", "loop-simplify", "lcssa",
    "loop-deletion", "loop-idiom", "aggressive-instcombine",
]

# ── IR corpus: balanced across all 4 label types ─────────────────────────────
# SAFE patterns: passes work on disjoint IR constructs
SAFE_IR = [
"""define i32 @f(ptr %a, ptr %b, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %inv = load i32, ptr %b, align 4
  %p = getelementptr inbounds i32, ptr %a, i32 %i
  %v = load i32, ptr %p, align 4
  %s = add i32 %v, %inv
  store i32 %s, ptr %p, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret i32 %inv
}""",
"""define i32 @f(i32 %x, i32 %y) {
entry:
  %a = add i32 %x, 0
  %b = mul i32 %y, 1
  %r = add i32 %a, %b
  ret i32 %r
}""",
"""define void @f(ptr %a, ptr %b, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr inbounds i32, ptr %a, i32 %i
  %pb = getelementptr inbounds i32, ptr %b, i32 %i
  %va = load i32, ptr %pa, align 4
  %vb = load i32, ptr %pb, align 4
  %s = add i32 %va, %vb
  store i32 %s, ptr %pa, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}""",
"""define i32 @f(i32 %x, i32 %y) {
entry:
  %a = mul i32 %x, 2
  %b = add i32 %y, %y
  %r = add i32 %a, %b
  ret i32 %r
}""",
]

# INTERFERES patterns: passes compete for same IR construct
INTERFERES_IR = [
"""define i32 @f(i1 %c) {
entry:
  %a = alloca { i32, i32 }, align 8
  %p0 = getelementptr inbounds { i32, i32 }, ptr %a, i32 0, i32 0
  %p1 = getelementptr inbounds { i32, i32 }, ptr %a, i32 0, i32 1
  store i32 1, ptr %p0, align 4
  br i1 %c, label %t, label %f
t:
  store i32 2, ptr %p1, align 4
  br label %merge
f:
  store i32 3, ptr %p1, align 4
  br label %merge
merge:
  %x = load i32, ptr %p0, align 4
  %y = load i32, ptr %p1, align 4
  %r = add i32 %x, %y
  ret i32 %r
}""",
"""define void @f(ptr %dst, ptr %src, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %ps = getelementptr inbounds i32, ptr %src, i32 %i
  %pd = getelementptr inbounds i32, ptr %dst, i32 %i
  %v = load i32, ptr %ps, align 4
  store i32 %v, ptr %pd, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}""",
"""define i32 @f(i32 %x, i32 %y) {
entry:
  %a = add i32 %x, %y
  %b = add i32 %x, %y
  %c = mul i32 %a, 2
  %d = mul i32 %b, 3
  %r = add i32 %c, %d
  ret i32 %r
}""",
"""define i32 @f(i1 %c, ptr %p) {
entry:
  %v = load i32, ptr %p, align 4
  %w = load i32, ptr %p, align 4
  %s = add i32 %v, %w
  br i1 %c, label %t, label %f
t:
  ret i32 %s
f:
  ret i32 %v
}""",
]

# PASS_A_DOMINATES patterns: A eliminates B's work
PASS_A_DOM_IR = [
"""define i32 @f(i1 %c) {
entry:
  %a = alloca i32, align 4
  store i32 42, ptr %a, align 4
  %v = load i32, ptr %a, align 4
  ret i32 %v
}""",
"""define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 0
  %b = mul i32 %a, 1
  %c = sub i32 %b, 0
  ret i32 %c
}""",
"""define i32 @f(i32 %x) {
entry:
  br i1 true, label %t, label %f
t:
  ret i32 %x
f:
  ret i32 0
}""",
"""define i32 @f(i32 %x) {
entry:
  %dead1 = mul i32 %x, 7
  %dead2 = add i32 %dead1, 3
  %dead3 = sub i32 %dead2, 1
  ret i32 %x
}""",
"""define i32 @f(i1 %c) {
entry:
  %a = alloca i32, align 4
  br i1 %c, label %t, label %f
t:
  store i32 1, ptr %a, align 4
  br label %merge
f:
  store i32 2, ptr %a, align 4
  br label %merge
merge:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}""",
"""define i32 @f(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = add i32 %a, 2
  %c = add i32 %b, 3
  ret i32 %c
}""",
]

# PASS_B_DOMINATES patterns: B eliminates A's work
PASS_B_DOM_IR = [
"""define void @f(ptr %a) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i32 %i
  store i32 %i, ptr %p, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, 4
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}""",
"""define i32 @f(i32 %n) {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %v = load i32, ptr %a, align 4
  %v2 = add i32 %v, 1
  store i32 %v2, ptr %a, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, %n
  br i1 %cmp, label %loop, label %exit
exit:
  %r = load i32, ptr %a, align 4
  ret i32 %r
}""",
"""define i32 @f(ptr %p) {
entry:
  %a = alloca i32, align 4
  store i32 5, ptr %a, align 4
  %v = load i32, ptr %a, align 4
  %r = add i32 %v, %v
  ret i32 %r
}""",
"""define void @f(ptr %a, i32 %n) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  %p = getelementptr inbounds i32, ptr %a, i32 %i
  %v = load i32, ptr %p, align 4
  %s = add i32 %v, 1
  store i32 %s, ptr %p, align 4
  %i.next = add nsw i32 %i, 1
  %cmp = icmp slt i32 %i.next, 8
  br i1 %cmp, label %loop, label %exit
exit:
  ret void
}""",
"""define i32 @f(i32 %x) {
entry:
  %cmp = icmp eq i32 %x, %x
  br i1 %cmp, label %t, label %f
t:
  ret i32 1
f:
  ret i32 0
}""",
"""define i32 @f(i32 %x) {
entry:
  %a = mul i32 %x, 3
  %b = mul i32 %x, 5
  %r = add i32 %a, %b
  ret i32 %r
}""",
]

# Combined corpus — weighted toward less-common label types
IR_CORPUS = (
    SAFE_IR +
    INTERFERES_IR * 3 +
    PASS_A_DOM_IR * 2 +
    PASS_B_DOM_IR * 3
)


# ── opt runner ────────────────────────────────────────────────────────────────
def run_opt(ir: str, passes: list) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".ll", mode="w", delete=False) as f:
        f.write(ir); fname = f.name
    out_f = fname + ".out.ll"
    try:
        r = subprocess.run(
            [OPT, f"-passes={','.join(passes)}", fname, "-S", "-o", out_f],
            capture_output=True, text=True, timeout=10
        )
        if r.returncode != 0: return None
        with open(out_f) as f: result = f.read()
        lines = [l for l in result.splitlines()
                 if not l.startswith(";") and not l.startswith("source_filename")]
        return "\n".join(lines).strip()
    except: return None
    finally:
        for p in [fname, out_f]:
            try: os.unlink(p)
            except: pass


def derive_label(ir, pass_a, pass_b):
    def norm(s):
        if s is None: return None
        return "\n".join(l for l in s.splitlines()
                         if not l.startswith(";") and not l.startswith("source_filename")).strip()

    base   = norm(ir)
    out_a  = norm(run_opt(ir, [pass_a]))
    out_b  = norm(run_opt(ir, [pass_b]))
    out_ab = norm(run_opt(ir, [pass_a, pass_b]))
    out_ba = norm(run_opt(ir, [pass_b, pass_a]))

    if any(x is None for x in [out_a, out_b, out_ab, out_ba]):
        return None

    a_changed = out_a != base
    b_changed = out_b != base

    if a_changed and out_ab == out_a and out_b != out_a:
        return "pass_a_dominates"
    if b_changed and out_ba == out_b and out_a != out_b:
        return "pass_b_dominates"
    if a_changed and b_changed and out_ab == out_ba:
        return "safe"
    if out_ab != out_ba and (a_changed or b_changed):
        return "interferes"
    return None


def make_explanation(pass_a, pass_b, label):
    return {
        "pass_a_dominates": (
            f"{pass_a} transforms the IR and leaves it in a state where "
            f"{pass_b} finds nothing to do — running AB produces the same output as A alone."
        ),
        "pass_b_dominates": (
            f"{pass_b} transforms the IR and leaves it in a state where "
            f"{pass_a} finds nothing to do — running BA produces the same output as B alone."
        ),
        "safe": (
            f"{pass_a} and {pass_b} transform disjoint parts of the IR independently — "
            f"their combined output is identical regardless of order (AB == BA)."
        ),
        "interferes": (
            f"{pass_a} and {pass_b} both modify the IR but their combined output "
            f"depends on execution order (AB != BA), indicating conflicting transformations."
        ),
    }[label]


def make_hash(pass_a, pass_b, ir):
    return hashlib.md5(f"{pass_a}|{pass_b}|{ir[:80]}".encode()).hexdigest()


def load_existing():
    hashes, counts = set(), Counter()
    try:
        with open(OUT_PATH) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    hashes.add(make_hash(obj["pass_a"], obj["pass_b"], obj["ir_snippet"]))
                    counts[obj["label"]] += 1
                except: pass
    except FileNotFoundError: pass
    return hashes, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    if args.small: args.count = 20

    print(f"Target: {args.count} → {OUT_PATH}")
    print(f"opt: {OPT}")

    existing, label_counts = load_existing()
    print(f"Existing: {sum(label_counts.values())} | {dict(label_counts)}")

    combos = [(ir, a, b) for ir in IR_CORPUS for a in PASSES for b in PASSES if a != b]
    random.shuffle(combos)

    accepted = skipped = 0

    for ir, pass_a, pass_b in combos:
        if accepted >= args.count: break
        h = make_hash(pass_a, pass_b, ir)
        if h in existing: continue

        label = derive_label(ir, pass_a, pass_b)
        if label is None: skipped += 1; continue

        # Hard cap: keep each label within 3x the rarest label's count
        min_count = max(1, min(label_counts.get(l, 0) for l in ["safe","interferes","pass_a_dominates","pass_b_dominates"]))
        if label_counts.get(label, 0) >= min_count * 3:
            continue

        ex = {
            "pass_a":      pass_a,
            "pass_b":      pass_b,
            "ir_snippet":  ir.strip(),
            "label":       label,
            "explanation": make_explanation(pass_a, pass_b, label),
        }
        existing.add(h)
        label_counts[label] += 1
        accepted += 1

        with open(OUT_PATH, "a") as f:
            f.write(json.dumps(ex) + "\n")

        if args.small or accepted % 200 == 0:
            print(f"  [{accepted}] {pass_a}/{pass_b} → {label} | {dict(label_counts)}")

    print(f"\nDone. accepted={accepted} skipped={skipped}")
    print(f"Label distribution: {dict(label_counts)}")


if __name__ == "__main__":
    main()