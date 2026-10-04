#!/usr/bin/env python3
"""Score the English->Luganda outputs of each quantization against the
reference translations (chrF, BLEU) and against the unquantized F16 outputs
(exact-match rate). Prints a markdown table.

Text is normalized before scoring: typographic apostrophes/quotes become plain
ones and whitespace is collapsed, so formatting differences are not counted as
translation errors.
"""
import json
import os
import re
import sys

import sacrebleu

OUT = os.path.expanduser("~/ml/quant-sweep/outputs")
ORDER = [("f16", "F16 (unquantized)"), ("Q8_0", "Q8_0"), ("Q6_K", "Q6_K"),
         ("Q5_K_M", "Q5_K_M"), ("Q4_K_M", "Q4_K_M (deployed)"), ("Q4_0", "Q4_0"),
         ("Q3_K_M", "Q3_K_M"), ("Q2_K", "Q2_K")]
SIZES = {"f16": "sunflower-gemma4-e2b-f16.gguf", "Q8_0": "sunflower-gemma4-e2b-Q8_0.gguf",
         "Q6_K": "sunflower-gemma4-e2b-Q6_K.gguf", "Q5_K_M": "sunflower-gemma4-e2b-Q5_K_M.gguf",
         "Q4_K_M": "sunflower-gemma4-e2b-Q4_K_M.gguf", "Q4_0": "sunflower-gemma4-e2b-Q4_0.gguf",
         "Q3_K_M": "sunflower-gemma4-e2b-Q3_K_M.gguf", "Q2_K": "sunflower-gemma4-e2b-Q2_K.gguf"}


def norm(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip()


def bootstrap_diff(hyps, base_hyps, refs, n_boot=1000, seed=0):
    """95% CI for chrF(model) - chrF(F16) by resampling sentences with replacement."""
    import random
    rng = random.Random(seed)
    idx = range(len(refs))
    diffs = []
    for _ in range(n_boot):
        pick = [rng.choice(idx) for _ in idx]
        a = sacrebleu.corpus_chrf([hyps[i] for i in pick], [[refs[i] for i in pick]]).score
        b = sacrebleu.corpus_chrf([base_hyps[i] for i in pick], [[refs[i] for i in pick]]).score
        diffs.append(a - b)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def load(label):
    path = f"{OUT}/{label}.jsonl"
    if not os.path.exists(path):
        return None
    return [json.loads(l) for l in open(path, encoding="utf-8")]


runs = {k: load(k) for k, _ in ORDER}
base = runs.get("f16")
rows = []
for key, name in ORDER:
    data = runs[key]
    if not data:
        continue
    hyps = [norm(r["hyp"]) for r in data]
    refs = [norm(r["ref"]) for r in data]
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    same, ci = "", ""
    if base and key != "f16":
        bh = [norm(r["hyp"]) for r in base]
        same = f"{100 * sum(a == b for a, b in zip(hyps, bh)) / len(hyps):.0f}%"
        lo, hi = bootstrap_diff(hyps, bh, refs)
        ci = f"{chrf - base_chrf:+.1f} [{lo:+.1f}, {hi:+.1f}]"
    elif key == "f16":
        base_chrf = chrf
    gb = os.path.getsize(os.path.expanduser(f"~/ml/gguf/{SIZES[key]}")) / 1e9
    rows.append((name, gb, chrf, bleu, same, ci, len(data)))

print("| Format | Size (GB) | chrF | BLEU | chrF vs F16 [95% CI] | Same output as F16 | Sentences |")
print("|---|---:|---:|---:|---:|---:|---:|")
for name, gb, chrf, bleu, same, ci, n in rows:
    print(f"| {name} | {gb:.2f} | {chrf:.1f} | {bleu:.1f} | {ci or '—'} | {same or '—'} | {n} |")
