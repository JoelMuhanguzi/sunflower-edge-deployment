#!/usr/bin/env python3
"""Summarize bench_compare.py results: medians, accuracy vs reference, tokens/s, temperature.
Usage: bench_summary.py bench_results/*.jsonl
"""
import json, re, statistics as st, sys

def norm(s): return re.sub(r"[^\w\s']", " ", s.lower()).split()

def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)

runs = {}
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        runs.setdefault(r["label"], []).append(r)

def med(xs): return st.median(xs) if xs else float("nan")
order = [k for k in ("A", "B", "C-5s", "C-tuned", "C-final") if k in runs]
print(f"{'setup':<9}{'n':>2} {'listen':>8} {'transl.':>8} {'voice':>7} {'TOTAL':>7} {'mean':>7} {'max':>7} {'startup':>8} {'WER':>5} {'repeats':>8} {'maxC':>5}")
for k in order:
    rs = [r for r in runs[k] if not r["warmup"] and "error" not in r]
    L = [r["times"]["listen"] for r in rs]; T = [r["times"]["translate"] for r in rs]
    V = [r["times"]["voice"] for r in rs]; tot = [r["total_s"] for r in rs]
    w = [wer(r["reference"], r.get("heard", "")) for r in rs]
    rep = sum(len(norm(r.get("heard", ""))) >= 1.7 * len(norm(r["reference"])) for r in rs)
    print(f"{k:<9}{len(rs):>2} {med(L):8.1f} {med(T):8.1f} {med(V):7.1f} {med(tot):7.1f} {st.mean(tot):7.1f} {max(tot):7.1f} "
          f"{rs[0]['setup_startup_s']:8.0f} {st.mean(w):5.2f} {rep:>5}/{len(rs)} {max(r['temp_end'] for r in rs):>5}")

print("\nA without its cold first call:", end=" ")
a = [r for r in runs.get("A", []) if r["clip"] != "eng_1.wav" and "error" not in r]
if a: print(f"median total {med([r['total_s'] for r in a]):.1f}s (listen {med([r['times']['listen'] for r in a]):.1f}, "
            f"translate {med([r['times']['translate'] for r in a]):.1f}, voice {med([r['times']['voice'] for r in a]):.1f})")

print("\nTokens per second (generation) where recorded:")
for k in order:
    rs = [r for r in runs[k] if not r["warmup"] and "error" not in r]
    for stage in ("listen", "translate"):
        vals, ns = [], []
        for r in rs:
            g = (r["stats"].get(stage) or {}).get("gen") or {}
            if g.get("tps"): vals.append(g["tps"])
            elif g.get("n") and r["times"].get(stage): vals.append(g["n"] / r["times"][stage])
            if g.get("n"): ns.append(g["n"])
        if vals:
            print(f"  {k:<8}{stage:<10} median {med(vals):6.2f} tok/s over {len(vals)} runs  (median {med(ns):.0f} tokens/run)")

print("\nPer clip (transcript vs reference):")
for k in order:
    for r in runs[k]:
        if r["warmup"]: continue
        print(f"  {k:<8}{r['clip']:<9} WER {wer(r['reference'], r.get('heard','')):4.2f} listen {r['times'].get('listen','-'):>6}s | {r.get('heard','')[:90]!r}")
