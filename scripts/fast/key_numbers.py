#!/usr/bin/env python3
"""Rebuild results/headtohead.csv from the raw JSON lines of every benchmark set.

    python3 scripts/fast/key_numbers.py            # prints the CSV
    python3 scripts/fast/key_numbers.py --write    # writes results/headtohead.csv

One row per setup: medians over the scored (non-warm-up) clips, word error rate against the reference
sentences, resident memory where recorded, CPU start temperatures and the throttle flags seen.
"""
import csv, glob, json, os, re, statistics as st, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")

# (folder, file prefix glob, device, conditions, setup, description)
SETS = [
    ("pi4-headtohead-2026-10-06", "A_*", "Pi 4", "original run", "A", "Gemma per tap, Sunbird VITS per tap"),
    ("pi4-headtohead-2026-10-06", "B_*", "Pi 4", "original run", "B", "Gemma loaded, Sunbird VITS worker"),
    ("pi4-headtohead-2026-10-06", "C-5s_*", "Pi 4", "original run", "C-5s", "Whisper 5 s window + NLLB, Sunbird VITS"),
    ("pi4-headtohead-2026-10-06", "C-tuned_*", "Pi 4", "original run", "C-tuned", "Whisper 6 s window + options, NLLB, Sunbird VITS"),
    ("pi4-headtohead-2026-10-06", "C-final_*", "Pi 4", "original run", "C", "Whisper + NLLB, Sunbird VITS (as shipped)"),
    ("pi4-headtohead-2026-10-06", "D-onnx-voice_*", "Pi 4", "original run, hot CPU", "D", "Whisper + NLLB, ONNX voice"),
] + [("pi4-clean-2026-10-08", f"pi4clean-{k}_2026*", "Pi 4", "matched rerun", k, d) for k, d in [
    ("A", "Gemma per tap, Sunbird VITS per tap"), ("B", "Gemma loaded, Sunbird VITS worker"),
    ("E", "Gemma loaded, ONNX voice"), ("C", "Whisper + NLLB, Sunbird VITS"), ("D", "Whisper + NLLB, ONNX voice"),
    ("C-beam1", "Whisper + NLLB (beam 1), Sunbird VITS"), ("D-beam1", "Whisper + NLLB (beam 1), ONNX voice")]
] + [("pi5-clean-2026-10-08", f"pi5clean-{k}_2026*", "Pi 5", "5 V/5 A, active cooler", k, d) for k, d in [
    ("A", "Gemma per tap, Sunbird VITS per tap"), ("B", "Gemma loaded, Sunbird VITS worker"),
    ("E", "Gemma loaded, ONNX voice"), ("C", "Whisper + NLLB, Sunbird VITS"), ("D", "Whisper + NLLB, ONNX voice"),
    ("C-beam1", "Whisper + NLLB (beam 1), Sunbird VITS"), ("D-beam1", "Whisper + NLLB (beam 1), ONNX voice")]
] + [("pi5-2026-10-08", f"pi5-{k}_2026*", "Pi 5", "5 V/3 A, no fan (throttled)", k, d) for k, d in [
    ("A", "Gemma per tap, Sunbird VITS per tap"), ("B", "Gemma loaded, Sunbird VITS worker"),
    ("E", "Gemma loaded, ONNX voice"), ("D", "Whisper + NLLB, ONNX voice")]]


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


rows = []
for folder, pat, device, cond, setup, desc in SETS:
    paths = sorted(glob.glob(os.path.join(ROOT, folder, pat + ".jsonl")))
    if not paths:
        continue
    recs = [json.loads(l) for l in open(paths[0]) if l.strip()]
    s = [r for r in recs if not r["warmup"] and "error" not in r]
    med = lambda f: round(st.median(f(r) for r in s), 1)
    rows.append({
        "device": device, "conditions": cond, "setup": setup, "description": desc, "clips": len(s),
        "listen_s": med(lambda r: r["times"]["listen"]), "translate_s": med(lambda r: r["times"]["translate"]),
        "voice_s": med(lambda r: r["times"]["voice"]), "total_s": med(lambda r: r["total_s"]),
        "slowest_total_s": max(round(r["total_s"], 1) for r in s), "startup_s": round(s[0]["setup_startup_s"]),
        "word_error_rate": round(st.mean(wer(r["reference"], r["heard"]) for r in s), 2),
        "rss_mb": "", "cpu_start_c": f'{min(r["temp_start"] for r in s)}-{max(r["temp_start"] for r in s)}',
        "throttle_flags": " ".join(sorted({r["throttled"] for r in s})),
        "source": os.path.relpath(paths[0], os.path.join(ROOT, "..")),
    })
    mem = sorted(glob.glob(os.path.join(ROOT, folder, f"pi5mem-{setup}_*.jsonl")))
    if device == "Pi 5" and "clean" in paths[0] and mem:
        rows[-1]["rss_mb"] = round(st.median(json.loads(l)["rss_mb"] for l in open(mem[0]) if '"warmup": false' in l))

out = sys.stdout
if "--write" in sys.argv:
    out = open(os.path.join(ROOT, "headtohead.csv"), "w", newline="")
w = csv.DictWriter(out, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if out is not sys.stdout:
    print("wrote", out.name, len(rows), "rows")
