#!/usr/bin/env python3
"""Rebuild the head-to-head rows of results/key-numbers.csv from the raw JSON lines.

    python3 scripts/fast/key_numbers.py            # prints the CSV rows
    python3 scripts/fast/key_numbers.py --write    # writes results/headtohead.csv

Medians over the scored (non-warm-up) clips of each run file in results/pi4-headtohead-2026-10-06/.
"""
import csv, glob, json, os, statistics as st, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
RUNS = [("A", "A_*"), ("B", "B_*"), ("C-5s", "C-5s_*"), ("C-tuned", "C-tuned_*"),
        ("C-final", "C-final_*"), ("D", "D-onnx-voice_*")]

rows = []
for name, pat in RUNS:
    path = sorted(glob.glob(os.path.join(ROOT, "pi4-headtohead-2026-10-06", pat + ".jsonl")))[0]
    recs = [json.loads(l) for l in open(path) if l.strip()]
    scored = [r for r in recs if not r["warmup"] and "error" not in r]
    med = lambda f: round(st.median(f(r) for r in scored), 1)
    rows.append({
        "setup": name, "clips": len(scored),
        "listen_s": med(lambda r: r["times"]["listen"]), "translate_s": med(lambda r: r["times"]["translate"]),
        "voice_s": med(lambda r: r["times"]["voice"]), "total_s": med(lambda r: r["total_s"]),
        "slowest_total_s": max(round(r["total_s"], 1) for r in scored),
        "startup_s": scored[0]["setup_startup_s"],
        "cpu_start_temp_c_min": min(r["temp_start"] for r in scored),
        "cpu_start_temp_c_max": max(r["temp_start"] for r in scored),
        "source": os.path.relpath(path, os.path.join(ROOT, "..")),
    })

if "--write" in sys.argv:
    out = os.path.join(ROOT, "headtohead.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)
else:
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
