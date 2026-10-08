#!/usr/bin/env python3
"""Translation only: time and compare NLLB beam sizes on the benchmark sentences.

    nllb_beam_test.py MODEL_DIR CLIPS_JSON [--repeat N]

Translates the reference sentence of each clip (English to Luganda, Luganda to English) with beam
sizes 5, 2 and 1, N timed repeats each after one untimed call, and prints median seconds, tokens
and the outputs so beam 1 can be compared with beam 5.
"""
import json, statistics as st, sys, time
sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])
from nllb_translate import Translator

model_dir, clips_path = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 3
clips = json.load(open(clips_path))
ISO = {"English": "eng", "Luganda": "lug"}
tr = Translator(model_dir)
tr.translate("hello", "eng", "lug")  # untimed warm-up
rows = {}
for beam in (5, 2, 1):
    for c in clips:
        src = ISO[c["lang"]]; dst = "lug" if src == "eng" else "eng"
        times = []
        for _ in range(repeat):
            t = time.perf_counter(); out = tr.translate(c["text"], src, dst, beam_size=beam)
            times.append(time.perf_counter() - t)
        rows[(beam, c["text"])] = (st.median(times), tr.last_tokens, out)
for beam in (5, 2, 1):
    ts = [rows[(beam, c["text"])][0] for c in clips]
    print(f"beam {beam}: median {st.median(ts):.2f}s  (per sentence: " + ", ".join(f"{t:.2f}" for t in ts) + ")")
print()
for c in clips:
    outs = {b: rows[(b, c["text"])][2] for b in (5, 2, 1)}
    same = "same" if outs[1] == outs[5] else "DIFFERENT"
    print(f"{c['lang'][:3]}: {c['text']}\n   beam5: {outs[5]}\n   beam2: {outs[2]}\n   beam1: {outs[1]}   [{same}]")
