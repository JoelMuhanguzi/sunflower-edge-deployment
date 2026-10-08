#!/usr/bin/env python3
"""Where does NLLB's translation time go, and does the weight precision matter?

    nllb_diagnose.py MODEL_DIR CLIPS_JSON [--repeat N] [--types int8,int8_float32,float32]

For each CTranslate2 compute type (the saved model is int8; float32 converts the weights on load):
  1. SPLIT: forces the decoder to produce exactly N tokens (min = max decoding length) for N = 2, 8, 16 on one
     sentence at beam 5 and beam 1, and fits time = fixed + per_token * N. `fixed` is the encoder plus set-up;
     `per_token` is one decoding step.
  2. SENTENCES: times the six reference sentences at beam 5 and beam 1, alternating the two so both see
     the same temperature, N repeats each, and prints medians.
"""
import json, statistics as st, sys, time
sys.path.insert(0, __file__.rsplit("/experiments/", 1)[0])
import ctranslate2
from nllb_translate import Translator, LANGUAGE_TOKENS

model_dir, clips_path = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 3
types = (sys.argv[sys.argv.index("--types") + 1] if "--types" in sys.argv else "int8,int8_float32,float32").split(",")
clips = json.load(open(clips_path))
ISO = {"English": "eng", "Luganda": "lug"}
LONG = "I am very sick, please take me to the hospital and call my family as soon as you can."


def forced(tr, text, src, dst, beam, n):
    ids = tr.tok(text)["input_ids"]; ids[0] = LANGUAGE_TOKENS[src]
    toks = tr.tok.convert_ids_to_tokens(ids)
    prefix = [tr.tok.convert_ids_to_tokens(LANGUAGE_TOKENS[dst])]
    t = time.perf_counter()
    tr.tr.translate_batch([toks], target_prefix=[prefix], beam_size=beam, min_decoding_length=n, max_decoding_length=n)
    return time.perf_counter() - t


for ct in types:
    t0 = time.time()
    tr = Translator(model_dir, compute_type=ct)
    print(f"\n##### compute_type={ct}  (load {time.time() - t0:.1f}s)", flush=True)
    tr.translate("hello", "eng", "lug")
    print("SPLIT  (forced output length N; time = fixed + per_token * N)")
    for beam in (5, 1):
        pts = []
        for n in (2, 8, 16):
            forced(tr, LONG, "eng", "lug", beam, n)
            pts.append((n, st.median(forced(tr, LONG, "eng", "lug", beam, n) for _ in range(repeat))))
        slope = (pts[2][1] - pts[0][1]) / (pts[2][0] - pts[0][0])
        fixed = pts[0][1] - slope * pts[0][0]
        print(f"  beam {beam}: " + ", ".join(f"N={n}: {t:.2f}s" for n, t in pts) +
              f"  ->  fixed (encoder + set-up) {fixed:.2f}s, per token {slope * 1000:.0f} ms")
    print("SENTENCES  (beams alternated per sentence)")
    res = {5: [], 1: []}
    for c in clips:
        src = ISO[c["lang"]]; dst = "lug" if src == "eng" else "eng"
        for beam in (5, 1):
            res[beam].append(st.median(
                (lambda t=time.perf_counter(): (tr.translate(c["text"], src, dst, beam_size=beam), time.perf_counter() - t)[1])()
                for _ in range(repeat)))
    for beam in (5, 1):
        print(f"  beam {beam}: median {st.median(res[beam]):.2f}s   per sentence: " + ", ".join(f"{t:.2f}" for t in res[beam]))
    del tr
