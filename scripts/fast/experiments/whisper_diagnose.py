#!/usr/bin/env python3
"""Where does Whisper's transcription time go (features, encoder, decoder), and does precision matter?

    whisper_diagnose.py MODEL_DIR CLIPS_JSON [--repeat N] [--types int8,int8_float32,float32]

Wraps faster-whisper's feature extractor and encoder with timers; decoder time is the remainder of the total.
Uses the app's settings (6 s window, beam 1, no timestamps, no-repeat 3-grams, 64 token cap). The 30 s stock
window is also timed for the first compute type. The saved model is int8; float32 converts the weights on load.
"""
import json, statistics as st, sys, time
import faster_whisper.transcribe as ft
from faster_whisper import WhisperModel

model_dir, clips_path = sys.argv[1], sys.argv[2]
repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 2
types = (sys.argv[sys.argv.index("--types") + 1] if "--types" in sys.argv else "int8,int8_float32,float32").split(",")
clips = json.load(open(clips_path))
codes = json.load(open(f"{model_dir}/language_map.json"))
LANG = {"English": codes["eng"], "Luganda": codes["lug"]}

timers = {"features": 0.0, "encoder": 0.0}
orig_pad = ft.pad_or_trim
orig_encode = WhisperModel.encode


def set_window(seconds):
    frames = int(seconds * 100)
    ft.pad_or_trim = lambda arr, length=3000, **kw: orig_pad(arr, frames, **kw)


def timed_encode(self, features):
    t = time.perf_counter(); out = orig_encode(self, features); timers["encoder"] += time.perf_counter() - t
    return out


WhisperModel.encode = timed_encode


def run(model, clip):
    timers["features"] = timers["encoder"] = 0.0
    fe = model.feature_extractor; orig_call = fe.__call__

    def timed_call(*a, **k):
        t = time.perf_counter(); out = orig_call(*a, **k); timers["features"] += time.perf_counter() - t
        return out
    fe.__class__.__call__ = lambda self, *a, **k: timed_call(*a, **k)
    t = time.perf_counter()
    segs, _ = model.transcribe(clip["file"], language=LANG[clip["lang"]], beam_size=1, condition_on_previous_text=False,
                               without_timestamps=True, no_repeat_ngram_size=3, max_new_tokens=64)
    segs = list(segs)
    total = time.perf_counter() - t
    fe.__class__.__call__ = orig_call.__func__ if hasattr(orig_call, "__func__") else orig_call
    return total, timers["features"], timers["encoder"], sum(len(s.tokens) for s in segs)


for n, ct in enumerate(types):
    t0 = time.time()
    model = WhisperModel(model_dir, device="cpu", compute_type=ct, cpu_threads=4)
    print(f"\n##### compute_type={ct}  (load {time.time() - t0:.1f}s)", flush=True)
    for window in ((6, 30) if n == 0 else (6,)):
        set_window(window)
        run(model, clips[0])  # untimed warm-up
        rows = []
        for c in clips:
            r = [run(model, c) for _ in range(repeat)]
            rows.append([st.median(x[i] for x in r) for i in range(4)])
        med = lambda i: st.median(r[i] for r in rows)
        total, feat, enc, tok = med(0), med(1), med(2), med(3)
        dec = total - feat - enc
        print(f"  window {window:>2} s: total {total:5.2f}s = features {feat:4.2f} + encoder {enc:5.2f} + decoder/other {dec:5.2f}"
              f"   ({tok:.0f} tokens, {dec / max(tok, 1) * 1000:.0f} ms per token)", flush=True)
    del model
