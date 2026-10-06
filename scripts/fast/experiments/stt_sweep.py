#!/usr/bin/env python3
"""Sweep Whisper encoder window and decoding options on the recorded clips; score vs reference.
Usage: stt_sweep.py <model_dir> <clips.json>   (clips.json 'file' paths are remapped to its folder)
"""
import json, os, re, sys, time
import faster_whisper.transcribe as ft
from faster_whisper import WhisperModel

model_dir, clips_json = sys.argv[1:3]
folder = os.path.dirname(clips_json)
clips = json.load(open(clips_json))
codes = json.load(open(f"{model_dir}/language_map.json"))
LANG = {"English": "en", "Luganda": codes["lug"]}
model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
orig = ft.pad_or_trim


def norm(s):
    return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


SETTINGS = {
    "default":           {},
    "no-timestamps":     {"without_timestamps": True},
    "no-repeat-3":       {"no_repeat_ngram_size": 3},
    "rep-penalty-1.3":   {"repetition_penalty": 1.3},
    "no-ts+no-repeat-3": {"without_timestamps": True, "no_repeat_ngram_size": 3},
}
print(f"{'window':>6} {'setting':<18} {'mean WER':>8} {'mean s':>7}  repeats  transcripts (eng_1, lug_4)")
for window in (5, 6, 8, 10, 30):
    ft.pad_or_trim = lambda a, length=3000, w=window, **k: orig(a, int(w * 100), **k)
    for name, opts in SETTINGS.items():
        wers, secs, repeats, shown = [], [], 0, []
        for c in clips:
            path = os.path.join(folder, os.path.basename(c["file"]))
            t = time.time()
            segs, _ = model.transcribe(path, language=LANG[c["lang"]], beam_size=1,
                                       condition_on_previous_text=False, **opts)
            text = " ".join(s.text.strip() for s in segs)
            secs.append(time.time() - t)
            wers.append(wer(c["text"], text))
            n, h = len(norm(c["text"])), norm(text)
            repeats += len(h) >= 1.7 * n
            if os.path.basename(c["file"]) in ("eng_1.wav", "lug_4.wav"):
                shown.append(text[:48])
        print(f"{window:>5}s {name:<18} {sum(wers)/len(wers):8.2f} {sum(secs)/len(secs):7.2f}  {repeats}/{len(clips)}      {shown}", flush=True)
