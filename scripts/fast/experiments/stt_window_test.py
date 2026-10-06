#!/usr/bin/env python3
"""Does faster-whisper (CTranslate2) work with an encoder window shorter than 30 s?

Replaces the 30 s padding (3000 mel frames) with a configurable length and reports
time and transcript for each. Usage: stt_window_test.py <model_dir> <iso3> file.wav ...
"""
import json
import sys
import time

import numpy as np
import faster_whisper.transcribe as ft
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

model_dir, lang, *files = sys.argv[1:]
code = json.load(open(f"{model_dir}/language_map.json"))[lang]
model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)

orig_pad = ft.pad_or_trim
for path in files:
    dur = len(decode_audio(path, sampling_rate=16000)) / 16000
    print(f"== {path.split('/')[-1]} ({dur:.1f}s audio)")
    for frames in (3000, 1500, 1000, 600, 500):
        ft.pad_or_trim = lambda a, length=frames, **k: orig_pad(a, length, **k)
        t0 = time.time()
        try:
            segs, _ = model.transcribe(path, language=code, beam_size=1,
                                       condition_on_previous_text=False)
            text = " ".join(s.text.strip() for s in segs)
            print(f"  window {frames / 100:4.0f}s: {time.time() - t0:5.2f}s -> {text!r}")
        except Exception as e:  # report engine errors instead of crashing
            print(f"  window {frames / 100:4.0f}s: ERROR {type(e).__name__}: {str(e)[:150]}")
