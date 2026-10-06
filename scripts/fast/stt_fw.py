#!/usr/bin/env python3
"""Transcribe with Sunbird's faster-whisper (CTranslate2), timing load and inference.

Usage: stt_fw.py <model_dir> <iso3 language> <compute_type> <beam> file.wav [file.wav ...]
Language is the ISO 639-3 code (lug, nyn, ach, eng, ...); it is mapped through the
model's language_map.json because the model reuses Whisper's language tokens.
"""
import json
import os
import sys
import time

import faster_whisper.transcribe as ft
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

model_dir, lang, compute_type, beam, *files = sys.argv[1:]

# Encoder window in seconds. Whisper normally pads every clip to 30 s; the app records
# 5 s, so a 5 s window does about a sixth of the encoder work. 30 = stock behaviour.
WINDOW_SECONDS = float(os.environ.get("WINDOW_SECONDS", "5"))
_orig_pad = ft.pad_or_trim
ft.pad_or_trim = lambda a, length=3000, **k: _orig_pad(a, int(WINDOW_SECONDS * 100), **k)
code = json.load(open(f"{model_dir}/language_map.json"))[lang]

t0 = time.time()
model = WhisperModel(model_dir, device="cpu", compute_type=compute_type, cpu_threads=4)
print(f"load ({compute_type}): {time.time() - t0:.1f}s   language {lang} -> token {code}   window {WINDOW_SECONDS:g}s")

for path in files:
    dur = len(decode_audio(path, sampling_rate=16000)) / 16000
    for run in ("cold", "warm"):
        t0 = time.time()
        segs, _ = model.transcribe(path, language=code, beam_size=int(beam),
                                   condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segs)   # generator: decoding happens here
        dt = time.time() - t0
        print(f"{path.split('/')[-1]} ({dur:.1f}s audio) {run}: {dt:.2f}s  [{dt / dur:.1f}x audio]  -> {text!r}")
