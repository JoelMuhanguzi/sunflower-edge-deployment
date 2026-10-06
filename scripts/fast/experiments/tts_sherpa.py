#!/usr/bin/env python3
"""Try the jq VITS ONNX voice through sherpa-onnx (the runtime its metadata points to).
Usage: tts_sherpa.py <model_dir> <outdir>
Tries: plain (tokens only) and espeak (system espeak-ng-data); prints duration and time."""
import os
import sys
import time
import wave

import numpy as np
import sherpa_onnx

model_dir, outdir = sys.argv[1:3]
os.makedirs(outdir, exist_ok=True)
ESPEAK = "/usr/lib/aarch64-linux-gnu/espeak-ng-data"
TEXTS = {"lug": "Oli otya leero? Nsanyuse okukulaba.", "eng": "Hello, how are you today?"}


def build(data_dir):
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=f"{model_dir}/vits-lug-eng.onnx", tokens=f"{model_dir}/tokens.txt",
        lexicon="", data_dir=data_dir)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, provider="cpu", num_threads=4),
        max_num_sentences=1)
    if not cfg.validate():
        raise RuntimeError("sherpa config invalid")
    return sherpa_onnx.OfflineTts(cfg)


MODES = {"plain": "", "espeak": ESPEAK}
for mode in sys.argv[3:] or MODES:
    data_dir = MODES[mode]
    print(f"== mode {mode}")
    try:
        t0 = time.time()
        tts = build(data_dir)
        print(f"  load {time.time() - t0:.1f}s, sample rate {tts.sample_rate}, speakers {tts.num_speakers}")
    except Exception as e:
        print(f"  could not build: {type(e).__name__}: {str(e)[:200]}")
        continue
    for lang, text in TEXTS.items():
        try:
            t0 = time.time()
            audio = tts.generate(text, sid=0, speed=1.0)
            dt = time.time() - t0
            x = np.array(audio.samples, dtype=np.float32)
            secs = len(x) / audio.sample_rate
            path = f"{outdir}/{mode}_{lang}.wav"
            with wave.open(path, "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(audio.sample_rate)
                w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
            print(f"  {lang}: {dt:.2f}s for {secs:.2f}s audio -> {path}")
        except Exception as e:
            print(f"  {lang}: failed {type(e).__name__}: {str(e)[:200]}")
