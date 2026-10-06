#!/usr/bin/env python3
"""Speak text with the small VITS ONNX voice (jq/vits-tts-lug-eng-onnx) via onnxruntime.

Usage: tts_onnx.py <model_dir> <int8|fp32> out.wav "text" ["text" ...]
Characters are mapped straight to token ids from tokens.txt (no phonemizer).
Prints load time, synthesis time and the real-time factor (synthesis / audio length).
"""
import sys
import time

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile

model_dir, precision, out_path, *texts = sys.argv[1:]
onnx_file = "vits-lug-eng.int8.onnx" if precision == "int8" else "vits-lug-eng.onnx"

tok = {}
for line in open(f"{model_dir}/tokens.txt", encoding="utf-8"):
    line = line.rstrip("\n")
    if not line:
        continue
    sym, _, idx = line.rpartition(" ")
    tok[sym if sym else " "] = int(idx)

t0 = time.time()
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(f"{model_dir}/{onnx_file}", opts, providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
rate = int(meta.get("sample_rate", 22050))
print(f"load ({precision}): {time.time() - t0:.2f}s  rate {rate}  language {meta.get('language')}")

chunks = []
for text in texts:
    ids = [tok[c] for c in text if c in tok]
    skipped = sorted({c for c in text if c not in tok})
    x = np.array([ids], dtype=np.int64)
    t0 = time.time()
    audio = sess.run(None, {"input": x, "input_lengths": np.array([len(ids)], dtype=np.int64),
                            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)})[0]
    dt = time.time() - t0
    audio = audio.squeeze()
    secs = len(audio) / rate
    print(f"{text!r}: {dt:.2f}s for {secs:.1f}s audio  [{dt / secs:.2f}x real time]  skipped chars: {skipped}")
    chunks += [audio, np.zeros(int(rate * 0.3), dtype=audio.dtype)]

scipy.io.wavfile.write(out_path, rate, np.concatenate(chunks))
print("wrote", out_path)
