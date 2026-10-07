#!/usr/bin/env python3
"""Time the character-level ONNX voice (jq/sherpa-vits-tts-lug-eng, the one the Sunbird app uses).

    tts_char_onnx.py MODEL_DIR [OUT_DIR] [--repeat N] [--file vits-lug-eng.int8.onnx]

Input recipe (from SunbirdAI/sunflower-app, lib/tts/tts_engine.dart): lowercase the text, map each
character to its id in tokens.txt, no blanks, scales [0.667, 1.0, 0.8], sample rate 22050.
"""
import sys, time, os
import numpy as np, onnxruntime as ort
from scipy.io import wavfile

args = [a for a in sys.argv[1:] if not a.startswith("--")]
onnx_file = sys.argv[sys.argv.index("--file") + 1] if "--file" in sys.argv else "vits-lug-eng.fp32.onnx"
if "--file" in sys.argv:
    args.remove(onnx_file)
repeat = int(sys.argv[sys.argv.index("--repeat") + 1]) if "--repeat" in sys.argv else 3
if "--repeat" in sys.argv:
    args.remove(sys.argv[sys.argv.index("--repeat") + 1])
model_dir = args[0]
out_dir = args[1] if len(args) > 1 else None
RATE = 22050
SENTENCES = [
    ("eng", "hello, how are you today?"),
    ("eng", "the weather is very nice this morning and we are going to the market"),
    ("lug", "oli otya, nsanyuse okukulaba leero"),
    ("lug", "ndi mukozi wa gavumenti era nnyumba yange eri e kampala"),
]
tok = {}
for line in open(f"{model_dir}/tokens.txt", encoding="utf8"):
    line = line.rstrip("\n")
    if line:
        sym, _, idx = line.rpartition(" ")
        tok[sym or " "] = int(idx)

t0 = time.time()
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
sess = ort.InferenceSession(f"{model_dir}/{onnx_file}", opts, providers=["CPUExecutionProvider"])
print(f"{onnx_file} load: {time.time() - t0:.2f}s", flush=True)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

for n, (lang, text) in enumerate(SENTENCES):
    ids = [tok[c] for c in text.lower() if c in tok]
    feed = {"input": np.array([ids], dtype=np.int64), "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array([0.667, 1.0, 0.8], dtype=np.float32)}
    times = []
    for _ in range(repeat):
        t = time.time()
        audio = sess.run(None, feed)[0].squeeze()
        times.append(time.time() - t)
    secs = len(audio) / RATE
    print(f"{lang} {len(text.split())} words, {secs:.1f}s audio: " + ", ".join(f"{x:.2f}s" for x in times)
          + f"  (first {times[0]:.2f}s = {times[0] / secs:.2f}x real time)", flush=True)
    if out_dir:
        wavfile.write(f"{out_dir}/{n}_{lang}.wav", RATE, audio)
