"""Step 1 of the wake-word app: listen on the microphone and report wake-word scores.

Plumbing test with the stock "hey_jarvis" model (a custom "hey sunflower" model
is trained later; pass its path with --model). Prints a line whenever the score
passes the threshold, plus CPU use at the end.

  wake-venv/bin/python wake_test.py --seconds 30
"""
import argparse
import os
import subprocess
import time

import numpy as np
from openwakeword.model import Model
from scipy.signal import butter, sosfilt, sosfilt_zi

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="hey_jarvis", help="stock name or path to an .onnx model")
ap.add_argument("--mic", default=os.environ.get("SUNFLOWER_MIC", "plughw:2,0"))
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()

# The USB mic picks up 50 Hz mains hum (and harmonics) strong enough to swamp speech;
# a 150 Hz high-pass removes it.
HP = butter(4, 150, "hp", fs=16000, output="sos")
zi = sosfilt_zi(HP) * 0
CHUNK = 1280  # 80 ms at 16 kHz, what openWakeWord expects
model = Model(wakeword_models=[args.model], inference_framework="onnx")
name = list(model.models.keys())[0]

mic = subprocess.Popen(
    ["arecord", "-q", "-D", args.mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
    stdout=subprocess.PIPE,
)
t0, cpu0 = time.time(), time.process_time()
peak = 0.0
print(f"listening for '{name}' for {args.seconds:.0f} s ...", flush=True)
while time.time() - t0 < args.seconds:
    raw = mic.stdout.read(CHUNK * 2)
    if len(raw) < CHUNK * 2:
        break
    x, zi = sosfilt(HP, np.frombuffer(raw, dtype=np.int16).astype(np.float64), zi=zi)
    score = model.predict(np.clip(x, -32768, 32767).astype(np.int16))[name]
    peak = max(peak, score)
    if score >= args.threshold:
        print(f"  WAKE  t={time.time() - t0:5.1f}s  score={score:.2f}", flush=True)
mic.terminate()
wall = time.time() - t0
print(f"peak score {peak:.2f}; detector CPU use {100 * (time.process_time() - cpu0) / wall:.0f}% of one core")
