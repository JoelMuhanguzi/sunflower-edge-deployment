"""Live "hey sunflower" detector: microphone -> 150 Hz high-pass -> openWakeWord embedding -> small classifier.

  python wake_detect.py --model wake_hey_sunflower.npz --seconds 60 --threshold 0.9

The WakeWord class is what the voice app imports: feed() takes 80 ms chunks of 16 kHz int16 audio
and returns True when the phrase is heard (several windows in a row above the threshold, then a refractory pause).
"""
import argparse
import os
import subprocess
import time

import numpy as np
from openwakeword.utils import AudioFeatures
from scipy.signal import butter, sosfilt, sosfilt_zi

CHUNK = 1280  # 80 ms at 16 kHz


class WakeWord:
    def __init__(self, model_path, threshold=0.9, consecutive=2, refractory_s=2.0):
        m = np.load(model_path)
        self.kind, self.frames = str(m["kind"]), int(m["frames"])
        if self.kind == "mlp":  # one hidden ReLU layer, sigmoid output
            self.W1, self.b1, self.W2, self.b2 = m["W1"], m["b1"], m["W2"], float(m["b2"])
        else:
            self.W, self.b = m["W"], float(m["b"])
        self.threshold, self.consecutive, self.refractory_s = threshold, consecutive, refractory_s
        self.feats = AudioFeatures(inference_framework="onnx")
        self.hp = butter(4, 150, "hp", fs=16000, output="sos")  # the USB mic has 50 Hz hum
        self.zi = sosfilt_zi(self.hp) * 0
        self.run, self.quiet, self.warmup, self.last_prob = 0, 0, 20, 0.0  # warmup: 1.6 s while the feature buffer fills
        self.refractory = int(refractory_s / 0.08)

    def prob(self, chunk):
        x, self.zi = sosfilt(self.hp, chunk.astype(np.float64), zi=self.zi)
        self.feats(np.clip(x, -32768, 32767).astype(np.int16))
        f = self.feats.get_features(self.frames).reshape(-1)
        if self.kind == "mlp":
            z = np.maximum(f @ self.W1 + self.b1, 0) @ self.W2 + self.b2
        else:
            z = f @ self.W + self.b
        self.last_prob = float(1 / (1 + np.exp(-np.clip(z, -50, 50))))
        return self.last_prob

    def feed(self, chunk):
        p = self.prob(chunk)
        if self.warmup > 0 or self.quiet > 0:
            self.warmup, self.quiet, self.run = max(0, self.warmup - 1), max(0, self.quiet - 1), 0
            return False
        self.run = self.run + 1 if p >= self.threshold else 0
        if self.run >= self.consecutive:
            self.run, self.quiet = 0, self.refractory
            return True
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mic", default=os.environ.get("SUNFLOWER_MIC", "plughw:2,0"))
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--threshold", type=float, default=0.9)
    ap.add_argument("--consecutive", type=int, default=2)
    ap.add_argument("--beep", action="store_true", help="play a confirmation beep on each wake event")
    args = ap.parse_args()
    ww = WakeWord(args.model, args.threshold, args.consecutive)
    mic = subprocess.Popen(["arecord", "-q", "-D", args.mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
                           stdout=subprocess.PIPE)
    t0, cpu0, peak, wakes = time.time(), time.process_time(), 0.0, 0
    print(f"listening for {args.seconds:.0f} s (threshold {args.threshold}, {args.consecutive} in a row)", flush=True)
    while time.time() - t0 < args.seconds:
        raw = mic.stdout.read(CHUNK * 2)
        if len(raw) < CHUNK * 2:
            break
        hit = ww.feed(np.frombuffer(raw, dtype=np.int16))
        peak = max(peak, ww.last_prob)
        if hit:
            wakes += 1
            print(f"  WAKE  t={time.time() - t0:5.1f}s  prob={ww.last_prob:.2f}", flush=True)
            if args.beep:
                subprocess.run(["pw-play", "/tmp/_beep.wav"], check=False)  # confirm audibly
    mic.terminate()
    print(f"done: {wakes} wake events, peak prob {peak:.2f}, "
          f"CPU {100 * (time.process_time() - cpu0) / (time.time() - t0):.0f}% of one core", flush=True)
