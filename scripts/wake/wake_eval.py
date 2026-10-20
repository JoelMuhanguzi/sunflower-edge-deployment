"""Scripted live test of the wake-word detector: quiet / normal talking / the phrase, with spoken prompts.

Counts wake events per phase (events during a prompt itself are ignored). Run on the Pi:
  wake-venv/bin/python wake_eval.py --model wake_hey_sunflower.npz --prompts /tmp/prompts
where /tmp/prompts holds p1.wav ... p4.wav (the spoken announcements).
"""
import argparse
import os
import subprocess
import time

import numpy as np
from wake_detect import CHUNK, WakeWord

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--mic", default=os.environ.get("SUNFLOWER_MIC", "plughw:2,0"))
ap.add_argument("--threshold", type=float, default=0.9)
ap.add_argument("--consecutive", type=int, default=2)
ap.add_argument("--save", default=None, help="folder to save each phase's raw 16 kHz audio (for retraining)")
args = ap.parse_args()

PHASES = [("quiet", "p1.wav", 15), ("talking", "p2.wav", 20), ("phrase", "p3.wav", 25)]
ww = WakeWord(args.model, args.threshold, args.consecutive)
mic = subprocess.Popen(["arecord", "-q", "-D", args.mic, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
                       stdout=subprocess.PIPE)


audio = []


def listen(seconds, play=None):
    """Return wake-event times (s) within the window; the prompt's own duration is not counted."""
    events, t0 = [], time.time()
    prompt = subprocess.Popen(["pw-play", os.path.join(args.prompts, play)]) if play else None
    counting_from = None
    while True:
        now = time.time() - t0
        if prompt is not None and prompt.poll() is not None:
            prompt = None
            counting_from = now + 0.7  # let the room echo die away
        if prompt is None and counting_from is None:
            counting_from = 0
        if counting_from is not None and now - counting_from > seconds:
            break
        raw = mic.stdout.read(CHUNK * 2)
        if counting_from is not None and now >= counting_from:
            audio.append(raw)
        hit = ww.feed(np.frombuffer(raw, dtype=np.int16))
        if hit and counting_from is not None and now >= counting_from:
            events.append(round(now - counting_from, 1))
    return events


result = {}
for name, wav, secs in PHASES:
    audio.clear()
    result[name] = listen(secs, wav)
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        with open(os.path.join(args.save, name + ".raw"), "wb") as f:
            f.write(b"".join(audio))
    print(f"{name}: {len(result[name])} wake events at {result[name]} s", flush=True)
subprocess.run(["pw-play", os.path.join(args.prompts, "p4.wav")])
mic.terminate()
