"""Record real "hey sunflower" samples for training the wake-word model.

Plays a short beep, then records 2.5 s from the microphone. Samples are taken
in three blocks of 10 (near the mic, an arm's length away, across the room);
a long double beep and an 8 s pause mark a change of block.

  python3 record_samples.py --out ~/ml/wake_samples/joel
"""
import argparse
import math
import os
import struct
import subprocess
import time
import wave

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--per-block", type=int, default=10)
ap.add_argument("--mic", default=os.environ.get("SUNFLOWER_MIC", "plughw:2,0"))
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def tone(path, hz, secs):
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(22050)
        w.writeframes(b"".join(struct.pack("<h", int(9000 * math.sin(2 * math.pi * hz * i / 22050)))
                               for i in range(int(22050 * secs))))


tone("/tmp/_beep.wav", 880, 0.25)
tone("/tmp/_long.wav", 520, 0.8)


def play(path):
    subprocess.run(["pw-play", path], check=False)


for block, name in enumerate(["near", "arm", "far"]):
    play("/tmp/_long.wav"); play("/tmp/_long.wav")
    time.sleep(8)
    for i in range(args.per_block):
        play("/tmp/_beep.wav")
        out = os.path.join(args.out, f"{name}_{i + 1:02d}.wav")
        subprocess.run(["arecord", "-q", "-D", args.mic, "-f", "S16_LE", "-r", "16000",
                        "-c", "1", "-d", "3", out], check=False)
        print(out, flush=True)
        time.sleep(1.2)
play("/tmp/_long.wav")
print("DONE", flush=True)
