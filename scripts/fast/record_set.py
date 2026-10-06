#!/usr/bin/env python3
"""Record the benchmark clips: six 5-second sentences (3 English, 3 Luganda).

Run on the Pi in a terminal:  python3 ~/ml/record_set.py
For each sentence: read it, press Enter, wait for "SPEAK NOW", say it within 5 s.
Afterwards press Enter to keep it or r + Enter to record it again.
Clips and the reference text go to ~/ml/bench_clips/ (clips.json lists them).
Uses the same mic device, gain and format as the touchscreen apps (16 kHz mono, 5 s).
"""
import json
import os
import subprocess
import time

MIC_DEVICE = "plughw:3,0"
MIC_GAIN_PERCENT = 62
SECONDS = 5
OUT = os.path.expanduser("~/ml/bench_clips")

SENTENCES = [
    ("English", "The hospital is near the market."),
    ("English", "My name is Joel, I am happy to see you."),
    ("English", "I am very sick, please take me to the hospital."),
    ("Luganda", "Oli otya leero? Nsanyuse okukulaba."),
    ("Luganda", "Ndi mulwadde nnyo, ntwala mu ddwaliro."),
    ("Luganda", "Abaana bagenda ku ssomero."),
]

os.makedirs(OUT, exist_ok=True)
card = MIC_DEVICE.split(":")[1].split(",")[0]
subprocess.run(["amixer", "-c", card, "sset", "Mic", f"{MIC_GAIN_PERCENT}%"], capture_output=True)

clips = []
for i, (lang, text) in enumerate(SENTENCES, 1):
    path = f"{OUT}/{lang[:3].lower()}_{i}.wav"
    while True:
        print(f"\n[{i}/{len(SENTENCES)}] {lang}:  {text}")
        input("Press Enter, then speak when you see SPEAK NOW ")
        for n in (3, 2, 1):
            print(n, flush=True)
            time.sleep(1)
        print(f"SPEAK NOW ({SECONDS} s)", flush=True)
        r = subprocess.run(["arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1",
                            "-d", str(SECONDS), path], capture_output=True, text=True)
        if r.returncode != 0:
            print("recording failed:", r.stderr.strip())
            continue
        if input("Done. Enter = keep, r = record again: ").strip().lower() != "r":
            break
    clips.append({"file": path, "lang": lang, "text": text})

json.dump(clips, open(f"{OUT}/clips.json", "w"), indent=2, ensure_ascii=False)
print(f"\nSaved {len(clips)} clips and clips.json in {OUT}")
