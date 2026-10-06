#!/usr/bin/env python3
"""Time requests to a running llama-server (Gemma4-E2B Q4_K_M + mmproj) on 127.0.0.1:8081.

Text requests use the same prompt shape as the touchscreen app ("Translate to X: ...").
The audio request uses the app's transcription prompt. Standard library only.
"""
import base64
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8081/v1/chat/completions"


def chat(content, max_tokens=128):
    body = json.dumps({"messages": [{"role": "user", "content": content}],
                       "max_tokens": max_tokens, "temperature": 0.3}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.load(r)
    dt = time.time() - t0
    return out["choices"][0]["message"]["content"].strip(), dt, out.get("timings", {})


print("--- text translation (same prompts as the NLLB test)")
for text in ["Where is the hospital?", "How are you today?",
             "The children are going to school.", "Please bring me a glass of water."]:
    reply, dt, tm = chat(f"Translate to Luganda: {text}")
    print(f"{text!r} -> {reply!r}  ({dt:.1f}s, gen {tm.get('predicted_per_second', 0):.2f} tok/s)")

wav = sys.argv[1] if len(sys.argv) > 1 else None
if wav:
    print("--- audio transcription")
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    prompt = ("The speaker is speaking Luganda. Transcribe exactly what they said, written in "
              "Luganda. Respond with only the transcription, nothing else.")
    for run in ("first", "second"):
        reply, dt, _ = chat([{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                             {"type": "text", "text": prompt}])
        print(f"{wav.split('/')[-1]} {run}: {dt:.1f}s -> {reply!r}")
