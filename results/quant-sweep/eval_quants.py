#!/usr/bin/env python3
"""Translate English->Luganda test sentences with a given GGUF via llama-server
(temperature 0) and save the outputs. Scoring is done separately (score_quants.py).

usage: eval_quants.py <label> <model.gguf> [n_sentences] [port]
"""
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HOME = os.path.expanduser("~")
SERVER = f"{HOME}/ml/llama.cpp/build/bin/llama-server"
DATA = f"{HOME}/ml/vits-work/training/training_files"
OUT = f"{HOME}/ml/quant-sweep/outputs"
os.makedirs(OUT, exist_ok=True)

label, model = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
port = int(sys.argv[4]) if len(sys.argv) > 4 else 8089


def load_pairs(n):
    lug = {r["Key"]: r for r in csv.DictReader(open(f"{DATA}/Prompt-Luganda.csv", encoding="utf-8"))}
    eng = {r["Key"]: r for r in csv.DictReader(open(f"{DATA}/Prompt-English.csv", encoding="utf-8"))}
    keys = [k for k, r in lug.items() if r["split"] == "test" and k in eng]
    return [(k, eng[k]["Text"].strip(), lug[k]["Text"].strip()) for k in keys[:n]]


pairs = load_pairs(n)
log = open(f"{OUT}/{label}.server.log", "w")
srv = subprocess.Popen(
    [SERVER, "-m", model, "--port", str(port), "-c", "4096", "-np", "4", "--jinja"],
    stdout=log, stderr=log,
)
try:
    base = f"http://127.0.0.1:{port}"
    for _ in range(300):
        try:
            if requests.get(f"{base}/health", timeout=2).status_code == 200:
                break
        except requests.RequestException:
            pass
        if srv.poll() is not None:
            sys.exit(f"server exited early; see {OUT}/{label}.server.log")
        time.sleep(1)
    else:
        sys.exit("server did not become healthy")

    def translate(item):
        key, src, ref = item
        r = requests.post(f"{base}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"Translate to Luganda: {src}"}],
            "temperature": 0, "max_tokens": 160,
        }, timeout=600)
        r.raise_for_status()
        hyp = r.json()["choices"][0]["message"]["content"].strip()
        return {"key": key, "src": src, "ref": ref, "hyp": hyp}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(translate, pairs))
    wall = time.time() - t0
    with open(f"{OUT}/{label}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{label}: {len(rows)} sentences in {wall:.0f}s (4 parallel requests; not a speed benchmark)")
finally:
    srv.terminate()
    try:
        srv.wait(timeout=15)
    except subprocess.TimeoutExpired:
        srv.kill()
