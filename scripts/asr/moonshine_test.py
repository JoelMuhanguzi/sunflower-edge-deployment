"""Transcribe wav clips with the Moonshine streaming models and report word error rate and time.

  python moonshine_test.py --sizes tiny small medium --clips ~/ml/pipeline2/bench_clips/eng_*.wav

Reference sentences are looked up by clip name in REFERENCES (the benchmark sentences read by the author).
"""
import argparse
import re
import time

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration

REFERENCES = {
    "eng_1": "The hospital is near the market",
    "eng_2": "My name is Joel, I am happy to see you.",
    "eng_3": "I am very sick please take me to the hospital",
}


def normalize(s):
    return re.sub(r"[^a-z0-9' ]", "", s.lower().replace("-", " ")).split()


def wer(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(len(r) + 1), np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    return d[-1, -1] / max(1, len(r))


def load_audio(path, sr=16000):
    x, fs = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample_poly(x, sr, fs) if fs != sr else x


ap = argparse.ArgumentParser()
ap.add_argument("--sizes", nargs="+", default=["tiny", "small", "medium"])
ap.add_argument("--clips", nargs="+", required=True)
ap.add_argument("--repo", default="moonshine-ai/moonshine-streaming-{size}")
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()
torch.set_num_threads(args.threads)

for size in args.sizes:
    repo = args.repo.format(size=size)
    t0 = time.perf_counter()
    model = MoonshineStreamingForConditionalGeneration.from_pretrained(repo).float().eval()
    proc = AutoProcessor.from_pretrained(repo)
    load_s = time.perf_counter() - t0
    print(f"== {size}  ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters, loaded in {load_s:.1f} s)")
    errs, secs = [], []
    for path in args.clips:
        audio = load_audio(path)
        inputs = proc(audio, return_tensors="pt", sampling_rate=proc.feature_extractor.sampling_rate)
        max_len = int((inputs.attention_mask.sum(dim=-1) * 6.5 / proc.feature_extractor.sampling_rate).max().item())
        runs = []
        for _ in range(3):
            t = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(**inputs, max_length=max_len)
            runs.append(time.perf_counter() - t)
        text = proc.decode(ids[0], skip_special_tokens=True).strip()
        name = path.split("/")[-1].rsplit(".", 1)[0]
        ref = REFERENCES.get(name)
        w = wer(ref, text) if ref else None
        errs.append(w) if w is not None else None
        secs.append(float(np.median(runs)))
        print(f"  {name}: {len(audio) / 16000:.1f} s audio, {np.median(runs):.2f} s to transcribe "
              f"(runs {', '.join(f'{r:.2f}' for r in runs)}) WER {w if w is None else round(w, 2)} | {text!r}")
    print(f"  median {np.median(secs):.2f} s per clip; mean WER {np.mean(errs):.3f}" if errs else "")
