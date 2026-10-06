#!/usr/bin/env python3
"""Persistent Sunbird VITS synthesizer: loads checkpoints once, then answers requests.

Run with cwd = vits-work/training (it imports utils/models/text from there):
    python vits_worker.py <model_dir> [<model_dir> ...]
Protocol, one JSON object per line on stdin; replies on stdout prefixed with '@@':
    in : {"dir": ..., "text": ..., "out": "/path.wav"}
    out: @@RESULT {"ok": true, "synth_s": ..., "audio_s": ..., "load_s": ...}
The synthesis code is the same as run_inference.py (same noise scales, lower-casing, OOV filter).
"""
import glob
import json
import sys
import time

sys.path.insert(0, ".")
import torch
from scipy.io.wavfile import write

import utils
from models import SynthesizerTrn
from text.mappers import TextMapper

_models = {}


def load(model_dir):
    """Load (once) and return (net_g, text_mapper, hps) for a model directory."""
    if model_dir not in _models:
        with open(f"{model_dir}/config.json") as f:
            hps = utils.HParams(**json.load(f))
        text_mapper = TextMapper(f"{model_dir}/vocab.txt")
        net_g = SynthesizerTrn(
            len(text_mapper.symbols), hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length, **hps.model).to("cpu")
        net_g.eval()
        utils.load_checkpoint(sorted(glob.glob(f"{model_dir}/G_*.pth"))[-1], net_g, None)
        _models[model_dir] = (net_g, text_mapper, hps)
    return _models[model_dir]


def say(tag, obj):
    sys.stdout.write(f"@@{tag} {json.dumps(obj)}\n")
    sys.stdout.flush()


t0 = time.time()
for d in sys.argv[1:]:
    load(d)
say("READY", {"models": sys.argv[1:], "load_s": round(time.time() - t0, 2)})

for line in sys.stdin:  # exits when the app closes stdin
    try:
        req = json.loads(line)
        t0 = time.time()
        net_g, text_mapper, hps = load(req["dir"])
        load_s = time.time() - t0
        processed = text_mapper.filter_oov(req["text"].lower())
        stn = text_mapper.get_text(processed, hps)
        t1 = time.time()
        with torch.no_grad():
            audio = net_g.infer(stn.unsqueeze(0), torch.LongTensor([stn.size(0)]),
                                noise_scale=0.667, noise_scale_w=0.8, length_scale=1
                                )[0][0, 0].cpu().float().numpy()
        synth_s = time.time() - t1
        write(req["out"], hps.data.sampling_rate, audio)
        say("RESULT", {"ok": True, "synth_s": round(synth_s, 2),
                       "audio_s": round(len(audio) / hps.data.sampling_rate, 2),
                       "load_s": round(load_s, 2)})
    except Exception as e:
        say("RESULT", {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]})
