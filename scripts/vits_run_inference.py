import sys
import json
import time
import torch
import numpy as np
from scipy.io.wavfile import write

import utils
import commons
from models import SynthesizerTrn
from text.mappers import TextMapper

if len(sys.argv) != 4:
    print("Usage: python run_inference.py <model_dir> <text> <out_wav_path>")
    sys.exit(1)

MODEL_DIR = sys.argv[1]
TEXT = sys.argv[2]
OUT_PATH = sys.argv[3]

t0 = time.time()

with open(f"{MODEL_DIR}/config.json") as f:
    cfg = json.load(f)

hps = utils.HParams(**cfg)
text_mapper = TextMapper(f"{MODEL_DIR}/vocab.txt")
device = torch.device("cpu")

net_g = SynthesizerTrn(
    len(text_mapper.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
).to(device)
net_g.eval()

import glob
g_ckpt = sorted(glob.glob(f"{MODEL_DIR}/G_*.pth"))[-1]
utils.load_checkpoint(g_ckpt, net_g, None)

load_time = time.time() - t0

processed = TEXT.lower()
processed = text_mapper.filter_oov(processed)
print(f"Original: {TEXT!r}")
print(f"Processed/filtered: {processed!r}")

stn_tst = text_mapper.get_text(processed, hps)

t1 = time.time()
with torch.no_grad():
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
    audio = net_g.infer(
        x_tst, x_tst_lengths, noise_scale=0.667, noise_scale_w=0.8, length_scale=1
    )[0][0, 0].cpu().float().numpy()
inference_time = time.time() - t1

write(OUT_PATH, hps.data.sampling_rate, audio)
duration = len(audio) / hps.data.sampling_rate
print(f"Saved: {OUT_PATH}")
print(f"Sample rate: {hps.data.sampling_rate}")
print(f"Duration: {duration:.2f}s")
print(f"Model load time: {load_time:.2f}s")
print(f"Inference time: {inference_time:.2f}s")
print(f"Real-time factor: {inference_time / duration:.2f}x")
