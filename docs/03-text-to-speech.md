# Part 3 — Text-to-speech output

[← Back to the project README](../README.md)


The goal here was closing the loop: not just understanding speech/text, but speaking a response back. This required surveying what Sunbird (and the wider ecosystem) actually offers for TTS, since — unlike Part 1 and 2 — there was no single obvious model to reach for.

## Step 14 — Survey of Sunbird's Hugging Face catalog

Before picking a TTS model, we audited Sunbird's full Hugging Face organization (**50 model repos total**, not just the ~12 flagship ones visible on the main org page) to understand what else was available and verify claims independently rather than trust page summaries, which proved unreliable for exact model names/sizes on first pass.

Key findings relevant to this deployment:

| Model | Modality | Notes |
|---|---|---|
| `Sunbird/Sunflower-14B-GGUF` | text | Pre-quantized GGUF already exists (8 quant levels). q4_k_m is **9.00 GB** — larger than the Pi 4's total RAM, so not usable here despite being pre-converted. Smaller levels (`iq2_xxs` 4.3GB, `tq1_0` 3.87GB) would fit but with significant quality loss. |
| `Sunbird/Sunflower-Speech-Ultravox-32B` | audio+text→text | Different architecture (Ultravox) from Gemma4-E2B; the published file is only the ~1.3GB audio adapter sitting on top of the 61GB Sunflower-32B backbone — not standalone, not Pi-feasible. |
| `Sunbird/orpheus-3b-tts-multilingual` | text→**speech** | See Step 15 — evaluated and ruled out. |
| `Sunbird/spark-tts-salt` | text→**speech** | See Step 16 — evaluated and ruled out. |
| `Sunbird/tts-vits-{lug,ach,nyn,teo,eng,lug-eng}` | text→**speech** | See Step 17 — evaluated, **working**, used in this deployment. |

## Step 15 — Orpheus-3B TTS: ruled out before attempting

`Sunbird/orpheus-3b-tts-multilingual` (6.6GB BF16, Llama-based) generates audio *tokens* which a separate **SNAC** neural codec then decodes into a waveform. Before attempting any conversion work, we researched feasibility:

- SNAC itself is lightweight (19.8M params, convolutional, ~50–150ms CPU decode) — not the bottleneck.
- llama.cpp's native Orpheus+SNAC support is an **open, unresolved issue** ([#12476](https://github.com/ggml-org/llama.cpp/issues/12476)) with only a **stalled, unmerged draft PR** ([#12487](https://github.com/ggml-org/llama.cpp/pull/12487)) attempting it — the author explicitly reported CPU performance problems in the core compute graph and handed off ownership.
- Existing community "GGUF" versions of Orpheus only quantize the Llama backbone; SNAC decoding in all of them still runs through a separate PyTorch package, not llama.cpp — so even using community work, this isn't a single clean runtime.
- No GGUF of Sunbird's specific multilingual fine-tune exists; one would need to be built from scratch, unproven.
- Sunbird's own model card only documents GPU hardware requirements (8GB+ VRAM) — CPU inference isn't an officially supported path for this model.

**Verdict:** treated as a longer-term stretch goal contingent on upstream llama.cpp support maturing, not attempted on this hardware.

## Step 16 — Spark-TTS: attempted, abandoned

`Sunbird/spark-tts-salt` (1.9GB BF16, Qwen2-based) covers more Ugandan languages than the VITS repos (Acholi, Ateso, Luganda, Lugbara, Runyankole, Swahili). Unlike Orpheus, its linked GitHub repo ([SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)) genuinely contains the code its model card references (verified directly, not assumed) — including real Apple Silicon MPS and CPU fallback paths in the generic inference script.

However, Sunbird's specific usage example requires `unsloth` (a CUDA-oriented fine-tuning/inference library) and hardcodes `"cuda"` as the device for both model loading and the `BiCodecTokenizer`. Attempted installing `unsloth` on macOS (Apple Silicon, no CUDA):

```bash
pip install unsloth
```

This stalled for several minutes resolving its dependency tree (it pulls in CUDA-specific tooling largely irrelevant/non-functional outside an NVIDIA GPU environment) without completing. **Abandoned** — not worth the engineering cost of patching around a CUDA-first library when a simpler alternative (Step 17 / MMS-TTS below) was already confirmed working.

## Step 17 — Sunbird VITS: the model card's code was broken, but the model works

`Sunbird/tts-vits-{lug,ach,nyn,teo,eng,lug-eng}` are single-language VITS checkpoints. VITS is architecturally simple (single-pass text→audio, no separate vocoder stage), making it the lightest-weight TTS option evaluated.

**Problem:** the model card's documented usage code calls `VITSInfereceAdapterModel.from_pretrained(...)` — **this class does not exist anywhere in the linked repository**, [SunbirdAI/vits](https://github.com/SunbirdAI/vits). What the repo actually contains is unmodified, generic VITS research code circa 2020: hardcoded `.cuda()` calls throughout the demo notebook, a stale `torch==1.6.0` pin, and a Cython extension (`monotonic_align`) that must be compiled locally.

Rather than treat this as a dead end, we reverse-engineered a working path using what *was* genuinely present and consistent in the Hugging Face repo itself (not the GitHub code):

- `config.json` — a complete, valid VITS hyperparameter file (confirmed actual sample rate: 22500 Hz, not the more common 22050)
- `vocab.txt` — the model's real, compact character-level vocabulary (30 symbols, including Luganda's `ŋ`)
- `G_<iteration>.pth` — the generator checkpoint (the `D_<iteration>.pth` discriminator files are training-only and were not needed for inference)
- `training/text/mappers.py` in the GitHub repo contains a `TextMapper` class — unlike the demo notebook, this module already has a CPU device fallback and is driven by an external vocab file, architecturally mirroring Meta's MMS-TTS approach

**Setup (Mac, modern dependency versions — not the stale repo pins):**

```bash
git clone --depth 1 https://github.com/SunbirdAI/vits.git vits-work
python3 -m venv ~/ml/vits-venv
~/ml/vits-venv/bin/pip install torch numpy scipy librosa unidecode cython phonemizer tqdm

cd vits-work/training/monotonic_align
mkdir -p training/monotonic_align   # workaround: cythonize writes here due to how it infers the build path from cwd
~/ml/vits-venv/bin/python setup.py build_ext --inplace
# copy the compiled .so into the nested path training/monotonic_align/__init__.py actually imports from:
mkdir -p monotonic_align
cp training/monotonic_align/core.cpython-*.so monotonic_align/
touch monotonic_align/__init__.py
```

Also removed a leftover notebook-only import (`from IPython.display import Audio`) from `text/mappers.py` — unused outside a notebook, and not worth adding `ipython` as a dependency for.

Downloaded the actual checkpoint, config, and vocab directly from Hugging Face (not GitHub):

```bash
huggingface-cli download Sunbird/tts-vits-lug --local-dir ~/ml/models/tts-vits-lug
```

A small generalized inference script (`training/run_inference.py`, written for this project) loads `config.json` into the hyperparameter structure the original `SynthesizerTrn` model class expects, builds a `TextMapper` from `vocab.txt`, loads the checkpoint, and runs CPU inference — no CUDA, no modification to the `models.py`/`commons.py` core VITS implementation itself.

```bash
python run_inference.py ~/ml/models/tts-vits-lug "Oli otya leero?" out.wav
```

**Result:** loaded cleanly, no shape mismatches, generated 1.34s of audio. Subjectively judged to sound noticeably natural for Luganda specifically — Sunbird's own training data for this language appears to outperform the more general-purpose MMS checkpoint for this language (Step 18), though no formal quality metric was measured.

Also verified working for **English** (`tts-vits-eng`, checkpoint at iteration 100,000 — far more trained than the others) and **Runyankole** (`tts-vits-nyn`, iteration 4,000), using the same generalized script.

#### Pi deployment: a genuine Python-version bug, not present on the Mac

Transferred `vits-work/` and the three model directories to the Pi via USB (same method as Steps 9 and 13). Installed the same dependencies in a fresh venv, then recompiled the Cython extension **fresh for ARM64 Linux** — the Mac-compiled `.so` is platform-specific and does not run on the Pi.

```bash
cd ~/ml/vits-work/training/monotonic_align
mkdir -p training/monotonic_align
~/ml/vits-venv/bin/python setup.py build_ext --inplace
# -> produces core.cpython-313-aarch64-linux-gnu.so
```

This compiled cleanly, but the subsequent import failed on the Pi where it had succeeded on the Mac:

```
ModuleNotFoundError: No module named 'monotonic_align.core'
```

**Diagnosis:** the Pi runs Python 3.13.5; the Mac environment used Python 3.11. Loading the compiled `.so` directly via `importlib.util.spec_from_file_location` worked without issue, proving the extension itself was built correctly — the failure was specifically in Python 3.13 resolving the package-relative import (`from .monotonic_align.core import ...`) differently than 3.11 did.

**Fix applied:** patched `monotonic_align/__init__.py` to load the compiled extension directly by file path rather than via the relative import:

```python
import os as _os
import importlib.util as _ilu
_so_path = _os.path.join(_os.path.dirname(__file__), "monotonic_align", "core.cpython-313-aarch64-linux-gnu.so")
_spec = _ilu.spec_from_file_location("core", _so_path)
_core = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_core)
maximum_path_c = _core.maximum_path_c
```

After this fix, inference ran successfully on the Pi, producing byte-for-byte the same processed text and comparable output to the Mac run. Played back through the Pi's HDMI audio output (`aplay -D plughw:0,0`) and confirmed audible, correct Luganda speech on real hardware — not just a generated file inspected remotely.

**This is a reusable finding independent of this specific model**: any VITS-family repo using this common `monotonic_align` Cython pattern is likely to hit the same Python 3.13 import issue on newer Raspberry Pi OS images, and the same direct-file-load patch should resolve it.

## Step 18 — MMS-TTS (Meta): the reliable fallback that became a core part of the pipeline

While investigating alternatives to the broken `tts-vits-lug` documentation, we found that Meta's MMS (Massively Multilingual Speech) project publishes individual small TTS checkpoints per language under the standard Hugging Face `transformers` library — `facebook/mms-tts-lug`, `-ach`, `-nyn`, `-teo`, `-eng`, and many more, each a standalone `VitsModel`.

This is **not a Sunbird model** — worth being explicit about for any paper framing — but it is real, verified, officially maintained, and dramatically simpler to deploy than any of the Sunbird-published TTS options:

- **36.3M parameters per language** (~140MB on disk) — about 1/30th the size of Sunbird's VITS checkpoints
- Standard `transformers.VitsModel` + `AutoTokenizer` — no custom repo, no Cython compilation, no system-level `espeak`/`phonemizer` dependency
- CPU-native by design

```python
from transformers import VitsModel, AutoTokenizer
import torch, scipy.io.wavfile

model = VitsModel.from_pretrained("facebook/mms-tts-lug")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-lug")
inputs = tokenizer("Oli otya leero?", return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform
scipy.io.wavfile.write("out.wav", rate=model.config.sampling_rate, data=output.squeeze().numpy())
```

Tested on both Mac (Luganda and English) and Pi (Luganda) — all produced correct, intelligible speech.

**Pi performance:** generating 1.63s of Luganda audio took **5.40s of inference time** (~3.3x slower than real-time) on the Pi 4's CPU, including first-run model download. Confirmed working via direct HDMI audio playback (`aplay -D plughw:0,0`) on the Pi itself, not just file inspection.

### TTS comparison summary

| Approach | Status | Size (per language) | Pi dependencies | Quality (subjective) |
|---|---|---|---|---|
| Orpheus-3B (Sunbird) | Not attempted — blocked upstream | 6.6GB + SNAC | None installed | Unknown |
| Spark-TTS (Sunbird) | Attempted, abandoned | 1.9GB | `unsloth` install stalled | Unknown |
| Sunbird VITS | **Working** | ~450MB/language | `torch`, `phonemizer`, `espeak`, compiled Cython ext | Good — judged more natural, esp. Luganda |
| MMS-TTS (Meta) | **Working** | ~145MB/language | `torch`, `transformers` only | Good — simpler install, slightly more generic-sounding |

Both working options are kept in the deployment: **MMS-TTS as the simple, broad-coverage default**, and **Sunbird's own VITS for languages where a dedicated checkpoint exists and quality is prioritized** (confirmed: Luganda, English, Runyankole).

