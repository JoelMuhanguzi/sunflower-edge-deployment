# Deploying Sunflower-Gemma4-E2B to a Raspberry Pi 4

This document records the full, unedited process of taking [Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B) — a multimodal fine-tune of a Gemma-4-class model trained on 69 African languages — from its original Hugging Face checkpoint to a working, offline, native **speech-in / speech-out** pipeline on a Raspberry Pi 4 (8GB RAM).

It is written as a reproducible log, including the problems we hit and how they were resolved, so it can serve as a reference for deploying similar African-language LLMs to edge hardware.

## Summary

| | |
|---|---|
| Source model | `Sunbird/Sunflower-Gemma4-E2B` (BF16, ~10.2 GB, multimodal: text+image+audio) |
| Text/audio-understanding artifact | GGUF, Q4_K_M quantization (3.16 GB) + F16 multimodal projector (985 MB) |
| Inference engine (text + audio-in) | [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-cli`, `llama-mtmd-cli`) |
| Speech output (TTS) | `facebook/mms-tts-*` (Meta, via `transformers`) and Sunbird's own VITS checkpoints |
| Target device | Raspberry Pi 4 Model B Rev 1.5, 8GB RAM, Debian 13 (trixie), 64-bit |
| Result | Full pipeline — speech/text in, understanding, speech out — confirmed working natively on-device, fully offline |

This project ended up covering three connected pieces, each documented in its own part below:

1. **[Part 1](#part-1--text-understanding-and-generation)** — deploying the text backbone (translation, chat)
2. **[Part 2](#part-2--audio-input-speech-understanding)** — adding audio input (speech → text) via llama.cpp's multimodal projector
3. **[Part 3](#part-3--text-to-speech-output)** — adding speech output (text → audio), after surveying Sunbird's wider model catalog and ruling out two other TTS approaches

## Why this path

The model card describes the model as suitable for edge devices, but the Hugging Face repository only hosts the full-precision BF16 safetensors checkpoint — there is no pre-quantized GGUF release. A 10.2 GB BF16 file will not fit usefully in an 8GB Pi's RAM. The deployment therefore required:

1. Converting the HF checkpoint to GGUF (llama.cpp's native format)
2. Quantizing it to 4-bit to shrink memory/storage footprint
3. Building `llama.cpp` natively for ARM64 on the Pi
4. Transferring the quantized model to the device

We also evaluated two alternative pre-quantized releases before settling on this approach — see [Alternatives Considered](#alternatives-considered).

---

## Environment

**Host machine (conversion/quantization work):**
- MacBook Air, Apple M4, 24GB RAM, macOS 27.0.1 (Sequoia-era build)
- Used for all heavy lifting: download, GGUF conversion, quantization, and a correctness test, before shipping the final artifact to the Pi.

**Target device:**
- Raspberry Pi 4 Model B Rev 1.5, 8GB RAM
- Debian GNU/Linux 13 (trixie), 64-bit (aarch64)
- 4x Cortex-A72 cores @ 1.5GHz (no dotprod/i8mm ARM extensions — those arrived with Cortex-A76 in the Pi 5, so this is a slower baseline than newer ARM boards)

---

## Part 1 — Text understanding and generation

## Step 1 — Install build tooling (Mac)

```bash
brew install cmake git-lfs
git lfs install
```

## Step 2 — Clone and build llama.cpp (Mac)

```bash
mkdir -p ~/ml && cd ~/ml
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

`-DGGML_METAL=ON` enables Apple Silicon GPU acceleration for local testing. This flag is Mac-only and is **not** used in the Pi build (see Step 7).

> **Note:** our first clone attempt stalled/hung on a flaky connection and left a broken `.git` state. If `git clone` appears to hang indefinitely, check with `du -sh llama.cpp` in another terminal — if size isn't growing, kill it and retry with `--depth 1` (shallow clone), which is faster and sufficient since we don't need history.

## Step 3 — Set up the Python conversion environment (Mac)

llama.cpp's HF→GGUF converter needs `torch`, `transformers`, `gguf`, `sentencepiece`, and `protobuf`.

```bash
python3 -m venv ~/ml/venv
~/ml/venv/bin/pip install --upgrade pip
~/ml/venv/bin/pip install -r ~/ml/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

## Step 4 — Authenticate with Hugging Face and download the model

The repository is gated, requiring accepted access terms plus a token.

```bash
huggingface-cli login
# paste a Hugging Face access token (Settings -> Access Tokens -> Read) when prompted
```

```bash
mkdir -p ~/ml/models
huggingface-cli download Sunbird/Sunflower-Gemma4-E2B \
  --local-dir ~/ml/models/Sunflower-Gemma4-E2B
```

This pulls down ~9.6 GB across `model.safetensors`, tokenizer files, and configs.

## Step 5 — Convert HF safetensors to GGUF (Mac)

First, confirm the architecture is supported by the converter:

```bash
cd ~/ml/llama.cpp
~/ml/venv/bin/python convert_hf_to_gguf.py --print-supported-models | grep -i gemma
```

This confirmed `Gemma4ForConditionalGeneration` — matching the model's `config.json` — is natively supported.

### Problem: tokenizer config incompatibility

The first conversion attempt failed with:

```
AttributeError: 'list' object has no attribute 'keys'
```

Root cause: the checkpoint's `tokenizer_config.json` stores `extra_special_tokens` as a list (`["<|video|>"]`), but the installed `transformers` version expects a dict (e.g. `{"video_token": "<|video|>"}`). This is a version-compatibility mismatch between how the checkpoint was saved and the transformers release used locally, not a problem with the model weights themselves.

**Fix applied** (local-only, not uploaded anywhere): backed up the original config, then replaced the malformed field with an empty dict. Since this deployment only needed text input/output (not video), dropping the video special-token mapping had no effect on functionality.

```bash
cp ~/ml/models/Sunflower-Gemma4-E2B/tokenizer_config.json{,.bak}
python3 -c "
import json
path = '/Users/joelmuhanguzi/ml/models/Sunflower-Gemma4-E2B/tokenizer_config.json'
with open(path) as f:
    d = json.load(f)
if isinstance(d.get('extra_special_tokens'), list):
    d['extra_special_tokens'] = {}
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
"
```

### Running the conversion

```bash
mkdir -p ~/ml/gguf
~/ml/venv/bin/python convert_hf_to_gguf.py \
  ~/ml/models/Sunflower-Gemma4-E2B \
  --outfile ~/ml/gguf/sunflower-gemma4-e2b-f16.gguf \
  --outtype f16
```

Output: `sunflower-gemma4-e2b-f16.gguf`, 541 tensors, **9.27 GB**. This extracts and converts only the text-generation backbone — the audio and vision towers present in the original checkpoint are not part of the standard GGUF text conversion path and were not carried over.

## Step 6 — Quantize to Q4_K_M (Mac)

```bash
./build/bin/llama-quantize \
  ~/ml/gguf/sunflower-gemma4-e2b-f16.gguf \
  ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  Q4_K_M
```

Result: **8828.84 MiB -> 3242.78 MiB** (16.00 BPW -> 5.88 BPW), final file **3.16 GB**.

`Q4_K_M` ("K-quant, medium") quantizes most tensors to 4 bits while keeping select sensitive tensors (e.g. `ffn_down`) at 6 bits, which preserves more quality than a flat 4-bit scheme at a modest size cost. It's llama.cpp's standard recommended default for resource-constrained inference.

## Step 7 — Validate locally before shipping to the Pi (Mac)

```bash
./build/bin/llama-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  -p "Translate to Luganda: How are you today?" \
  -n 64 --temp 0.3
```

**Output:** `Oli otya leero?` — the correct, idiomatic Luganda translation. This confirmed the conversion and quantization preserved the model's actual translation capability before investing time in the Pi transfer. Ran at 68.5 tok/s prompt / 47.2 tok/s generation on the M4 (Metal-accelerated) — recorded here only as a sanity baseline, not representative of Pi performance.

## Step 8 — Set up llama.cpp on the Pi

Over SSH:

```bash
ssh <user>@<pi-ip>
sudo apt-get update
sudo apt-get install -y cmake   # git and gcc/g++ were already present on this Pi OS image

mkdir -p ~/ml && cd ~/ml
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

No `-DGGML_METAL` flag here (Mac-only). CMake auto-detected the Pi's Cortex-A72 cores and correctly disabled ARM dot-product/matmul-int8 extensions (`nodotprod`, `noi8mm`) that the A72 doesn't support — this is expected and not an error; it just means the build targets the CPU's actual capabilities. A warning about missing OpenSSL appeared too — this only disables HTTPS support in the optional `llama-server` HTTP component and did not block the build or affect `llama-cli`.

Verify the build:

```bash
~/ml/llama.cpp/build/bin/llama-cli --version
```

## Step 9 — Transfer the quantized model to the Pi

### Attempt 1: `scp` over Wi-Fi — failed

```bash
scp ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf <user>@<pi-ip>:~/ml/gguf/
```

This transfer died mid-copy when the laptop briefly left the Pi's network. `scp` does not resume — the partial file had to be deleted and the transfer restarted from zero.

### Attempt 2: `rsync --partial` over Wi-Fi — too slow

```bash
rsync -avh --progress --partial \
  ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  <user>@<pi-ip>:~/ml/gguf/
```

`rsync` is resumable, which solved the drop-out problem, but measured throughput on this network was only ~800 KB/s — on the order of an hour-plus for 3.16 GB. For a one-off transfer, this made a physical alternative worth it.

### Attempt 3: USB flash drive — used in the end

1. Copied the GGUF file onto a USB flash drive via macOS Finder.
   - Note: direct command-line `cp`/`touch` to the mounted USB volume failed with `Operation not permitted` due to macOS file-access permissions on the terminal session in use; Finder drag-and-drop bypassed this cleanly.
   - Also note: the target filesystem was FAT32, which has a 4 GB per-file limit. Our 3.16 GB file fit; the unquantized 9.27 GB F16 file would **not** have fit and would need exFAT/ext4 or splitting.
2. Physically moved the drive to the Pi and plugged it in — it auto-mounted at `/media/<user>/<volume-name>`.
3. Copied from the USB mount to the Pi's internal storage (its SD card root filesystem) over a local `cp`:

```bash
cp /media/<user>/<volume-name>/sunflower-gemma4-e2b-Q4_K_M.gguf ~/ml/gguf/
```

This local disk-to-disk copy ran at roughly 25 MB/s — more than 30x faster than the Wi-Fi transfer — and completed in about two minutes.

File integrity was confirmed by comparing byte-for-byte file size against the source (`3,416,120,736` bytes on both ends).

## Step 10 — Run inference natively on the Pi

```bash
cd ~/ml/llama.cpp
./build/bin/llama-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  -p "Translate to Luganda: How are you today?" \
  -n 64 --temp 0.3
```

**Output:** `Oli otya leero?` — identical, correct output to the Mac test.

**Performance on Raspberry Pi 4 (CPU-only, no accelerator):**

| Metric | Value |
|---|---|
| Prompt processing | 6.0 tok/s |
| Generation | 2.3 tok/s |

### Interactive use

For multi-turn use without relaunching the binary each time, omit `-p`:

```bash
./build/bin/llama-cli -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf -n 512 --temp 0.5 -c 2048
```

- `-n` — max tokens generated per response
- `--temp` — sampling temperature (lower = more deterministic/focused, higher = more varied)
- `-c` — context window size (conversation memory length)

Exit with `Ctrl+C` or `/exit`.

---

## Part 2 — Audio input (speech understanding)

Sunflower-Gemma4-E2B's `config.json` declares an `audio_config` (model type `gemma4_audio`) alongside the text backbone — the checkpoint supports speech input natively, not just text. Part 1 only converted the text decoder; this part adds the audio (and vision) encoder so the same model can listen to spoken audio and respond in text.

## Step 11 — Confirm llama.cpp support and export the multimodal projector (Mac)

llama.cpp represents multimodal encoders as a separate **mmproj** (multimodal projector) GGUF file, loaded alongside the main text GGUF. Support is architecture-specific and declared explicitly in the converter:

```bash
cd ~/ml/llama.cpp
~/ml/venv/bin/python convert_hf_to_gguf.py --print-supported-models | grep -i gemma4
```

This surfaced a dedicated `Gemma4VisionAudioModel` class registered against `Gemma4ForConditionalGeneration` (the exact architecture of our model) with `has_audio_encoder = True` and `has_vision_encoder = True` — i.e. this isn't a generic "might work" path, llama.cpp has purpose-built support for this model's audio tower.

Export it from the same HF checkpoint already on disk (no re-download needed):

```bash
~/ml/venv/bin/python convert_hf_to_gguf.py \
  ~/ml/models/Sunflower-Gemma4-E2B \
  --outfile ~/ml/gguf/mmproj-sunflower-gemma4-e2b-f16.gguf \
  --mmproj --outtype f16
```

Output: `mmproj-sunflower-gemma4-e2b-f16.gguf`, **985 MB** (vision + audio encoders combined, F16).

## Step 12 — Test audio input locally (Mac)

llama.cpp's multimodal CLI is a separate binary, `llama-mtmd-cli`, built automatically as part of the normal `cmake --build` (Step 2) — no extra build flags needed.

Generated a test clip with macOS's built-in TTS, resampled to the 16kHz mono the model's audio encoder expects:

```bash
say -o test_audio.aiff "How are you today?"
ffmpeg -y -i test_audio.aiff -ar 16000 -ac 1 -c:a pcm_s16le test_audio.wav
```

```bash
./build/bin/llama-mtmd-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  --mmproj ~/ml/gguf/mmproj-sunflower-gemma4-e2b-f16.gguf \
  --audio test_audio.wav \
  --jinja \
  -p "What did the speaker say? Respond in English." \
  -n 64 --temp 0.3
```

**Problem:** the first attempt crashed with `std::runtime_error: this custom template is not supported, try using --jinja` — the model's chat template requires the `--jinja` flag (Jinja2-based template rendering) rather than the CLI's built-in template handling. Adding `--jinja` fixed it.

**Output:** `Hello! How are you today?` — a coherent, relevant response to the actual spoken content (not a generic fallback), confirming real audio understanding. Audio encoding itself took ~190ms. llama.cpp prints a warning that audio input is "in experimental stage and may have reduced quality" — worth keeping in mind, but it worked correctly on every phrase tested.

## Step 13 — Transfer mmproj and test on the Pi

Transferred via the same USB-drive method as Step 9 (Wi-Fi on this network was not retested — already known to be unreliable/slow for large files). Verified byte-identical file size after the USB → Pi-internal-storage copy.

```bash
cd ~/ml/llama.cpp
./build/bin/llama-mtmd-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  --mmproj ~/ml/gguf/mmproj-sunflower-gemma4-e2b-f16.gguf \
  --audio ~/ml/test_audio.wav \
  --jinja \
  -p "What did the speaker say? Respond in English." \
  -n 64 --temp 0.3
```

**Output:** `How are you today` — correct, matching the Mac result. Audio batch encoding took 1475ms on the Pi's CPU (vs. ~190ms on the M4 — expected, given no Metal/GPU acceleration on the Pi). Total model+mmproj load time was ~2 minutes 14 seconds.

**Confirmed: real-time audio understanding works natively on the Pi 4, fully offline, using the same single `llama-mtmd-cli` binary already built in Part 1.**

### Memory footprint with both files loaded

| Component | Size on disk | Approx. resident RAM |
|---|---|---|
| Text model (Q4_K_M) | 3.16 GB | ~3.2–3.5 GB |
| mmproj (vision+audio, F16) | 985 MB | ~1.0–1.2 GB |
| **Total** | **~4.1 GB** | **~4.5 GB** |
| Pi 4 available RAM | — | 7.6 GB usable |

Comfortable headroom — well under half of available RAM, with room to spare for a TTS process running alongside (Part 3).

---

## Part 3 — Text-to-speech output

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

---

## Alternatives considered

Two other release artifacts were evaluated before settling on the manual-quantization path:

1. **`Sunbird/translate-nllb-3.3b-salt`** — a 3.3B NLLB-based translation model also published by Sunbird. Ruled out because NLLB is an encoder-decoder (seq2seq) architecture; llama.cpp's GGUF converter only supports decoder-only causal LM architectures. This model would require a different native runtime path (e.g. direct PyTorch/Transformers inference with dynamic int8 quantization), not GGUF/llama.cpp.
2. **`ak3ra/sunflower-cactus-int4`** — a pre-quantized (int4) 4.24 GB archive referencing the [Cactus](https://github.com/cactus-compute/cactus) edge-AI engine. Cactus is a legitimate edge-inference framework with its own model format and runtime, explicitly supporting Gemma-family models — but it primarily targets mobile/wearable platforms (iOS, Android), and Raspberry Pi/Linux support was not clearly documented. The repository also had no model card describing its contents. Given the lack of documentation and unconfirmed Pi compatibility, we chose the better-established llama.cpp path instead.

See also [Part 3's alternatives](#step-15--orpheus-3b-tts-ruled-out-before-attempting) (Orpheus-3B, Spark-TTS) for the TTS-specific evaluation.

## Known limitations / follow-ups

- Generation speed (~2.3 tok/s text, ~3.3x real-time for TTS) is workable for short exchanges but slow for long-form content. Lighter quantizations (e.g. `Q4_0`, `Q3_K_M` for the text model) would trade quality for speed and have not yet been benchmarked.
- The tokenizer config fix (Part 1, Step 5) and the `monotonic_align` import fix (Part 3, Step 17) were applied to local copies only; neither was upstreamed or reported to the respective maintainers.
- No persistent serving layer (e.g. `llama-server` with an OpenAI-compatible HTTP API) has been set up yet — current usage is via interactive CLI tools over SSH.
- Live microphone input and real-time speaker output (as opposed to file-based audio in/out) have not yet been tested — next planned step is a USB microphone (ALSA, likely via `arecord`) and Bluetooth speaker (via `bluetoothctl` pairing + PulseAudio/PipeWire routing).
- Audio input quality was only tested with clear, single-speaker, English TTS-generated clips — not yet validated against real recorded speech in any of the 69 supported African languages, background noise, or multiple speakers.
- The three Sunbird VITS languages not yet tested (Acholi, Ateso, Lug+Eng bilingual) remain to be verified using the same process documented in Step 17.
- TTS voice quality (Sunbird VITS vs. MMS-TTS) was judged subjectively by ear, not measured with any formal metric (e.g. MOS, PESQ) — worth doing properly for the paper.

## References

- Model: [huggingface.co/Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B)
- Inference engine: [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
- Org: [huggingface.co/Sunbird](https://huggingface.co/Sunbird)
- Sunbird VITS: [github.com/SunbirdAI/vits](https://github.com/SunbirdAI/vits)
- MMS-TTS: [huggingface.co/facebook/mms-tts-lug](https://huggingface.co/facebook/mms-tts-lug) (and other `facebook/mms-tts-*` language checkpoints)
- SNAC codec (relevant to Orpheus): [github.com/hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)
- llama.cpp Orpheus/SNAC tracking issue: [#12476](https://github.com/ggml-org/llama.cpp/issues/12476), draft PR [#12487](https://github.com/ggml-org/llama.cpp/pull/12487)
- Spark-TTS: [github.com/SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
