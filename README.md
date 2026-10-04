# Sunflower on a Raspberry Pi: offline speech translation for African languages

A working prototype that runs [Sunbird AI's Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B), a Gemma 4–based model for 69 African languages, **entirely on a Raspberry Pi 4**: speak into a microphone, see what it heard, get a translation, and hear it spoken back. After the one-off setup it needs no internet connection.

This repository is a reproducible **log of how that was done**, including the dead ends. The model's repository ships only a 10 GB full-precision checkpoint, so most of the work was quantizing it, adding audio input and speech output, and making it usable on a small touchscreen.

> **Status: working prototype, not a product.** Speech accuracy varies a lot by language (see [what to expect](#what-to-expect)), the touch calibration is only partly verified, and the device has no battery or enclosure yet.

## What it does

| You do | The device does |
|---|---|
| Type text | Translates or chats (llama.cpp, quantized model) |
| Speak (USB mic) | Shows a transcript, optionally a translation |
| Choose **Translate** | Speaks the translation (Sunbird VITS or Meta MMS-TTS) |
| Choose **Transcribe** | Shows the transcript only |

```
 USB mic ──► record 5 s ──► Gemma4-E2B + audio encoder ──► transcript
                                                              │
                              (Translate mode only)           ▼
 headset ◄── speak ◄── VITS / MMS-TTS ◄── translation ◄── Gemma4-E2B (text)
```

## Results at a glance

Measured on a Raspberry Pi 4 Model B (8 GB), CPU only. Mostly **single runs** on a busy device, so read them as orders of magnitude ([why](docs/benchmarks.md)).

| | |
|---|---|
| Model size | 10.21 GB original → **3.42 GB** (Q4_K_M, 66.5% smaller) + 0.99 GB audio encoder |
| Quantization level | Q4_K_M compared with seven other levels: **no detectable translation-quality loss** against the unquantized model and no broken outputs; higher levels were 20–40% slower on the Pi ([details](docs/quantization-sweep.md)) |
| Text generation (Q4_K_M) | ~2.45 tokens/s generation, ~6.0 prompt (3-repetition benchmark, ±0.02) |
| Memory during audio + text inference | system memory in use peaked at ~2.6 GB of 7.6 GB, excluding cached file pages (≈0.75 GB of that is idle baseline); the text model process's peak resident memory, including its mapped file, is ~3.5 GB |
| Speech → transcript | ~30 s warm, ~2 min on a cold start |
| Transcript → translation | ~18 s warm |
| Speech synthesis | ~3–4.5× slower than real time |

### What to expect

The model card's own evaluation (full-precision weights; not re-measured for our 4-bit version) gives word error rates of **0.15 for English, 0.16 for Swahili, about 0.50 for Luganda and Runyankole**. Expect usable transcripts in the first two and noticeable errors in the others, which carry into the translation. Details and per-language figures: [docs/02-audio-input.md](docs/02-audio-input.md#which-languages-does-audio-input-cover).

## Read this next

| | |
|---|---|
| **[Key findings](docs/findings.md)** | The 24 non-obvious lessons, one page |
| [Part 1: text model](docs/01-text-model.md) | Convert, quantize, deploy (Steps 1–10) |
| [Part 2: audio input](docs/02-audio-input.md) | Speech understanding via llama.cpp's multimodal projector (Steps 11–13) |
| [Part 3: text-to-speech](docs/03-text-to-speech.md) | Three TTS approaches tried; two kept (Steps 14–18) |
| [Part 4: live audio and demo](docs/04-live-audio-demo.md) | Real mic and headset, terminal demo (Steps 19–22) |
| [Part 5: touchscreen device](docs/05-touchscreen-device.md) | Display, touch fix, UI, launcher (Steps 23–26) |
| [Benchmarks](docs/benchmarks.md) | Timing, memory, size |
| [Quantization comparison](docs/quantization-sweep.md) | Eight levels: size, translation quality, Pi speed and memory |
| [Rebuilding from scratch](docs/rebuild-from-scratch.md) | A blank-card rebuild: what was identical, what differed, damaged files |
| [LiteRT-LM evaluation](docs/litert-lm-evaluation.md) | Google's runtime as an alternative; not adopted |
| [Alternatives and limitations](docs/limitations.md) | What was ruled out, and what is still open |

## Running it

Assumes the layout used throughout the docs (`~/ml/...`) on a Pi, after completing Parts 1–3 to produce the model files and TTS environments. To rebuild a Pi on a blank card instead, see [Rebuilding from scratch](docs/rebuild-from-scratch.md) and `scripts/setup-pi.sh` (**not yet run end to end on a blank card**; it automates the steps that were done by hand). Model files are never included.

```bash
python3 scripts/sunflower_touch_ui.py   # 480x320 touchscreen app
python3 scripts/sunflower_demo.py       # terminal menu (text/speech, in/out)
```

The touchscreen app's title bar uses Sunbird's logos, which are **not** in this repository. Fetch them with `scripts/fetch_assets.sh`; without them the app shows a text-only title. To install a desktop launcher: `scripts/pi/install_launcher.sh`. To preview the UI layout on a desktop machine: `python3 scripts/preview_ui.py`.

## Repository map

```
README.md          this page
docs/              the write-up, split by part
scripts/
  sunflower_touch_ui.py     touchscreen app (Translate / Transcribe)
  sunflower_demo.py         terminal menu version
  vits_run_inference.py     CPU inference for Sunbird VITS checkpoints
  touch_test.py             draws a marker where each touch lands (for calibration)
  preview_ui.py             desktop preview of the touchscreen UI
  fetch_assets.sh           downloads the logos (not stored here)
  setup-pi.sh               rebuild a Pi: packages, llama.cpp, Python envs, VITS, display, touch, launcher, mic
  pi/                       udev touch rule, launcher installer, VITS import fix (patch_monotonic_align.py)
  quant-sweep/              make/evaluate/score the quantization levels; Pi speed benchmark
```

`vits_run_inference.py` expects a checkout of [SunbirdAI/vits](https://github.com/SunbirdAI/vits) and the setup described in [Part 3](docs/03-text-to-speech.md). Model weights are not included.

## References

- Model: [Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B) · Org: [huggingface.co/Sunbird](https://huggingface.co/Sunbird)
- Inference engine: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Sunbird VITS: [github.com/SunbirdAI/vits](https://github.com/SunbirdAI/vits) · MMS-TTS: [facebook/mms-tts-lug](https://huggingface.co/facebook/mms-tts-lug) (and other `facebook/mms-tts-*` checkpoints)
- Orpheus-related: [SNAC codec](https://github.com/hubertsiuzdak/snac), llama.cpp tracking issue [#12476](https://github.com/ggml-org/llama.cpp/issues/12476), draft PR [#12487](https://github.com/ggml-org/llama.cpp/pull/12487)
- [Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
- Google's [gemma-translator](https://github.com/google-gemma/gemma-translator) reference project
- [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and [litert-torch](https://github.com/google-ai-edge/litert-torch) (formerly ai-edge-torch)
