# Sunflower at the edge: offline speech translation for African languages on small boards

A working prototype that translates speech between English and African languages **entirely on a Raspberry Pi 4**: speak into a microphone, see what it heard, get a translation, and hear it spoken back. After the one-off setup it needs no internet connection.

Two pipelines were built and compared on the same device:

- **The first app, "Sunflower"** uses one model, [Sunbird AI's Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B) (a Gemma 4–based model for 69 African languages), for both listening and translating (Parts 1–5).
- **The second app, "Sunflower Fast"** swaps that for Sunbird's smaller specialist models, a Whisper-based speech recogniser and an NLLB translator, kept loaded in memory, and speaks English and Luganda with a small character-level ONNX voice. On the Pi 4 it took **a median 41 s per spoken round trip against 122 s** for the first app as originally built (Part 6).
- **A third app, "Sunflower Chat"**, is for conversation with Gemma: speak or type a question and read the reply as it streams in (Part 7).

All of it was measured on a **Raspberry Pi 4 and, in Part 8, a Raspberry Pi 5** under matching conditions. On the Pi 5 the fastest setup (Gemma kept loaded) takes **12 s per spoken round trip** against 63–66 s on the Pi 4, and the Whisper + NLLB pipeline takes 17–20 s against 37–40 s.

This repository is a reproducible **log of how that was done**, including the dead ends. The Gemma repository ships only a 10 GB full-precision checkpoint, so much of the work was quantizing it, adding audio input and speech output, and making it usable on a small touchscreen. Part 6 then asked whether the specialist models do better and measured both pipelines on the same recordings.

> **Status: working prototype, not a product.** Speech accuracy varies a lot by language (see [what to expect](#what-to-expect)); the accuracy comparison between pipelines rests on six recordings of one speaker; the touch calibration is only partly verified; and the device has no battery or enclosure yet.

## What it does

| You do | The device does |
|---|---|
| Type text | Translates or chats (llama.cpp, quantized model) |
| Speak (USB mic) | Shows a transcript, optionally a translation |
| Choose **Translate** | Speaks the translation (Sunbird VITS or Meta MMS-TTS) |
| Choose **Transcribe** | Shows the transcript only |
| Open **Sunflower Chat**, then speak or type | Answers as a conversation, streaming the reply word by word |

```
 First app (Sunflower)
   USB mic ─► record 5 s ─► Gemma4-E2B + audio encoder ─► transcript ─► Gemma4-E2B (text) ─► translation ─┐
 Second app (Sunflower Fast)                                                                              │
   USB mic ─► record 5 s ─► Whisper large-v3 int8 ─────► transcript ─► NLLB-1.3B int8 ────► translation ─┤
                                                                                                          ▼
                                       headset ◄─ speak ◄─ Sunbird VITS / Meta MMS-TTS (first app) ◄──┤
                                       headset ◄─ speak ◄─ ONNX voice, English + Luganda (second app) ◄┘
```

## Results at a glance

### Pi 4 and Pi 5, measured the same way (Part 8)

Median seconds for one spoken round trip (listen, translate, speak) over six recorded 5-second clips, both boards headless from a cold boot with the same scripts. Pi 4: heatsinks and two fans (the soft temperature limit was reached on most runs). Pi 5: active cooler, 5 V / 5 A supply, never throttled. [Full tables, stage splits and caveats](docs/08-pi5-and-matched-rerun.md); raw data in [`results/`](results/README.md).

| Setup | Pi 4 total | Pi 5 total | Pi 5: listen / translate / voice | Pi 5 memory |
|---|---|---|---|---|
| A: first app as built (Gemma started per tap) | 98.4 | 51.7 | 27.5 / 10.7 / 12.8 | – |
| B: Gemma kept loaded, Sunbird VITS voices | 66.1 | **12.3** | 7.1 / 2.7 / 2.6 | 6.6 GB |
| E: Gemma kept loaded, ONNX voice | 63.0 | 12.9 | 7.4 / 3.0 / 1.7 | 6.1 GB |
| C: Whisper + NLLB int8, Sunbird VITS voices | 40.0 | 20.1 | 9.0 / 7.4 / 3.2 | 4.4 GB |
| D: Whisper + NLLB int8, ONNX voice | **36.6** | 17.5 | 8.8 / 6.6 / 1.8 | 3.7 GB |

- **The best pipeline depends on the board.** On the Pi 4 the specialist models (Whisper + NLLB) are faster; on the Pi 5 Gemma kept loaded is, because its generation speed rose 3.2× (2.45 to 7.80 tokens/s) and its prompt processing 6.7×, while the CTranslate2 decoders gained about 2×.
- **Keeping models loaded is the biggest single gain on the Pi 5** (A to B: 51.7 s to 12.3 s).
- **Where the time goes:** at the 6 s window the app uses, Whisper spends 82% of its time in the decoder and 18% in the encoder; NLLB spends nearly all of it in the decoding loop (about half a second per token). int8 is not the cause: float32 is 2–4× slower for both.
- **The ONNX voice** (109 MB, English and Luganda) synthesizes faster than real time on the Pi 5 and about 2× real time on the Pi 4; its int8 version is 3–6× *slower*.
- Word error rate: 0.06 for every Whisper setup; Gemma's varies between runs (0.08–0.29) because it samples. One speaker, six clips: "comparable", not a ranking.

### The original Pi 4 comparison (Part 6)

Measured on a Raspberry Pi 4 Model B (8 GB), CPU only, with heatsinks and fans. The headline comparison uses **six 5-second recordings of one speaker** (three English, three Luganda) put through each setup in the same way, with the benchmark waiting up to 15 minutes for the CPU to fall below 60 °C before each setup (per-clip start temperatures were 48–79 °C for setups A to C-final, so some runs were warm); figures are medians in seconds ([method and raw results](docs/06-fast-pipeline.md#head-to-head-on-the-pi-4-six-recorded-clips)).

| Setup | Listen | Translate | Voice | **Spoken round trip** | Start-up |
|---|---|---|---|---|---|
| First app as originally built (Gemma started afresh on every tap) | 78.8 | 21.3 | 24.1 | **121.6 s** | none |
| First app now (Gemma and two voices kept loaded) | 43.6 | 12.4 | 9.6 | **65.1 s** | 1–2 min |
| **Second app** (Whisper + NLLB int8, kept loaded) | **19.3** | **10.7** | **9.9** | **40.8 s** | about 100 s |
| Second app with the ONNX voice (run on a hotter CPU, so compare only the voice column) | 20.6 | 11.7 | **7.1** | 40.8 s | about 83 s |

The second app is about **3× faster** than the first as built and 1.6× faster than Gemma kept loaded. Word error rates on these clips were comparable (0.09, 0.09 and 0.06), but that is one speaker and six sentences, so read it as "not obviously worse", not as a ranking. The middle row is the benchmark's setup with Gemma kept loaded; the changed app itself took 61–71 s on three of the clips.

**What produced the speed-up**
- **Loading models once** took the same Gemma model from 122 s to 65 s per round trip (at the cost of about 4.3 GB held in memory and a 1–2 minute start-up).
- **Smaller specialist models** took it from 65 s to 41 s: Whisper int8 (1.56 GB) and NLLB int8 (1.38 GB) against Gemma Q4_K_M plus its audio encoder (4.41 GB).
- **Converting Whisper to int8 on the Mac** cut its load on the Pi from 82 s to 2.8 s, and **shortening its 30-second window** cut transcription of a short clip from about 70 s to about 14 s (a median of 19 s over the recorded clips with the final 6 s window). A window exactly as long as the recording made Whisper repeat sentences (one run looped for 300 s), so the app uses 6 s plus a token cap and a repeat guard.

| | |
|---|---|
| Gemma size | 10.21 GB original → **3.42 GB** (Q4_K_M, 66.5% smaller) + 0.99 GB audio encoder |
| Gemma quantization level | Q4_K_M compared with seven other levels: **no detectable translation-quality loss** against the unquantized model and no broken outputs; higher levels were 20–40% slower on the Pi ([details](docs/quantization-sweep.md)) |
| Gemma generation speed | ~2.45 tokens/s generation, ~6.0 prompt (3-repetition benchmark, ±0.02); unchanged by keeping it loaded. Whisper and NLLB stage rates (about 0.7 and 1.0 tokens/s end to end) include their encoders and are not comparable |
| Memory | First app with Gemma kept loaded: ~4.3 GB resident in the server. Second app: 3.9 GB in use with Whisper and NLLB, ~4.6 GB after a run with the voices. Original per-tap Gemma run: ~2.6 GB peak in use. The two apps cannot be open together |
| Chat (Sunflower Chat, Pi 4) | First word after ~9 s, then ~2.2–2.3 tokens/s; a short answer takes 15–20 s ([Part 7](docs/07-sunflower-chat.md)) |
| Speech synthesis | Sunbird VITS: 2.6–3.7× the audio length on the Pi 4 (a background worker saves the per-tap load of about 10 s, not the synthesis). The small ONNX voice (109 MB, English and Luganda) runs at about 2×, taking 3–8 s per sentence; its int8 version is 4–5× slower. Nobody has judged its sound quality ([details](docs/06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model)) |
| A 4 GB board | On an Orange Pi Zero 2W, Whisper int8 took ~24 s per clip and NLLB 9–20 s per sentence, with throttling; Gemma plus its audio encoder cannot fit |

Mostly small samples on a device that throttles under sustained load (the CPU reached 72–79 °C), so read the figures as the right order of magnitude; the earlier single-run figures are in [Benchmarks](docs/benchmarks.md). An earlier claim of "about 30 s to listen" came from a clip about a second long and has been corrected: with 5 s recordings the original first app took about 78 s.

### Models and sizes

All on disk, as used on the Pi 4. Nothing here is stored in this repository.

| Role | Model | Precision | Size | Used by |
|---|---|---|---|---|
| Listening and translating | [Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B), converted to GGUF | Q4_K_M (original BF16: 10.21 GB) | 3.42 GB | first app, chat |
| Audio encoder for Gemma | mmproj from the same model | F16 | 0.99 GB | first app |
| Listening | Sunbird SunflowerASR-51 (Whisper large-v3 fine-tune), converted with CTranslate2 | int8 (float32 original: 6.17 GB) | 1.56 GB | second app |
| Translating | Sunbird NLLB-1.3B (SALT), CTranslate2 | int8 | 1.38 GB | second app |
| Voice, English and Luganda | [jq/sherpa-vits-tts-lug-eng](https://huggingface.co/jq/sherpa-vits-tts-lug-eng), character-level VITS in ONNX | fp32 (an int8 copy we made is 38 MB and slower) | 109 MB | second app |
| Voice, per language | Sunbird VITS (English, Luganda, Runyankole) | fp32 | 145 MB, 450 MB, ~450 MB | first app; second app for Runyankole |
| Voice, other languages | Meta MMS-TTS | fp32 | 139 MB each | both apps |

Models in use: first app 4.41 GB for Gemma and its audio encoder; second app 2.94 GB for Whisper and NLLB, plus the voice. The measured figures that go with these are in [Benchmarks](docs/benchmarks.md#comparing-the-two-pipelines-part-6).

### Where the raw data is

[`results/`](results/README.md) holds the raw measurements behind every table: the six-clip head-to-head (JSON lines per run), the voice timings, and the quantization sweep's per-sentence outputs and scores. `results/headtohead.csv`, `models.csv` and `voice-pi4.csv` hold the headline figures as plain tables; `scripts/fast/key_numbers.py` regenerates the first from the raw files.

### What to expect

The Gemma model card's own evaluation (full-precision weights; not re-measured for our 4-bit version) gives word error rates of **0.15 for English, 0.16 for Swahili, about 0.50 for Luganda and Runyankole**. On our six clips both pipelines transcribed English exactly and Luganda with word error rates between 0.00 and 0.40 per sentence, better than the model card's corpus-level figure suggests, but six hand-picked short sentences are not comparable with a corpus. Expect noticeable errors in Luganda and Runyankole, which carry into the translation. The second app's translation model does not cover Swahili. Details and per-language figures: [docs/02-audio-input.md](docs/02-audio-input.md#which-languages-does-audio-input-cover).

## Read this next

| | |
|---|---|
| **[Key findings](docs/findings.md)** | The 43 non-obvious lessons, one page |
| [Part 1: text model](docs/01-text-model.md) | Convert, quantize, deploy (Steps 1–10) |
| [Part 2: audio input](docs/02-audio-input.md) | Speech understanding via llama.cpp's multimodal projector (Steps 11–13) |
| [Part 3: text-to-speech](docs/03-text-to-speech.md) | Three TTS approaches tried; two kept (Steps 14–18) |
| [Part 4: live audio and demo](docs/04-live-audio-demo.md) | Real mic and headset, terminal demo (Steps 19–22) |
| [Part 5: touchscreen device](docs/05-touchscreen-device.md) | Display, touch fix, UI, launcher (Steps 23–26) |
| [Part 6: a second, faster pipeline](docs/06-fast-pipeline.md) | Whisper + NLLB int8 instead of Gemma for listening and translating, kept loaded in memory: about 3× faster per spoken round trip on the Pi 4 in a six-clip head-to-head; a small ONNX voice for English and Luganda; also Orange Pi and Mac measurements |
| [Part 7: Sunflower Chat](docs/07-sunflower-chat.md) | A conversation app: speak or type to Gemma on the touchscreen, replies streamed, optional spoken replies |
| [Part 8: Pi 5 and a matched Pi 4 rerun](docs/08-pi5-and-matched-rerun.md) | Both boards measured the same way: six setups, where Whisper's and NLLB's time goes, int8 against float32, Gemma's generation speed, memory |
| [Benchmarks](docs/benchmarks.md) | Timing, memory, size, including the two pipelines side by side |
| [Quantization comparison](docs/quantization-sweep.md) | Eight levels: size, translation quality, Pi speed and memory |
| [Rebuilding from scratch](docs/rebuild-from-scratch.md) | A blank-card rebuild: what was identical, what differed, damaged files |
| [LiteRT-LM evaluation](docs/litert-lm-evaluation.md) | Google's runtime as an alternative; not adopted |
| [Alternatives and limitations](docs/limitations.md) | What was ruled out, and what is still open |

## Running it

Assumes the layout used throughout the docs (`~/ml/...`) on a Pi, after completing Parts 1–3 to produce the model files and TTS environments. To rebuild a Pi on a blank card instead, see [Rebuilding from scratch](docs/rebuild-from-scratch.md) and `scripts/setup-pi.sh` (**not yet run end to end on a blank card**; it automates the steps that were done by hand). Model files are never included.

```bash
python3 scripts/sunflower_touch_ui.py   # 480x320 touchscreen app (Gemma kept loaded; start-up 1-2 min)
scripts/fast/sunflower_fast_ui.py       # second app, Whisper + NLLB (see Part 6; needs its own Python environment)
python3 scripts/sunflower_chat_ui.py    # third app, chat with Gemma (see Part 7)
python3 scripts/sunflower_demo.py       # terminal menu (text/speech, in/out)
python3 scripts/fast/bench_compare.py D # benchmark setups A, B, C or D on a Pi (see results/README.md)
```

The touchscreen app's title bar uses Sunbird's logos, which are **not** in this repository. Fetch them with `scripts/fetch_assets.sh`; without them the app shows a text-only title. To install a desktop launcher: `scripts/pi/install_launcher.sh`. To preview the UI layout on a desktop machine: `python3 scripts/preview_ui.py`.

## Repository map

```
README.md          this page
LICENSE            MIT (code and docs; the models keep their own licenses)
results/           raw measurements and CSV tables for the paper (see results/README.md)
docs/              the write-up, split by part
scripts/
  sunflower_touch_ui.py     touchscreen app (Translate / Transcribe); Gemma and two voices kept loaded
  sunflower_chat_ui.py      chat app: speak or type to Gemma, streamed replies (Part 7)
  vits_worker.py            background voice synthesizer shared by the touchscreen apps
  baseline/                 frozen copy of the first app as originally benchmarked (Gemma started per tap)
  sunflower_demo.py         terminal menu version
  vits_run_inference.py     CPU inference for Sunbird VITS checkpoints
  touch_test.py             draws a marker where each touch lands (for calibration)
  preview_ui.py             desktop preview of the touchscreen UI
  fetch_assets.sh           downloads the logos (not stored here)
  fast/                     second pipeline: Whisper + NLLB app, ONNX voice, benchmark scripts, diagnosis experiments (Parts 6 and 8)
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

## License

The code and documentation in this repository are released under the [MIT License](LICENSE). The models it uses (Sunflower/Gemma, Sunbird Whisper, NLLB, VITS, MMS-TTS) are not part of this repository and keep their own licenses; check each model's page before reuse. The NLLB base model is published under CC-BY-NC.
