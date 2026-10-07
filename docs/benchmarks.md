# Benchmarks

[← Back to the project README](../README.md)


All Pi numbers below were measured on the Raspberry Pi 4 Model B Rev 1.5 (8GB RAM, Cortex-A72 @ 1.5GHz, CPU-only, Debian 13/trixie) described in [Environment](01-text-model.md#environment). Mac numbers are from the Apple M4 host used for development, included only as a sanity baseline — they are not representative of edge-device performance and should not be read as "the model is this fast," only "the conversion/quantization didn't break anything."

Apart from the [pipeline comparison](#comparing-the-two-pipelines-part-6) just below (medians of six runs per setup), everything here is a single-sample measurement per configuration, not an average over many runs, except where noted (MMS-TTS). Edge-hardware benchmarks are noisy — see the variance note below — so treat single numbers as indicative, not precise.

### Comparing the two pipelines (Part 6)

Full method, caveats and raw data: [Part 6](06-fast-pipeline.md) and `results/pi4-headtohead-2026-10-06/`. The first pipeline (Gemma for listening and translating) is compared with the second (Whisper for listening, NLLB for translating) on the Pi 4, with the same six recorded 5 s clips (three English, three Luganda; one speaker) and identical stage timing. Medians in seconds; the benchmark waited up to 15 minutes for the CPU to fall below 60 °C before each setup, but per-clip start temperatures were 48–79 °C for setups A to C-final, so some runs were warm.

**Spoken round trip on the Pi 4**

| Setup | Listen | Translate | Voice | **Total** | Slowest run | Start-up | Word error rate | Clips with repeats |
|---|---|---|---|---|---|---|---|---|
| A: first app as originally built (Gemma started per tap) | 78.8 | 21.3 | 24.1 | **121.6** | 200.5 (cold first call) | none | 0.09 | 0 of 6 |
| B: Gemma kept loaded in `llama-server`, voices preloaded (what the first app now does) | 43.6 | 12.4 | 9.6 | **65.1** | 70.8 | 56 s | 0.09 | 0 of 6 |
| C-5s: second app, 5 s window | 27.0 | 15.0 | 13.5 | 56.2 | 335.7 (a 299 s loop) | 96 s | 1.03 | 3 of 6 |
| C-tuned: second app, 6 s window + decoding options | 19.8 | 11.6 | 10.1 | 42.6 | 45.2 | 107 s | 0.23 | 1 of 6 |
| **C-final: second app as shipped** | **19.3** | **10.7** | **9.9** | **40.8** | 43.5 | 102 s | **0.06** | 0 of 6 |

The first app after the change took 61–71 s on three of the clips in a headless run (start-up 111 s for Gemma and 26 s for the voices from a cold cache).

**Models and memory**

| | First pipeline | Second pipeline |
|---|---|---|
| Listening and translating models on disk | Gemma4-E2B Q4_K_M 3.42 GB + audio encoder 0.99 GB = **4.41 GB** | Whisper int8 1.56 GB + NLLB int8 1.38 GB = **2.94 GB** |
| Voices (same in both apps) | Sunbird VITS English 145 MB, Luganda 450 MB, Runyankole ~450 MB; MMS-TTS 139 MB each | same; the second app now uses the character-level ONNX voice (109 MB, English and Luganda) in place of the English and Luganda VITS voices |
| Memory in use, kept loaded | ~4.3 GB resident in the server (includes the mapped model file) | 3.9 GB with Whisper and NLLB; 4.6 GB after a run with the voices |
| Generation speed | 2.40–2.47 tokens/s (setups A and B; `llama-bench`: 2.45) | not comparable: Whisper 0.7 and NLLB 1.0 tokens/s end to end including encoder and beam search |
| Languages | Sunflower's 69 | transcription as Gemma's; translation limited to eng, ach, lgg, lug, nyn, teo (**no Swahili**) |

**Whisper int8, English clip of 1.1 s (single runs)**

| | Mac | Pi 4 | Orange Pi Zero 2W (4 GB) |
|---|---|---|---|
| Model load | 0.4 s | **2.8 s** (82 s when quantized on the fly from float16) | 53 s |
| 30 s window (stock) | 5.8 s | 66–72 s | 149–151 s |
| 5 s window | 1.0 s | 13.9–14.1 s | 24.3–24.5 s |

**NLLB int8 translation, English to Luganda (single runs)**

| | Mac | Pi 4 | Orange Pi |
|---|---|---|---|
| Load | 0.8 s | 39 s | 21–43 s |
| Per sentence (beam 5) | 0.4–0.6 s | 5.7–12.7 s | 9.2–19.5 s |
| For comparison: Gemma translating the same sentences, kept loaded | – | 7.6–11.0 s | – |

**Voice on the Pi 4, one Luganda sentence of 9 words**

| Path | Time |
|---|---|
| New process per tap (standalone) | 31.3 s (load 3.9 + synthesis 21.5 + about 6 s start-up) |
| New process per tap (inside the app) | 53.3 s (not reproduced standalone) |
| Preloaded worker (standalone) | 14.3 s and 19.6 s |
| Preloaded worker (inside the app) | 15.4 s |
| Within the six-clip benchmark | 5.7–11.8 s for 2.0–4.4 s of speech, about 2.6–3.7× the audio length |

**Character-level ONNX voice** (`jq/sherpa-vits-tts-lug-eng`, one model for English and Luganda; method and caveats in [Part 6](06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model)). Synthesis time on the Pi 4, three runs per sentence, CPU below 60 °C at the start:

| Sentence | Audio | fp32 (109 MB) | int8 (38 MB) |
|---|---|---|---|
| English, 5 words | 1.4 s | 3.2, 3.0, 2.8 s | 14.9, 12.7, 12.0 s |
| English, 14 words | 3.6 s | 7.2, 6.6, 7.3 s | 33.8, 31.0, 31.5 s |
| Luganda, 5 words | 2.3 s | 5.6, 4.6, 5.0 s | 20.1, 19.9, 20.9 s |
| Luganda, 10 words | 3.8 s | 7.4, 7.2, 8.4 s | 33.4, 31.3, 25.3 s |

fp32 is about 2× real time against 2.6–3.7× for the Sunbird VITS voices; int8 is four to five times slower than fp32 and not recommended. Load time 3.0 s (fp32) and 3.8 s (int8). Mac: fp32 0.70 s (English) and 0.41 s (Luganda), int8 1.27 s and 1.71 s. Inside the second app (setup D, a hot CPU) the voice step's median was 7.1 s against 9.9 s for the Sunbird VITS voices.

The other ONNX repository, `jq/vits-tts-lug-eng-onnx`, holds an IPA-phoneme model that cannot be fed plain text; timings reported for it in earlier versions of this page were measured on output that was not speech and have been removed.

### Text generation (Gemma4-E2B, Q4_K_M)

| | Mac (M4, Metal) | Raspberry Pi 4 (CPU) |
|---|---|---|
| Prompt processing | 68.5 tok/s | 6.0 tok/s |
| Generation | 47.2 tok/s | 2.2–2.3 tok/s |
| End-to-end (load + 64 tokens) | — (not isolated) | 19.4s |

The Pi runs at roughly **9–20x slower** than the M4 depending on phase, consistent with the loss of GPU/Metal acceleration and the Cortex-A72's lack of the ARM dot-product/matmul-int8 extensions that help quantized inference on newer ARM cores (e.g. Cortex-A76 in the Pi 5).

On the rebuilt Pi (newer llama.cpp build), a 3-repetition `llama-bench` of Q4_K_M gave **5.99 ± 0.02 prompt and 2.45 ± 0.01 generation tokens/s**, slightly better than the single-run figures above. See the [quantization comparison](quantization-sweep.md#speed-and-memory-on-the-raspberry-pi-4) for the same measurement across five levels.

### Audio input (Gemma4-E2B + mmproj)

| | Mac (M4, Metal) | Raspberry Pi 4 (CPU) |
|---|---|---|
| Audio batch encoding | ~190 ms | ~1475–1595 ms |
| Full load (text model + mmproj) | — (not isolated) | ~2 min 14s (cold) |
| Peak resident RAM (text + mmproj + audio encode) | — (not measured) | **~2.6 GB** (sampled live via `free -m` during inference; idle baseline was ~0.75 GB) |

The ~2.6 GB peak is well under the earlier size-based estimate of ~4.5 GB in [Part 2](02-audio-input.md) — that estimate summed on-disk file sizes plus a flat overhead assumption, which overstated actual resident memory. The empirical number is the one to trust; it leaves roughly 5 GB of the Pi's 7.6 GB usable RAM free for a TTS process to run alongside in the same session.

**Two different memory measures appear in these docs.** The ~2.6 GB above is system memory *in use* (`free -m`), which excludes cached file pages. The [quantization comparison](quantization-sweep.md) reports the model process's *peak resident memory*, which includes the memory-mapped model file (3.47 GB for Q4_K_M text-only). Both are valid; they are not interchangeable.

### Text-to-speech inference (Raspberry Pi 4, CPU)

| Model | Language | Audio duration | Model load | Inference time | Real-time factor |
|---|---|---|---|---|---|
| Sunbird VITS | Luganda | 1.55s | 3.65s | 6.69s | 4.32x |
| Sunbird VITS | English | 2.37s | 6.21s | 7.03s | 2.97x |
| Sunbird VITS | Runyankole | 1.50s | 14.97s | 6.70s | 4.46x |
| MMS-TTS (Meta) | Luganda (run 1) | 1.63s | 7.68s† | 5.40s‡ | 3.31x |
| MMS-TTS (Meta) | Luganda (run 2) | 1.70s | 7.68s | 8.58s | 5.06x |
| MMS-TTS (Meta) | Luganda (run 3) | 1.68s | 3.89s | 5.04s | 3.00x |

† First measured run; ‡ this specific run also included the one-time Hub download, so its inference time is directly comparable across runs but its *load* time is not (download time is excluded from "load" above — see note below).

**Real-time factor** = inference time ÷ audio duration. A factor of 1.0x would mean generation keeps pace with playback; everything measured here is **3–4.5x slower than real-time**, meaning a short sentence takes several seconds to synthesize after the text is ready — workable for a "type/speak, wait, hear the answer" interaction, not for live streaming synthesis.

**Run-to-run variance is real and non-trivial**: MMS-TTS inference time ranged from 5.04s to 8.58s across three runs of the identical input on the same device with a warm model cache — a ~70% spread. Runyankole's VITS load time (14.97s) was more than double Luganda's and English's (~4–6s) despite a similarly-sized checkpoint, for reasons not yet diagnosed (possibly thermal throttling from back-to-back runs, or background system load — the Pi was not benchmarked in an isolated/idle state). **Treat all single-sample timings in this document as indicative of rough order of magnitude, not precise, reproducible figures** — a proper benchmark would run each configuration 10+ times and report mean ± standard deviation, which has not been done here.

### Model size and quantization

| Artifact | Size | Reduction vs. BF16 source |
|---|---|---|
| Original BF16 safetensors | 10.21 GB | — |
| GGUF, F16 | 9.27 GB | 9.2% smaller |
| GGUF, Q4_K_M (deployed) | **3.42 GB** | **66.5% smaller** (63.2% smaller than the F16 GGUF) |
| mmproj, F16 (audio+vision) | 0.99 GB | — (not quantized; F16 only, no Q4 mmproj path attempted) |

Internally, `llama-quantize` reported the per-tensor-weighted shift as **8828.84 MiB → 3242.78 MiB**, i.e. **16.00 bits/weight → 5.88 bits/weight** — close to but above the nominal "4-bit" the format is named for, because Q4_K_M keeps a subset of more sensitive tensors (e.g. `ffn_down`) at 6-bit precision rather than quantizing everything uniformly.

**Other quantization levels** were compared in a separate [quantization comparison](quantization-sweep.md): eight levels for translation quality (on a Mac) and five for Pi speed and memory. Q4_K_M, the level deployed here, held up best overall. Not yet covered: the IQ-series formats, quantizing the audio encoder, other languages, and speech transcription under quantization.

### TTS model size comparison

| | Sunbird VITS | MMS-TTS (Meta) |
|---|---|---|
| Size per language | ~450 MB (generator only; discriminator `.pth` excluded, training-only) | ~145 MB |
| Ratio | **3.1x larger** than MMS-TTS per language | — |
| Languages tested | Luganda, English, Runyankole | Luganda, English |

