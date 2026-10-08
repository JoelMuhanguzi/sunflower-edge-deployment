# Part 8: The Raspberry Pi 5, and the Pi 4 measured again under matching conditions

[← Back to the project README](../README.md)

Part 6 compared the two pipelines on the Pi 4. This part repeats the comparison on a **Raspberry Pi 5** and, on the same day with the same scripts, on the **Pi 4 again**, so the two boards can be read side by side. It also answers three questions earlier parts could only guess at: where NLLB's and Whisper's time goes, whether int8 is what makes them slow, and how fast Gemma generates on the newer board.

## The two boards and how they were run

| | Raspberry Pi 4 Model B Rev 1.5 | Raspberry Pi 5 Model B Rev 1.1 |
|---|---|---|
| CPU | 4 × Cortex-A72, 1.5 GHz | 4 × Cortex-A76, 2.4 GHz (has the int8 dot-product and fp16 instructions) |
| Memory | 8 GB | 8 GB |
| Cooling and power | heatsinks on the main chips and two fans; the soft temperature limit was reached during most runs (throttle flag `0x80000`) | active cooler; 5.1 V / 5 A supply; never throttled (flag `0x0`), CPU start temperatures 55–71 °C |
| OS | Debian 13 (trixie) | Debian 13 (trixie), Python 3.13 |
| llama.cpp | `a7b94df` | `a7b94df` (same commit) |

Both ran **headless** (the desktop session stopped) from a fresh boot, so the first run of each is a true cold start. The same scripts and the same six recorded clips as in [Part 6](06-fast-pipeline.md#head-to-head-on-the-pi-4-six-recorded-clips) were used (`scripts/fast/bench_compare.py`), with a wait for the CPU to fall below 60 °C before each setup; the clips themselves still start at 55–80 °C because there is no cooling between them. Raw results: `results/pi4-clean-2026-10-08/` and `results/pi5-clean-2026-10-08/`.

> **A first Pi 5 attempt was invalid and is kept only as a record.** It used a 5 V / 3 A adapter and no fan: `dmesg` logged "Undervoltage detected!", the CPU sat at 82–86 °C with frequency capping, and the Gemma numbers came out up to 1.5× slower (B 18.0 s against 12.3 s, E 17.3 s against 12.9 s; D 19.3 s against 17.5 s) (`results/pi5-2026-10-08/`, marked superseded). The Pi 5 wants 5 V / 5 A.

The setups are those of Part 6 plus two: **A** the first app as originally built (Gemma started per tap, Sunbird VITS voice per tap); **B** Gemma kept loaded in `llama-server` with the Sunbird VITS voices preloaded; **E** B with the [character-level ONNX voice](06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model); **C** Whisper int8 + NLLB int8 with the Sunbird VITS voices; **D** C with the ONNX voice. Beam size is 5 unless marked.

## Results: spoken round trip, medians of six clips, seconds

| Setup | | Listen | Translate | Voice | **Total** | Slowest | Start-up | Word error rate |
|---|---|---|---|---|---|---|---|---|
| **A** Gemma per tap | Pi 4 | 54.5 | 19.9 | 21.6 | **98.4** | 117.3 | none | 0.13 |
| | Pi 5 | 27.5 | 10.7 | 12.8 | **51.7** | 62.0 | none | 0.19 |
| **B** Gemma loaded, Sunbird VITS | Pi 4 | 43.8 | 13.0 | 7.5 | **66.1** | 74.0 | 46 s | 0.08 |
| | Pi 5 | 7.1 | 2.7 | 2.6 | **12.3** | 15.3 | 44 s | 0.29 |
| **E** Gemma loaded, ONNX voice | Pi 4 | 43.2 | 12.0 | 5.2 | **63.0** | 66.6 | 120 s | 0.09 |
| | Pi 5 | 7.4 | 3.0 | 1.7 | **12.9** | 13.3 | 68 s | 0.09 |
| **C** Whisper + NLLB, Sunbird VITS | Pi 4 | 19.3 | 10.9 | 8.5 | **40.0** | 43.3 | 114 s | 0.06 |
| | Pi 5 | 9.0 | 7.4 | 3.2 | **20.1** | 21.1 | 64 s | 0.06 |
| **D** Whisper + NLLB, ONNX voice | Pi 4 | 19.3 | 10.8 | 5.9 | **36.6** | 38.5 | 87 s | 0.06 |
| | Pi 5 | 8.8 | 6.6 | 1.8 | **17.5** | 18.4 | 46 s | 0.06 |
| **C**, beam 1 | Pi 4 | 19.3 | 8.8 | 7.9 | 37.2 | 40.0 | 67 s | 0.06 |
| | Pi 5 | 10.4 | 6.5 | 2.8 | 20.0 | 21.1 | 62 s | 0.06 |
| **D**, beam 1 | Pi 4 | 19.2 | 8.8 | 5.3 | 33.4 | 36.8 | 86 s | 0.06 |
| | Pi 5 | 9.4 | 6.0 | 2.0 | 17.4 | 18.8 | 41 s | 0.06 |

(`results/headtohead.csv` has these and every earlier run in one table, regenerated from the raw files by `scripts/fast/key_numbers.py`.)

**What the table says**
- **The Pi 5 is 2× to 5.4× faster, depending on the pipeline.** Gemma kept loaded gains most (B 66.1 s → 12.3 s, E 63.0 s → 12.9 s), the Whisper + NLLB pipelines about 2× (C 40.0 s → 20.1 s, D 36.6 s → 17.5 s), and the per-tap first app 1.9× (98.4 s → 51.7 s).
- **The ranking of the two pipelines flips between the boards.** On the Pi 4 Whisper + NLLB is faster than Gemma kept loaded (36.6 to 40.0 s against 63.0 to 66.1 s); on the Pi 5 Gemma kept loaded is faster (12.3 to 12.9 s against 17.5 to 20.1 s). Gemma's generation speed jumped 3.2× and its prompt processing 6.7× (below), while the CTranslate2 decoders in Whisper and NLLB gained about 2×.
- **Keeping models loaded matters more than anything else on the Pi 5:** A → B is 4.2× (51.7 s → 12.3 s); on the Pi 4 it is 1.5×.
- **The ONNX voice** takes 2.6 s off the Whisper + NLLB pipeline on the Pi 5 (D against C) and 3.4 s on the Pi 4, and makes no clear difference inside the Gemma pipeline on the Pi 5 (E 12.9 s against B 12.3 s).
- **Beam 1 helps the Pi 4 a little and the Pi 5 hardly at all.** Translation falls from 10.8–10.9 s to 8.8 s on the Pi 4 (about 19%; totals 40.0 → 37.2 s and 36.6 → 33.4 s) but only from 6.6–7.4 s to 6.0–6.5 s on the Pi 5 (totals unchanged, 17.5 → 17.4 s and 20.1 → 20.0 s). The six translations were the same at beams 1, 2 and 5 except one wording difference (below).
- **Accuracy.** Whisper's word error rate was 0.06 in every setup on both boards (deterministic). Gemma's varies between runs (0.08 to 0.29 for the same model and clips) because it samples at temperature 0.3; its translations are also less stable. One speaker and six clips, so read these as "comparable", not as a ranking.
- **Reading the transcripts and translations.** The author, a Luganda speaker and the speaker on the recordings, read the transcripts and translations of all six sentences from the Whisper + NLLB runs and judged them correct and sensible. This is one reader who is not independent of the recordings, so it is not a substitute for an independent evaluation.

## Where the time goes (Pi 5, measured with the diagnosis scripts in `scripts/fast/experiments/`)

**Gemma generation speed** (`llama-bench`, Q4_K_M, 4 threads, 64-token prompt, 32 generated, 3 repetitions):

| | Prompt processing | Generation |
|---|---|---|
| Pi 4 | 5.99 ± 0.02 tokens/s | 2.45 ± 0.01 tokens/s |
| Pi 5 | **40.17 ± 0.02** tokens/s | **7.80 ± 0.02** tokens/s |
| Ratio | 6.7× | 3.2× |

**Whisper int8, one clip, the app's settings** (`whisper_diagnose.py`; features, encoder and decoder timed separately):

| Window | Compute type | Total | Features | Encoder | Decoder | Per token |
|---|---|---|---|---|---|---|
| 6 s (the app) | int8 | 10.09 s | 0.01 s | 1.78 s (18%) | 8.30 s (82%) | 572 ms |
| 6 s | int8_float32 | 10.03 s | 0.01 s | 1.80 s | 8.22 s | 567 ms |
| 6 s | float32 | 37.84 s | 0.01 s | 5.86 s | 31.97 s | 2,205 ms |
| 30 s (stock) | int8 | 26.79 s | 0.00 s | 17.27 s (65%) | 9.52 s | 793 ms |

At the stock 30 s window the encoder is the larger part, as Part 6 inferred; at the 6 s window the app uses, the **decoder is 82%** of the time. Shortening the window removes encoder work but leaves the decoder.

**NLLB int8, where the time goes** (`nllb_diagnose.py`: decoder forced to produce exactly N tokens; the time for N tokens is fit as a fixed part plus a cost per token):

| Compute type | Beam | Fixed (encoder + set-up) | Per generated token | Six sentences, median |
|---|---|---|---|---|
| int8 | 5 | about 0 s | 800 ms | 11.9 s |
| int8 | 1 | 0.5 s | 503 ms | 7.3 s |
| int8_float32 | 5 | 0.1 s | 734 ms | 11.5 s |
| int8_float32 | 1 | 0.5 s | 495 ms | 7.2 s |
| float32 | 5 | 1.2 s | 1,713 ms | 27.6 s |
| float32 | 1 | 1.3 s | 1,586 ms | 23.6 s |

- **Nearly all of NLLB's time is the decoding loop,** about half a second per token with a 1.3 B-parameter model. The encoder is under 0.5 s.
- **int8 is not what makes it slow; float32 is 2 to 3× slower still** (and Whisper float32 is 3.8× slower than int8). Decoding is limited by how many bytes of weights are read per token, so fewer bits per weight help: this is consistent with Gemma's Q4_K_M being faster per token than either. (The ONNX voice is the opposite case, where int8 hurts; see below.)
- **Gemma generates a token in about 130 ms on the same board, 4× less than NLLB's 503 ms at beam 1.** We have not found out why CTranslate2's decoding is so much slower per token than llama.cpp's; this is an open question.
- **Unexplained:** the same `translate` call took 11.9 s per sentence at beam 5 in this isolated test (and in the throttled run the night before, identically) but only 6.6–7.4 s per sentence inside the full pipeline at beam 5. The test ran the beams interleaved. We have not established the difference.

**The ONNX voice, fp32 against int8** (`tts_char_onnx.py`, three runs per sentence, seconds):

| | Pi 4 fp32 | Pi 4 int8 | Pi 5 fp32 | Pi 5 int8 |
|---|---|---|---|---|
| Load | 5.4 | 4.5 | 2.3 | 2.0 |
| English, 5 words | 3.0–3.6 | 12.6–14.5 | 1.0–1.2 | 5.1–5.4 |
| English, 14 words | 7.2–10.0 | 30.0–32.5 | 2.6–2.7 | 12.3–13.3 |
| Luganda, 5 words | 4.4–5.0 | 17.7–19.4 | 1.4 | 7.6–8.4 |
| Luganda, 10 words | 6.4–8.0 | 28.3–35.4 | 2.1–2.2 | 11.4–13.2 |
| Time ÷ audio length | 1.7–2.5× | 7–9× | **0.64–0.74×** | 3.1–3.9× |

On the Pi 5 the voice is faster than real time at fp32. int8 stays 3 to 6 times slower than fp32 on both boards, even on the Cortex-A76 with the int8 dot-product instruction, so the Pi 4's missing instruction is not the explanation.

**Memory** (resident memory of the benchmark process and everything it started, including the memory-mapped model files it touched; median of the first scored clips, `results/pi5-clean-2026-10-08/pi5mem-*.jsonl`):

| Setup | Resident memory (Pi 5) |
|---|---|
| D Whisper + NLLB + ONNX voice | 3.7 GB |
| C Whisper + NLLB + Sunbird VITS | 4.4 GB |
| E Gemma loaded + ONNX voice | 6.1 GB |
| B Gemma loaded + Sunbird VITS | 6.6 GB |

The Gemma setups hold about 4.3 GB in the server plus the voices; the ONNX voice is about 0.6 GB lighter than the Sunbird VITS worker (which loads PyTorch). All fit in 8 GB; B leaves about 1.4 GB.

## The Pi 5's GPU

The VideoCore VII supports OpenGL ES and Vulkan, not CUDA, and as far as we know not OpenCL. ONNX Runtime and CTranslate2 (which Whisper and NLLB use) have no backend for it, so those models cannot use it. llama.cpp has a Vulkan backend that could in principle run Gemma; **we did not test it** and have no evidence it would beat the CPU (the GPU shares the same memory). The GPU drives the display only.

## Caveats

- One speaker, six clips, one run per setup, medians of six. The beam and window settings were chosen while looking at the same clips.
- Both boards still warmed during a run (Pi 5 start temperatures 55–71 °C, Pi 4 56–80 °C with the soft limit reached on most Pi 4 runs). Nothing throttled on the Pi 5.
- Memory is resident-set size, which counts touched file pages of the memory-mapped model.
- Not measured: independent human evaluation of translation and voice quality; more speakers; the Vulkan GPU path; why CTranslate2 decodes slower per token than llama.cpp.
