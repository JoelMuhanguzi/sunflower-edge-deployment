# Benchmarks

[← Back to the project README](../README.md)


All Pi numbers below were measured on the Raspberry Pi 4 Model B Rev 1.5 (8GB RAM, Cortex-A72 @ 1.5GHz, CPU-only, Debian 13/trixie) described in [Environment](01-text-model.md#environment). Mac numbers are from the Apple M4 host used for development, included only as a sanity baseline — they are not representative of edge-device performance and should not be read as "the model is this fast," only "the conversion/quantization didn't break anything."

Everything here is a single-sample measurement per configuration, not an average over many runs, except where noted (MMS-TTS). Edge-hardware benchmarks are noisy — see the variance note below — so treat single numbers as indicative, not precise.

### Text generation (Gemma4-E2B, Q4_K_M)

| | Mac (M4, Metal) | Raspberry Pi 4 (CPU) |
|---|---|---|
| Prompt processing | 68.5 tok/s | 6.0 tok/s |
| Generation | 47.2 tok/s | 2.2–2.3 tok/s |
| End-to-end (load + 64 tokens) | — (not isolated) | 19.4s |

The Pi runs at roughly **9–20x slower** than the M4 depending on phase, consistent with the loss of GPU/Metal acceleration and the Cortex-A72's lack of the ARM dot-product/matmul-int8 extensions that help quantized inference on newer ARM cores (e.g. Cortex-A76 in the Pi 5).

### Audio input (Gemma4-E2B + mmproj)

| | Mac (M4, Metal) | Raspberry Pi 4 (CPU) |
|---|---|---|
| Audio batch encoding | ~190 ms | ~1475–1595 ms |
| Full load (text model + mmproj) | — (not isolated) | ~2 min 14s (cold) |
| Peak resident RAM (text + mmproj + audio encode) | — (not measured) | **~2.6 GB** (sampled live via `free -m` during inference; idle baseline was ~0.75 GB) |

The ~2.6 GB peak is well under the earlier size-based estimate of ~4.5 GB in [Part 2](02-audio-input.md) — that estimate summed on-disk file sizes plus a flat overhead assumption, which overstated actual resident memory. The empirical number is the one to trust; it leaves roughly 5 GB of the Pi's 7.6 GB usable RAM free for a TTS process to run alongside in the same session.

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

**Not yet measured**: alternative quantization levels (`Q4_0`, `Q5_K_M`, `Q8_0`, `IQ2_XXS`, etc.) were not benchmarked against Q4_K_M for this model — Q4_K_M was chosen as llama.cpp's standard recommended default without a comparative sweep. A proper size-vs-quality-vs-speed table across quantization levels is flagged as follow-up work (see [Known limitations](limitations.md#known-limitations--follow-ups)) and would strengthen any claims about this being an "optimal" quantization choice rather than a reasonable default.

### TTS model size comparison

| | Sunbird VITS | MMS-TTS (Meta) |
|---|---|---|
| Size per language | ~450 MB (generator only; discriminator `.pth` excluded, training-only) | ~145 MB |
| Ratio | **3.1x larger** than MMS-TTS per language | — |
| Languages tested | Luganda, English, Runyankole | Luganda, English |

