# Alternatives considered and known limitations

[← Back to the project README](../README.md)

## Alternatives considered

Two other release artifacts were evaluated before settling on the manual-quantization path:

1. **`Sunbird/translate-nllb-3.3b-salt`** — a 3.3B NLLB-based translation model also published by Sunbird. Ruled out because NLLB is an encoder-decoder (seq2seq) architecture; llama.cpp's GGUF converter only supports decoder-only causal LM architectures. This model would require a different native runtime path (e.g. direct PyTorch/Transformers inference with dynamic int8 quantization), not GGUF/llama.cpp.
2. **`ak3ra/sunflower-cactus-int4`** — a pre-quantized (int4) 4.24 GB archive referencing the [Cactus](https://github.com/cactus-compute/cactus) edge-AI engine. Cactus is a legitimate edge-inference framework with its own model format and runtime, explicitly supporting Gemma-family models — but it primarily targets mobile/wearable platforms (iOS, Android), and Raspberry Pi/Linux support was not clearly documented. The repository also had no model card describing its contents. Given the lack of documentation and unconfirmed Pi compatibility, we chose the better-established llama.cpp path instead.

See also [Part 3's alternatives](03-text-to-speech.md#step-15--orpheus-3b-tts-ruled-out-before-attempting) (Orpheus-3B, Spark-TTS) for the TTS-specific evaluation.

## Known limitations / follow-ups

- Generation speed (~2.2–2.3 tok/s text, ~3–4.5x real-time for TTS) is workable for short exchanges but slow for long-form content.
- **No quantization-level comparison sweep.** Only Q4_K_M was benchmarked for the text model. Alternative levels (`Q4_0`, `Q5_K_M`, `Q8_0`, `IQ2_XXS`, etc.) would trade size/speed/quality differently and have not been measured — see [Benchmarks](benchmarks.md#model-size-and-quantization). This is the single highest-value follow-up for strengthening any "optimal quantization" claim in the paper.
- **Benchmarks reported here are single-sample, not averaged.** Measured run-to-run variance was substantial (MMS-TTS inference time varied ~70% across three identical runs; see [Benchmarks](benchmarks.md#text-to-speech-inference-raspberry-pi-4-cpu)). A rigorous benchmark would run each configuration 10+ times on an idle, thermally-stable device and report mean ± standard deviation.
- The mmproj multimodal projector was only benchmarked at F16 — no quantized (Q4/Q8) mmproj was built or tested, so its ~1GB size and RAM footprint may be reducible.
- The tokenizer config fix (Part 1, Step 5) and the `monotonic_align` import fix (Part 3, Step 17) were applied to local copies only; neither was upstreamed or reported to the respective maintainers.
- No persistent serving layer (e.g. `llama-server` with an OpenAI-compatible HTTP API) has been set up yet — current usage is via interactive CLI tools over SSH, or the menu-driven `sunflower_demo.py` (Part 4) launched locally on the Pi.
- Audio input quality was tested with clear, single-speaker, near-field USB mic recordings (both TTS-generated and live spoken) — not yet validated against background noise, multiple speakers, a lower-quality/distant mic, or real recorded speech in any of the 69 supported African languages beyond Luganda/English.
- The three Sunbird VITS languages not yet tested (Acholi, Ateso, Lug+Eng bilingual) remain to be verified using the same process documented in Step 17.
- TTS voice quality (Sunbird VITS vs. MMS-TTS) was judged subjectively by ear, not measured with any formal metric (e.g. MOS, PESQ) — worth doing properly for the paper.
- The USB microphone's capture gain (Part 4, Step 20) was set manually via `amixer` and does not persist across reboots; a portable/unattended deployment would need `alsactl store` or an equivalent startup hook.
- Offline operation: inference itself requires no network connectivity on either device. The one exception is MMS-TTS's first run, which downloads its checkpoint (~145MB) from the Hugging Face Hub via `transformers`' `from_pretrained`; it is cached locally afterward and all subsequent runs are fully offline. Sunbird's VITS and Gemma4-E2B checkpoints were downloaded once, manually, ahead of time, and never require network access thereafter.
- The touchscreen and its UI are built ([Part 5](05-touchscreen-device.md)), but battery power, a dedicated speaker and an enclosure are not — see [Part 4](04-live-audio-demo.md#looking-ahead-a-dedicated-handheld-device).
- **Touch accuracy was only partly verified.** The top-left corner was checked explicitly and ~40 logged taps looked consistent with the UI responding correctly; there was no formal four-corner test. The outer ~20–35 px of each edge do not register on this panel, and no scaling calibration was applied for that.
- **Transcription prompt not yet compared with the trained one.** The app uses its own transcription wording; the model card says the model was trained with `Please transcribe this <Language> audio.` and warns of poorer results otherwise. Not yet tested on real Luganda/Runyankole speech ([Part 2](02-audio-input.md#which-languages-does-audio-input-cover)).
- **Speech accuracy in Luganda and Runyankole is limited by the model itself** (about 50% word error in the model card's own evaluation of the full-precision weights); the effect of our 4-bit quantization on those figures is unmeasured.
- **Two-step Translate mode takes longer than a single call** (~30 s transcription plus ~18 s translation on a warm run, ~2 min cold; single measurements). Long sentences may be cut off in the results panel, which fits about two lines per text.
- **Single hardware fault, one data point.** The first touchscreen unit's touch stopped working; a second unit worked on the same configuration. We believe the first was faulty but did not diagnose it physically.
- The LiteRT-LM evaluation only tested text generation; the audio-encoder export path for this architecture is very new (merged upstream roughly a week before this evaluation) and was not attempted, given the text-only result was already not competitive with the deployed GGUF pipeline on size, reliability, or conversion time.

