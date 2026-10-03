# Part 2 — Audio input (speech understanding)

[← Back to the project README](../README.md)


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


## Which languages does audio input cover?

The model card ([Sunbird/Sunflower-Gemma4-E2B](https://huggingface.co/Sunbird/Sunflower-Gemma4-E2B)) says the model understands text and speech across **69 African languages**, with audio input limited to **30 seconds** of 16 kHz mono audio. The two claims differ in scope:

- **69 languages** are listed for "translation and speech evaluation" overall.
- **51 languages** have reported **speech transcription** results (110 dataset configurations from Sunbird's `Sunbird/speech` data, 5,201 scored examples). For the other ~18 there is no published speech score, so speech support there is unverified.

### How accurate is it?

Macro-averaged over the 51 languages: **word error rate (WER) 0.47**, **character error rate (CER) 0.19** (lower is better; WER can exceed 1.0 when the model inserts more words than the reference has). In plain terms, about half the words are wrong on average, though spellings are often close.

The languages offered in our touchscreen app, plus the range:

| Language | WER | CER | Scored examples |
|---|---:|---:|---:|
| French (best) | 0.10 | 0.05 | 50 |
| Afrikaans | 0.14 | 0.06 | 170 |
| English | 0.15 | 0.08 | 100 |
| Swahili | 0.16 | 0.06 | 99 |
| Acholi | 0.41 | 0.18 | 92 |
| Runyankole | 0.50 | 0.16 | 94 |
| Luganda | 0.50 | 0.17 | 266 |
| Kwamba (worst) | 1.23 | 0.74 | 49 |

Take-aways: high-resource languages (English, Swahili, Afrikaans, French) are quite usable; Luganda and Runyankole are roughly 50% word error, so a translation made from their transcript inherits those errors. The model card itself warns that "the low-resource end of the table is genuinely weak, and output there should be verified by a speaker before use."

### Caveats on these numbers

- They describe the **original full-precision model**. Nobody has measured how much our 4-bit (Q4_K_M) conversion changes them.
- The card describes the weights as a **preview** ("output will change before the stable release").
- **Prompt format matters.** The card says the model "was trained with prompts that explicitly name the spoken language" and gives the form `Please transcribe this Luganda audio.`, warning it "may perform poorly when the spoken language is omitted or named incorrectly." Our app currently uses its own, longer wording ("The speaker is speaking X. Transcribe exactly what they said, written in X. Respond with only the transcription, nothing else."). That wording worked for English, but we have **not** compared it against the trained phrasing on real Luganda or Runyankole speech. This is the highest-value follow-up for transcription accuracy.
