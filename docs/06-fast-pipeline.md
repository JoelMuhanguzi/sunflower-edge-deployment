# Part 6: A second pipeline, built from Sunbird's smaller specialist models

[← Back to the project README](../README.md)

The first pipeline ([Parts 1–5](01-text-model.md)) uses one model, Sunflower-Gemma4-E2B, for both listening and translating. After it worked, Sunbird's other published models suggested a second route: a Whisper-based speech recogniser and an NLLB translation model, each smaller than the Gemma checkpoint, which we could keep loaded in memory. This page records how that pipeline was built, how it was measured, and what is and is not established. **Both apps are kept** so the pipelines can be compared on the same device (Raspberry Pi 4, 8 GB, CPU only).

> **Status: preliminary.** The Pi 4 end-to-end figures come from a person speaking into the touchscreen app, a few runs on one sentence. They show that the second pipeline is faster, not by exactly how much. A controlled head-to-head (same recorded clips through both pipelines, no human in the loop) is still to be done; see [Not yet established](#not-yet-established).

## The two pipelines

| Stage | First app (`sunflower_touch_ui.py`) | Second app (`scripts/fast/sunflower_fast_ui.py`) |
|---|---|---|
| Speech → text | Gemma4-E2B Q4_K_M + audio encoder (`llama-mtmd-cli`) | Sunbird faster-whisper (Whisper large-v3 fine-tune for 51 African languages), **int8**, 5 s window |
| Translation | Gemma4-E2B Q4_K_M (`llama-cli`) | Sunbird NLLB-1.3B (SALT), **int8** (CTranslate2) |
| Speech out | Sunbird VITS / Meta MMS, started per tap | Same voices; English and Luganda **kept loaded** in a background worker |
| Model loading | Every tap (new process each time) | Once, at start-up (about 70 s) |
| Size on disk (listen + translate) | 3.42 GB + 0.99 GB audio encoder = **4.41 GB** | 1.56 GB + 1.38 GB = **2.94 GB** |
| Languages | English, Luganda, Runyankole, Swahili, Acholi | Same for transcription. **Translation excludes Swahili** (NLLB-SALT covers eng, ach, lgg, lug, nyn, teo) |

Speech output was deliberately left unchanged so the comparison isolates the listening and translating stages. The small ONNX voice (`jq/vits-tts-lug-eng-onnx`) was tried and **not adopted**: its repository has no model card, the tokens are an IPA phoneme set, `espeak-ng` has no Luganda, and `sherpa-onnx` refused it ([experiments](#the-small-onnx-voice-not-adopted)).

## How the models were made

Neither model could be used as published on the Pi.

- **NLLB.** Sunbird's published 8-bit file (`translate-nllb-1.3b-salt-8bit`) uses bitsandbytes, which needs a GPU. We downloaded the full-precision `Sunbird/translate-nllb-1.3b-salt` (5.1 GB) and converted it on the Mac with `ct2-transformers-converter --quantization int8`, copying the tokenizer files. Result: 1.38 GB. The converter needed `transformers==4.56.2` plus `protobuf`; transformers 5.x could not load the tokenizer and an older 4.4x lacked an argument the converter passes.
- **Whisper.** The faster-whisper repository stores float16 (3.09 GB) and quantizes on the fly when loaded, which on the Pi 4 took **82 s** to load and spiked memory. We instead converted the original `Sunbird/SunflowerASR-51-african-languages` (6.17 GB, float32) to a saved int8 model (**1.56 GB**) with the same converter (`--low_cpu_mem_usage`). Two fixes were needed: rename `extra_special_tokens` to `additional_special_tokens` in the original's `tokenizer_config.json` (it was written by a newer transformers), and copy `language_map.json`, `preprocessor_config.json`, `vocab.json` and `merges.txt` from the faster-whisper repository. Transcripts matched the on-the-fly version on four clips, apart from a comma. Pi 4 load time fell from 82 s to **2.8 s**.
- **Language tokens.** The Whisper model reuses Whisper's unused language tokens for African languages (`lug`→`sd`, `nyn`→`si`, `ach`→`su`). The model card warns against automatic language detection, so the app always passes the token from `language_map.json`.
- **The 5 s window.** Whisper pads every clip to 30 s before its encoder, so time barely depends on clip length (Mac: 1.3 s clip 6.5 s, 4 s clip 6.4 s). The apps record exactly 5 s, so the encoder window is shortened to 5 s by replacing faster-whisper's `pad_or_trim` (`WINDOW_SECONDS` in `stt_fw.py`; `FAST_WINDOW` in the app). **Accuracy risk:** on a 4 s clip the Mac transcript changed from "How are you today?" to "Hawaii today" at every shorter window, including 15 s. It has not been tested on enough real speech.

## Measurements

All times are seconds. "Cold/warm" are given only where measured; Whisper and NLLB showed no warm-up benefit. Single-run figures carry the usual caveat: the Pi 4 reached its temperature limit (84–85 °C) in every Whisper run, so these include thermal throttling.

### Speech to text: Whisper int8, English clip of 1.1 s

| | Mac | Pi 4 | Orange Pi Zero 2W (4 GB) |
|---|---|---|---|
| Model load | 0.4 | **2.8** (was 82 with on-the-fly quantization) | 53 |
| Transcription, stock 30 s window | 5.8 | 66–72 | 149–151 |
| Transcription, **5 s window** | 1.0 | **13.9–14.1** | **24.3–24.5** |
| Peak memory in use (30 s / 5 s) | – | 2.61 / 2.15 GB | 2.88 / 2.33 GB (swap 204 / 257 MB) |
| Max temperature (30 s / 5 s) | – | 85 / 84 °C | 86 / 84 °C |

Inside the app, with real 5 s recordings, Listen took **20–29 s** on the Pi 4. Longer, unfamiliar sentences cost more because the decoder writes tokens one at a time (Luganda needs more tokens per sentence), which the window cut does not touch.

### Translation, English → Luganda

| | Mac | Pi 4 | Orange Pi |
|---|---|---|---|
| NLLB int8 load | 0.8 | 39 | 21–43 |
| NLLB int8, per sentence (beam 5) | 0.4–0.6 | 5.7–12.7 (mean about 9); 5.8–6.0 for a 4-word sentence; 14.5–14.7 in the app for a longer (about 13-word) sentence | 9.2–19.5 |
| Gemma Q4_K_M via a persistent `llama-server`, per sentence | – | 7.6–11.0 | – |
| Gemma Q4_K_M, new process per call (first app) | – | about 18 (warm, README figure) | – |

NLLB and Gemma gave word-for-word identical Luganda on three of four test sentences; on the fourth they differed ("Nsaba mundeetere egiraasi y'amazzi" from NLLB, a garbled-looking rendering from Gemma). This was not judged by a Luganda speaker. On the Orange Pi the clock dropped from 1512 to 1200 MHz during the run (temperature 71 → 85 °C).

### Speech output: Sunbird VITS, Luganda, 9-word sentence on the Pi 4

| Path | Voice time | Detail |
|---|---|---|
| New process per tap (standalone) | 31.3 | load 3.9 + synthesis 21.5 (4.3 s of audio) + about 6 s starting Python and torch |
| New process per tap (inside the app) | **53.3** | not reproduced standalone; the extra time is unexplained (heat, memory, or cold libraries are candidates) |
| Preloaded worker (standalone) | 14.3 and 19.6 | 3.9 s and 5.0 s of audio; VITS output length varies run to run |
| Preloaded worker (inside the app) | **15.4** | 4.05 s of audio, about 3.8× real time |

Loading the English and Luganda voices into the worker takes about 26 s once at start-up. The synthesis cost (about 3.7–5× the audio length) is unchanged; the worker removes the per-tap load and process start-up (about 10 s).

### Whole run on the Pi 4 (fast app, Translate English → Luganda)

The same spoken phrase ("I am very sick and not feeling well, take me to the hospital") twice:

| | Listen | Translate | Voice | **Total** |
|---|---|---|---|---|
| Voice loaded per tap | 20.9 | 14.5 | 53.3 | **88.7** |
| Voice preloaded | 20.3 | 14.7 | 15.4 | **50.3** |

The two transcripts were close but not identical ("…very sick and am not feeling well…" against "…very sick and not feeling well…"); both translations began "Ndi mulwadde nnyo era siwulira bulungi ntwale…".

### Comparison with the first pipeline

The first app has no timing display, so **no same-sentence, same-session run of it exists**. The estimate below uses its earlier benchmarks and should be replaced by a measured head-to-head:

| | First app (earlier benchmarks) | Second app (measured above) |
|---|---|---|
| Speech → text | about 30 s warm, about 2 min cold | 20–29 s in the app (12.6–14 s for a short clip) |
| Translation | about 18 s warm | 6–15 s, depending on sentence length |
| Voice | about 30 s with a per-tap load (estimate for the same sentence) | about 15 s, preloaded |
| Start-up | none, but every tap loads | about 70 s, then resident |
| Memory | one model process at a time | 3.9 GB in use with Whisper and NLLB loaded; 4.6 GB in use after a run with the voice worker (single readings of `free`) |
| **Estimated whole run** | **roughly 75–100 s** (estimate, not measured) | **50.3 s** (measured, one run) |

### Tokens per second

The second app now records `tokens.listen_tokens` (Whisper decoder tokens, including timestamp tokens) and `tokens.translate_tokens` (NLLB tokens) with each run in `~/ml/demo_scratch/fast_pipeline_log.jsonl`, next to the stage times, so tokens per second can be computed per run. **No such figures exist yet**; they appear after the next runs. For the first pipeline, `llama-bench` (3 repetitions, Q4_K_M) measured **5.99 ± 0.02 prompt and 2.45 ± 0.01 generated tokens/s**. Whisper's time includes its fixed encoder cost, so tokens/s for it is a per-run figure, not a model constant.

### The persistent Gemma server (a side test)

To check whether loading Gemma once would help the first pipeline, `llama-server` ran Q4_K_M with its audio encoder on the Pi 4: ready in 19 s, 4.27 GB resident. Translation fell to 7.6–11 s per sentence (from about 18 s). Audio transcription of a new clip took 17.8–37.2 s on the first request (an identical repeat took 2.8–4.5 s, **because of caching**, so those are not honest figures). Loading once helps translation a lot and audio little.

## The small ONNX voice (not adopted)

`jq/vits-tts-lug-eng-onnx` (114 MB, or 38 MB at int8) loads and runs under `onnxruntime`: about 2 s per sentence on the Pi 4 at full precision, and **int8 was slower** than full precision on both boards (Pi 4 7.6–9.9 s; Orange Pi 18–29 s against 3–7 s). But the output was never confirmed to be speech: the model expects phonemes, and without them our character input produced clips of about a second for two sentences. `sherpa-onnx` refused it twice (plain mode: not a character model, a lexicon is needed; with eSpeak data: a duplicate apostrophe token at ids 157 and 159). The question of how it should be fed is open.

## The Orange Pi Zero 2W

Four Cortex-A53 cores at 1.5 GHz and 3.9 GB of memory, no passwordless `sudo`. Whisper int8 and NLLB int8 both ran (see the tables); Gemma Q4_K_M (3.42 GB) plus its audio encoder (0.99 GB) does not fit, and the smaller quantizations that do fit are too lossy (Q3_K_M −4.3 chrF, Q2_K −25.5; see [quantization-sweep.md](quantization-sweep.md)). It runs about 1.7–2.2× slower than the Pi 4 on Whisper and throttles quickly (85 °C, 1200 MHz). Its Gemma llama.cpp build was not attempted.

## Not yet established

- A **controlled head-to-head** of both pipelines on identical recorded clips with all stages timed the same way.
- **Accuracy** of the 5 s window and of the int8 Whisper on real English and Luganda speech (word error rate against references), and translation quality judged by Luganda speakers.
- How the NLLB beam size (5 now) trades time for quality.
- The effect of **cooling**: every Pi 4 run hit its temperature limit.
- Runs on the **Pi 5**, and whether the ONNX voice can be used at all.
- Whisper encoder and decoder time measured **separately** (the encoder is only inferred to dominate at the stock window).

## Reproduce

`scripts/fast/` holds the app (`sunflower_fast_ui.py`), the translator (`nllb_translate.py`), the voice worker (`vits_worker.py`), the Whisper script (`stt_fw.py`) and a launcher template. Failed or side experiments are in `scripts/fast/experiments/`. The app expects `~/ml/pipeline2/` with `venv/` (faster-whisper, ctranslate2, transformers, sentencepiece, protobuf; `av<17` on the Pi because faster-whisper 1.2.1 fails with newer PyAV), `models/whisper-51-int8/` and `models/nllb-salt-ct2-int8/`. Models are never included. Verify copies with SHA-256: one 1.4 GB file was silently damaged on the same USB drive that damaged files before ([rebuild-from-scratch.md](rebuild-from-scratch.md)).
