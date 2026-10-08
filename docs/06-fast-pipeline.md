# Part 6: A second pipeline, built from Sunbird's smaller specialist models

[← Back to the project README](../README.md)

The first pipeline ([Parts 1–5](01-text-model.md)) uses one model, Sunflower-Gemma4-E2B, for both listening and translating. After it worked, Sunbird's other published models suggested a second route: a Whisper-based speech recogniser and an NLLB translation model, each smaller than the Gemma checkpoint, which we could keep loaded in memory. This page records how that pipeline was built, how it was measured, and what is and is not established. **Both apps are kept** so the pipelines can be compared on the same device (Raspberry Pi 4, 8 GB, CPU only).

> **Status.** The second pipeline was timed against the first on the Pi 4 with the same six recorded clips and a controlled setup ([head-to-head](#head-to-head-on-the-pi-4-six-recorded-clips)): about **three times faster** for a spoken round trip, with comparable accuracy on those clips. The sample is small (one speaker, six clips), so read the accuracy as "not obviously worse", not as a measured ranking; see [Not yet established](#not-yet-established).

## The two pipelines

| Stage | First app as benchmarked (setup A, frozen in `scripts/baseline/`) | Second app (`scripts/fast/sunflower_fast_ui.py`) |
|---|---|---|
| Speech → text | Gemma4-E2B Q4_K_M + audio encoder (`llama-mtmd-cli`) | Sunbird faster-whisper (Whisper large-v3 fine-tune for 51 African languages), **int8**, 6 s window, no timestamp tokens, repeat guards |
| Translation | Gemma4-E2B Q4_K_M (`llama-cli`) | Sunbird NLLB-1.3B (SALT), **int8** (CTranslate2) |
| Speech out | Sunbird VITS / Meta MMS, started per tap | Same voices; English and Luganda **kept loaded** in a background worker |
| Model loading | Every tap (new process each time) | Once, at start-up (about 70 s) |
| Size on disk (listen + translate) | 3.42 GB + 0.99 GB audio encoder = **4.41 GB** | 1.56 GB + 1.38 GB = **2.94 GB** |
| Languages | English, Luganda, Runyankole, Swahili, Acholi | Same for transcription. **Translation excludes Swahili** (NLLB-SALT covers eng, ach, lgg, lug, nyn, teo) |

**The first app has since been changed** to keep Gemma, its audio encoder and the English and Luganda voices loaded (a local `llama-server` plus the shared voice worker), which is setup B below made permanent. A headless run of the changed app over three of the recorded clips took 61–71 s per round trip, matching setup B (median 65.1 s), after a cold start-up of 111 s for Gemma and 26 s for the voices. The frozen per-tap version remains the baseline (setup A).

Speech output was deliberately left unchanged so the comparison isolates the listening and translating stages. A smaller, faster voice was tested afterwards and added to the second app for English and Luganda: a character-level ONNX model, one 109 MB file for both languages ([below](#a-faster-voice-a-character-level-onnx-model)). An earlier attempt with a different ONNX model had failed because it was the wrong model for text input; that is explained in the same section.

## How the models were made

Neither model could be used as published on the Pi.

- **NLLB.** Sunbird's published 8-bit file (`translate-nllb-1.3b-salt-8bit`) uses bitsandbytes, which needs a GPU. We downloaded the full-precision `Sunbird/translate-nllb-1.3b-salt` (5.1 GB) and converted it on the Mac with `ct2-transformers-converter --quantization int8`, copying the tokenizer files. Result: 1.38 GB. The converter needed `transformers==4.56.2` plus `protobuf`; transformers 5.x could not load the tokenizer and an older 4.4x lacked an argument the converter passes.
- **Whisper.** The faster-whisper repository stores float16 (3.09 GB) and quantizes on the fly when loaded, which on the Pi 4 took **82 s** to load and spiked memory. We instead converted the original `Sunbird/SunflowerASR-51-african-languages` (6.17 GB, float32) to a saved int8 model (**1.56 GB**) with the same converter (`--low_cpu_mem_usage`). Two fixes were needed: rename `extra_special_tokens` to `additional_special_tokens` in the original's `tokenizer_config.json` (it was written by a newer transformers), and copy `language_map.json`, `preprocessor_config.json`, `vocab.json` and `merges.txt` from the faster-whisper repository. Transcripts matched the on-the-fly version on four clips, apart from a comma. Pi 4 load time fell from 82 s to **2.8 s**.
- **Language tokens.** The Whisper model reuses Whisper's unused language tokens for African languages (`lug`→`sd`, `nyn`→`si`, `ach`→`su`). The model card warns against automatic language detection, so the app always passes the token from `language_map.json`.
- **The encoder window.** Whisper pads every clip to 30 s before its encoder, so time barely depends on clip length (Mac: 1.3 s clip 6.5 s, 4 s clip 6.4 s). The apps record 5 s, so the window is shortened by replacing faster-whisper's `pad_or_trim` (`FAST_WINDOW` in the app, `WINDOW_SECONDS` in `stt_fw.py`). **A window exactly as long as the recording is a trap:** Whisper often repeated the sentence (3 of 6 clips), and one run looped for 300 s. A sweep over the six recorded clips (Mac, int8, `scripts/fast/experiments/stt_sweep.py`) picked a 6 s window with no timestamp tokens and no repeated 3-grams:

  | Window | Setting | Mean word error rate | Mean time per clip | Clips with repeated text |
  |---|---|---|---|---|
  | 5 s | default | 1.03 | 4.0 s | 3 of 6 |
  | 5 s | no timestamps + no repeat | 0.53 | 1.6 s | 3 of 6 |
  | 6 s | default | 0.42 | 1.6 s | 1 of 6 |
  | **6 s** | **no timestamps + no repeat** | **0.23** | **1.6 s** | 1 of 6 |
  | 8 s | no timestamps | 0.26 | 2.0 s | 1 of 6 |
  | 30 s (stock) | any of five | 0.18–0.23 | 7.7–8.1 s | 0 of 6 |

  Two guards sit on top: a cap of 64 new tokens, and removal of a repeated sentence (`collapse_repeats` in the app; only sentences of three or more words are compared). Separately, on a 4 s clip a short window changed "How are you" to "Hawaii" at every window up to 15 s; that effect has not been measured on a larger set.

## Measurements

All times are seconds. "Cold/warm" are given only where measured; Whisper and NLLB showed no warm-up benefit. Single-run figures carry the usual caveat: the Pi 4 reached its temperature limit (84–85 °C) in every Whisper run, so these include thermal throttling, even though the board has heatsinks and two fans (normal for sustained four-core load).

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

### Head-to-head on the Pi 4, six recorded clips

**Method.** Six 5-second recordings of one speaker (three English, three Luganda; `scripts/fast/record_set.py`) went through each setup on the same Pi 4: English clips were translated into Luganda, Luganda clips into English, and the translation was spoken. `scripts/fast/bench_compare.py` timed the three stages identically in every setup, one setup per process so memory was freed between them, waiting up to 15 minutes for the CPU to fall below 60 °C before each (per-clip start temperatures were 48–79 °C for setups A to C-final, so some runs were warm) and an unscored warm-up run for the setups that stay loaded. Setups:

- **A** the first app as it was before it was changed to keep models loaded (a new process per call, voice loaded per tap; frozen in `scripts/baseline/sunflower_touch_ui_per_tap.py`, git tag `v1-first-app-per-tap`);
- **B** the first app's model kept loaded in `llama-server` (this is what the first app now does) (prompt cache **off**, so repeated inputs are not flattered) with the English and Luganda voices preloaded;
- **C-5s** the second app with a 5 s window; **C-tuned** with a 6 s window, no timestamp tokens and no repeated 3-grams passed in from outside; **C-final** the finished app with those as defaults plus the two guards;
- **D** C-final with the character-level ONNX voice for English and Luganda, run a day later (see [the voice section](#a-faster-voice-a-character-level-onnx-model)). **Its CPU was hotter than C-final's** (clips started at 73–84 °C against 58–70 °C), so compare its voice column, not its total.

**Median per clip, seconds** (raw runs in `results/pi4-headtohead-2026-10-06/`):

| Setup | Listen | Translate | Voice | **Total** | Slowest run | Start-up | Word error rate | Clips with repeats | Max CPU temp |
|---|---|---|---|---|---|---|---|---|---|
| **A** first app | 78.8 | 21.3 | 24.1 | **121.6** | 200.5 (cold first call) | none | 0.09 | 0 of 6 | 78 °C |
| **B** Gemma kept loaded | 43.6 | 12.4 | 9.6 | **65.1** | 70.8 | 56 s | 0.09 | 0 of 6 | 79 °C |
| **C-5s** second app, 5 s window | 27.0 | 15.0 | 13.5 | 56.2 | **335.7** (a 299 s loop) | 96 s | 1.03 | 3 of 6 | 72 °C |
| **C-tuned** 6 s window + options | 19.8 | 11.6 | 10.1 | 42.6 | 45.2 | 107 s | 0.23 | 1 of 6 | 72 °C |
| **C-final** finished app | **19.3** | **10.7** | **9.9** | **40.8** | 43.5 | 102 s | **0.06** | 0 of 6 | 72 °C |
| **D** C-final + ONNX voice (hot CPU, see note) | 20.6 | 11.7 | **7.1** | 40.8 | 47.4 | 83 s | not scored | not checked | 84 °C |

Without its cold first call, A's median total is 121.3 s (listen 78.4, translate 22.3, voice 20.5).

- **Speed.** C-final is about **3.0× faster than A** (121.6 s to 40.8 s) and **1.6× faster than B** (65.1 s to 40.8 s). Keeping Gemma loaded alone (A to B) nearly halves the time, so part of the second app's gain comes from keeping models resident and part from the smaller models: the model choice is worth the 65 s to 41 s step.
- **Cost.** The second app needs about 100 s to start and keeps its models in memory (3.9 GB in use with Whisper and NLLB loaded, 4.6 GB after a run with the voices). The first app starts instantly but pays its loads on every tap.
- **Accuracy.** English was transcribed exactly by A, B and C-final. In Luganda, A and B scored 0.40, 0.17 and 0.00 on the three clips and C-final 0.20, 0.17 and 0.00. Word error rate is computed against the sentences the speaker was asked to read, ignoring punctuation and case; Luganda spelling variants count as errors. **This is six clips from one speaker, and the repeat guard was designed while looking at the same clips, so C-final's score is optimistic.** Translation quality was not scored.
- **Repeats.** In C-5s the repeated sentence was also translated and spoken twice, so a repeat cost time in all three stages; the 299 s run is one clip looping until the length limit.
- **Temperature.** Every setup ran sustained on four cores; A and B reached 78–79 °C with the soft temperature limit active at times, C about 72 °C.

**Hand-driven runs in the touchscreen app** (a person speaking, one sentence) before the benchmark: voice loaded per tap, listen 20.9 + translate 14.5 + voice 53.3 = **88.7 s**; with the voice worker, 20.3 + 14.7 + 15.4 = **50.3 s**. The 53 s voice step was not reproduced when the same sentence was synthesized outside the app (31 s); the extra time is unexplained.

### Tokens per second

Generation speed of the first pipeline's model was **2.40–2.47 tokens/s** in setups A and B (taken from llama.cpp's own timing output), matching the 2.45 from `llama-bench`; keeping it loaded does not change generation speed, it removes load and prompt costs. (Setup A's listening step does not report its figures in a form the script parses, so that stage has none.) For the second pipeline the benchmark records the tokens each stage produced; dividing by the stage's whole time gives **about 0.7 tokens/s for Whisper** (median 14 tokens in 19.3 s) and **about 1.0 for NLLB** (12 tokens in 10.7 s). Those are end-to-end stage rates including the encoder, the beam search and tokenisation, not decoding speeds, so they are not comparable with Gemma's 2.4; the comparable quantity is seconds per stage (the table above). The voice runs at about **2.6–3.7× the audio length** (6.1–11.8 s for 2.0–4.4 s of speech).

### The persistent Gemma server (a side test)

To check whether loading Gemma once would help the first pipeline, `llama-server` ran Q4_K_M with its audio encoder on the Pi 4: ready in 19 s, 4.27 GB resident. Translation fell to 7.6–11 s per sentence (from about 18 s). Audio transcription of a new clip took 17.8–37.2 s on the first request (an identical repeat took 2.8–4.5 s, **because of caching**, so those are not honest figures). Loading once helps translation a lot and audio little.

## A faster voice: a character-level ONNX model

**Result.** The ONNX voice `jq/sherpa-vits-tts-lug-eng` speaks English and Luganda from one 109 MB file and synthesizes at about **2× the audio length on the Pi 4**, against 2.6–3.7× for the Sunbird VITS voices (about 450 MB per language). In the six-clip benchmark the voice step's median fell from 9.9 s to 7.1 s (setup D above). The second app now uses it for English and Luganda; `FAST_VOICE=vits` switches back, and Runyankole still uses its Sunbird VITS voice because this model does not speak it.

**The model.** `vits-lug-eng.fp32.onnx` from [`jq/sherpa-vits-tts-lug-eng`](https://huggingface.co/jq/sherpa-vits-tts-lug-eng), 114,026,066 bytes, SHA-256 `45083e0c27d3f857fc3c5342b06aef1a088eadf8ba414a9f84358facd11bf5cc`, plus its 39-symbol `tokens.txt` (lowercase letters, space and a few symbols). It is the model the [Sunbird tutor app](https://github.com/SunbirdAI/sunflower-app) downloads. Its metadata reads `model_type: vits`, `frontend: characters`, `add_blank: 0`, one speaker. The metadata gives a sample rate of 22500 but the app, and we, play it at 22050 Hz and Whisper transcribed the result correctly.

**How to feed it** (from `lib/tts/tts_engine.dart` in the tutor app, implemented as `CharVoice` in `scripts/fast/sunflower_fast_ui.py`): lowercase the text, map each character to its id in `tokens.txt`, skip unknown characters, no blank tokens between them, and pass `input` (ids), `input_lengths` and `scales = [0.667, 1.0, 0.8]`. Text is split at punctuation and into chunks of 10 words, as in the tutor app.

**Pi 4 timings** (`scripts/fast/experiments/tts_char_onnx.py`, 4 threads, three runs per sentence, CPU below 60 °C at the start; raw output in `results/voice-onnx-2026-10-07/`):

| Sentence | Audio | fp32 synthesis time | int8 synthesis time |
|---|---|---|---|
| English, 5 words | 1.4 s | 3.2, 3.0, 2.8 s | 14.9, 12.7, 12.0 s |
| English, 14 words | 3.6 s | 7.2, 6.6, 7.3 s | 33.8, 31.0, 31.5 s |
| Luganda, 5 words | 2.3 s | 5.6, 4.6, 5.0 s | 20.1, 19.9, 20.9 s |
| Luganda, 10 words | 3.8 s | 7.4, 7.2, 8.4 s | 33.4, 31.3, 25.3 s |

Load takes 3.0 s (fp32) and 3.8 s (int8). An earlier fp32 run gave the same figures within noise. On the Mac the fp32 model took 0.70 s (English) and 0.41 s (Luganda) and int8 1.27 s and 1.71 s.

**int8 is not worth it.** We quantized the model ourselves with ONNX Runtime dynamic quantization (38 MB). It was four to five times *slower* than fp32 on the Pi 4 and also slower on the Mac. The reason is not established. A first guess was that the Cortex-A72 lacks int8 dot-product instructions, but a preliminary run on a Raspberry Pi 5 (Cortex-A76, which has them) was also 3–6× slower with int8, so that does not explain it; more likely ONNX Runtime's dynamically quantized kernels, or the quantize and dequantize steps around the convolutions, cost more than they save for this model. The int8 file saves 76 MB of disk and nothing else.

**Is the output speech?** On the Mac, the Sunbird Whisper transcribed the fp32 output of "hello, how are you today?" as "Hello how are you today." and of "oli otya, nsanyuse okukulaba leero" as "Oli otyaasanyisa okukulaba leero?"; the int8 output gave "Hello, how are you today?" and "Oli wetyaamuzanisa okukulaba leero?". English is right; the Luganda is close but not exact in both. **Nobody has listened to the voice for quality, and no Luganda speaker has judged it.** Memory use of the voice was not measured separately (the second app holds about 3.4 GB resident with Whisper, NLLB and this voice loaded).

### The earlier attempt: a different model that expects phonemes

We first tested `jq/vits-tts-lug-eng-onnx` and concluded the ONNX voice could not be used. That was **the wrong model**, and the conclusion was ours, not the model's. The two repositories look alike but differ in the input vocabulary:

| | `jq/sherpa-vits-tts-lug-eng` (works with text) | `jq/vits-tts-lug-eng-onnx` (expects phonemes) |
|---|---|---|
| Files | `vits-lug-eng.fp32.onnx` (114,026,066 bytes) | `vits-lug-eng.onnx` (114,325,834 bytes), `vits-lug-eng.int8.onnx` (38,314,074 bytes) |
| Graph | VITS, 6,476 nodes, input dimensions `N`, `L` | VITS, 6,474 nodes, input dimensions `batch_size`, `phonemes` |
| Vocabulary | 39 tokens: lowercase letters, space, symbols | 162 tokens: IPA symbols, stress marks, letters and `<blk>` |
| Metadata | `frontend: characters` | no `frontend` key |
| Uploaded | 13 May 2026 | 15 May 2026, after several re-uploads |

Fed lowercase characters, with or without blank tokens, the phoneme model produced clips of about one to two seconds that Whisper transcribed as nonsense ("vac.", "11."). `sherpa-onnx` refused it twice (plain mode: not a character model; with eSpeak data: a duplicate apostrophe token at ids 157 and 159). We did not find a Luganda voice in `espeak-ng`'s voice list on the Pi (the check was a search of that list for a few names). Whether the phoneme model is a newer retrain meant to replace the character one, and which phonemizer it expects, is **unanswered**: neither repository has a model card.

The timings quoted for that model in earlier versions of these docs (about 2 s per sentence at full precision, int8 slower) were measured on clips that were not speech and should not be used.

## The Orange Pi Zero 2W

Four Cortex-A53 cores at 1.5 GHz and 3.9 GB of memory, no passwordless `sudo`. Whisper int8 and NLLB int8 both ran (see the tables); Gemma Q4_K_M (3.42 GB) plus its audio encoder (0.99 GB) does not fit, and the smaller quantizations that do fit are too lossy (Q3_K_M −4.3 chrF, Q2_K −25.5; see [quantization-sweep.md](quantization-sweep.md)). It runs about 1.7–2.2× slower than the Pi 4 on Whisper and throttles quickly (85 °C, 1200 MHz). Its Gemma llama.cpp build was not attempted.

## Not yet established

- **More speakers and more clips.** Six clips from one speaker, with reference sentences we chose, show the setups are comparable, not which is more accurate. The repeat guard and window were tuned on these clips.
- **Translation quality** judged independently (the author, a Luganda speaker and the speaker on the recordings, judged the six sentences' transcripts and translations correct; no one else has); NLLB's beam size (5 now) against time.
- **Whisper encoder and decoder time** are now measured on the Pi 5 ([Part 8](08-pi5-and-matched-rerun.md)): the encoder is 65% of the time at the stock 30 s window and 18% at the 6 s window the app uses.
- **Pi 5 runs** are in [Part 8](08-pi5-and-matched-rerun.md).
- **The ONNX voice's sound quality** (judged by a Luganda speaker) and its memory use; a clean benchmark of setup D with the CPU cooled between clips (the first run was hot); whether the phoneme-based ONNX model can be fed with a phonemizer.
- ~~int8 against float32 for Whisper and NLLB~~: measured on the Pi 5 in [Part 8](08-pi5-and-matched-rerun.md); float32 is 2–4× slower than int8 for both.
- **The unexplained 53 s voice step** inside the app (31 s standalone).
- Cooling is not an open question: the Pi 4 has heatsinks and two fans and still reached 78–79 °C under the first app's sustained load.

## Reproduce

`scripts/fast/` holds the app (`sunflower_fast_ui.py`), the translator (`nllb_translate.py`), the Whisper script (`stt_fw.py`) and a launcher template. Failed or side experiments are in `scripts/fast/experiments/`. The voice worker, `scripts/vits_worker.py`, is shared with the first app. The app expects `~/ml/pipeline2/` with `venv/` (faster-whisper, ctranslate2, transformers, sentencepiece, protobuf; `av<17` on the Pi because faster-whisper 1.2.1 fails with newer PyAV), `models/whisper-51-int8/` and `models/nllb-salt-ct2-int8/`. Models are never included. Verify copies with SHA-256: one 1.4 GB file was silently damaged on the same USB drive that damaged files before ([rebuild-from-scratch.md](rebuild-from-scratch.md)).

**The benchmark.** `scripts/fast/record_set.py` records the six clips on the Pi; `bench_compare.py A|B|C` times one setup per run (`run_bench_all.sh` runs all four); `bench_summary.py` prints the medians, word error rates and tokens/s from the JSON lines in `results/`. The reference sentences are the `SENTENCES` list in `record_set.py`. The clips themselves (a person's voice) are not included.
