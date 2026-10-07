# Results: raw data and tables for the paper

Everything here was measured on the hardware named in the main [README](../README.md) (Raspberry Pi 4 Model B 8 GB unless stated). The tables in the docs are built from these files.

| File or folder | What it is | Used in |
|---|---|---|
| `headtohead.csv` | Medians per setup (A, B, C-5s, C-tuned, C-final, D): listen, translate, voice, total, slowest run, start-up, CPU start temperatures. **Generated**: `python3 scripts/fast/key_numbers.py --write` | [Part 6](../docs/06-fast-pipeline.md#head-to-head-on-the-pi-4-six-recorded-clips), [Benchmarks](../docs/benchmarks.md) |
| `models.csv` | Every model used: role, source, precision, size on disk. Hand-written | README "Models and sizes" |
| `voice-pi4.csv` | ONNX voice timings on the Pi 4 (fp32 and int8, three runs per sentence). Hand-copied from the raw text in `voice-onnx-2026-10-07/` | [Part 6](../docs/06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model) |
| `pi4-headtohead-2026-10-06/` | One JSON line per clip per setup, including heard text, translation, per-stage times, CPU temperature and throttle flags; `app-session-log.jsonl` is the app's own log. Has its own README | head-to-head tables |
| `voice-onnx-2026-10-07/` | Raw output of the ONNX voice tests (Pi 4 timings, Mac listening check) | Part 6 voice section |
| `quant-sweep/` | Gemma quantization sweep: per-sentence outputs for each level, scores, Pi `llama-bench` JSON, the scripts and logs | [Quantization comparison](../docs/quantization-sweep.md) |

## Caveats to carry into any paper

- The head-to-head uses **six 5-second clips from one speaker** (three English, three Luganda). The reference sentences are the `SENTENCES` list in `scripts/fast/record_set.py`; the audio is not included. The Whisper repeat guard was tuned on these same clips.
- The benchmark waited up to 15 minutes for the CPU to fall below 60 °C once per setup, not before each clip. Per-clip start temperatures were 48–84 °C, so some runs were throttled. Setup D ran hotter than the others (73–84 °C).
- Times are medians of six; the voice timings are three runs per sentence. Treat the figures as the right order of magnitude.
- **Not measured:** translation quality judged by Luganda speakers; Whisper and NLLB under any quantization other than int8; Whisper encoder and decoder time separately; the voices' sound quality; memory of the ONNX voice on its own.

## Reproducing

1. Record clips with `scripts/fast/record_set.py` (or use your own and edit `clips.json`).
2. `python3 scripts/fast/bench_compare.py A|B|C|D` on the Pi, one setup per run; results go to `~/ml/bench_results/`.
3. `python3 scripts/fast/bench_summary.py` for medians and word error rates; `python3 scripts/fast/key_numbers.py --write` for the CSV.
4. `python3 scripts/fast/experiments/tts_char_onnx.py MODEL_DIR --repeat 3` for the voice timings.
