# Results: raw data and tables for the paper

Everything here was measured on the hardware named in the main [README](../README.md) (Raspberry Pi 4 Model B 8 GB unless stated). The tables in the docs are built from these files.

| File or folder | What it is | Used in |
|---|---|---|
| `headtohead.csv` | One row per setup, device and condition: medians of listen, translate, voice, total, slowest run, start-up, word error rate, memory, CPU start temperatures, throttle flags. **Generated**: `python3 scripts/fast/key_numbers.py --write` | [Part 6](../docs/06-fast-pipeline.md#head-to-head-on-the-pi-4-six-recorded-clips), [Part 8](../docs/08-pi5-and-matched-rerun.md), [Benchmarks](../docs/benchmarks.md) |
| `pi4-clean-2026-10-08/` | The Pi 4 rerun under matching conditions: raw JSON lines for setups A, B, C, D, E and C, D with beam 1, the voice timings, the run log | [Part 8](../docs/08-pi5-and-matched-rerun.md) |
| `pi5-clean-2026-10-08/` | The Pi 5 results (5 V / 5 A supply, active cooler): the same setups, the NLLB and Whisper diagnosis logs (`nllb_diagnose.log`, `whisper_diagnose.log`), `llama_bench_q4km.json`, and the short memory runs (`pi5mem-*.jsonl`, with `rss_mb`) | [Part 8](../docs/08-pi5-and-matched-rerun.md) |
| `pi5-2026-10-08/` | **Superseded.** The first Pi 5 attempt on a 5 V / 3 A adapter with no fan (under-voltage, thermal throttling); kept as a record | Part 8, note |
| `models.csv` | Every model used: role, source, precision, size on disk. Hand-written | README "Models and sizes" |
| `voice-timings.csv` | ONNX voice timings (fp32 and int8, three runs per sentence) on both boards and every condition tested. **Generated** from the raw text logs listed in its `source` column | [Part 6](../docs/06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model), [Part 8](../docs/08-pi5-and-matched-rerun.md) |
| `pi4-headtohead-2026-10-06/` | One JSON line per clip per setup, including heard text, translation, per-stage times, CPU temperature and throttle flags; `app-session-log.jsonl` is the app's own log. Has its own README | head-to-head tables |
| `voice-onnx-2026-10-07/` | Raw output of the ONNX voice tests (Pi 4 timings, Mac listening check) | Part 6 voice section |
| `quant-sweep/` | Gemma quantization sweep: per-sentence outputs for each level, scores, Pi `llama-bench` JSON, the scripts and logs | [Quantization comparison](../docs/quantization-sweep.md) |

## Caveats to carry into any paper

- The head-to-head uses **six 5-second clips from one speaker** (three English, three Luganda). The reference sentences are the `SENTENCES` list in `scripts/fast/record_set.py`; the audio is not included. The Whisper repeat guard was tuned on these same clips.
- The benchmark waited up to 15 minutes for the CPU to fall below 60 °C once per setup, not before each clip. Per-clip start temperatures were 48–84 °C, so some runs were throttled. Setup D ran hotter than the others (73–84 °C).
- Times are medians of six; the voice timings are three runs per sentence. Treat the figures as the right order of magnitude.
- **Judged by the author only:** the transcripts and translations of the six sentences were read by the author, a Luganda speaker and the speaker on the recordings, and judged correct. That is not an independent evaluation.
- **Not measured:** an independent human evaluation of translation and voice quality; Whisper and NLLB under quantization other than int8, float32 and int8_float32 compute (on the Pi 5 only); the voices' sound quality; more speakers; the Pi 5's GPU.

## Reproducing

1. Record clips with `scripts/fast/record_set.py` (or use your own and edit `clips.json`).
2. `python3 scripts/fast/bench_compare.py A|B|C|D` on the Pi, one setup per run; results go to `~/ml/bench_results/`.
3. `python3 scripts/fast/bench_summary.py` for medians and word error rates; `python3 scripts/fast/key_numbers.py --write` for the CSV.
4. `python3 scripts/fast/experiments/tts_char_onnx.py MODEL_DIR --repeat 3 --file vits-lug-eng.int8.onnx` for the voice timings.
5. `scripts/fast/experiments/nllb_diagnose.py` and `whisper_diagnose.py` for the NLLB and Whisper time splits and compute-type comparison; `llama-bench -m <Q4_K_M gguf> -t 4 -p 64 -n 32 -r 3 -o json` for Gemma's token rates; `bench_compare.py <setup> --limit 2` records resident memory (`rss_mb`).
