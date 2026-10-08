# Raspberry Pi 5 (8 GB), preliminary runs, 2026-10-07/08

**Superseded by `../pi5-clean-2026-10-08/`; kept as a record, do not quote.** The board was powered by a 5 V / 3 A adapter (the Pi 5 wants 5 V / 5 A) and had no fan or heatsink. `dmesg` logged "Undervoltage detected!", the throttle word had the under-voltage bit set, and the CPU reached 82-86 °C with the soft temperature limit and frequency capping active during runs. Real figures on a properly powered, cooled board should be faster. Pi 5 Model B Rev 1.1, Cortex-A76 at up to 2.4 GHz, Debian 13 trixie, Python 3.13.

- `pi5-D_20261007_220828.jsonl`: setup D (Whisper int8 + NLLB int8 + ONNX voice, six clips, one warm-up) with `scripts/fast/bench_compare.py`. Medians: listen 9.6 s, translate 7.0 s, voice 2.2 s, total 19.3 s (Pi 4: 20.6, 11.7, 7.1, 40.8). Start-up of 4.1 s is not comparable with the Pi 4's 83 s because the model files were in the page cache after being checksummed.
- `voice-onnx-fp32-int8.txt`: ONNX voice timings, `tts_char_onnx.py`, three runs per sentence, CPU below 60 °C at the start of each model. fp32 0.9-3.0 s per sentence (0.6-0.7× the audio length); int8 5.1-12.9 s (2.6-3.5×), still 4-6× slower than fp32 even though the A76 has the int8 dot-product instruction (`asimddp`).

Model files were copied by rsync and every file's SHA-256 matched the Mac's. To be repeated with proper power and cooling.

## Setups E, B and A (Gemma side), 2026-10-07 evening

Same board, same supply, same six clips and the same script (`scripts/fast/bench_compare.py E|B|A`), llama.cpp pinned to `a7b94df` (the build behind the Pi 4's rebuilt-Pi numbers), Gemma Q4_K_M with its F16 audio encoder. Medians of six, seconds:

| Setup | Listen | Translate | Voice | Total | Slowest | Start-up | CPU start temp |
|---|---|---|---|---|---|---|---|
| A: Gemma started per tap, Sunbird VITS per tap (`pi5-A_*.jsonl`) | 27.0 | 15.2 | 11.2 | **53.7** | 69.0 | none | 59-83 °C |
| B: Gemma kept loaded, Sunbird VITS worker (`pi5-B_*.jsonl`) | 10.2 | 3.5 | 3.4 | **18.0** | 20.2 | 56 s | 80-85 °C |
| E: Gemma kept loaded, ONNX voice (`pi5-E_*.jsonl`) | 10.2 | 3.9 | 2.3 | **17.3** | 18.0 | 41 s | 80-86 °C |
| D: Whisper + NLLB + ONNX voice (`pi5-D_*.jsonl`) | 9.6 | 7.0 | 2.2 | **19.3** | 20.8 | 4 s (cached) | 73-86 °C |

Same caveats: 5 V / 3 A supply, no fan, soft temperature limit and frequency capping active in most scored clips. Translation token counts per clip (from the logs) were 7-22 for Gemma and 5-14 for NLLB, so NLLB's slower translate step is not explained by longer outputs.
