# Quantization sweep: raw data

Raw files behind [docs/quantization-sweep.md](../../docs/quantization-sweep.md).

- `outputs/<level>.jsonl`: per-sentence translations from each quantization level (200 sentences each), with the server logs.
- `scores.md`: scores computed from those outputs, including the count of broken outputs per level.
- `pi-bench/<level>.json`: `llama-bench` results on the Raspberry Pi 4 (speed and memory).
- `eval_quants.py`, `score_quants.py`, `run_all.sh`: the scripts that produced them. `quantize.log` and `run_all.log` are the run logs.
