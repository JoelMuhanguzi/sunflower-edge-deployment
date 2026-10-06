# Pi 4 head-to-head, 2026-10-06

Raw output of `scripts/fast/bench_compare.py` (one JSON line per run; first line of each
non-A file is an unscored warm-up) and `scripts/fast/bench_summary.py` over them. Setups: A first app
as it is, B Gemma kept loaded in `llama-server` with prompt cache off and voices preloaded, C-5s the
second app with a 5 s window, C-tuned the second app with a 6 s window, no timestamps and no repeated
3-grams, C-final the finished app with those as defaults plus a token cap and a repeated-sentence guard. `app-session-log.jsonl` is the second app's own log from hand-driven touchscreen runs.

The clips were six 5-second recordings of one speaker (three English, three Luganda) made with
`scripts/fast/record_set.py`; the audio is not included. The reference sentences are in each record's `reference`.
