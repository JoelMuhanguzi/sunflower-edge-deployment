# ONNX voice tests, 2026-10-07

Raw output behind [Part 6, "A faster voice"](../../docs/06-fast-pipeline.md#a-faster-voice-a-character-level-onnx-model).

- `pi4-fp32-first-run.txt`: Pi 4, character model fp32, three runs per sentence.
- `pi4-int8-and-fp32-same-session.txt`: Pi 4, int8 then fp32 in one session, CPU below 60 °C at the start of each. The first two lines are the SHA-256 prefixes of the int8 file on the Mac and on the Pi (they match).
- `mac-whisper-check.txt`: the Mac listening test: Whisper transcribing the synthesized audio, for the wrong (phoneme) model, the working character model and its int8 version.
- Setup D of the six-clip benchmark is `../pi4-headtohead-2026-10-06/D-onnx-voice_20261007_113719.jsonl`.

Script: `scripts/fast/experiments/tts_char_onnx.py`. Models: `jq/sherpa-vits-tts-lug-eng` (SHA-256 `45083e0c…`). The int8 file was made by us; it is not a download.
