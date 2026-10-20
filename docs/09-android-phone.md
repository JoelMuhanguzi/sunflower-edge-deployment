# Part 9: Running on an Android phone (Samsung Galaxy S23 Ultra)

Device: Galaxy S23 Ultra, Snapdragon 8 Gen 2 (Adreno 740 GPU, Hexagon NPU), 12 GB RAM.
Model: Sunflower Gemma 4 E2B (the same Q4_K_M GGUF used on the Pis), llama.cpp commit `a7b94df`.
Raw logs are in [`results/phone-s23-2026-10/`](../results/phone-s23-2026-10/); the tables below are in
`phone-speed.csv`, `phone-speech.csv` and `phone-voice.csv` in that folder.
Battery temperature is the phone's own sensor, readable at the time of each run (it climbed from 32 °C to 42 °C during the session).

## 1. The headline

| What | Pi 4 | Pi 5 | S23 Ultra |
|---|---|---|---|
| Gemma generation, Q4_K_M (tokens/s) | 2.45 | 7.79 | **14.8** (KleidiAI CPU build) |
| Prompt processing, Q4_K_M (tokens/s) | 5.99 | 40.1 | **56.6** |
| Speech in, Luganda/English audio to Gemma (median) | n/a | n/a | **9.3 s listen+translate** (encoder on GPU) |
| Voice, ONNX fp32, ~3.6 s of audio | ~7 s | see `results/voice-timings.csv` | 1.9 s (int8: **0.8 s**) |

Pi numbers are from `results/quant-speed.csv`; the phone used the same `llama-bench` settings (64-token prompt, 32 generated). The phone is the fastest of the three devices for generation, mostly because its CPU has int8 matrix instructions and a very fast memory system.

## 2. CPU build matters more than anything else (Q4_K_M, 4 threads)

| llama.cpp CPU build | prompt tok/s | generation tok/s |
|---|---:|---:|
| base (no i8mm) | 15.7 | 10.0 |
| i8mm kernels | 54.5 | 13.9 |
| **KleidiAI** | **56.6** | **14.8** |

Prompt processing is 3.6x faster with the int8-matrix build. 4 threads beat 6 and 8: using the efficiency cores slows generation (8 threads: 6.6 tok/s).

## 3. Does the GPU (Adreno, OpenCL) help?

The OpenCL backend works (Adreno 740 detected), but putting the language model on the GPU makes generation slower:

| Quant | LM on CPU: pp64 / tg32 | LM on GPU: pp64 / tg32 |
|---|---:|---:|
| Q4_0 | 75.6 / 15.6 | 27.3 / 11.0 |
| Q4_K_M | 51.0 / 14.2 | 46.7 / 4.5 |
| Q5_K_M | 38.6 / 10.3 | 29.0 / 2.9 |
| Q6_K | 38.2 / 9.5 | 24.1 / 2.8 |
| Q8_0 | 108.9 / 14.4 | 17.1 / 7.1 |

Conclusion: keep the language model on the CPU. The OpenCL kernels are tuned for Adreno 750 and newer, and the S23 (Adreno 740) is older.
Q4_0 gives the fastest generation (15.6 tok/s) and Q8_0 the fastest prompt processing (109 tok/s), but Q8_0 (4.6 GiB) thrashed memory when the audio encoder was also loaded, so Q4_K_M or Q4_0 are the practical choices.

## 4. Speech in: where the GPU does help

Gemma 4 has its own audio encoder, so one model both listens and translates. The encoder can run on the GPU while the language model stays on the CPU.
Six clips (3 English, 3 Luganda), recorded by the author:

| Configuration | median listen | median translate | total | mean WER |
|---|---:|---:|---:|---:|
| CPU only (KleidiAI) | 8.6 s | 1.9 s | 11.0 s | 0.16 |
| OpenCL build, all on CPU | 8.9 s | 1.9 s | 11.3 s | 0.16 |
| **Encoder on GPU, LM on CPU** | **7.0 s** | 2.1 s | **9.3 s** | **0.13** |
| Everything on GPU | 21.6 s | 15.0 s | about 37 s | n/a (crashed) |

The GPU speeds up listening by about 20% and changes no answers. Putting everything on the GPU is 3x slower and the server died on the fourth clip.
WER is word error rate on the transcript; the Luganda clip with WER 0.6 to 0.8 is a short greeting with names. Judgement of correctness was by the author, a Luganda speaker.

## 5. Sustained load and heat

12 back-to-back 256-token generations: 13.3 tok/s at the start, about 9.8 tok/s at the end (first three average 12.2, last three 9.9), a drop of roughly 19% as the battery temperature rose from 38.8 °C to 41.7 °C. The phone throttles, the Pi 5 with its active cooler did not.

## 6. Voice on the phone (ONNX Runtime, jq/sherpa-vits-tts-lug-eng)

| Model, provider | 23 chars (eng) | 68 chars (eng) | 33 chars (lug) | 55 chars (lug) |
|---|---:|---:|---:|---:|
| fp32, CPU | 0.71 s | 1.92 s | 1.17 s | 1.77 s |
| fp32, XNNPACK | 0.74 s | 1.96 s | 1.14 s | 1.69 s |
| fp32, NNAPI | 0.72 s | 1.85 s | 1.15 s | 1.73 s |
| **int8, CPU** | **0.29 s** | **0.81 s** | **0.48 s** | **0.78 s** |

Unlike the Pis (where our int8 conversion was 3 to 6x slower than fp32), on the Snapdragon int8 is about 2.5x faster than fp32. XNNPACK and NNAPI give no benefit over plain CPU. The int8 model's audio quality was not formally compared; check by ear before relying on it.

## 7. The on-phone app

`scripts/android/` holds a small working app: a web page served by `llama-server` on the phone itself, which records the microphone, sends audio to Gemma, then speaks the translation with the ONNX voice. It was tested end to end on the S23: voice 1.0 to 1.4 s, translation 1.8 s.

- `start_phone_app.sh`: pushes the page and starts both servers over `adb` (needs USB debugging).
- `tts_server.cpp`, `tts_bench.cpp`: the C++ voice server and benchmark (ONNX Runtime Android 1.31.0, NDK r28c). Build instructions are in the comments; the compiled binary is not committed.
- `index.html`: the page. `getUserMedia` works because the page is served from `localhost`.
- `scripts/fast/phone_bench.py` runs the speech timing test against the phone's server.

## 8. What was not tried or did not work

- **Hexagon NPU**: llama.cpp has a Hexagon backend, but it needs the Hexagon SDK (Docker toolchain). Not tried.
- **iPhone XR (Safari, WebAssembly voice)**: model loading ended with the tab being killed (3 GB RAM). Unresolved.
- **Q8_0 together with the audio encoder** exhausted memory and heated the phone to 48 °C; it was stopped.
- Phone timings were taken on a warm phone (32 to 42 °C battery sensor) and a single device, one session.

## 9. Cleaning up

Everything pushed to the phone (`/data/local/tmp/llama.cpp`, `/tmp/tts`, `/tmp/webapp`) was removed afterwards.
