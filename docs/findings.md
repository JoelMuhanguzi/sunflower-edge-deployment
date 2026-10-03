# Key findings

[← Back to the project README](../README.md)

The non-obvious things we learned deploying an African-language speech model to a Raspberry Pi 4. Each item links to the page with the evidence. Single-run measurements are marked as such; see [Benchmarks](benchmarks.md) for why that matters.

## Model deployment

1. **"Edge-ready" on a model card does not mean an edge-ready download.** The Hugging Face repo for Sunflower-Gemma4-E2B contains only a 10.2 GB BF16 checkpoint: no GGUF, no pre-quantized file. We converted and quantized it ourselves ([Part 1](01-text-model.md)). Q4_K_M took it from 10.21 GB to 3.42 GB (66.5% smaller) and gave a near-identical answer to the F16 version on one simple knowledge question ("transform" vs "convert" the only difference). That is one data point, not a quality evaluation.
2. **Architecture decides the toolchain before size does.** Sunbird's NLLB translation model cannot become a GGUF at all (it is encoder-decoder; llama.cpp converts decoder-only models). Check the architecture first ([alternatives](limitations.md#alternatives-considered)).
3. **Audio input needs a second file and one flag.** llama.cpp loads Gemma 4's audio encoder as a separate ~1 GB "mmproj" GGUF, and the model's chat template needs `--jinja` ([Part 2](02-audio-input.md)). llama.cpp itself labels audio input "experimental".
4. **Measure RAM, don't add up file sizes.** Summing file sizes suggested ~4.5 GB for text model plus audio encoder; sampling `free -m` during a real run showed system memory in use peaking at **~2.6 GB** (about 0.75 GB of which was the idle baseline) ([Benchmarks](benchmarks.md)).
5. **Speech accuracy varies enormously by language.** From the model card: word error 0.15 for English and 0.16 for Swahili, but about 0.50 for Luganda and Runyankole, over 1.0 for the worst ([Part 2](02-audio-input.md#which-languages-does-audio-input-cover)). Those are for the full-precision model; the effect of 4-bit quantization on them has not been measured.

## Documentation vs reality

6. **A model card's example code may not exist.** `Sunbird/tts-vits-lug`'s card calls a class that is absent from the linked repository. The checkpoint, `config.json` and `vocab.txt` on Hugging Face were enough to reverse-engineer a working CPU path ([Part 3](03-text-to-speech.md)).
7. **Old research code plus new Python breaks in subtle ways.** A compiled Cython extension imported fine on Python 3.11 and failed with `ModuleNotFoundError` on the Pi's Python 3.13; loading the `.so` by file path fixed it ([Part 3](03-text-to-speech.md)).
8. **Upstream gaps are real blockers.** Orpheus-3B text-to-speech was ruled out because llama.cpp's SNAC support is an unmerged draft with unresolved CPU performance problems ([Part 3](03-text-to-speech.md)).
9. **Vendor driver scripts age badly.** The display maker's install script (its repository references a 2017-era OS) is reported by open issues to fail on current Raspberry Pi OS; a built-in kernel overlay (`piscreen`) did the job with one config line ([Part 5](05-touchscreen-device.md)).

## Hardware

10. **The Pi 4's headphone jack cannot record.** `arecord -l` listed no capture devices until a USB microphone was added ([Part 4](04-live-audio-demo.md)). A headset's microphone plugged into that jack is simply unused.
11. **A cheap USB microphone at 100% gain was noisy; 62% was clean** ([Part 4](04-live-audio-demo.md)).
12. **Touch can fail in hardware while everything in software looks fine.** The driver loaded cleanly and reported a touch device, yet produced zero events, even on a rebuilt-from-scratch configuration. A second screen worked immediately on the same configuration ([Part 5](05-touchscreen-device.md)).
13. **A mirrored touchscreen is fixed at the touch layer, not by rotating the display.** The overlay's `rotate` setting affects only the display, and a rotation cannot cancel a single-axis mirror. The overlay's own `invy` option killed touch events here; a libinput calibration matrix applied via a udev rule worked, but only took effect after a reboot ([Part 5](05-touchscreen-device.md)).
14. **Input device numbers move between boots** (`event4` one boot, `event5` the next). Match devices by a stable path, not their number ([Part 5](05-touchscreen-device.md)).

## Software on a touchscreen

15. **Drop-down menus do not work reliably with touch under Wayland/XWayland.** Tk popup menus are separate windows; taps opened them but selections were not captured. An in-app full-screen picker fixed it ([Part 5](05-touchscreen-device.md)).
16. **Separate "transcribe" and "translate" steps** cost about 20 s extra on the Pi but give a transcript to display and check (single warm runs: ~30 s and ~18 s; a cold start takes ~2 min) ([Part 5](05-touchscreen-device.md)).

## Alternative runtime

17. **Google's LiteRT-LM converted the same checkpoint but was a worse fit.** The output was 5.07 GB vs 3.42 GB for GGUF, conversion needed an undocumented flag, the chat template failed in its template engine, and a repeat attempt ran out of disk (our estimate from the intermediate file sizes is 35–40 GB needed at peak) ([LiteRT-LM evaluation](litert-lm-evaluation.md)).

## Measurement caveats

18. **Edge timings are noisy.** Identical MMS-TTS runs varied by about 70% (5.0–8.6 s). Treat single numbers as order-of-magnitude ([Benchmarks](benchmarks.md)).
