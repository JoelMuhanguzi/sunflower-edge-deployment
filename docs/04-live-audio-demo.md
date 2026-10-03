# Part 4 — Live audio I/O and an interactive demo

[← Back to the project README](../README.md)


Parts 1–3 validated every capability using pre-recorded or generated audio files. This part moves to **live hardware**: a real USB microphone for input and a 3.5mm headset for output, connected directly to the Pi, with no file staging from another machine.

### Step 19 — Audio hardware: what the Pi 4 actually supports

```bash
aplay -l   # list playback devices
arecord -l # list capture devices
```

**Finding: the Raspberry Pi 4's built-in 3.5mm AV jack is output-only.** `arecord -l` reported zero capture devices before any USB microphone was connected — not a missing driver, but a hardware limitation of the board (the analog jack has no input circuitry; this changed on later Pi models with different jack designs, but not the 4). A connected headset's microphone side cannot be used via the AV jack regardless of software configuration.

**Fix:** a USB "PnP" microphone, plugged into one of the Pi's USB ports. Class-compliant USB audio devices need no driver installation on Linux — `arecord -l` detected it immediately as a new ALSA card:

```
card 3: Device [USB PnP Sound Device], device 0: USB Audio [USB Audio]
```

Final working device assignment on this Pi (note: USB device card numbers are not guaranteed stable across reboots or different USB ports — verify with `arecord -l`/`aplay -l` after any hardware change):

| | ALSA device |
|---|---|
| Microphone (USB) | `plughw:3,0` |
| Speaker (headset, 3.5mm jack) | `plughw:2,0` |

### Step 20 — Recording, and a real noise problem

```bash
arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 4 out.wav
```

16kHz mono, matching the audio encoder's expected input format directly — no resampling step needed for this particular USB mic, unlike the macOS-generated test clips in Part 2 which needed an `ffmpeg` conversion.

**Problem:** the first several recordings had audible electrical/hiss noise. Checked the mixer:

```bash
amixer -c 3
# Mono: Capture 16 [100%] [23.81dB] [on]
```

The capture gain was at its maximum (100%, 23.81dB) — overdriving the mic input, which commonly produces exactly this kind of noise on inexpensive USB mics. Reduced it:

```bash
amixer -c 3 sset Mic 60%   # landed at 62% / 14.88dB
```

Noise cleared on the next recording and stayed clear across subsequent tests. **This gain setting does not persist across reboots by default** — `alsactl store` (or an equivalent startup script) would be needed to make it permanent for an unattended/portable deployment.

### Step 21 — Full live voice-to-voice round trip

With a working mic and speaker, ran the complete loop for the first time with no pre-recorded files involved at any stage:

1. **Record** a live spoken question through the USB mic (`arecord`)
2. **Understand + translate** via `llama-mtmd-cli` with an instruction prompt ("Translate what the speaker said into Luganda...")
3. **Synthesize speech** from the model's text reply via Sunbird's VITS
4. **Play** the result through the headset (`aplay`)

Spoken English question → model heard and translated it → **"Oli otya leero?"** (correct Luganda) → synthesized and played back audibly through the headset, confirmed by ear. All four steps ran natively on the Pi, fully offline, no file ever leaving the device.

This is the first point in the project where every stage — capture, understanding, translation, synthesis, playback — happened as a live, continuous interaction on the target hardware, rather than a test against a prepared artifact.

### Step 22 — A unified demo script

Driving each stage by hand over SSH does not scale to an actual demo. `sunflower_demo.py` wraps the whole pipeline in one menu-driven script, reusing the exact binaries and Python environments already proven in Parts 1–3 (no model code was rewritten — this is an orchestration layer, calling `llama-cli` / `llama-mtmd-cli` / the VITS and MMS-TTS inference paths as subprocesses).

Four modes:

1. **Text → Text** — interactive translate/chat loop, typed input
2. **Speech → Text** — record via mic, print the model's text response
3. **Speech → Speech** — the full round trip from Step 21, as a repeatable menu option
4. **Text → Speech** — type text, choose a voice (Sunbird VITS for Luganda/English/Runyankole, or MMS-TTS by language code for anything else), hear it spoken

Two implementation issues surfaced while building it, both informative about the underlying CLI tools' actual (vs. documented) behavior:

- **`llama-cli`'s non-interactive flag is `--single-turn`, not `-no-cnv`** — the latter doesn't exist on this build and errors immediately. `--single-turn` combined with a predefined `-p` prompt runs one exchange and exits cleanly.
- **`llama-cli` and `llama-mtmd-cli` split their output differently between stdout and stderr.** `llama-mtmd-cli` cleanly isolates the model's reply to stdout, with all logging/chat-template-example noise on stderr — trivial to parse. `llama-cli` prints its banner, ASCII art, and prompt echo to stdout as well, requiring the script to locate the line following the echoed `> <prompt>` and read until the trailing `[ Prompt: ... t/s ]` stats line. This was discovered by inspecting raw captured output rather than assumed from either tool's `--help` text, which doesn't document this split.

A matching `.desktop` launcher (`~/Desktop/SunflowerDemo.desktop`) and wrapper shell script (`run_sunflower_demo.sh`) make the menu launchable from the Pi's graphical desktop, following the same pattern as a launcher already present on this device, rather than requiring an SSH session for every interaction.

### Looking ahead: a dedicated handheld device

Google's [`google-gemma/gemma-translator`](https://github.com/google-gemma/gemma-translator) project (Google Creative Labs) independently builds a similar speech-translator concept onto a small screen and speaker, as a 3D-printed handheld unit. It is a useful reference point but, on inspection, leaves most of the physical build undocumented: no specific display model or part number is named (only "a 480x320 kiosk-style display"), no battery/power design is described anywhere in the repo, and the three included STL enclosure files do not include a battery compartment. It targets a **Raspberry Pi 5** specifically, with no stated reason or benchmark comparison against a Pi 4, and uses a different runtime entirely (Google's LiteRT-LM rather than llama.cpp — see the [LiteRT-LM evaluation](litert-lm-evaluation.md) below for why we did not adopt it).

The small screen and touch interface have since been built (see [Part 5](05-touchscreen-device.md)); a dedicated speaker, battery power and an enclosure are still to do, informed by what this reference project leaves unresolved rather than a ready template to copy.

