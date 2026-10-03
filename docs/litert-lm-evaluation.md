# Alternative runtime evaluated: LiteRT-LM

[← Back to the project README](../README.md)


Google's `gemma-translator` reference project (see [Part 4](04-live-audio-demo.md#looking-ahead-a-dedicated-handheld-device)) uses **LiteRT-LM**, Google AI Edge's successor to TFLite for on-device LLM inference, rather than llama.cpp. Since we already had a working Sunbird checkpoint and were curious whether it would convert and run with this alternative toolchain — and whether it might perform better, particularly given the acquisition of a Raspberry Pi 5 for a future iteration — we attempted a side-by-side evaluation on the Mac (text-only, following the same validate-before-deploying discipline as the rest of this project).

### Setup

The conversion CLI is a separate tool from the runtime, both installed via `uv` (not plain `pip`):

```bash
uv tool install litert-torch-nightly  # conversion: HF checkpoint -> .litertlm
uv tool install litert-lm             # runtime: run a .litertlm file
```

### Attempt 1: missing undocumented flag

```bash
litert-torch export_hf ~/ml/models/Sunflower-Gemma4-E2B ~/ml/litert-out --task=text_generation
```

Failed partway through export:

```
AssertionError: External embedder is required for Gemma4.
```

Not mentioned in the `--help` output's flag descriptions as Gemma4-specific, but present (and used, without explanation of why it's required) in Google's own tutorial example. Adding `--externalize_embedder` resolved it.

### Attempt 2: conversion succeeds, chat template does not

```bash
litert-torch export_hf ~/ml/models/Sunflower-Gemma4-E2B ~/ml/litert-out \
  --task=text_generation --externalize_embedder
```

This completed — roughly 20 minutes, with a detailed multi-stage log (load weights, export prefill/decode graphs, lower to MLIR, quantize, package). Produced `model.litertlm`, **5.07 GB**, using a `dynamic_wi8_afp32` (8-bit weights, fp32 activations) quantization recipe — reported in-process as "4.0x smaller" than the unquantized intermediate, though that comparison is against LiteRT-LM's own unquantized `.tflite` stage, not against our deployed GGUF baseline.

Running it with a normal prompt failed:

```
litert-lm run ~/ml/litert-out/model.litertlm --prompt "Translate to Luganda: How are you today?"
# E0000 ... Failed to apply template: unknown method: map has no method named get
```

Sunbird's `chat_template.jinja` is a large (~17KB), feature-rich Jinja2 template (nested macros, `dictsort`, tool-calling support) — the kind of template complexity common in modern instruction-tuned model releases. LiteRT-LM renders templates with **minijinja**, a lighter Rust-based Jinja subset, which does not support the full template's method calls. Bypassing the chat template entirely with `--no-template` (sending the raw prompt with no conversation formatting) did produce output: **"Oli bulungi?"** — a different, but also plausible, Luganda phrasing from our GGUF pipeline's consistent **"Oli otya leero?"** for the same request. Reproduced identically across two runs at different temperatures, so not simply sampling noise; the divergence is more likely attributable to the missing chat-template context, the different quantization scheme, or both — not root-caused further.

### Attempt 3: the experimental fix, and a disk-space failure

The converter exposes `--experimental_transpile_chat_template_for_minijinja` (default `False`), apparently intended to address exactly this gap. Re-running with it added:

```bash
litert-torch export_hf ~/ml/models/Sunflower-Gemma4-E2B ~/ml/litert-out-v2 \
  --task=text_generation --externalize_embedder \
  --experimental_transpile_chat_template_for_minijinja
```

This run **exhausted available disk space** before completing ("`ENOSPC: no space left on device`"). The conversion process holds substantial intermediate state simultaneously on disk: unquantized `model.tflite` (9.1 GB) and `per_layer_embedder.tflite` (9.4 GB) alongside their quantized counterparts and the original ~10GB source checkpoint — an estimated peak (summing those files) on the order of 35–40 GB, against a final packaged output of only ~5 GB. (A second, independent lesson here, unrelated to LiteRT-LM specifically: background model-conversion tasks can silently exhaust disk on a shared development machine faster than interactive use would suggest — worth checking `df -h` before any multi-gigabyte conversion, not just after one fails.)

### Verdict: not adopted

| | llama.cpp / GGUF (deployed) | LiteRT-LM |
|---|---|---|
| Final model size | **3.42 GB** | 5.07 GB (48% larger) |
| Required undocumented flags to convert at all | No | Yes (`--externalize_embedder`) |
| Chat template support | Works as-is | Broken; experimental fix crashed on disk space, unresolved |
| Output for identical prompt | "Oli otya leero?" (consistent, used throughout this project) | "Oli bulungi?" (different, not root-caused) |
| Conversion time | <1 minute (quantize step) | ~20 minutes |
| Maturity signals | Stable, widely used, `--print-supported-models` confirms explicit Gemma4 support | Dated nightly build; several flags/behaviors marked "experimental"; required flag undocumented for this architecture |

LiteRT-LM is real, actively developed, and did ultimately produce working (if imperfect) output on a genuinely custom third-party checkpoint — not nothing. But on every axis that matters for this deployment — size, reliability, speed to convert, and output consistency with the rest of the project — it currently underperforms the llama.cpp path already in production use. Not pursued further. Revisiting LiteRT-LM is reasonable once (a) the chat-template transpilation matures past "experimental," and (b) there is a specific reason to believe it meaningfully outperforms llama.cpp on the target hardware — e.g. once the Raspberry Pi 5 is available for direct comparison, since LiteRT-LM's binaries may be built assuming CPU features (such as those added in Cortex-A76) that the Pi 4 used throughout this project lacks.

