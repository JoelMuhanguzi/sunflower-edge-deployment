# Part 1 — Deploying the text model to a Raspberry Pi 4

[← Back to the project README](../README.md)

## Why this path

The model card describes the model as suitable for edge devices, but the Hugging Face repository only hosts the full-precision BF16 safetensors checkpoint — there is no pre-quantized GGUF release. A 10.2 GB BF16 file will not fit usefully in an 8GB Pi's RAM. The deployment therefore required:

1. Converting the HF checkpoint to GGUF (llama.cpp's native format)
2. Quantizing it to 4-bit to shrink memory/storage footprint
3. Building `llama.cpp` natively for ARM64 on the Pi
4. Transferring the quantized model to the device

We also evaluated two alternative pre-quantized releases before settling on this approach — see [Alternatives Considered](limitations.md#alternatives-considered).

---

## Environment

**Host machine (conversion/quantization work):**
- MacBook Air, Apple M4, 24GB RAM, macOS 27.0.1 (Sequoia-era build)
- Used for all heavy lifting: download, GGUF conversion, quantization, and a correctness test, before shipping the final artifact to the Pi.

**Target device:**
- Raspberry Pi 4 Model B Rev 1.5, 8GB RAM
- Debian GNU/Linux 13 (trixie), 64-bit (aarch64)
- 4x Cortex-A72 cores @ 1.5GHz (no dotprod/i8mm ARM extensions — those arrived with Cortex-A76 in the Pi 5, so this is a slower baseline than newer ARM boards)



## Step 1 — Install build tooling (Mac)

```bash
brew install cmake git-lfs
git lfs install
```

## Step 2 — Clone and build llama.cpp (Mac)

```bash
mkdir -p ~/ml && cd ~/ml
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

`-DGGML_METAL=ON` enables Apple Silicon GPU acceleration for local testing. This flag is Mac-only and is **not** used in the Pi build (see Step 7).

> **Note:** our first clone attempt stalled/hung on a flaky connection and left a broken `.git` state. If `git clone` appears to hang indefinitely, check with `du -sh llama.cpp` in another terminal — if size isn't growing, kill it and retry with `--depth 1` (shallow clone), which is faster and sufficient since we don't need history.

## Step 3 — Set up the Python conversion environment (Mac)

llama.cpp's HF→GGUF converter needs `torch`, `transformers`, `gguf`, `sentencepiece`, and `protobuf`.

```bash
python3 -m venv ~/ml/venv
~/ml/venv/bin/pip install --upgrade pip
~/ml/venv/bin/pip install -r ~/ml/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

## Step 4 — Authenticate with Hugging Face and download the model

The repository is gated, requiring accepted access terms plus a token.

```bash
huggingface-cli login
# paste a Hugging Face access token (Settings -> Access Tokens -> Read) when prompted
```

```bash
mkdir -p ~/ml/models
huggingface-cli download Sunbird/Sunflower-Gemma4-E2B \
  --local-dir ~/ml/models/Sunflower-Gemma4-E2B
```

This pulls down ~9.6 GB across `model.safetensors`, tokenizer files, and configs.

## Step 5 — Convert HF safetensors to GGUF (Mac)

First, confirm the architecture is supported by the converter:

```bash
cd ~/ml/llama.cpp
~/ml/venv/bin/python convert_hf_to_gguf.py --print-supported-models | grep -i gemma
```

This confirmed `Gemma4ForConditionalGeneration` — matching the model's `config.json` — is natively supported.

### Problem: tokenizer config incompatibility

The first conversion attempt failed with:

```
AttributeError: 'list' object has no attribute 'keys'
```

Root cause: the checkpoint's `tokenizer_config.json` stores `extra_special_tokens` as a list (`["<|video|>"]`), but the installed `transformers` version expects a dict (e.g. `{"video_token": "<|video|>"}`). This is a version-compatibility mismatch between how the checkpoint was saved and the transformers release used locally, not a problem with the model weights themselves.

**Fix applied** (local-only, not uploaded anywhere): backed up the original config, then replaced the malformed field with an empty dict. Since this deployment only needed text input/output (not video), dropping the video special-token mapping had no effect on functionality.

```bash
cp ~/ml/models/Sunflower-Gemma4-E2B/tokenizer_config.json{,.bak}
python3 -c "
import json, os
path = os.path.expanduser('~/ml/models/Sunflower-Gemma4-E2B/tokenizer_config.json')
with open(path) as f:
    d = json.load(f)
if isinstance(d.get('extra_special_tokens'), list):
    d['extra_special_tokens'] = {}
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
"
```

### Running the conversion

```bash
mkdir -p ~/ml/gguf
~/ml/venv/bin/python convert_hf_to_gguf.py \
  ~/ml/models/Sunflower-Gemma4-E2B \
  --outfile ~/ml/gguf/sunflower-gemma4-e2b-f16.gguf \
  --outtype f16
```

Output: `sunflower-gemma4-e2b-f16.gguf`, 541 tensors, **9.27 GB**. This extracts and converts only the text-generation backbone — the audio and vision towers present in the original checkpoint are not part of the standard GGUF text conversion path and were not carried over.

## Step 6 — Quantize to Q4_K_M (Mac)

```bash
./build/bin/llama-quantize \
  ~/ml/gguf/sunflower-gemma4-e2b-f16.gguf \
  ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  Q4_K_M
```

Result: **8828.84 MiB -> 3242.78 MiB** (16.00 BPW -> 5.88 BPW), final file **3.16 GB**.

`Q4_K_M` ("K-quant, medium") quantizes most tensors to 4 bits while keeping select sensitive tensors (e.g. `ffn_down`) at 6 bits, which preserves more quality than a flat 4-bit scheme at a modest size cost. It's llama.cpp's standard recommended default for resource-constrained inference.

## Step 7 — Validate locally before shipping to the Pi (Mac)

```bash
./build/bin/llama-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  -p "Translate to Luganda: How are you today?" \
  -n 64 --temp 0.3
```

**Output:** `Oli otya leero?` — the correct, idiomatic Luganda translation. This confirmed the conversion and quantization preserved the model's actual translation capability before investing time in the Pi transfer. Ran at 68.5 tok/s prompt / 47.2 tok/s generation on the M4 (Metal-accelerated) — recorded here only as a sanity baseline, not representative of Pi performance.

## Step 8 — Set up llama.cpp on the Pi

Over SSH:

```bash
ssh <user>@<pi-ip>
sudo apt-get update
sudo apt-get install -y cmake   # git and gcc/g++ were already present on this Pi OS image

mkdir -p ~/ml && cd ~/ml
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j4
```

No `-DGGML_METAL` flag here (Mac-only). CMake auto-detected the Pi's Cortex-A72 cores and correctly disabled ARM dot-product/matmul-int8 extensions (`nodotprod`, `noi8mm`) that the A72 doesn't support — this is expected and not an error; it just means the build targets the CPU's actual capabilities. A warning about missing OpenSSL appeared too — this only disables HTTPS support in the optional `llama-server` HTTP component and did not block the build or affect `llama-cli`.

Verify the build:

```bash
~/ml/llama.cpp/build/bin/llama-cli --version
```

## Step 9 — Transfer the quantized model to the Pi

### Attempt 1: `scp` over Wi-Fi — failed

```bash
scp ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf <user>@<pi-ip>:~/ml/gguf/
```

This transfer died mid-copy when the laptop briefly left the Pi's network. `scp` does not resume — the partial file had to be deleted and the transfer restarted from zero.

### Attempt 2: `rsync --partial` over Wi-Fi — too slow

```bash
rsync -avh --progress --partial \
  ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  <user>@<pi-ip>:~/ml/gguf/
```

`rsync` is resumable, which solved the drop-out problem, but measured throughput on this network was only ~800 KB/s — on the order of an hour-plus for 3.16 GB. For a one-off transfer, this made a physical alternative worth it.

### Attempt 3: USB flash drive — used in the end

1. Copied the GGUF file onto a USB flash drive via macOS Finder.
   - Note: direct command-line `cp`/`touch` to the mounted USB volume failed with `Operation not permitted` due to macOS file-access permissions on the terminal session in use; Finder drag-and-drop bypassed this cleanly.
   - Also note: the target filesystem was FAT32, which has a 4 GB per-file limit. Our 3.16 GB file fit; the unquantized 9.27 GB F16 file would **not** have fit and would need exFAT/ext4 or splitting.
2. Physically moved the drive to the Pi and plugged it in — it auto-mounted at `/media/<user>/<volume-name>`.
3. Copied from the USB mount to the Pi's internal storage (its SD card root filesystem) over a local `cp`:

```bash
cp /media/<user>/<volume-name>/sunflower-gemma4-e2b-Q4_K_M.gguf ~/ml/gguf/
```

This local disk-to-disk copy ran at roughly 25 MB/s — more than 30x faster than the Wi-Fi transfer — and completed in about two minutes.

File integrity was confirmed by comparing byte-for-byte file size against the source (`3,416,120,736` bytes on both ends).

## Step 10 — Run inference natively on the Pi

```bash
cd ~/ml/llama.cpp
./build/bin/llama-cli \
  -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf \
  -p "Translate to Luganda: How are you today?" \
  -n 64 --temp 0.3
```

**Output:** `Oli otya leero?` — identical, correct output to the Mac test.

**Performance on Raspberry Pi 4 (CPU-only, no accelerator):**

| Metric | Value |
|---|---|
| Prompt processing | 6.0 tok/s |
| Generation | 2.3 tok/s |

### Interactive use

For multi-turn use without relaunching the binary each time, omit `-p`:

```bash
./build/bin/llama-cli -m ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf -n 512 --temp 0.5 -c 2048
```

- `-n` — max tokens generated per response
- `--temp` — sampling temperature (lower = more deterministic/focused, higher = more varied)
- `-c` — context window size (conversation memory length)

Exit with `Ctrl+C` or `/exit`.

