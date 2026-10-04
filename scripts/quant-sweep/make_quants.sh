#!/bin/sh
# Create every quantization level from the F16 GGUF (Part 1 shows how F16 is made).
# Each takes 15-40 s on an Apple M4; combined output is about 25 GB.
set -e
LLAMA="${LLAMA:-$HOME/ml/llama.cpp}"
G="${GGUF_DIR:-$HOME/ml/gguf}"
for q in Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K; do
  "$LLAMA/build/bin/llama-quantize" "$G/sunflower-gemma4-e2b-f16.gguf" "$G/sunflower-gemma4-e2b-$q.gguf" "$q"
done
# Optional: split Q8_0 (4.95 GB) so it fits on a FAT32 USB drive (4 GB per-file limit).
# "$LLAMA/build/bin/llama-gguf-split" --split --split-max-size 3G \
#     "$G/sunflower-gemma4-e2b-Q8_0.gguf" q8-split/sunflower-gemma4-e2b-Q8_0
