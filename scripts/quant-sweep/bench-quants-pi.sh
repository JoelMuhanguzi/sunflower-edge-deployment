#!/bin/sh
# Speed + peak-memory benchmark of each quantization level on this Pi (CPU, 4 threads).
cd "$HOME/ml/bench"
BIN="$HOME/ml/llama.cpp/build/bin/llama-bench"
echo "pi: $(tr -d 0 < /proc/device-tree/model)"
echo "kernel: $(uname -r)  llama.cpp: $(cd $HOME/ml/llama.cpp && git log -1 --format=%h)"
for q in Q4_0 Q4_K_M Q5_K_M Q6_K Q8_0; do
  m="$HOME/ml/gguf/sunflower-gemma4-e2b-$q.gguf"
  # Q8_0 is >4 GB, so it travels as two shards on FAT32 media; llama.cpp loads the
  # whole set when pointed at the first shard.
  [ -f "$m" ] || m="$HOME/ml/gguf/sunflower-gemma4-e2b-$q-00001-of-00002.gguf"
  echo "=== $q  start $(date +%H:%M:%S)  temp $(vcgencmd measure_temp)  throttled $(vcgencmd get_throttled)"
  "$BIN" -m "$m" -p 64 -n 32 -r 3 -t 4 -o json > "$q.json" 2> "$q.err" &
  pid=$!
  peak=0
  while kill -0 $pid 2>/dev/null; do
    v=$(awk "/VmHWM/{print \$2}" /proc/$pid/status 2>/dev/null)
    [ -n "$v" ] && peak=$v
    sleep 1
  done
  echo "=== $q  done  $(date +%H:%M:%S)  temp $(vcgencmd measure_temp)  throttled $(vcgencmd get_throttled)  peak_rss_kb $peak"
done
echo "BENCH-ALL-DONE"
