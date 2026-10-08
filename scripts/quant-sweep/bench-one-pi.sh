#!/bin/sh
# usage: bench_one.sh LEVEL   (same method as scripts/quant-sweep/bench-quants-pi.sh, one level)
q=$1; mkdir -p "$HOME/ml/bench"; cd "$HOME/ml/bench"
BIN="$HOME/ml/llama.cpp/build/bin/llama-bench"
m="$HOME/ml/gguf/sunflower-gemma4-e2b-$q.gguf"
until [ $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) -lt 55 ]; do sleep 5; done
echo "pi: $(tr -d '\0' < /proc/device-tree/model)  kernel: $(uname -r)  llama.cpp: $(cd $HOME/ml/llama.cpp && git log -1 --format=%h)" > "$q.log"
echo "=== $q  start $(date +%H:%M:%S)  temp $(vcgencmd measure_temp)  throttled $(vcgencmd get_throttled)" >> "$q.log"
"$BIN" -m "$m" -p 64 -n 32 -r 3 -t 4 -o json > "$q.json" 2> "$q.err" &
pid=$!; peak=0
while kill -0 $pid 2>/dev/null; do
  v=$(awk '/VmHWM/{print $2}' /proc/$pid/status 2>/dev/null); [ -n "$v" ] && peak=$v; sleep 1
done
echo "=== $q  done  $(date +%H:%M:%S)  temp $(vcgencmd measure_temp)  throttled $(vcgencmd get_throttled)  peak_rss_kb $peak" >> "$q.log"
