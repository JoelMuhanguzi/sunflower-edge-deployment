#!/bin/bash
# Start the Sunflower web app on an Android phone connected by USB (adb).
# Needs on the phone, in /data/local/tmp: llama.cpp/ (bin_opencl build, Q4_K_M model, mmproj) and tts/ (tts_server,
# libonnxruntime.so, tokens.txt, vits-lug-eng.fp32.onnx). See docs/09-android-phone.md.
set -e
L=/data/local/tmp/llama.cpp; T=/data/local/tmp/tts; W=/data/local/tmp/webapp
HERE="$(cd "$(dirname "$0")" && pwd)"
adb shell "mkdir -p $W" && adb push -q "$HERE/index.html" $W/
[ -f "$HERE/tts_server" ] && adb push -q "$HERE/tts_server" $T/ && adb shell "chmod +x $T/tts_server"
adb shell "pkill -f llama-server; pkill -f tts_server" >/dev/null 2>&1 || true
sleep 1
adb shell "cd $T && (LD_LIBRARY_PATH=. nohup ./tts_server . 8082 > tts_server.log 2>&1 < /dev/null &)"
# The language model stays on the CPU; the audio encoder uses the Adreno GPU (-ngl 0 keeps the model weights off the GPU).
adb shell "cd $L && (LD_LIBRARY_PATH=bin_opencl nohup bin_opencl/llama-server -m sunflower-gemma4-e2b-Q4_K_M.gguf \
  --mmproj mmproj-sunflower-gemma4-e2b-f16.gguf --jinja --host 127.0.0.1 --port 8081 -c 2048 -t 4 -ngl 0 \
  --path $W > server_app.log 2>&1 < /dev/null &)"
adb forward tcp:8081 tcp:8081 >/dev/null
echo -n "waiting for the model server"
for i in $(seq 1 100); do curl -s -m 2 http://127.0.0.1:8081/health | grep -q '"ok"' && break; echo -n "."; sleep 3; done; echo
curl -s http://127.0.0.1:8081/health; echo
[ "$1" = "--open" ] && adb shell am start -a android.intent.action.VIEW -d http://localhost:8081/ >/dev/null && echo "opened in the phone's browser"
echo "open http://localhost:8081 in Chrome on the phone"
