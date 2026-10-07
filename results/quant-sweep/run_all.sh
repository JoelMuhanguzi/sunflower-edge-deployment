#!/bin/sh
cd "$HOME/ml/quant-sweep"
for q in f16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0; do
  echo "[$(date +%H:%M:%S)] start $q"
  "$HOME/ml/venv/bin/python" eval_quants.py "$q" "$HOME/ml/gguf/sunflower-gemma4-e2b-$q.gguf" 200 || echo "FAILED $q"
done
echo "[$(date +%H:%M:%S)] ALL DONE"
