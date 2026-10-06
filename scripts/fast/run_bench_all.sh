#!/bin/sh
# Runs the four setups one after another (memory is freed between them). Run on the Pi 4.
cd ~/ml/pipeline2
PY=./venv/bin/python
rm -f bench_all.done
$PY bench_compare.py A > bench_A.log 2>&1
$PY bench_compare.py B > bench_B.log 2>&1
$PY bench_compare.py C --label C-5s > bench_C5.log 2>&1
FAST_WINDOW=6 FAST_NO_TIMESTAMPS=1 FAST_NO_REPEAT=3 $PY bench_compare.py C --label C-tuned > bench_Ct.log 2>&1
echo done > bench_all.done
