#!/bin/sh
# Rebuild the Sunflower speech translator on a fresh Raspberry Pi 4 (64-bit
# Raspberry Pi OS Desktop, "trixie"). Run ON THE PI as the normal user (needs sudo).
#
# This script was assembled from the commands run, one at a time, on a freshly
# flashed card (see docs/rebuild-from-scratch.md). Each step checks before it
# changes anything, so it can be re-run. It does NOT download or contain model
# files: see "Models" below.
#
#   sh setup-pi.sh                      # every step
#   STEPS="llamacpp venvs" sh setup-pi.sh   # only the named steps
#   SKIP_UPGRADE=1 sh setup-pi.sh       # skip the (long) full apt upgrade
#
# Steps: apt llamacpp venvs vits display touch launcher mic
#
# Models (copy these into ~/ml yourself, then verify with sha256sum against the
# source machine; sizes alone do not catch damaged copies):
#   ~/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf
#   ~/ml/gguf/mmproj-sunflower-gemma4-e2b-f16.gguf
#   ~/ml/models/tts-vits-{lug,eng,nyn}/{config.json,vocab.txt,G_*.pth}
#   ~/ml/models/mms-tts-{swh,ach,lug,eng,nyn}/        (offline MMS voices)
#   ~/ml/vits-work/                                    (SunbirdAI/vits checkout)
#   ~/ml/sunflower_touch_ui.py, sunflower_chat_ui.py, sunflower_demo.py, vits_worker.py, assets/
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ML="$HOME/ml"
STEPS="${STEPS:-apt llamacpp venvs vits display touch launcher mic}"
want() { echo " $STEPS " | grep -q " $1 "; }
say() { printf '\n== %s\n' "$*"; }

if want apt; then
  say "apt: system packages"
  sudo apt-get update -qq
  if [ -z "${SKIP_UPGRADE:-}" ]; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get -y -qq full-upgrade
  fi
  sudo DEBIAN_FRONTEND=noninteractive apt-get -y -qq install \
    git build-essential cmake python3-venv python3-dev espeak evtest libportaudio2 rsync
fi

if want llamacpp; then
  say "llama.cpp: clone and build (about 15-20 minutes on a Pi 4)"
  mkdir -p "$ML"
  [ -d "$ML/llama.cpp" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$ML/llama.cpp"
  ( cd "$ML/llama.cpp" && echo "llama.cpp commit: $(git log -1 --format=%h)" \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j4 )
  [ -x "$ML/llama.cpp/build/bin/llama-mtmd-cli" ] || { echo "build failed: llama-mtmd-cli missing" >&2; exit 1; }
fi

if want venvs; then
  say "python environments (CPU-only torch)"
  # vits-venv: Sunbird VITS (needs Cython to build monotonic_align)
  [ -x "$ML/vits-venv/bin/python" ] || python3 -m venv "$ML/vits-venv"
  "$ML/vits-venv/bin/pip" install -q --upgrade pip
  "$ML/vits-venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
  "$ML/vits-venv/bin/pip" install -q numpy scipy librosa unidecode phonemizer tqdm cython
  # tts-venv: Meta MMS-TTS via transformers
  [ -x "$ML/tts-venv/bin/python" ] || python3 -m venv "$ML/tts-venv"
  "$ML/tts-venv/bin/pip" install -q --upgrade pip
  "$ML/tts-venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
  "$ML/tts-venv/bin/pip" install -q transformers scipy
fi

if want vits; then
  say "VITS: compile monotonic_align for this CPU and fix the Python 3.13 import"
  MA="$ML/vits-work/training/monotonic_align"
  [ -f "$MA/core.pyx" ] || { echo "missing $MA/core.pyx: copy vits-work into $ML first" >&2; exit 1; }
  # cythonize writes under a path derived from the working directory, so the
  # target folders must exist first (same workaround as on the Mac).
  mkdir -p "$MA/training/monotonic_align" "$MA/monotonic_align"
  ( cd "$MA" && "$ML/vits-venv/bin/python" setup.py build_ext --inplace >/dev/null )
  cp "$MA"/training/monotonic_align/core*.so "$MA/monotonic_align/"
  touch "$MA/monotonic_align/__init__.py"
  python3 "$HERE/pi/patch_monotonic_align.py" "$MA/__init__.py"
  ( cd "$ML/vits-work/training" && "$ML/vits-venv/bin/python" -c \
    "from monotonic_align import maximum_path_c; print('monotonic_align import OK')" )
fi

if want display; then
  say "display: MHS-3.5in SPI panel via the stock piscreen overlay (reboot needed)"
  CFG=/boot/firmware/config.txt
  [ -f "$HOME/config.txt.backup-orig" ] || sudo cp "$CFG" "$HOME/config.txt.backup-orig"
  sudo sed -i 's/^#dtparam=spi=on/dtparam=spi=on/' "$CFG"
  if ! grep -q '^dtoverlay=piscreen' "$CFG"; then
    printf '\n# MHS-3.5inch SPI display (ILI9486 + XPT2046 touch), stock piscreen overlay\ndtoverlay=piscreen,drm,speed=32000000,fps=30\n' | sudo tee -a "$CFG" >/dev/null
    echo "added piscreen overlay; reboot to activate"
  else
    echo "piscreen overlay already present"
  fi
fi

if want touch; then
  say "touch: flip the Y axis with a libinput calibration rule (reboot needed)"
  sudo cp "$HERE/pi/90-touchscreen-flip-y.rules" /etc/udev/rules.d/
  sudo udevadm control --reload-rules
fi

if want launcher; then
  say "desktop launcher, and stop the file manager asking how to run it"
  sh "$HERE/pi/install_launcher.sh"
  mkdir -p "$HOME/.config/libfm"
  # pcmanfm rewrites its settings when it exits, so change the file in the gap
  # between the old process exiting and the supervisor (lwrespawn) restarting it.
  pkill -x pcmanfm 2>/dev/null || true
  n=0; while pgrep -x pcmanfm >/dev/null && [ "$n" -lt 100 ]; do sleep 0.05; n=$((n+1)); done
  F="$HOME/.config/libfm/libfm.conf"
  if [ -f "$F" ]; then
    sed -i 's/^quick_exec=0/quick_exec=1/' "$F"
    grep -q '^quick_exec' "$F" || sed -i '/^\[config\]/a quick_exec=1' "$F"
  else
    printf '[config]\nquick_exec=1\n' > "$F"
  fi
  grep '^quick_exec' "$F"
fi

if want mic; then
  say "microphone gain (USB mic defaults to 100%, which was noisy)"
  CARD=$(arecord -l 2>/dev/null | sed -n 's/^card \([0-9]*\):.*USB.*/\1/p' | head -1)
  if [ -n "$CARD" ]; then
    amixer -c "$CARD" sset Mic 62% | grep 'Mono:' || true
    sudo alsactl store
    echo "note: this does not survive a reboot; the apps set 62% themselves before each recording"
    echo "USB mic is card $CARD; the apps expect plughw:3,0 for the mic and plughw:2,0 for the headphones"
  else
    echo "no USB capture device found; plug the microphone in and re-run: STEPS=mic sh $0"
  fi
fi

say "done. Reboot if the display or touch steps changed anything: sudo reboot"
