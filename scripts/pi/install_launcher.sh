#!/bin/sh
# Install the Sunflower and Sunflower Chat desktop launchers for the current user.
# Assumes the scripts live in ~/ml (adjust the templates if they do not).
# Sunflower Fast has its own template in scripts/fast/ (it uses a different Python).
set -e
mkdir -p "$HOME/Desktop"
for name in Sunflower SunflowerChat; do
  sed "s|__HOME__|$HOME|g" "$(dirname "$0")/$name.desktop.template" > "$HOME/Desktop/$name.desktop"
  chmod +x "$HOME/Desktop/$name.desktop"
  # Mark trusted so the file manager treats it as a launcher (silently ignored if unavailable).
  gio set "$HOME/Desktop/$name.desktop" metadata::trusted true 2>/dev/null || true
  echo "Installed $HOME/Desktop/$name.desktop"
done
echo "If double-clicking shows an Execute / Open dialog, see docs/05-touchscreen-device.md (Step 26)."
