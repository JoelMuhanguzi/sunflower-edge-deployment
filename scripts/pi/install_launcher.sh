#!/bin/sh
# Install the Sunflower desktop launcher for the current user.
# Assumes the scripts live in ~/ml (adjust the template if they do not).
set -e
mkdir -p "$HOME/Desktop"
sed "s|__HOME__|$HOME|g" "$(dirname "$0")/Sunflower.desktop.template" > "$HOME/Desktop/Sunflower.desktop"
chmod +x "$HOME/Desktop/Sunflower.desktop"
# Mark trusted so the file manager treats it as a launcher (silently ignored if unavailable).
gio set "$HOME/Desktop/Sunflower.desktop" metadata::trusted true 2>/dev/null || true
echo "Installed $HOME/Desktop/Sunflower.desktop"
echo "If double-clicking shows an Execute / Open dialog, see docs/05-touchscreen-device.md (Step 26)."
