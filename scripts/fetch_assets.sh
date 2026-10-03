#!/bin/sh
# Download the Sunbird and Sunflower logos and produce the title-bar-sized
# PNGs that sunflower_touch_ui.py looks for in ./assets.
#
# The logos belong to Sunbird AI and are NOT stored in this repository.
# Requires: curl, rsvg-convert (librsvg), and Python with Pillow.
#   macOS: brew install librsvg && pip install pillow
set -e

OUT="${ASSETS_DIR:-$(dirname "$0")/assets}"
PY="${PYTHON:-python3}"
mkdir -p "$OUT"

for tool in curl rsvg-convert; do
  command -v "$tool" >/dev/null || { echo "missing: $tool" >&2; exit 1; }
done
"$PY" -c "import PIL" 2>/dev/null || { echo "missing: Pillow for $PY" >&2; exit 1; }

# Look the URLs up from the live pages: the Sunflower asset URL contains a
# build hash that changes whenever the site is redeployed.
SUNBIRD_SVG=$(curl -sL https://sunbird.ai/ | grep -o 'https://sunbird.ai/[^"]*Sunbird-Horizontal-logo-white\.svg' | head -1)
FLOWER_PATH=$(curl -sL https://sunflower.sunbird.ai/ | grep -o '/_next/static/media/logo_icon\.[A-Za-z0-9_-]*\.png' | head -1)
[ -n "$SUNBIRD_SVG" ] && [ -n "$FLOWER_PATH" ] || { echo "could not locate logo URLs (site layout changed?)" >&2; exit 1; }

curl -sLo "$OUT/sunbird-white.svg" "$SUNBIRD_SVG"
curl -sLo "$OUT/sunflower-icon.png" "https://sunflower.sunbird.ai$FLOWER_PATH"

rsvg-convert --height 20 -f png -o "$OUT/sunbird-logo-20.png" "$OUT/sunbird-white.svg"
"$PY" - "$OUT" <<'PYEOF'
import sys
from PIL import Image
out = sys.argv[1]
im = Image.open(f"{out}/sunflower-icon.png").convert("RGBA")
h = 24
im.resize((round(im.width * h / im.height), h), Image.LANCZOS).save(f"{out}/sunflower-icon-24.png")
PYEOF
echo "Assets written to $OUT"
ls -la "$OUT"
