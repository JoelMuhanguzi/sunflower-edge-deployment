# Part 5 — A touchscreen translator device

[← Back to the project README](../README.md)

Parts 1–4 produced a working speech pipeline driven from a terminal. This part puts a small touchscreen in front of it: a 3.5" SPI display, a touch UI with on-screen language selection, and a desktop launcher. Most of the effort went into getting the display and touch to behave, so most of this page is about that.

## Hardware

| | |
|---|---|
| Board | Raspberry Pi 4 Model B, 8 GB, Raspberry Pi OS (Debian 13 "trixie", 64-bit) |
| Display | "MHS-3.5inch" SPI TFT, 480×320, ILI9486 display controller, XPT2046 resistive touch |
| Audio in | USB "PnP" microphone (see [Part 4](04-live-audio-demo.md)) |
| Audio out | 3.5 mm headset jack |
| Desktop | Wayland compositor `labwc` (the current Raspberry Pi OS default), not X11 |

## Step 23 — Getting the display working: use the built-in overlay, not the vendor script

The vendor's leaflet and wiki point to `git clone https://github.com/goodtft/LCD-show` followed by `sudo ./MHS35-show`. We did **not** use it:

- The repo's open issues report it failing on Bookworm and Trixie: it hardcodes the old `/boot/config.txt` path (current systems use `/boot/firmware/config.txt`) and references packages that no longer resolve.
- The vendor wiki's newest tested OS is Raspberry Pi OS Bookworm (Nov 2024), **32-bit**. This Pi runs a newer, 64-bit release, and the wiki makes no mention of Wayland.
- The script rewrites boot configuration, which is the wrong thing to run blind on a headless device.

Raspberry Pi OS already ships a mainline overlay for this chip pair, `piscreen`, which needs no download. It is listed in `dtoverlay -h piscreen`:

```
Params: speed, rotate, fps, debug, xohms, drm, invx, invy, swapxy
```

Setup, with `config.txt` backed up first:

```bash
sudo cp /boot/firmware/config.txt ~/config.txt.backup
sudo sed -i 's/^#dtparam=spi=on/dtparam=spi=on/' /boot/firmware/config.txt
echo 'dtoverlay=piscreen,drm,speed=32000000,fps=30' | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

`dmesg` then shows both drivers loading (`ili9486` for the display, `ads7846` for touch, since the XPT2046 is register-compatible). `wlr-randr` reports a single 480×320 mode.

Two settings that came up and turned out not to apply to this screen: `hdmi_group` / `hdmi_mode` ("CEA mode") and underscan are HDMI-output settings and do nothing for an SPI panel; `disable_overscan=1` was already set.

## Step 24 — Touch: from "dead" to "mirrored" to working

This was the slow part. In order:

1. **First screen: display worked, touch worked, then stopped.** After a series of config changes (rotation values, a calibration rule, reboots), the touch controller stopped producing any events at all. `evtest` on the touch device showed **zero** events, even after we removed every change we had made and rebuilt the configuration from scratch. A software cause was ruled out because the clean configuration is the same one that worked earlier the same day. A second, new screen worked immediately on that same configuration (571 events in an 8-second `evtest` run). The most likely explanation is a hardware fault in the first unit, though we did not open it up to confirm.

2. **On the working screen, touch was mirrored.** A visual test app (`scripts/touch_test.py`) draws a circle where each touch lands and logs the coordinates. Tapping the top-left corner logged `x≈40, y≈285` on a 480×320 screen: left/right correct, **top/bottom flipped**.

3. **`rotate` does not fix a mirror.** We first assumed rotating the display would help. Decompiling the overlay (`dtc -I dtb -O dts piscreen.dtbo`) shows `rotate` only changes the *display*; the touch axes have their own parameters (`invx`, `invy`, `swapxy`). A 180° rotation also cannot cancel a single-axis mirror: it just turns a vertical mirror into a horizontal one.

4. **`invy=1` in the overlay killed touch.** Adding `,invy` to the overlay line produced zero touch events after reboot. Reverting the line restored them (170 events). We did not investigate why, and did not retry it, so this is an observation and not a root cause.

5. **The fix that worked: a libinput calibration matrix.** Instead of changing the kernel driver, flip Y one layer up with a udev rule that sets `LIBINPUT_CALIBRATION_MATRIX` (`1 0 0 0 -1 1`, i.e. y' = 1 − y). The rule is in `scripts/pi/90-touchscreen-flip-y.rules`:

   ```bash
   sudo cp scripts/pi/90-touchscreen-flip-y.rules /etc/udev/rules.d/
   sudo reboot
   ```

   **Reboot, don't just reload the rules.** `libinput list-devices` showed the new matrix immediately, but taps were still wrong until the next boot. Our reading (not independently verified) is that the desktop applies the calibration when it first detects the touchscreen at startup.

   After the reboot, a tap in the top-left corner logged `x=41, y=47`, as expected, and the user reported that taps matched much better than before.

**What was and wasn't verified:** the top-left corner was checked explicitly; the other corners were judged from the spread of ~40 logged taps and from the user's report that the UI buttons now respond where they are tapped. We did not run a formal four-corner accuracy test. The same investigation was repeated on a rebuilt card with the same result ([rebuild](rebuild-from-scratch.md)).

**Edge dead zone:** the logged touch range was roughly x 36–453 and y 31–299 on a 480×320 panel, so the outer ~20–35 px of each edge do not register. Keep important buttons away from the edges. (The quit button sits well inside the right edge for this reason.)

### Dead ends worth knowing about

- **`xinput_calibrator`** (the tool a popular tutorial recommends) installs fine but is X11-only. Under Wayland it only sees the virtual `xwayland-touch` device the compositor creates, not the real touchscreen, so its results cannot be applied. We installed it, confirmed this, and did not use it.
- **Matching the udev rule by device number is fragile.** The touchscreen was `event4` on one boot and `event5` on the next. The rule matches `ID_PATH=="platform-fe204000.spi-cs-1"`, which is stable. A first attempt that matched by hardware ID (`hwdb`) never applied, because SPI input devices have no `MODALIAS` to match.
- **`labwc` already maps touch to the screen.** On first login the OS wrote `~/.config/labwc/rc.xml` with `<touch deviceName="ADS7846 Touchscreen" mapToOutput="SPI-1" mouseEmulation="yes"/>`. That handles *which screen* touch controls, not axis direction.

## Step 25 — The touchscreen app

`scripts/sunflower_touch_ui.py` is a Tkinter app sized for 480×320, reusing the same command-line tools as the earlier demo (llama.cpp for speech and text, Sunbird VITS or Meta MMS for speech output). It is an orchestration layer; no model code was changed.

**Two modes**, chosen with a toggle at the top:

- **Translate:** record 5 seconds → transcribe in the speaker's language → translate that text → speak the translation. The screen shows both the transcript ("Heard") and the translation.
- **Transcribe:** record 5 seconds → show the transcript. No translation, no spoken reply.

**Why two model steps instead of one.** An earlier version asked the audio model to hear and translate in a single call, which gives no transcript to show. Splitting it costs extra time but gives a visible, checkable transcript and lets the second step use the fast text-only model. Measured on the Pi with a warm model cache (single runs): transcription **~30 s**, translation **~18 s**. A cold first run takes about 2 minutes. If the speaker's language and target language are the same, translation is skipped.

**Design points that mattered on a touchscreen:**

- **No drop-down menus.** Tk popup menus are separate windows and did not work reliably with touch under XWayland: the menu opened, but the selection was not captured. Each language button instead opens a full-screen picker drawn inside the app, with large buttons.
- **Worker threads must not touch Tk.** The first version read Tk variables from the worker thread, which Tk only permits on the main thread (it happened to work, but is fragile). Values are now read on the main thread and passed in, and all screen updates go through `root.after`. A stubbed test (fake microphone, models and speaker, driven through a real main loop) verified the Translate, Transcribe and same-language flows.
- **Brand elements.** Colours (`#ffaa28` orange, the most frequent colour in sunbird.ai's page source) and logos in the title bar. The record button is labelled "TAP TO SPEAK" because it records a fixed 5 seconds after one tap; it does not respond to holding.
- **Desktop preview.** `scripts/preview_ui.py` shows the real UI at 480×320 on a desktop machine, so layout changes can be checked before copying to the Pi. Only the visuals work off-device; recording and models need the Pi.
- **Offline voices.** Swahili and Acholi speech use Meta's MMS-TTS. The app loads each from a local folder (`~/ml/models/mms-tts-<code>`) when it exists and only downloads otherwise, so these voices work with no network ([Part 3](03-text-to-speech.md)).

## Step 26 — Launching from the desktop

`scripts/pi/install_launcher.sh` creates the desktop launcher (from `Sunflower.desktop.template`, filling in the home directory). On first use, double-clicking it showed a dialog: *Execute / Execute in terminal / Open / Cancel*. That dialog is the file manager's `quick_exec=0` setting ("ask how to run executables"), not a fault in the launcher.

It can be switched off in **File Manager → Edit → Preferences → General → "Don't ask options on launch of executable file"**. To do it over SSH, the catch is that `pcmanfm` holds the setting in memory and rewrites `~/.config/libfm/libfm.conf`. On Raspberry Pi OS it is run by a supervisor (`lwrespawn`) that restarts it one second after it exits, so the setting can be changed in that gap:

```bash
cp ~/.config/libfm/libfm.conf ~/.config/libfm/libfm.conf.bak
pkill -x pcmanfm
while pgrep -x pcmanfm >/dev/null; do sleep 0.05; done
sed -i 's/^quick_exec=0/quick_exec=1/' ~/.config/libfm/libfm.conf
```

The setting was still intact after the file manager restarted. It applies to every desktop shortcut, not just this one.

## Branding assets

The Sunbird and Sunflower logos are not included in this repository; they belong to Sunbird. `scripts/fetch_assets.sh` downloads them from the public sites and produces the title-bar-sized PNGs the app looks for in `assets/`. If the files are missing the app falls back to a text-only title bar.

## Not yet done

- Battery power, a speaker, and an enclosure. Google's [`gemma-translator`](https://github.com/google-gemma/gemma-translator) reference project documents none of these (see [Part 4](04-live-audio-demo.md#looking-ahead-a-dedicated-handheld-device)).
- A formal four-corner touch accuracy check, and a scaling calibration for the edge dead zone.
- Comparing our transcription prompt against the model's trained prompt format (see [Part 2](02-audio-input.md#which-languages-does-audio-input-cover)).
