# Rebuilding the whole system on a blank card

[← Back to the project README](../README.md)

After the first install had accumulated a lot of experiments, the whole system was rebuilt on a freshly flashed 64 GB card, following the steps in these docs. This page records what that showed: what worked unchanged, what differed, and what it cost.

The rebuild was done step by step by hand. `scripts/setup-pi.sh` was then written from the commands actually run. **It has not been run end to end on a blank card**; it was only re-run on the already-configured Pi to check that each step is safe to repeat (see [below](#how-the-setup-script-was-tested)).

## What was identical

| | First install | Rebuild |
|---|---|---|
| Board | Raspberry Pi 4 Model B Rev 1.5, 8 GB | same |
| OS | Raspberry Pi OS Desktop 64-bit (Debian 13 "trixie") | same release |
| Python | 3.13.5 | 3.13.5 |
| Kernel | 6.18.34 | 6.18.50 (a later release) |
| llama.cpp | an earlier build (`b1-552f18f`) | commit `a7b94df` (4 Oct 2026), a newer build |

Text generation measured **6.1 prompt / 2.4 generation tokens/s** (first-install figures: 6.0 / 2.3), and audio encoding took **1.45 s** (1.48 s before). The new numbers are marginally better; the newer llama.cpp build is a plausible reason, but we did not isolate it.

## What differed, and why it matters

- **`sudo` asks for a password on the new image,** where the first card had passwordless `sudo`. Remote, scripted setup stalls on this, so one command had to be run at the Pi to allow it.
- **Choosing "password" for SSH in Raspberry Pi Imager meant no key was installed.** Authentication was fixed with `ssh-copy-id`, then password login was switched off (`PasswordAuthentication no` in a `sshd_config.d` file named so it sorts before the cloud-init one, because the first matching value wins).
- **The old SSH host key no longer matched.** A reflashed Pi on the same IP has new host keys, so the stored entry had to be replaced. Do this only when you know you reflashed.
- **A weak default password** was chosen at first; changing it was part of the setup.
- **The USB microphone's gain defaults to 100% again** (the noisy setting). It was set to ~62% and saved with `alsactl store`, but **that did not survive a reboot**: the gain read 100% again afterwards (the `alsa-restore` service was active). The touchscreen app therefore sets the gain to 62% with `amixer` before every recording (`MIC_GAIN_PERCENT` in `sunflower_touch_ui.py`); this was tested by forcing 100% and recording through the app's own code. The terminal demo does the same.

## Files damaged in transit

The model files were moved from the development Mac to the Pi on a USB flash drive. Three files arrived **damaged although their sizes were exactly right**:

| File | What we saw |
|---|---|
| Audio encoder (`mmproj`, 1 GB) | three different checksums for "the same" file: Mac, the drive, and the Pi's copy |
| Runyankole MMS voice (`model.safetensors`, 145 MB) | checksum differed from the Mac's |
| Q8_0 model, second piece (1.9 GB) | the drive's copy already differed from the Mac's; the Pi copied it faithfully |

Each damaged file read back the same way every time it was read, so these were wrong *contents*, not unreliable reads. Everything else (the four other model files, the voices, the other MMS models, the apps) matched. The Pi's own log showed the drive had been mounted with *"Volume was not properly unmounted. Some data may be corrupt."* A second drive, which mounted without that warning, delivered the same Q8_0 piece correctly on the first try, which points at the first drive (or how it was handled), not at the method.

What worked: compare **SHA-256** on both machines for anything large; resend a bad file from the original with `rsync --checksum` (only the differing parts travel). Over home Wi-Fi the 1 GB encoder took 11 seconds, far faster than the USB copy (~17 MB/s). Do not trust file sizes.

## Sequence that worked

1. Flash with Raspberry Pi Imager (hostname, Wi-Fi, SSH set in the advanced options); install the SSH key; allow passwordless `sudo`; disable SSH password login.
2. `apt full-upgrade`, then `cmake espeak evtest libportaudio2`.
3. Clone and build llama.cpp (about 15 minutes).
4. Copy the models in; **verify every large file with `sha256sum`**.
5. Create the two Python environments (CPU-only `torch`).
6. Compile the VITS extension and apply the import fix.
7. Test in order: text → audio in → speech out (VITS, then MMS **with network access disabled**, `HF_HUB_OFFLINE=1`).
8. Display overlay and reboot; confirm both drivers loaded and raw touch events arrive; install the touch rule and reboot; check the corners.
9. Desktop launcher, file-manager setting, microphone gain; a live voice test.

The display and touch behaved the same on both cards: the same working display configuration (`piscreen`, no `rotate=`), the same mirrored vertical touch before the fix (a first tap in the top-left read `x=33, y=288` on the 480×320 panel), and the same cure (`x=33, y=22` afterwards). The other three corners also read plausibly in tap order (for example bottom-right `x=452, y=293`), but which corner each tap was aimed at is inferred from the order, not recorded, so this is not a formal accuracy test.

## How the setup script was tested

`scripts/setup-pi.sh` bundles steps 2, 3, 5, 6, 8 and 9 as separate, re-runnable stages (`STEPS="llamacpp venvs"` runs a subset). Its helper `scripts/pi/patch_monotonic_align.py` was tested on a copy: it patches once, recognises an already-patched file, imports correctly afterwards, and refuses a file it does not recognise.

The whole script was then **re-run on the already-configured Pi** (skipping the system upgrade) to check that each stage detects finished work and leaves it alone. That tests repeatability, **not** a blank-card install. Model files are never fetched by the script.
