"""Wake word by transcription: wake when Moonshine hears "sunflower" (or "flower") in a short stretch of speech.

A voice-activity detector (Silero, from openWakeWord) cuts the microphone stream into stretches of speech. Each
stretch (at most 3 s, from the first sound) is written down by a Moonshine model and the wake word is looked for in
the text. Hum, silence and noise never reach Moonshine, and ordinary speech only wakes it if the word "flower"
(or a close mishearing such as "flour") appears, so it cannot chime by itself.

  det = MoonshineWake(transcribe)          # transcribe(float32 16 kHz audio) -> text
  for chunk in mic_chunks_80ms:            # int16, 1280 samples
      if det.feed(chunk): ...              # True once per wake
"""
import re
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

CHUNK = 1280  # 80 ms at 16 kHz


def is_wake_text(text):
    t = re.sub(r"[^a-z ]", "", text.lower())
    nospace = t.replace(" ", "")
    return "flower" in t or "flour" in t or "sunflo" in nospace or re.search(r"\bflow(ing|s)?\b", t) is not None


class MoonshineWake:
    def __init__(self, transcribe, log=None, max_segment_s=3.0, end_quiet_s=0.5, save_dir=None, pre_roll_s=0.8):
        from openwakeword.vad import VAD
        self.transcribe, self.log, self.save_dir = transcribe, log, save_dir
        self.pre_chunks = int(pre_roll_s / 0.08)
        self.vad = VAD()
        self.hp = butter(4, 250, "hp", fs=16000, output="sos")  # mains hum has strong harmonics up to about 200 Hz
        self.zi = sosfilt_zi(self.hp) * 0
        self.max_chunks, self.end_quiet = int(max_segment_s / 0.08), int(end_quiet_s / 0.08)
        self.warmup, self.pre, self.seg = 12, [], []
        self.run = self.quiet = 0
        self.started = self.handled = False
        self.last_text = ""

    def _prep(self, chunks):
        a = np.concatenate(chunks).astype(np.float64)
        pk = np.abs(a).max()
        if pk > 0:
            a *= min(8.0, 0.8 * 32767 / pk)  # the voice is faint once the hum is filtered out
        return (a / 32768.0).astype("float32")

    def _save(self, name):
        import os, wave
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, name)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.concatenate(self.seg[:self.max_chunks]).tobytes())  # filtered, not level-normalised
        return path

    def _check(self):
        stamp = time.strftime("%H%M%S")
        saved = self._save(f"seg_{stamp}.wav") if self.save_dir else None
        text = self.transcribe(self._prep(self.seg[:self.max_chunks])).strip()
        self.last_text = text
        hit = is_wake_text(text)
        if self.log:
            self.log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "heard": text, "wake": hit, "seconds": round(len(self.seg) * 0.08, 1), "peak": int(np.abs(np.concatenate(self.seg[:self.max_chunks])).max()), "file": saved})
        return hit

    def feed(self, chunk):
        x, self.zi = sosfilt(self.hp, chunk.astype(np.float64), zi=self.zi)
        x = np.clip(x, -32768, 32767).astype(np.int16)
        if self.warmup > 0:  # the filter settles
            self.warmup -= 1
            return False
        p = self.vad.predict(x[:960], frame_size=480)
        if not self.started:
            self.pre = (self.pre + [x])[-self.pre_chunks:]  # lead-in before the speech started
            self.run = self.run + 1 if p > 0.35 else 0
            if self.run >= 1:
                self.started, self.handled, self.quiet, self.seg = True, False, 0, list(self.pre)
            return False
        self.seg.append(x)
        self.quiet = self.quiet + 1 if p < 0.4 else 0
        hit = False
        if not self.handled and (self.quiet >= self.end_quiet or len(self.seg) >= self.max_chunks):
            hit = self._check()
            self.handled = True  # one look per stretch of speech
        if self.quiet >= self.end_quiet:
            self.started, self.run, self.seg = False, 0, []
        return hit
