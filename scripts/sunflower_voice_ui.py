#!/usr/bin/env python3
"""
Sunflower Voice: hands-free, voice to voice. Say "Hey Sunflower", ask something, hear the answer.

  waiting   : the microphone feeds a small wake-word detector (wake_detect.py)
  listening : a chime, then the question is recorded until you stop talking (Silero voice-activity detector)
  thinking  : Gemma writes down what was said, then answers (the conversation is kept)
  speaking  : the answer is spoken sentence by sentence by the ONNX voice while Gemma is still writing

The screen shows the state as an animated circle. Tapping the circle acts as the wake word (a fallback
for a noisy room). It shares the first app's Gemma server (as Sunflower Chat does) and must not run
together with Sunflower Fast (memory). Timings of every turn are logged, including wake to first sound.

VOICE_MOCK=1 replaces the model, microphone and voice with stand-ins for previewing the screen on a desktop;
VOICE_WINDOWED=1 opens a window instead of fullscreen.
"""

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "wake"), os.path.expanduser("~/ml")):
    sys.path.insert(0, p)
import sunflower_touch_ui as base  # shared server, voice, recording devices, constants

MOCK = os.environ.get("VOICE_MOCK") == "1"
WINDOWED = os.environ.get("VOICE_WINDOWED") == "1"
WAKE_MODEL = os.environ.get("SUNFLOWER_WAKE_MODEL", os.path.expanduser("~/ml/wake_hey_sunflower.npz"))
CONTEXT = 4096
HISTORY_MESSAGES = 6
MAX_REPLY_TOKENS = 160
MAX_QUESTION_S = 10          # a question is cut after this long
NO_SPEECH_S = 5              # give up if nobody speaks within this long after the chime
END_SILENCE_S = 1.0          # the question ends after this much silence
CHUNK = 1280                 # 80 ms at 16 kHz
LOG_PATH = f"{base.SCRATCH}/voice_log.jsonl"
CHIME = "/tmp/sunflower_chime.wav"

BG, FG, GREY = "#1a1a1a", "#f0f0f0", "#8a8a8a"
COLORS = {"loading": "#555555", "waiting": "#d98b43", "listening": "#3fbf6f", "thinking": "#4aa3ff", "speaking": "#ffaa28"}
LABELS = {"loading": "Loading...", "waiting": 'Say "Hey Sunflower"', "listening": "Listening...",
          "thinking": "Thinking...", "speaking": "Speaking..."}


def instruction(lang):
    return (f"You are Sunflower, a friendly assistant from Sunbird AI, talking out loud with someone. "
            f"Reply in {lang}. Keep answers short: one to three plain sentences, with no lists or symbols.")


def clean_for_speech(text):
    """Remove markdown and emoji so the voice reads plain words."""
    text = re.sub(r"[*_#`>~]+", "", text)
    text = re.sub(r"[^\x00-ɏ‘-”]", "", text)  # keep Latin letters and common quotes
    return " ".join(text.split())


class SentenceSplitter:
    """Collects streamed text and hands back complete sentences (or a long clause) as soon as they are ready."""

    def __init__(self):
        self.buf = ""

    def add(self, piece):
        self.buf += piece
        out = []
        while True:
            m = re.search(r"[.?!]+[\"')\]]*\s|\n", self.buf)
            if m is None:
                if len(self.buf) > 140 and "," in self.buf[60:]:  # a long clause: speak it, do not wait
                    i = self.buf.index(",", 60) + 1
                    out.append(self.buf[:i]); self.buf = self.buf[i:]
                    continue
                break
            out.append(self.buf[:m.end()]); self.buf = self.buf[m.end():]
        return [s.strip() for s in out if s.strip()]

    def rest(self):
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []


class Mic:
    """A running arecord process delivering 80 ms chunks of 16 kHz mono int16 audio."""

    def __init__(self, device):
        self.proc = subprocess.Popen(["arecord", "-q", "-D", device, "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self):
        raw = self.proc.stdout.read(CHUNK * 2)
        if len(raw) < CHUNK * 2:
            raise RuntimeError("microphone stopped")
        return np.frombuffer(raw, dtype=np.int16)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
        self.proc.wait()


def make_chime(path=CHIME):
    """A short two-note chime, played after the wake word."""
    sr = 22050
    parts = []
    for hz, secs in ((660, 0.12), (880, 0.18)):
        t = np.arange(int(sr * secs)) / sr
        parts.append(np.sin(2 * np.pi * hz * t) * np.minimum(1, np.minimum(t, secs - t) * 80) * 9000)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(np.concatenate(parts).astype(np.int16).tobytes())


class MockServer:
    def transcribe(self, audio_path, lang):
        time.sleep(0.8)
        return "What is photosynthesis?"

    def stream_chat(self, messages, **kw):
        for w in "Photosynthesis is how plants turn sunlight, water and air into food. It happens in the leaves.".split():
            time.sleep(0.12)
            yield w + " "


class VoiceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sunflower Voice")
        if not WINDOWED:
            self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)
        self.state = "loading"
        self.level = 0.0          # 0..1 loudness while listening, drives the circle
        self.lang = tk.StringVar(value="English")
        self.history = []
        self.server = self.voice = self.wake = None
        self.manual = threading.Event()
        self.closing = False
        self.t_anim = 0.0
        self._build_ui()
        self._animate()
        threading.Thread(target=self._main_loop, daemon=True).start()

    # ------------------------------------------------------------------ screen
    def _build_ui(self):
        bar = tk.Frame(self.root, bg=BG, height=34)
        bar.pack(fill="x"); bar.pack_propagate(False)
        tk.Label(bar, text="Sunflower Voice", bg=BG, fg="#ffaa28", font=("DejaVu Sans", 13, "bold")).pack(side="left", padx=10)
        tk.Button(bar, text="x", font=("DejaVu Sans", 11, "bold"), fg=FG, bg=BG, activebackground="#333", relief="flat",
                  bd=0, width=3, command=self.close).pack(side="right", padx=(2, 22), pady=2)  # touch panel ignores the outer ~25 px
        tk.Button(bar, textvariable=self.lang, font=("DejaVu Sans", 10), fg=FG, bg="#333", relief="flat", bd=0, padx=8,
                  command=self._toggle_lang).pack(side="right", padx=4, pady=3)
        tk.Button(bar, text="New chat", font=("DejaVu Sans", 10), fg=FG, bg="#333", relief="flat", bd=0, padx=8,
                  command=self._clear).pack(side="right", padx=4, pady=3)
        self.canvas = tk.Canvas(self.root, bg=BG, height=170, highlightthickness=0)
        self.canvas.pack(fill="x")
        self.canvas.bind("<Button-1>", lambda e: self.manual.set())  # tapping the circle is a manual wake
        self.you_var, self.bot_var, self.status_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        tk.Label(self.root, textvariable=self.you_var, bg=BG, fg=GREY, font=("DejaVu Sans", 10), wraplength=456,
                 justify="left", anchor="w").pack(fill="x", padx=12)
        tk.Label(self.root, textvariable=self.bot_var, bg=BG, fg=FG, font=("DejaVu Sans", 11), wraplength=456,
                 justify="left", anchor="nw", height=3).pack(fill="x", padx=12)
        tk.Label(self.root, textvariable=self.status_var, bg=BG, fg=GREY, font=("DejaVu Sans", 8), anchor="w").pack(
            fill="x", padx=12, side="bottom")

    def _toggle_lang(self):
        self.lang.set("Luganda" if self.lang.get() == "English" else "English")

    def _clear(self):
        self.history = []
        self.you_var.set(""); self.bot_var.set("")
        self.status_var.set("New conversation")

    def _animate(self):
        """Redraw the state circle about 20 times a second."""
        c, w, h = self.canvas, self.canvas.winfo_width() or 480, 170
        cx, cy, t, color = w // 2, 78, self.t_anim, COLORS[self.state]
        c.delete("all")
        if self.state in ("waiting", "loading"):
            r = 38 + 4 * math.sin(t * 1.8)
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=3)
            c.create_oval(cx - 12, cy - 12, cx + 12, cy + 12, fill=color, outline="")
        elif self.state == "listening":
            r = 36 + 30 * self.level
            c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=4)
            c.create_oval(cx - 30, cy - 30, cx + 30, cy + 30, fill=color, outline="")
        elif self.state == "thinking":
            c.create_oval(cx - 34, cy - 34, cx + 34, cy + 34, outline="#2a3b4d", width=6)
            c.create_arc(cx - 34, cy - 34, cx + 34, cy + 34, start=-t * 240 % 360, extent=100, style="arc", outline=color, width=6)
        else:  # speaking: rings spreading outwards
            for k in range(3):
                phase = (t * 0.9 + k / 3) % 1
                r = 30 + 40 * phase
                c.create_oval(cx - r, cy - r, cx + r, cy + r, outline=color, width=max(1, int(4 * (1 - phase))))
            c.create_oval(cx - 26, cy - 26, cx + 26, cy + 26, fill=color, outline="")
        c.create_text(cx, 150, text=LABELS[self.state], fill=color, font=("DejaVu Sans", 13, "bold"))
        self.t_anim += 0.05
        if not self.closing:
            self.root.after(50, self._animate)

    def ui(self, fn, *a):
        if not self.closing:
            self.root.after(0, lambda: fn(*a))

    def set_state(self, s):
        self.state = s

    def status(self, msg):
        self.ui(self.status_var.set, msg)

    def close(self):
        self.closing = True
        self.root.destroy()

    # ------------------------------------------------------------------ models
    def _load(self):
        t0 = time.perf_counter()
        if MOCK:
            self.server = MockServer()
            return
        probe = base.GemmaServer.__new__(base.GemmaServer)
        probe.url = f"http://127.0.0.1:{base.SERVER_PORT}"
        if not probe._healthy():
            free = base.mem_available_gb()
            if free < base.MIN_FREE_GB:
                raise RuntimeError(f"Only {free:.1f} GB free. Close the other Sunflower app, then reopen this one.")
        self.status("Loading Gemma (1 to 2 minutes)...")
        self.server = base.GemmaServer(ctx=CONTEXT)
        self.status("Loading voice...")
        if base.CharVoice is None:
            raise RuntimeError("The ONNX voice needs the pipeline2 Python (onnxruntime, scipy).")
        self.voice = base.CharVoice()
        make_chime()
        self.status(f"Ready (loaded in {time.perf_counter() - t0:.0f} s)")

    def _new_wake(self):
        from wake_detect import WakeWord
        return WakeWord(WAKE_MODEL, threshold=0.9, consecutive=2)

    # ------------------------------------------------------------------ main loop
    def _main_loop(self):
        try:
            self._load()
        except Exception as e:
            self.status(f"Start-up failed: {e}")
            print("start-up failed:", repr(e), flush=True)
            return
        while not self.closing:
            try:
                self._wait_for_wake()
                if self.closing:
                    break
                self._turn()
            except Exception as e:
                self.status(f"Error: {type(e).__name__}: {e}")
                print("voice error:", repr(e), flush=True)
                time.sleep(2)

    def _wait_for_wake(self):
        self.set_state("waiting")
        self.manual.clear()
        if MOCK:
            while not self.manual.wait(0.2):
                if self.closing:
                    return
            self.wake_time = time.perf_counter()
            return
        ww, mic = self._new_wake(), Mic(base.MIC_DEVICE)
        try:
            while not self.closing:
                chunk = mic.read()
                if ww.feed(chunk) or self.manual.is_set():
                    self.wake_time = time.perf_counter()
                    return
        finally:
            mic.close()

    def _capture(self):
        """Record one question: from the first speech until END_SILENCE_S of quiet. Returns a wav path or None."""
        from openwakeword.vad import VAD
        from scipy.signal import butter, sosfilt, sosfilt_zi
        hp = butter(4, 120, "hp", fs=16000, output="sos")  # mains hum
        zi, vad = sosfilt_zi(hp) * 0, VAD()
        mic, chunks, pre, started, quiet, speech_run = Mic(base.MIC_DEVICE), [], [], False, 0, 0
        t0 = time.perf_counter()
        try:
            while not self.closing:
                x, zi = sosfilt(hp, mic.read().astype(np.float64), zi=zi)
                x = np.clip(x, -32768, 32767).astype(np.int16)
                p = vad.predict(x[:960], frame_size=480)
                self.level = min(1.0, float(np.sqrt((x.astype(np.float64) ** 2).mean())) / 4000)
                if not started:
                    pre = (pre + [x])[-6:]  # half a second before the speech started
                    speech_run = speech_run + 1 if p > 0.5 else 0
                    if speech_run >= 2:
                        started, chunks, quiet = True, list(pre), 0
                    elif time.perf_counter() - t0 > NO_SPEECH_S:
                        return None
                else:
                    chunks.append(x)
                    quiet = quiet + 1 if p < 0.4 else 0
                    if quiet >= END_SILENCE_S / 0.08 or len(chunks) * 0.08 > MAX_QUESTION_S:
                        break
        finally:
            mic.close()
            self.level = 0.0
        path = f"{base.SCRATCH}/voice_input.wav"
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(np.concatenate(chunks).tobytes())
        return path

    def _messages(self, lang):
        msgs = [dict(m) for m in self.history[-HISTORY_MESSAGES:]]
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        if msgs:
            msgs[0]["content"] = f"{instruction(lang)}\n\n{msgs[0]['content']}"
        return msgs

    def _turn(self):
        lang = self.lang.get()
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "lang": lang}
        self.set_state("listening")
        if not MOCK:
            base.play_audio(CHIME)
            time.sleep(0.4)  # the Bluetooth speaker lags a little; do not record the chime
        t0 = time.perf_counter()
        audio = "mock.wav" if MOCK else self._capture()
        if audio is None:
            self.status("Heard nothing. Say \"Hey Sunflower\" again.")
            return
        rec["question_s"] = round(time.perf_counter() - t0, 2)
        self.set_state("thinking")
        t1 = time.perf_counter()
        text = self.server.transcribe(audio, lang)
        rec["listen_s"] = round(time.perf_counter() - t1, 2)
        if not text:
            self.status("Could not make out any speech.")
            return
        rec["question"] = text
        self.ui(self.you_var.set, "You: " + text)
        self.ui(self.bot_var.set, "")
        self.history.append({"role": "user", "content": text})

        # speak sentences while Gemma is still writing: a worker synthesizes and plays them in order
        sentences, first_sound = queue.Queue(), []
        speaker = threading.Thread(target=self._speaker, args=(sentences, lang, first_sound, rec), daemon=True)
        speaker.start()
        splitter, reply, t2, t_first = SentenceSplitter(), "", time.perf_counter(), None
        for piece in self.server.stream_chat(self._messages(lang), max_tokens=MAX_REPLY_TOKENS):
            if t_first is None:
                t_first = time.perf_counter()
            reply += piece
            self.ui(self.bot_var.set, reply.strip()[-200:])
            for s in splitter.add(piece):
                sentences.put(s)
        for s in splitter.rest():
            sentences.put(s)
        sentences.put(None)
        rec["first_word_s"] = round((t_first or time.perf_counter()) - t2, 2)
        rec["reply"] = reply.strip()
        self.history.append({"role": "assistant", "content": reply.strip()})
        speaker.join()
        if first_sound:
            rec["wake_to_first_sound_s"] = round(first_sound[0] - self.wake_time, 2)
            rec["question_end_to_first_sound_s"] = round(first_sound[0] - (t0 + rec["question_s"]), 2)
        rec["total_after_question_s"] = round(time.perf_counter() - t1, 2)
        self.status(f"Listen {rec['listen_s']}s | first word {rec['first_word_s']}s | "
                    f"first sound {rec.get('question_end_to_first_sound_s', '-')}s after you stopped")
        self._log(rec)

    def _speaker(self, sentences, lang, first_sound, rec):
        while True:
            s = sentences.get()
            if s is None:
                return
            text = clean_for_speech(s)
            if not text:
                continue
            if MOCK:
                self.set_state("speaking"); time.sleep(1.0)
                first_sound.append(time.perf_counter()) if not first_sound else None
                continue
            if lang not in base.CHAR_VOICE_LANGS:
                continue
            out = f"{base.SCRATCH}/voice_reply.wav"
            if self.voice.speak(text, out).get("ok"):
                self.set_state("speaking")
                if not first_sound:
                    first_sound.append(time.perf_counter())
                base.play_audio(out)

    def _log(self, rec):
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    if WINDOWED:
        root.geometry("480x320")
    VoiceApp(root)
    root.mainloop()
