#!/usr/bin/env python3
"""
Sunflower "fast pipeline" touchscreen app for the MHS-3.5" display (480x320).
Same screens and voice output as sunflower_touch_ui.py, but with the other models:
  speech -> text : Sunbird faster-whisper (Whisper large-v3 fine-tune), int8, 5 s window
  translation    : Sunbird NLLB-1.3B (SALT), CTranslate2 int8
  speech output  : English and Luganda: a character-level ONNX voice kept loaded in the app (FAST_VOICE=vits
                   for the Sunbird VITS worker); other languages load Sunbird VITS / Meta MMS per tap
Both models are loaded once at start-up; each stage's time is shown and logged.
"""

import difflib
import json
import os
import re
import subprocess
import sys
import threading
import time
import tkinter as tk

HOME = os.path.expanduser("~")
PIPELINE2 = f"{HOME}/ml/pipeline2"
WHISPER_DIR = f"{PIPELINE2}/models/whisper-51-int8"
NLLB_DIR = f"{PIPELINE2}/models/nllb-salt-ct2-int8"
VITS_DIR = f"{HOME}/ml/vits-work/training"
VITS_VENV_PY = f"{HOME}/ml/vits-venv/bin/python"
TTS_VENV_PY = f"{HOME}/ml/tts-venv/bin/python"

MIC_DEVICE = "plughw:3,0"
SPEAKER_DEVICE = "plughw:2,0"
RECORD_SECONDS = 5
MIC_GAIN_PERCENT = 62
# Whisper normally pads every clip to 30 s, which is most of its cost on the Pi.
# The app records RECORD_SECONDS and the encoder gets one second more: a window of exactly
# the recording length made Whisper repeat sentences (see docs/06-fast-pipeline.md).
# FAST_WINDOW=30 runs the stock window for comparison.
WINDOW_SECONDS = float(os.environ.get("FAST_WINDOW", RECORD_SECONDS + 1))
# Whisper decoding settings found on six recorded clips (docs/06-fast-pipeline.md): no timestamp
# tokens and no repeated 3-grams, plus a cap on new tokens so a runaway loop cannot cost minutes.
# Override with FAST_NO_TIMESTAMPS=0, FAST_NO_REPEAT=0, FAST_MAX_TOKENS=<n>.
DECODE_OPTS = {"max_new_tokens": int(os.environ.get("FAST_MAX_TOKENS", "64"))}
if os.environ.get("FAST_NO_TIMESTAMPS", "1") == "1":
    DECODE_OPTS["without_timestamps"] = True
if int(os.environ.get("FAST_NO_REPEAT", "3")) > 0:
    DECODE_OPTS["no_repeat_ngram_size"] = int(os.environ.get("FAST_NO_REPEAT", "3"))
LOG_PATH = f"{HOME}/ml/demo_scratch/fast_pipeline_log.jsonl"
VITS_WORKER = f"{HOME}/ml/vits_worker.py"  # shared with the first app
PRELOADED_VOICES = ("English", "Luganda")  # kept in memory by the worker
# Character-level ONNX voice (jq/sherpa-vits-tts-lug-eng, as in the Sunbird app): one 109 MB model for
# English and Luganda, about 2x real time on the Pi 4 against 2.6-3.7x for the Sunbird VITS voices.
# FAST_VOICE=vits goes back to the Sunbird VITS worker for English and Luganda.
CHAR_VOICE_DIR = f"{PIPELINE2}/models/sherpa-vits-lug-eng"
CHAR_VOICE_LANGS = ("English", "Luganda")
USE_CHAR_VOICE = os.environ.get("FAST_VOICE", "onnx") == "onnx"

# ISO 639-3 codes. NLLB (SALT) covers only these four; Swahili is transcribe-only here.
ISO = {"English": "eng", "Luganda": "lug", "Runyankole": "nyn", "Acholi": "ach", "Swahili": "swh"}
TRANSLATABLE = {"English", "Luganda", "Runyankole", "Acholi"}

SCRATCH = f"{HOME}/ml/demo_scratch"
os.makedirs(SCRATCH, exist_ok=True)

# Languages offered in the UI. VITS entries point at a local Sunbird
# checkpoint; MMS entries give the facebook/mms-tts-<code> language code.
LANGUAGES = {
    "English": {"tts": "vits", "dir": f"{HOME}/ml/models/tts-vits-eng"},
    "Luganda": {"tts": "vits", "dir": f"{HOME}/ml/models/tts-vits-lug"},
    "Runyankole": {"tts": "vits", "dir": f"{HOME}/ml/models/tts-vits-nyn"},
    "Swahili": {"tts": "mms", "code": "swh"},
    "Acholi": {"tts": "mms", "code": "ach"},
}


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def record_audio(out_path, seconds=RECORD_SECONDS):
    # The USB mic reverts to 100% gain (noisy) after a reboot even with
    # `alsactl store`, so set it before every recording.
    card = MIC_DEVICE.split(":")[1].split(",")[0]
    run(["amixer", "-c", card, "sset", "Mic", f"{MIC_GAIN_PERCENT}%"])
    result = run([
        "arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(seconds), out_path,
    ])
    return result.returncode == 0


def play_audio(path):
    run(["aplay", "-D", SPEAKER_DEVICE, path])


def collapse_repeats(text):
    """Whisper on a short window sometimes says the sentence twice ("X. X."). Keep the first of
    each run of near-identical sentences (only sentences of 3+ words are compared)."""
    def norm(t):
        return re.sub(r"[^\w\s']", "", t.lower()).strip()
    kept = []
    for part in (p for p in re.split(r"(?<=[.?!])\s+", text.strip()) if p):
        if (kept and len(norm(part).split()) >= 3
                and difflib.SequenceMatcher(None, norm(kept[-1]), norm(part)).ratio() >= 0.8):
            continue
        kept.append(part)
    return " ".join(kept)


class Engine:
    """Whisper (speech -> text) and NLLB (translation), loaded once and kept in memory."""

    def __init__(self):
        sys.path.insert(0, PIPELINE2)
        import faster_whisper.transcribe as ft
        from faster_whisper import WhisperModel
        from nllb_translate import Translator

        frames = int(WINDOW_SECONDS * 100)  # 100 mel frames per second
        orig_pad = ft.pad_or_trim
        ft.pad_or_trim = lambda arr, length=3000, **kw: orig_pad(arr, frames, **kw)
        codes = json.load(open(f"{WHISPER_DIR}/language_map.json"))
        self.whisper_code = {name: codes[iso if iso != "swh" else "swa"] for name, iso in ISO.items()}
        self.whisper = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8", cpu_threads=4)
        self.translator = Translator(NLLB_DIR)

    def transcribe(self, audio_path, lang_name):
        """Speech -> text in the speaker's own language (no translation)."""
        segments, _ = self.whisper.transcribe(
            audio_path, language=self.whisper_code[lang_name], beam_size=1,
            condition_on_previous_text=False, **DECODE_OPTS)
        segments = list(segments)  # decoding happens while iterating
        self.last_tokens = sum(len(s.tokens) for s in segments)
        self.last_raw = " ".join(s.text.strip() for s in segments).strip()
        return collapse_repeats(self.last_raw)

    def translate(self, text, src_name, dst_name):
        out = self.translator.translate(" ".join(text.split()), ISO[src_name], ISO[dst_name])
        self.last_tokens = self.translator.last_tokens
        return out


def speak_text(text, lang_name, out_path):
    lang = LANGUAGES[lang_name]
    out_path = os.path.abspath(out_path)
    if lang["tts"] == "vits":
        result = run([
            VITS_VENV_PY, "run_inference.py", lang["dir"], text, out_path,
        ], cwd=VITS_DIR)
        return result.returncode == 0
    else:
        # Prefer a local copy (works offline); fall back to the Hugging Face hub.
        local = f"{HOME}/ml/models/mms-tts-{lang['code']}"
        ref = local if os.path.isdir(local) else f"facebook/mms-tts-{lang['code']}"
        script = f"""
from transformers import VitsModel, AutoTokenizer
import torch, scipy.io.wavfile
model = VitsModel.from_pretrained({ref!r})
tokenizer = AutoTokenizer.from_pretrained({ref!r})
inputs = tokenizer({text!r}, return_tensors="pt")
with torch.no_grad():
    output = model(**inputs).waveform
scipy.io.wavfile.write({out_path!r}, rate=model.config.sampling_rate, data=output.squeeze().numpy())
print("ok")
"""
        result = run([TTS_VENV_PY, "-c", script])
        return result.returncode == 0 and "ok" in result.stdout


class CharVoice:
    """In-process ONNX voice that reads plain characters (recipe from SunbirdAI/sunflower-app tts_engine.dart):
    lowercase the text, one token id per character, no blanks, scales [0.667, 1.0, 0.8], 22050 Hz."""
    RATE = 22050
    SCALES = (0.667, 1.0, 0.8)
    CHUNK_WORDS = 10

    def __init__(self, model_dir=CHAR_VOICE_DIR):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        self.tok = {}
        for line in open(f"{model_dir}/tokens.txt", encoding="utf8"):
            line = line.rstrip("\n")
            if line:
                sym, _, idx = line.rpartition(" ")
                self.tok[sym or " "] = int(idx)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.sess = ort.InferenceSession(f"{model_dir}/vits-lug-eng.fp32.onnx", opts,
                                         providers=["CPUExecutionProvider"])

    def _chunks(self, text):
        out = []
        for part in re.split(r"[.,!?;:]", text.lower()):
            words = part.split()
            out += [" ".join(words[i:i + self.CHUNK_WORDS]) for i in range(0, len(words), self.CHUNK_WORDS)]
        return out

    def speak(self, text, out_path):
        np = self.np
        t0 = time.perf_counter()
        pieces = []
        for chunk in self._chunks(text):
            ids = [self.tok[c] for c in chunk if c in self.tok]
            if not ids:
                continue
            audio = self.sess.run(None, {
                "input": np.array([ids], dtype=np.int64),
                "input_lengths": np.array([len(ids)], dtype=np.int64),
                "scales": np.array(self.SCALES, dtype=np.float32)})[0].squeeze()
            pieces += [audio, np.zeros(int(self.RATE * 0.15), dtype=audio.dtype)]
        if not pieces:
            return {"ok": False, "error": "no known characters"}
        audio = np.concatenate(pieces)
        from scipy.io import wavfile
        wavfile.write(out_path, self.RATE, (np.clip(audio, -1, 1) * 32767).astype(np.int16))
        return {"ok": True, "synth_s": round(time.perf_counter() - t0, 2), "audio_s": round(len(audio) / self.RATE, 2)}


class VitsWorker:
    """Background process that keeps the Sunbird VITS voices loaded (see vits_worker.py)."""

    def __init__(self, model_dirs):
        self.lock = threading.Lock()
        self.dirs = set(model_dirs)
        self.proc = subprocess.Popen(
            [VITS_VENV_PY, VITS_WORKER, *model_dirs], cwd=VITS_DIR, text=True, bufsize=1,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=open(f"{SCRATCH}/vits_worker.log", "w"))
        self.ready = self._read("READY")  # blocks until the voices are loaded

    def _read(self, tag):
        for line in self.proc.stdout:
            if line.startswith(f"@@{tag} "):
                return json.loads(line[len(tag) + 3:])
        raise RuntimeError("VITS worker exited")

    def speak(self, model_dir, text, out_path):
        with self.lock:
            self.proc.stdin.write(json.dumps({"dir": model_dir, "text": text, "out": out_path}) + "\n")
            self.proc.stdin.flush()
            return self._read("RESULT")


class SunflowerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sunflower Translator (fast pipeline)")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#1a1a1a")

        self.src_lang = tk.StringVar(value="English")
        self.dst_lang = tk.StringVar(value="Luganda")
        self.mode = tk.StringVar(value="Translate")  # "Translate" or "Transcribe"
        self.busy = False

        self._build_ui()

        self.engine = None
        self.vits = None
        self.char_voice = None
        self.last_voice = {}
        self.last_counts = {}
        self.busy = True
        self.record_btn.config(state="disabled", bg="#555")
        self.status_var.set("Loading models... (about 45 s)")
        threading.Thread(target=self._load_engine, daemon=True).start()

    def _load_engine(self):
        t0 = time.perf_counter()
        try:
            self.engine = Engine()
            self.set_status("Loading voices (English, Luganda)...")
            if USE_CHAR_VOICE:
                try:
                    self.char_voice = CharVoice()
                except Exception as e:  # fall back to the Sunbird VITS worker
                    print("ONNX voice unavailable:", repr(e), flush=True)
            if self.char_voice is None:
                try:
                    self.vits = VitsWorker([LANGUAGES[n]["dir"] for n in PRELOADED_VOICES])
                except Exception as e:  # fall back to loading a voice on every tap
                    print("voice worker unavailable:", repr(e), flush=True)
        except Exception as e:  # show the problem on screen instead of a silent dead app
            self.set_status(f"Model load failed: {type(e).__name__}")
            print("model load failed:", repr(e), flush=True)
            return
        self.busy = False
        self.set_status(f"Ready (models loaded in {time.perf_counter() - t0:.0f} s)")
        self.ui(lambda: self.record_btn.config(state="normal", bg="#ffaa28"))

    def _build_ui(self):
        # Sunbird AI brand colors (extracted from sunbird.ai stylesheet)
        FG = "#f0f0f0"
        BG = "#1a1a1a"
        ORANGE = "#ffaa28"
        ORANGE_DARK = "#d98b43"
        DARK_PANEL = "#111111"

        # Title bar: Sunflower icon + name on the left, Sunbird AI logo and
        # quit button on the right. Logos are pre-sized PNGs in ./assets; if a
        # file is missing the bar falls back to text only.
        title_bar = tk.Frame(self.root, bg=BG, height=36)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        self._logos = []  # keep references so Tk doesn't garbage-collect images
        assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

        def load_logo(name):
            try:
                img = tk.PhotoImage(file=os.path.join(assets, name))
                self._logos.append(img)
                return img
            except tk.TclError:
                return None

        flower = load_logo("sunflower-icon-24.png")
        if flower:
            tk.Label(title_bar, image=flower, bg=BG).pack(side="left", padx=(10, 4))
        tk.Label(title_bar, text="Sunflower Fast", bg=BG, fg=ORANGE,
                 font=("DejaVu Sans", 14, "bold")).pack(side="left", padx=(0 if flower else 10, 0))

        # The touch panel doesn't register the outer ~25px, so keep the quit
        # button well inside the right edge.
        exit_btn = tk.Button(title_bar, text="x", font=("DejaVu Sans", 11, "bold"), fg=FG,
                              bg=BG, activebackground="#333", relief="flat", bd=0,
                              width=3, command=self.root.destroy)
        exit_btn.pack(side="right", padx=(2, 22), pady=2)

        sunbird = load_logo("sunbird-logo-20.png")
        if sunbird:
            tk.Label(title_bar, image=sunbird, bg=BG).pack(side="right", padx=(0, 6))

        # Mode toggle: Translate (speech -> text -> translation -> speech) or
        # Transcribe (speech -> text in the same language, shown on screen).
        mode_row = tk.Frame(self.root, bg=BG)
        mode_row.pack(fill="x", padx=10, pady=(2, 2))
        mode_row.columnconfigure(0, weight=1)
        mode_row.columnconfigure(1, weight=1)
        self.mode_btns = {}
        for col, name in enumerate(("Translate", "Transcribe")):
            btn = tk.Button(
                mode_row, text=name, font=("DejaVu Sans", 11, "bold"),
                relief="flat", bd=0, command=lambda n=name: self._set_mode(n),
            )
            btn.grid(row=0, column=col, sticky="nsew", padx=(0, 2) if col == 0 else (2, 0), ipady=3)
            self.mode_btns[name] = btn

        # Language selectors. Tk popup menus are separate windows and don't
        # work reliably with touch under XWayland, so each button opens a
        # full-screen picker drawn inside the app instead (see _open_picker).
        lang_row = tk.Frame(self.root, bg=BG)
        lang_row.pack(fill="x", padx=10, pady=(2, 4))
        lang_row.columnconfigure(0, weight=1)
        lang_row.columnconfigure(1, weight=1)

        self.src_btn = tk.Button(
            lang_row, textvariable=self._labeled(self.src_lang, "Speak: "),
            font=("DejaVu Sans", 13, "bold"), bg=DARK_PANEL, fg=FG,
            activebackground="#222", activeforeground=FG, relief="flat", bd=0,
            command=lambda: self._open_picker("Speak in:", self.src_lang),
        )
        self.src_btn.grid(row=0, column=0, sticky="nsew", padx=(0, 4), ipady=6)

        self.dst_btn = tk.Button(
            lang_row, textvariable=self._labeled(self.dst_lang, "To: "),
            font=("DejaVu Sans", 13, "bold"), bg=DARK_PANEL, fg=ORANGE,
            activebackground="#222", activeforeground=ORANGE, relief="flat", bd=0,
            command=lambda: self._open_picker("Translate to:", self.dst_lang),
        )
        self.dst_btn.grid(row=0, column=1, sticky="nsew", padx=(4, 0), ipady=6)

        self.record_btn = tk.Button(
            self.root, text="TAP TO SPEAK", font=("DejaVu Sans", 18, "bold"),
            bg=ORANGE, fg="#1a1a1a", activebackground=ORANGE_DARK,
            relief="flat", command=self.on_record_tap,
        )
        self.record_btn.pack(expand=True, fill="both", padx=14, pady=4)

        # Results panel: what the model heard, and (in Translate mode) its translation.
        bottom = tk.Frame(self.root, bg=DARK_PANEL, height=122)
        bottom.pack(fill="x", side="bottom")
        bottom.pack_propagate(False)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(bottom, textvariable=self.status_var, bg=DARK_PANEL, fg=ORANGE,
                  font=("DejaVu Sans", 9), anchor="w").pack(fill="x", padx=10, pady=(3, 0))

        self.heard_title = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.heard_title, bg=DARK_PANEL, fg="#8a8a8a",
                  font=("DejaVu Sans", 8), anchor="w").pack(fill="x", padx=10)
        self.heard_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.heard_var, bg=DARK_PANEL, fg=FG,
                  font=("DejaVu Sans", 11), anchor="w", justify="left",
                  wraplength=440).pack(fill="x", padx=10)

        self.result_title = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.result_title, bg=DARK_PANEL, fg="#8a8a8a",
                  font=("DejaVu Sans", 8), anchor="w").pack(fill="x", padx=10)
        self.result_var = tk.StringVar(value="")
        tk.Label(bottom, textvariable=self.result_var, bg=DARK_PANEL, fg=ORANGE,
                  font=("DejaVu Sans", 12, "bold"), anchor="w", justify="left",
                  wraplength=440).pack(fill="x", padx=10)

        self._set_mode("Translate")

    def _open_picker(self, title, var):
        """Full-window language chooser: big tap targets, no popup windows."""
        ORANGE, BG, PANEL, FG = "#ffaa28", "#1a1a1a", "#111111", "#f0f0f0"

        overlay = tk.Frame(self.root, bg=BG)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()

        tk.Label(overlay, text=title, bg=BG, fg=ORANGE,
                 font=("DejaVu Sans", 14, "bold")).pack(pady=(8, 4))

        grid = tk.Frame(overlay, bg=BG)
        grid.pack(expand=True, fill="both", padx=16)
        cols = 2
        names = list(LANGUAGES.keys())
        rows = (len(names) + cols - 1) // cols
        for c in range(cols):
            grid.columnconfigure(c, weight=1, uniform="col")
        for r in range(rows):
            grid.rowconfigure(r, weight=1, uniform="row")

        def choose(name):
            var.set(name)
            overlay.destroy()

        for i, name in enumerate(names):
            selected = (name == var.get())
            tk.Button(
                grid, text=name, font=("DejaVu Sans", 14, "bold"),
                bg=ORANGE if selected else PANEL,
                fg="#1a1a1a" if selected else FG,
                activebackground="#d98b43", activeforeground="#1a1a1a",
                relief="flat", bd=0, command=lambda n=name: choose(n),
            ).grid(row=i // cols, column=i % cols, sticky="nsew", padx=4, pady=4)

        tk.Button(
            overlay, text="Cancel", font=("DejaVu Sans", 12),
            bg=BG, fg="#aaaaaa", activebackground="#333", activeforeground=FG,
            relief="flat", bd=0, command=overlay.destroy,
        ).pack(fill="x", padx=16, pady=(0, 8), ipady=6)

    def _labeled(self, var, prefix):
        """Returns a StringVar that mirrors `var` but displayed with a prefix."""
        labeled = tk.StringVar(value=f"{prefix}{var.get()}")
        var.trace_add("write", lambda *_: labeled.set(f"{prefix}{var.get()}"))
        return labeled

    def _set_mode(self, mode):
        """Switch between Translate and Transcribe; restyle buttons, hide 'To:'."""
        if self.busy:
            return
        self.mode.set(mode)
        for name, btn in self.mode_btns.items():
            on = (name == mode)
            btn.config(bg="#ffaa28" if on else "#111111",
                       fg="#1a1a1a" if on else "#cccccc",
                       activebackground="#d98b43" if on else "#222222")
        if mode == "Transcribe":
            self.dst_btn.config(state="disabled", fg="#555555", disabledforeground="#555555")
        else:
            self.dst_btn.config(state="normal", fg="#ffaa28")
        self._show_texts("", "", "", "")
        self.status_var.set("Ready")

    def _show_texts(self, heard_title, heard, result_title, result):
        self.heard_title.set(heard_title)
        self.heard_var.set(heard)
        self.result_title.set(result_title)
        self.result_var.set(result)

    def ui(self, fn, *args):
        """Run a UI update on Tk's main thread (safe to call from the worker)."""
        self.root.after(0, lambda: fn(*args))

    def set_status(self, text):
        self.ui(self.status_var.set, text)

    def on_record_tap(self):
        if self.busy:
            return
        self.busy = True
        self.record_btn.config(state="disabled", bg="#555")
        # Read Tk variables here on the main thread; the worker must not touch them.
        threading.Thread(
            target=self._run_pipeline,
            args=(self.mode.get(), self.src_lang.get(), self.dst_lang.get()),
            daemon=True,
        ).start()

    def _run_pipeline(self, mode, src, dst):
        times, heard, translated = {}, "", ""
        self.last_voice = {}
        self.last_counts = {}
        try:
            self.ui(self._show_texts, "", "", "", "")
            if mode == "Translate" and (src not in TRANSLATABLE or dst not in TRANSLATABLE):
                self.set_status("Swahili isn't covered by the translation model. Use Transcribe.")
                return

            self.set_status(f"Recording... ({RECORD_SECONDS}s) - speak now")
            audio_path = f"{SCRATCH}/ui_input.wav"
            if not record_audio(audio_path):
                self.set_status("Recording failed.")
                return

            # Step 1: speech -> text in the speaker's own language.
            self.set_status(f"Listening ({src})...")
            t0 = time.perf_counter()
            heard = self.engine.transcribe(audio_path, src)
            times["Listen"] = time.perf_counter() - t0
            self.last_counts["listen_tokens"] = self.engine.last_tokens
            if self.engine.last_raw != heard:
                self.last_counts["heard_raw"] = self.engine.last_raw
            if not heard:
                self.set_status("Couldn't make out any speech.")
                return
            self.ui(self._show_texts, f"Heard ({src}):", heard, "", "")

            if mode == "Transcribe":
                self.set_status(self._summary(times))
                return

            # Step 2: text -> translation (skipped if source == target language).
            if src == dst:
                translated = heard
            else:
                self.set_status(f"Translating to {dst}...  ({self._summary(times)})")
                t0 = time.perf_counter()
                translated = self.engine.translate(heard, src, dst)
                times["Translate"] = time.perf_counter() - t0
                self.last_counts["translate_tokens"] = self.engine.last_tokens
                if not translated:
                    self.set_status("Translation failed.")
                    return
            self.ui(self._show_texts, f"Heard ({src}):", heard,
                    f"Translation ({dst}):", translated)

            # Step 3: speak the translation (same voices as the first app).
            self.set_status(f"Speaking ({dst})...  ({self._summary(times)})")
            t0 = time.perf_counter()
            out_path = f"{SCRATCH}/ui_output.wav"
            ok = self._speak(translated, dst, out_path)
            times["Voice"] = time.perf_counter() - t0
            if ok:
                play_audio(out_path)
                self.set_status(self._summary(times))
            else:
                self.set_status("Speech synthesis failed.")
        finally:
            self._log(mode, src, dst, times, heard, translated)
            self.busy = False
            self.ui(lambda: self.record_btn.config(state="normal", bg="#ffaa28"))

    def _speak(self, text, lang_name, out_path):
        """Use the ONNX voice (or the preloaded worker) for English/Luganda; otherwise (or on error) load per tap."""
        if self.char_voice and lang_name in CHAR_VOICE_LANGS:
            try:
                res = self.char_voice.speak(text, out_path)
                self.last_voice = {"path": "onnx", **res}
                if res.get("ok"):
                    return True
            except Exception as e:
                self.last_voice = {"path": "onnx", "ok": False, "error": repr(e)[:150]}
        if self.vits and lang_name in PRELOADED_VOICES:
            try:
                res = self.vits.speak(LANGUAGES[lang_name]["dir"], text, out_path)
                self.last_voice = {"path": "worker", **res}
                if res.get("ok"):
                    return True
            except Exception as e:
                self.last_voice = {"path": "worker", "ok": False, "error": repr(e)[:150]}
        self.last_voice.setdefault("path", "per-tap")
        return speak_text(text, lang_name, out_path)

    @staticmethod
    def _summary(times):
        total = sum(times.values())
        return "Done: " + " | ".join(f"{k} {v:.1f}s" for k, v in times.items()) + f" | total {total:.1f}s"

    def _log(self, mode, src, dst, times, heard, translated):
        """One JSON line per run, so the two pipelines can be compared afterwards."""
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline": "whisper+nllb",
                    "window_s": WINDOW_SECONDS, "decode_opts": DECODE_OPTS, "mode": mode, "src": src, "dst": dst,
                    "seconds": {k: round(v, 2) for k, v in times.items()},
                    "voice": self.last_voice, "tokens": self.last_counts,
                    "heard": heard, "translated": translated}, ensure_ascii=False) + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SunflowerApp(root)
    root.mainloop()
