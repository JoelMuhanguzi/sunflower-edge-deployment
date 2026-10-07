#!/usr/bin/env python3
"""
Sunflower Chat: talk or type to Sunflower-Gemma4-E2B on the 480x320 touchscreen.

Input is either speech (record 5 s, shown as text) or typing on an on-screen keyboard.
The conversation is kept, the reply appears word by word as it is generated, and can
optionally be spoken (English/Luganda/Runyankole voices load on first use).

It shares the first app's Gemma server (sunflower_touch_ui.GemmaServer): if the first app is
already open, Chat attaches to its server; otherwise Chat starts one (context 4096). Whichever
app started the server stops it when it closes, so close Chat last if you opened it first.
Not for use together with Sunflower Fast (memory). Each reply's timing is logged, including
time to the first word and tokens per second.

CHAT_MOCK=1 replaces the model with a canned one for previewing the layout on a desktop;
CHAT_WINDOWED=1 opens a window instead of fullscreen.
"""

import json
import os
import sys
import threading
import time
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sunflower_touch_ui as base  # shared server, voices, recording, constants

MOCK = os.environ.get("CHAT_MOCK") == "1"
WINDOWED = os.environ.get("CHAT_WINDOWED") == "1"
CONTEXT = 4096
HISTORY_MESSAGES = 6        # last turns sent with each question (3 questions + 3 answers)
MAX_REPLY_TOKENS = 200
SPEAK_CHARS = 250           # a spoken reply is cut near this length (speech is slow)
LOG_PATH = f"{base.SCRATCH}/chat_log.jsonl"

BG, PANEL, FG = "#1a1a1a", "#111111", "#f0f0f0"
ORANGE, ORANGE_DARK, GREY = "#ffaa28", "#d98b43", "#8a8a8a"


class MockServer:
    """Stand-in model for previewing the UI without the Pi."""

    def transcribe(self, audio_path, lang_name):
        time.sleep(0.6)
        return "What is photosynthesis?"

    def stream_chat(self, messages, **kw):
        for word in "Photosynthesis is how plants turn sunlight, water and air into food.".split():
            time.sleep(0.15)
            yield word + " "


def instruction(lang):
    return (f"You are Sunflower, a friendly assistant from Sunbird AI. Reply in {lang}. "
            f"Keep answers short: one to three sentences.")


def first_sentences(text, limit=SPEAK_CHARS):
    """The start of a reply, ended at a sentence boundary when one is near the limit."""
    if len(text) <= limit:
        return text
    cut = max(text.rfind(c, 0, limit) for c in ".?!")
    return text[:cut + 1] if cut > 40 else text[:limit]


class ChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sunflower Chat")
        if not WINDOWED:
            self.root.attributes("-fullscreen", True)
        self.root.configure(bg=BG)

        self.lang = tk.StringVar(value="English")
        self._chat_lang = "English"  # language of the current conversation
        self.input_mode = "Speak"       # "Speak" or "Type"
        self.voice_reply = False
        self.history = []               # [{"role": "user"|"assistant", "content": text}]
        self.server = None
        self.vits = None
        self.busy = True

        self._build_ui()
        self.action_btn.config(state="disabled", bg="#555")
        self.status_var.set("Loading model... (1 to 2 minutes)")
        threading.Thread(target=self._load, daemon=True).start()

    # ---------------------------------------------------------------- start-up
    def _load(self):
        t0 = time.perf_counter()
        try:
            if MOCK:
                self.server = MockServer()
                base.record_audio = lambda *a, **k: True
            else:
                probe = base.GemmaServer.__new__(base.GemmaServer)
                probe.url = f"http://127.0.0.1:{base.SERVER_PORT}"
                if not probe._healthy():  # we will have to start it, so check the memory first
                    free = base.mem_available_gb()
                    if free < base.MIN_FREE_GB:
                        self.set_status(f"Only {free:.1f} GB free. Close the other Sunflower app, then reopen this one.")
                        return
                self.server = base.GemmaServer(ctx=CONTEXT)
        except Exception as e:
            self.set_status(f"Model load failed: {type(e).__name__}")
            print("model load failed:", repr(e), flush=True)
            return
        self.busy = False
        self.set_status(f"Ready (loaded in {time.perf_counter() - t0:.0f} s)")
        self.ui(lambda: self.action_btn.config(state="normal", bg=ORANGE))

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        title_bar = tk.Frame(self.root, bg=BG, height=36)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        self._logos = []
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
        tk.Label(title_bar, text="Sunflower Chat", bg=BG, fg=ORANGE,
                 font=("DejaVu Sans", 14, "bold")).pack(side="left", padx=(0 if flower else 10, 0))
        # The touch panel does not register the outer ~25 px, so keep the quit button inset.
        tk.Button(title_bar, text="x", font=("DejaVu Sans", 11, "bold"), fg=FG, bg=BG,
                  activebackground="#333", relief="flat", bd=0, width=3,
                  command=self.root.destroy).pack(side="right", padx=(2, 22), pady=2)
        sunbird = load_logo("sunbird-logo-20.png")
        if sunbird:
            tk.Label(title_bar, image=sunbird, bg=BG).pack(side="right", padx=(0, 6))

        controls = tk.Frame(self.root, bg=BG)
        controls.pack(fill="x", padx=10, pady=(2, 2))
        for col in range(4):
            controls.columnconfigure(col, weight=1, uniform="ctl")
        style = dict(font=("DejaVu Sans", 9, "bold"), bg=PANEL, fg=FG, activebackground="#222",
                     activeforeground=FG, relief="flat", bd=0)
        lang_style = {**style, "fg": ORANGE}
        self.mode_btn = tk.Button(controls, text="Input: Speak", command=self._toggle_mode, **style)
        self.lang_btn = tk.Button(controls, text="Chat: English", command=self._pick_language, **lang_style)
        self.voice_btn = tk.Button(controls, text="Voice reply: Off", command=self._toggle_voice, **style)
        self.new_btn = tk.Button(controls, text="New chat", command=self._new_chat, **style)
        for col, btn in enumerate((self.mode_btn, self.lang_btn, self.voice_btn, self.new_btn)):
            btn.grid(row=0, column=col, sticky="nsew", padx=2, ipady=4)
        self.lang.trace_add("write", lambda *_: self.lang_btn.config(text=f"Chat: {self.lang.get()}"))
        self.lang.trace_add("write", lambda *_: self._on_lang_change())

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var, bg=BG, fg=ORANGE, anchor="w",
                 font=("DejaVu Sans", 8)).pack(fill="x", padx=12)

        self.action_btn = tk.Button(
            self.root, text="TAP TO SPEAK", font=("DejaVu Sans", 16, "bold"), bg=ORANGE, fg=BG,
            activebackground=ORANGE_DARK, relief="flat", command=self._on_action)
        self.action_btn.pack(side="bottom", fill="x", padx=14, pady=(2, 8), ipady=6)

        self.chat = tk.Text(self.root, bg=PANEL, fg=FG, font=("DejaVu Sans", 10), wrap="word",
                            relief="flat", bd=0, padx=8, pady=4, state="disabled", cursor="arrow")
        self.chat.pack(fill="both", expand=True, padx=12, pady=(2, 2))
        self.chat.tag_configure("you", foreground="#cfcfcf")
        self.chat.tag_configure("bot", foreground=ORANGE)
        self.chat.tag_configure("who", foreground=GREY, font=("DejaVu Sans", 8, "bold"))

    # ---------------------------------------------------------------- small helpers
    def ui(self, fn, *args):
        """Run a UI update on Tk's main thread (safe to call from a worker)."""
        self.root.after(0, lambda: fn(*args))

    def set_status(self, text):
        self.ui(self.status_var.set, text)

    def _write(self, text, tag):
        self.chat.config(state="normal")
        self.chat.insert("end", text, tag)
        self.chat.see("end")
        self.chat.config(state="disabled")

    def _say(self, who, text, tag):
        self._write(f"{who}\n", "who")
        self._write(text, tag)

    def _toggle_mode(self):
        if self.busy:
            return
        self.input_mode = "Type" if self.input_mode == "Speak" else "Speak"
        self.mode_btn.config(text=f"Input: {self.input_mode}")
        self.action_btn.config(text="TAP TO SPEAK" if self.input_mode == "Speak" else "TAP TO TYPE")

    def _toggle_voice(self):
        if self.busy:
            return
        self.voice_reply = not self.voice_reply
        self.voice_btn.config(text=f"Voice reply: {'On' if self.voice_reply else 'Off'}",
                              fg=ORANGE if self.voice_reply else FG)
        if self.voice_reply:
            self.status_var.set("Voices load on the first spoken reply (about 30 s).")

    def _clear_chat(self, status):
        self.history = []
        self.chat.config(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.config(state="disabled")
        self.status_var.set(status)

    def _new_chat(self):
        if not self.busy:
            self._clear_chat("New chat")

    def _on_lang_change(self):
        """The instruction at the start of the conversation names the language, so changing it
        would make the server reread the whole history (about 25 s) in the wrong language anyway."""
        if self.lang.get() != self._chat_lang:
            self._chat_lang = self.lang.get()
            if self.history:
                self._clear_chat(f"Language changed: new {self._chat_lang} chat")

    def _pick_language(self):
        if self.busy:
            return
        # The first app's full-window picker only needs self.root, so it is reused as it is.
        base.SunflowerApp._open_picker(self, "Chat in:", self.lang)

    # ---------------------------------------------------------------- input
    def _on_action(self):
        if self.busy:
            return
        if self.input_mode == "Speak":
            self._start_turn("voice", None)
        else:
            self._open_keyboard()

    def _start_turn(self, kind, text):
        self.busy = True
        self.action_btn.config(state="disabled", bg="#555")
        # Read Tk variables here on the main thread; the worker must not touch them.
        threading.Thread(target=self._run_turn, args=(kind, text, self.lang.get(), self.voice_reply),
                         daemon=True).start()

    def _open_keyboard(self):
        """Full-window on-screen keyboard (no popup windows: they do not work with touch here)."""
        overlay = tk.Frame(self.root, bg=BG)
        overlay.place(x=0, y=0, relwidth=1, relheight=1)
        overlay.lift()
        typed, shift = [""], [False]

        top = tk.Frame(overlay, bg=BG)
        top.pack(fill="x", padx=22, pady=(8, 4))
        shown = tk.StringVar(value="|")
        tk.Label(top, textvariable=shown, bg=PANEL, fg=FG, font=("DejaVu Sans", 11), anchor="w",
                 justify="left", wraplength=330, height=2).pack(side="left", fill="x", expand=True)
        tk.Button(top, text="Cancel", font=("DejaVu Sans", 10), bg=BG, fg="#aaaaaa", relief="flat", bd=0,
                  activebackground="#333", command=overlay.destroy).pack(side="right", padx=(6, 0), ipady=8)

        def refresh():
            tail = typed[0][-70:]
            shown.set(tail + "|")

        keys = []

        def press(ch):
            typed[0] += ch.upper() if (shift[0] and ch.isalpha()) else ch
            if shift[0]:
                shift[0] = False
                relabel()
            refresh()

        def relabel():
            for btn, ch in keys:
                btn.config(text=ch.upper() if shift[0] and ch.isalpha() else ch)

        def backspace():
            typed[0] = typed[0][:-1]
            refresh()

        def send():
            text = typed[0].strip()
            overlay.destroy()
            if text:
                self._start_turn("typed", text)

        grid = tk.Frame(overlay, bg=BG)
        grid.pack(fill="both", expand=True, padx=22, pady=(0, 6))
        grid.columnconfigure(0, weight=1)
        rows = ["1234567890", "qwertyuiop", "asdfghjkl'", "zxcvbnm,.?ŋ"]
        for r, row in enumerate(rows):
            grid.rowconfigure(r, weight=1)
            frame = tk.Frame(grid, bg=BG)
            frame.grid(row=r, column=0, sticky="nsew")
            for c, ch in enumerate(row):
                frame.columnconfigure(c, weight=1, uniform=f"r{r}")
                btn = tk.Button(frame, text=ch, font=("DejaVu Sans", 12, "bold"), bg=PANEL, fg=FG,
                                activebackground=ORANGE_DARK, activeforeground=BG, relief="flat", bd=0,
                                command=lambda ch=ch: press(ch))
                btn.grid(row=0, column=c, sticky="nsew", padx=1, pady=1)
                keys.append((btn, ch))
            frame.rowconfigure(0, weight=1)
        grid.rowconfigure(len(rows), weight=1)
        bottom = tk.Frame(grid, bg=BG)
        bottom.grid(row=len(rows), column=0, sticky="nsew")
        bottom.rowconfigure(0, weight=1)
        spec = [("⇧", 1, lambda: (shift.__setitem__(0, not shift[0]), relabel())),
                ("space", 4, lambda: press(" ")), ("⌫", 1, backspace), ("Send", 2, send)]
        for c, (label, weight, cmd) in enumerate(spec):
            bottom.columnconfigure(c, weight=weight)
            tk.Button(bottom, text=label, font=("DejaVu Sans", 11, "bold"),
                      bg=ORANGE if label == "Send" else PANEL, fg=BG if label == "Send" else FG,
                      activebackground=ORANGE_DARK, relief="flat", bd=0, command=cmd
                      ).grid(row=0, column=c, sticky="nsew", padx=1, pady=1)

    # ---------------------------------------------------------------- one question and answer
    def _messages(self, lang):
        """The recent turns, starting with a user turn that carries the instruction."""
        recent = self.history[-HISTORY_MESSAGES:]
        while recent and recent[0]["role"] != "user":
            recent = recent[1:]
        msgs = [dict(m) for m in recent]
        if msgs:
            msgs[0]["content"] = f"{instruction(lang)}\n\n{msgs[0]['content']}"
        return msgs

    def _run_turn(self, kind, text, lang, speak):
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "lang": lang, "speak": speak}
        reply = ""
        try:
            if kind == "voice":
                self.set_status(f"Recording... ({base.RECORD_SECONDS}s) - speak now")
                audio_path = f"{base.SCRATCH}/chat_input.wav"
                if not base.record_audio(audio_path):
                    self.set_status("Recording failed.")
                    return
                self.set_status(f"Listening ({lang})...")
                t0 = time.perf_counter()
                text = self.server.transcribe(audio_path, lang)
                rec["listen_s"] = round(time.perf_counter() - t0, 2)
                if not text:
                    self.set_status("Couldn't make out any speech.")
                    return
            rec["question"] = text
            self.ui(self._say, "You", text + "\n", "you")
            self.history.append({"role": "user", "content": text})

            self.set_status("Thinking...")
            self.ui(self._say, "Sunflower", "", "bot")
            t_start, t_first, n = time.perf_counter(), None, 0
            for piece in self.server.stream_chat(self._messages(lang), max_tokens=MAX_REPLY_TOKENS):
                if t_first is None:
                    t_first = time.perf_counter()
                    self.set_status("Replying...")
                n += 1
                reply += piece
                self.ui(self._write, piece, "bot")
            t_end = time.perf_counter()
            self.ui(self._write, "\n\n", "bot")
            reply = reply.strip()
            self.history.append({"role": "assistant", "content": reply})
            rec.update({"reply": reply, "tokens": n,
                        "first_word_s": round((t_first or t_end) - t_start, 2),
                        "total_s": round(t_end - t_start, 2),
                        "tokens_per_s": round((n - 1) / (t_end - t_first), 2) if t_first and n > 1 and t_end > t_first else None})
            summary = f"First word {rec['first_word_s']}s | {n} tokens"
            if rec["tokens_per_s"]:
                summary += f" at {rec['tokens_per_s']} tok/s"
            self.set_status(summary + f" | {rec['total_s']}s")

            if speak and reply:
                self._speak(first_sentences(reply), lang, rec)
        except Exception as e:  # for example the Gemma server stopped: say so instead of dying silently
            self.set_status(f"Error: {type(e).__name__}")
            print("chat error:", repr(e), flush=True)
            rec["error"] = repr(e)[:200]
        finally:
            self._log(rec)
            self.busy = False
            self.ui(lambda: self.action_btn.config(state="normal", bg=ORANGE))

    def _speak(self, text, lang, rec):
        """Say the start of the reply. Voices for English/Luganda are loaded once, on first use."""
        t0 = time.perf_counter()
        out_path = f"{base.SCRATCH}/chat_reply.wav"
        if MOCK:
            time.sleep(0.5)
            return
        try:
            if lang in base.PRELOADED_VOICES and self.vits is None:
                self.set_status("Loading voices (English, Luganda)...")
                self.vits = base.VitsWorker([base.LANGUAGES[n]["dir"] for n in base.PRELOADED_VOICES])
            self.set_status(f"Speaking ({lang})...")
            if self.vits and lang in base.PRELOADED_VOICES:
                ok = bool(self.vits.speak(base.LANGUAGES[lang]["dir"], text, out_path).get("ok"))
            else:
                ok = base.speak_text(text, lang, out_path)
            if ok:
                base.play_audio(out_path)
        except Exception as e:
            print("speech error:", repr(e), flush=True)
            ok = False
        rec["voice_s"] = round(time.perf_counter() - t0, 2)
        self.set_status(f"Done | voice {rec['voice_s']}s")

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
    app = ChatApp(root)
    root.mainloop()
