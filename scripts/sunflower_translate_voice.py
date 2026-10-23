#!/usr/bin/env python3
"""
Sunflower Translate (hands-free): say "Hey Sunflower", say something in English, hear it in Luganda.

  wake word ("sunflower", found by Moonshine in a short stretch of speech)  ->  chime  ->  Moonshine streaming (int8, English) writes down what was said
             ->  Gemma (kept loaded, text only, no audio encoder) translates it into Luganda
             ->  the ONNX voice speaks the Luganda, sentence by sentence

This is pipeline G of Part 10 (the fastest measured on the Pi 5) wrapped in the hands-free loop of Sunflower Voice.
It reuses that app's screen, wake word and recording, so the same notes apply: the screen is optional, a chime
marks the wake word, three rising notes mean it is loaded. Run it with the Moonshine Python environment:

  ~/ml/moonshine-venv/bin/python sunflower_translate_voice.py

MOONSHINE_SIZE=tiny|small|medium (default medium) picks the recogniser. TRANSLATE_TARGET=English would translate
Luganda speech instead only if a Luganda recogniser were available; Moonshine is English only, so the direction is fixed.
Every turn is logged to demo_scratch/translate_voice_log.jsonl with the stage times.
"""

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sunflower_voice_ui as vui  # screen, wake word, recording, speaker queue
from sunflower_voice_ui import base, np, tk

MOONSHINE_SIZE = os.environ.get("MOONSHINE_SIZE", "medium")
MOONSHINE_DIR = os.environ.get("MOONSHINE_DIR", os.path.expanduser(f"~/ml/moonshine_models/{MOONSHINE_SIZE}"))
LOG_PATH = f"{base.SCRATCH}/translate_voice_log.jsonl"
TEXT_ONLY_PORT = base.SERVER_PORT


def start_text_only_server():
    """Gemma without its audio encoder (about 1 GB less memory). Returns the process, or None if one is already running."""
    url = f"http://127.0.0.1:{TEXT_ONLY_PORT}/health"
    try:
        if b"ok" in urllib.request.urlopen(url, timeout=3).read():
            return None
    except Exception:
        pass
    proc = subprocess.Popen(
        [base.LLAMA_SERVER, "-m", base.TEXT_MODEL, "--jinja", "--host", "127.0.0.1",
         "--port", str(TEXT_ONLY_PORT), "-c", "2048", "-t", "4"],
        stdout=open(f"{base.SCRATCH}/llama_server_text.log", "w"), stderr=subprocess.STDOUT)
    deadline = time.time() + 600
    while True:
        try:
            if b"ok" in urllib.request.urlopen(url, timeout=3).read():
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("llama-server exited")
        if time.time() > deadline:
            raise RuntimeError("llama-server did not become ready")
        time.sleep(3)


class TranslateApp(vui.VoiceApp):
    def __init__(self, root):
        self.srv_proc = None
        super().__init__(root)
        self.root.title("Sunflower Translate")

    # the base class builds a language toggle; this app is fixed English -> Luganda
    def _toggle_lang(self):
        pass

    def _load(self):
        import torch
        from transformers import AutoProcessor, MoonshineStreamingForConditionalGeneration
        t0 = time.perf_counter()
        vui.set_mic_levels()
        self.status("Loading Gemma (text only)...")
        self.srv_proc = start_text_only_server()
        self.server = base.GemmaServer(ctx=2048)  # finds the server above and reuses it
        self.status("Loading Moonshine...")
        torch.set_num_threads(4)
        model = MoonshineStreamingForConditionalGeneration.from_pretrained(MOONSHINE_DIR).float().eval()
        torch.backends.quantized.engine = "qnnpack"
        self.ms_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.ms_proc = AutoProcessor.from_pretrained(MOONSHINE_DIR)
        self.torch = torch
        self.status("Loading voice...")
        self.voice = base.CharVoice()
        vui.make_chime()
        vui.make_chime(vui.READY, ((523, 0.12), (659, 0.12), (784, 0.25)))
        self.status(f"Ready (loaded in {time.perf_counter() - t0:.0f} s)")

    def close(self):
        if self.srv_proc and self.srv_proc.poll() is None:
            self.srv_proc.terminate()
        super().close()

    def _hear(self, wav_path):
        """English speech -> text with Moonshine (a wav file)."""
        from scipy.io import wavfile
        rate, data = wavfile.read(wav_path)
        return self._hear_array(data.astype("float32") / 32768.0)

    def _hear_array(self, audio):
        """English speech -> text with Moonshine (float32 audio at 16 kHz)."""
        sr = self.ms_proc.feature_extractor.sampling_rate
        inputs = self.ms_proc(audio, return_tensors="pt", sampling_rate=sr)
        max_len = int((inputs.attention_mask.sum(dim=-1) * 6.5 / sr).max().item())
        with self.torch.no_grad():
            ids = self.ms_model.generate(**inputs, max_length=max_len)
        return self.ms_proc.decode(ids[0], skip_special_tokens=True).strip()

    def _wait_for_wake(self):
        """Wake on the word "sunflower": Moonshine writes down each short stretch of speech and the text is searched.
        Hum and silence never reach it (a voice-activity detector gates it), so it cannot chime by itself."""
        sys.path.insert(0, os.path.join(HERE, "wake"))
        from moonshine_wake import MoonshineWake
        self.set_state("waiting")
        self.manual.clear()
        det = MoonshineWake(self._hear_array, log=self._log_wake, save_dir=f"{base.SCRATCH}/wake_segments")
        mic = vui.Mic(base.MIC_DEVICE)
        try:
            while not self.closing:
                chunk = mic.read()
                if det.feed(chunk) or self.manual.is_set():
                    self.wake_time = time.perf_counter()
                    return
        finally:
            mic.close()

    def _log_wake(self, rec):
        try:
            with open(f"{base.SCRATCH}/wake_log.jsonl", "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _turn(self):
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "src": "English", "dst": "Luganda", "moonshine": MOONSHINE_SIZE}
        self.set_state("listening")
        base.play_audio(vui.CHIME)
        time.sleep(0.4)  # the Bluetooth speaker lags a little; do not record the chime
        t0 = time.perf_counter()
        audio = self._capture()
        if audio is None:
            self.status('Heard nothing. Say "Hey Sunflower" again.')
            return
        rec["question_s"] = round(time.perf_counter() - t0, 2)
        t_end = t0 + rec["question_s"]
        self.set_state("thinking")
        t1 = time.perf_counter()
        heard = vui.strip_tokens(self._hear(audio)).strip()
        rec["listen_s"] = round(time.perf_counter() - t1, 2)
        rec["heard"] = heard
        if not heard:
            self.status("Could not make out any speech.")
            self._log(rec)
            return
        self.ui(self.you_var.set, "English: " + heard)
        self.ui(self.bot_var.set, "")
        t2 = time.perf_counter()
        luganda = vui.strip_tokens(self.server.translate(heard, "Luganda")).strip()
        rec["translate_s"] = round(time.perf_counter() - t2, 2)
        rec["luganda"] = luganda
        self.ui(self.bot_var.set, luganda)
        sentences, first_sound = queue.Queue(), []
        speaker = threading.Thread(target=self._speaker, args=(sentences, "Luganda", first_sound, rec), daemon=True)
        speaker.start()
        splitter = vui.SentenceSplitter()
        for s in splitter.add(luganda + " ") + splitter.rest():
            sentences.put(s)
        sentences.put(None)
        t3 = time.perf_counter()
        speaker.join()
        if first_sound:
            rec["question_end_to_first_sound_s"] = round(first_sound[0] - t_end, 2)
            rec["wake_to_first_sound_s"] = round(first_sound[0] - self.wake_time, 2)
        rec["speak_total_s"] = round(time.perf_counter() - t3, 2)
        self.status(f"Moonshine {rec['listen_s']}s | translate {rec['translate_s']}s | "
                    f"first sound {rec.get('question_end_to_first_sound_s', '-')}s after you stopped")
        self._log(rec)

    def _log(self, rec):
        try:
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    root = tk.Tk()
    if vui.WINDOWED:
        root.geometry("480x320")
    TranslateApp(root)
    root.mainloop()
