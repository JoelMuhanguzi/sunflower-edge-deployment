"""Character-level ONNX voice shared by the Sunflower apps (English and Luganda).

Model: jq/sherpa-vits-tts-lug-eng (vits-lug-eng.fp32.onnx, 114 MB, one model for both languages), the
voice the Sunbird tutor app uses. On the Pi 4 it runs at about 2x real time against 2.6-3.7x for the Sunbird
VITS voices, and faster than real time on the Pi 5 (docs/06-fast-pipeline.md, docs/08-pi5-and-matched-rerun.md).

Needs numpy, scipy and onnxruntime, so the apps import it inside try/except and fall back to the Sunbird VITS
worker when they run under a Python without them (the system python3). SUNFLOWER_VOICE=vits (or the older
FAST_VOICE=vits) forces the Sunbird VITS voices.
"""
import os
import re
import time

HOME = os.path.expanduser("~")
CHAR_VOICE_DIR = os.environ.get("SUNFLOWER_VOICE_DIR", f"{HOME}/ml/pipeline2/models/sherpa-vits-lug-eng")
CHAR_VOICE_LANGS = ("English", "Luganda")
USE_CHAR_VOICE = os.environ.get("SUNFLOWER_VOICE", os.environ.get("FAST_VOICE", "onnx")) == "onnx"


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
