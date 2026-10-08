#!/usr/bin/env python3
"""Head-to-head benchmark of the two pipelines on the same recorded clips (run on the Pi 4).

    bench_compare.py A|B|C [--clips ~/ml/bench_clips/clips.json] [--repeat N]

  A  first app as it was (frozen copy sunflower_touch_ui_per_tap.py): Gemma (llama-mtmd-cli / llama-cli, new process per call) + VITS per call
  B  first app, loaded:    Gemma in a persistent llama-server (prompt cache OFF) + preloaded voices
  C  second app:           Whisper int8 (5 s window) + NLLB int8, loaded once + preloaded voices
  D  C with the character-level ONNX voice (jq/sherpa-vits-tts-lug-eng) for English and Luganda
  E  B (Gemma kept loaded) with the same ONNX voice

Each clip is spoken in English or Luganda and translated into the other one, then spoken.
Stages are timed identically in every setup: listen, translate, voice. One setup per
invocation so memory is freed between them. Results go to ~/ml/bench_results/<config>_<time>.jsonl.
"""
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

HOME = os.path.expanduser("~")
sys.path.insert(0, f"{HOME}/ml")
sys.path.insert(0, f"{HOME}/ml/pipeline2")

ap = argparse.ArgumentParser()
ap.add_argument("config", choices=["A", "B", "C", "D", "E"])
ap.add_argument("--clips", default=f"{HOME}/ml/bench_clips/clips.json")
ap.add_argument("--repeat", type=int, default=1)
ap.add_argument("--label", default=None, help="name for the result file (default: the config letter)")
ap.add_argument("--cool", type=float, default=60.0, help="wait until the CPU is below this many C")
ap.add_argument("--limit", type=int, default=0, help="use only the first N clips (0 = all six)")
args = ap.parse_args()

OUT_DIR = f"{HOME}/ml/bench_results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = f"{OUT_DIR}/{args.label or args.config}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
WAV_OUT = f"{HOME}/ml/demo_scratch/bench_out.wav"
os.makedirs(os.path.dirname(WAV_OUT), exist_ok=True)


def cpu_temp():
    return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000


def tree_rss_mb():
    """Resident memory (MB) of this process and all its descendants (llama-server, voice worker...)."""
    parent, rss = {}, {}
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            stat = open(f"/proc/{pid}/stat").read()
            parent[int(pid)] = int(stat[stat.rindex(")") + 2:].split()[1])
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    rss[int(pid)] = int(line.split()[1]) / 1024
        except (OSError, ValueError):
            pass
    mine, changed = {os.getpid()}, True
    while changed:
        changed = False
        for pid, ppid in parent.items():
            if ppid in mine and pid not in mine:
                mine.add(pid); changed = True
    return round(sum(rss.get(pid, 0) for pid in mine))


def throttled():
    r = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True)
    return r.stdout.strip().split("=")[-1]


def wait_cool():
    t0 = time.time()
    while cpu_temp() > args.cool and time.time() - t0 < 900:
        time.sleep(10)
    return round(time.time() - t0)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def parse_llama_stats(text):
    """Best-effort tokens/s from llama.cpp output (perf lines or the '[ Prompt: .. | Generation: .. ]' line)."""
    stats = {}
    for line in text.splitlines():
        m = re.search(r"(prompt eval time|eval time)\s*=\s*([\d.]+) ms /\s*(\d+) (?:tokens|runs).*?([\d.]+) tokens per second", line)
        if m:
            key = "prompt" if m.group(1).startswith("prompt") else "gen"
            stats[key] = {"n": int(m.group(3)), "tps": float(m.group(4))}
        m = re.search(r"\[ Prompt: ([\d.]+) t/s \| Generation: ([\d.]+) t/s \]", line)
        if m:
            stats.setdefault("prompt", {})["tps"] = float(m.group(1))
            stats.setdefault("gen", {})["tps"] = float(m.group(2))
    return stats


def audio_seconds(path):
    from scipy.io import wavfile
    rate, data = wavfile.read(path)
    return round(len(data) / rate, 2)


def transcribe_prompt(lang):
    return (f"The speaker is speaking {lang}. Transcribe exactly what they said, written in "
            f"{lang}. Respond with only the transcription, nothing else.")


def translate_prompt(text, dst):
    return f"Translate to {dst}: " + " ".join(text.split())


# ---------------------------------------------------------------- setups
class SetupA:
    """The first app's exact commands, one new process per call."""

    def __init__(self):
        import sunflower_touch_ui_per_tap as old
        self.old = old
        self.startup = 0.0

    def listen(self, wav, lang):
        r = run([self.old.MTMD_CLI, "-m", self.old.TEXT_MODEL, "--mmproj", self.old.MMPROJ,
                 "--audio", wav, "--jinja", "-p", transcribe_prompt(lang), "-n", "128", "--temp", "0.3"])
        return r.stdout.strip(), parse_llama_stats(r.stderr + r.stdout)

    def translate(self, text, src, dst):
        prompt = translate_prompt(text, dst)
        r = run([self.old.LLAMA_CLI, "-m", self.old.TEXT_MODEL, "-p", prompt, "-n", "128",
                 "--temp", "0.3", "--single-turn"])
        return self.old._extract_reply(r.stdout, prompt) or "", parse_llama_stats(r.stderr + r.stdout)

    def speak(self, text, dst):
        ok = self.old.speak_text(text, dst, WAV_OUT)
        return ok, {"path": "per-tap"}

    def close(self):
        pass


class VoiceWorkerMixin:
    def start_voices(self):
        import sunflower_fast_ui as fast
        self.fast = fast
        t0 = time.time()
        self.vits = fast.VitsWorker([fast.LANGUAGES[n]["dir"] for n in fast.PRELOADED_VOICES])
        return time.time() - t0

    def speak(self, text, dst):
        res = self.vits.speak(self.fast.LANGUAGES[dst]["dir"], text, WAV_OUT)
        return bool(res.get("ok")), {"path": "worker", **res}


class SetupB(VoiceWorkerMixin):
    """Gemma kept loaded in llama-server; prompt cache off so repeated audio is not flattered."""
    URL = "http://127.0.0.1:8081"

    def __init__(self):
        import sunflower_touch_ui_per_tap as old
        t0 = time.time()
        server = f"{HOME}/ml/llama.cpp/build/bin/llama-server"
        self.srv = subprocess.Popen(
            [server, "-m", old.TEXT_MODEL, "--mmproj", old.MMPROJ, "--jinja", "--host", "127.0.0.1",
             "--port", "8081", "-c", "2048", "-t", "4"],
            stdout=open(f"{OUT_DIR}/llama-server.log", "w"), stderr=subprocess.STDOUT)
        while True:
            try:
                if b"ok" in urllib.request.urlopen(self.URL + "/health", timeout=3).read():
                    break
            except Exception:
                time.sleep(3)
            if self.srv.poll() is not None:
                raise RuntimeError("llama-server exited")
        self.startup = (time.time() - t0) + self.start_voices()

    def _chat(self, content):
        body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 128,
                           "temperature": 0.3, "cache_prompt": False}).encode()
        req = urllib.request.Request(self.URL + "/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=900))
        tm = out.get("timings", {})
        stats = {"prompt": {"n": tm.get("prompt_n"), "tps": tm.get("prompt_per_second")},
                 "gen": {"n": tm.get("predicted_n"), "tps": tm.get("predicted_per_second")}}
        return out["choices"][0]["message"]["content"].strip(), stats

    def listen(self, wav, lang):
        b64 = base64.b64encode(open(wav, "rb").read()).decode()
        return self._chat([{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                           {"type": "text", "text": transcribe_prompt(lang)}])

    def translate(self, text, src, dst):
        return self._chat(translate_prompt(text, dst))

    def close(self):
        self.vits.proc.terminate()
        self.srv.terminate()


class SetupC(VoiceWorkerMixin):
    """The second app's engine: Whisper int8 (5 s window) + NLLB int8, loaded once."""

    def __init__(self):
        import sunflower_fast_ui as fast
        t0 = time.time()
        self.engine = fast.Engine()
        self.startup = (time.time() - t0) + self.start_voices()

    def listen(self, wav, lang):
        text = self.engine.transcribe(wav, lang)
        return text, {"gen": {"n": self.engine.last_tokens}}

    def translate(self, text, src, dst):
        out = self.engine.translate(text, src, dst)
        return out, {"gen": {"n": self.engine.last_tokens}}

    def close(self):
        self.vits.proc.terminate()


class SetupD(SetupC):
    """Setup C with the in-process character-level ONNX voice instead of the Sunbird VITS worker."""

    def start_voices(self):
        import sunflower_fast_ui as fast
        t0 = time.time()
        self.cv = fast.CharVoice()
        return time.time() - t0

    def speak(self, text, dst):
        res = self.cv.speak(text, WAV_OUT)
        return bool(res.get("ok")), {"path": "onnx", **res}

    def close(self):
        pass


class SetupE(SetupB):
    """Setup B (Gemma kept loaded in llama-server) with the in-process character-level ONNX voice."""

    def start_voices(self):
        import sunflower_fast_ui as fast
        t0 = time.time()
        self.cv = fast.CharVoice()
        return time.time() - t0

    def speak(self, text, dst):
        res = self.cv.speak(text, WAV_OUT)
        return bool(res.get("ok")), {"path": "onnx", **res}

    def close(self):
        self.srv.terminate()


# ---------------------------------------------------------------- run
clips = json.load(open(args.clips))
if args.limit:
    clips = clips[:args.limit]
waited = wait_cool()
print(f"config {args.config}: waited {waited}s for cooldown, CPU {cpu_temp():.0f} C, flags {throttled()}", flush=True)
t0 = time.time()
setup = {"A": SetupA, "B": SetupB, "C": SetupC, "D": SetupD, "E": SetupE}[args.config]()
print(f"setup ready: startup {setup.startup:.1f}s (models/voices loaded once; 0 for A)", flush=True)

plan = []
if args.config != "A":  # one unscored warm-up so first-use cache effects are not in the numbers
    plan.append((clips[0], True))
for _ in range(args.repeat):
    plan += [(c, False) for c in clips]

with open(OUT_PATH, "a") as out:
    for clip, warmup in plan:
        src = clip["lang"]
        dst = "Luganda" if src == "English" else "English"
        rec = {"config": args.config, "clip": os.path.basename(clip["file"]), "src": src, "dst": dst,
               "label": args.label or args.config, "env": {k: v for k, v in os.environ.items() if k.startswith("FAST_")},
               "reference": clip["text"], "warmup": warmup, "setup_startup_s": round(setup.startup, 1),
               "temp_start": round(cpu_temp()), "times": {}, "stats": {}}
        try:
            t = time.perf_counter()
            heard, rec["stats"]["listen"] = setup.listen(clip["file"], src)
            rec["times"]["listen"] = round(time.perf_counter() - t, 2)
            rec["heard"] = heard
            t = time.perf_counter()
            translated, rec["stats"]["translate"] = setup.translate(heard, src, dst)
            rec["times"]["translate"] = round(time.perf_counter() - t, 2)
            rec["translated"] = translated
            t = time.perf_counter()
            ok, rec["voice"] = setup.speak(translated, dst)
            rec["times"]["voice"] = round(time.perf_counter() - t, 2)
            rec["voice_ok"] = ok
            if ok:
                rec["audio_s"] = audio_seconds(WAV_OUT)
        except Exception as e:  # keep going; the failure is part of the record
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
        rec["total_s"] = round(sum(rec["times"].values()), 2)
        rec["temp_end"], rec["throttled"] = round(cpu_temp()), throttled()
        rec["rss_mb"] = tree_rss_mb()
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        tag = "warm-up" if warmup else "run    "
        print(f"{tag} {rec['clip']} {src[:3]}->{dst[:3]}  {rec['times']}  total {rec['total_s']}s  "
              f"{rec['temp_start']}->{rec['temp_end']}C {rec['throttled']}  heard={rec.get('heard', '')[:40]!r}", flush=True)

setup.close()
print(f"BENCH_DONE {args.config} wrote {OUT_PATH}", flush=True)
