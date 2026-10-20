#!/usr/bin/env python3
"""Time listen and translate on an Android phone running llama-server, with the same six clips, prompts
and request settings as bench_compare.py setup B (Gemma kept loaded, prompt cache off).

    adb forward tcp:8081 tcp:8081
    phone_bench.py CLIPS_JSON [--url http://127.0.0.1:8081] [--out results.jsonl]

The voice step is not measured here (it needs ONNX Runtime on the phone). Clip paths in the JSON are
resolved by file name next to the JSON file.
"""
import base64, json, os, re, statistics as st, sys, time, urllib.request

clips_path = sys.argv[1]
url = sys.argv[sys.argv.index("--url") + 1] if "--url" in sys.argv else "http://127.0.0.1:8081"
out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
clips = json.load(open(clips_path))
base = os.path.dirname(os.path.abspath(clips_path))


def norm(s): return re.sub(r"[^\w\s']", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def chat(content):
    body = json.dumps({"messages": [{"role": "user", "content": content}], "max_tokens": 128,
                       "temperature": 0.3, "cache_prompt": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    wall = time.perf_counter() - t
    tm = out.get("timings", {})
    return out["choices"][0]["message"]["content"].strip(), wall, tm


def run(clip, record):
    src = clip["lang"]; dst = "Luganda" if src == "English" else "English"
    wav = os.path.join(base, os.path.basename(clip["file"]))
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    heard, t_listen, tm_l = chat([
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
        {"type": "text", "text": f"The speaker is speaking {src}. Transcribe exactly what they said, written in "
                                 f"{src}. Respond with only the transcription, nothing else."}])
    translated, t_tr, tm_t = chat("Translate to " + dst + ": " + " ".join(heard.split()))
    rec = {"clip": os.path.basename(wav), "src": src, "dst": dst, "reference": clip["text"], "heard": heard,
           "translated": translated, "listen_s": round(t_listen, 2), "translate_s": round(t_tr, 2),
           "total_s": round(t_listen + t_tr, 2), "wer": round(wer(clip["text"], heard), 2),
           "listen_prompt_tps": round(tm_l.get("prompt_per_second", 0), 1), "listen_gen_tps": round(tm_l.get("predicted_per_second", 0), 1),
           "translate_gen_tps": round(tm_t.get("predicted_per_second", 0), 1), "warmup": not record}
    print(("run    " if record else "warm-up"), rec["clip"], f"listen {rec['listen_s']}s translate {rec['translate_s']}s "
          f"| WER {rec['wer']} | heard={heard[:50]!r}", flush=True)
    return rec


run(clips[0], False)
recs = [run(c, True) for c in clips]
if out_path:
    with open(out_path, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
m = lambda k: round(st.median(r[k] for r in recs), 1)
print(f"MEDIAN listen {m('listen_s')}s translate {m('translate_s')}s total {m('total_s')}s "
      f"| mean WER {round(st.mean(r['wer'] for r in recs), 2)} | n={len(recs)}")
