"""Train a small "hey sunflower" classifier on top of openWakeWord's audio embedding.

Inputs
  --data      folder from gen_synthetic.py (pos/, neg/)
  --real      folder of real positive clips recorded on the Pi (record_samples.py)
  --real-neg  optional folder of real negative recordings (normal talking, ambient) from the same mic
Output
  --out       .npz with the classifier weights (numpy only, so the Pi needs no scikit-learn)

Every clip is placed on a 2.5 s canvas of real mic noise with a random gain and offset, high-passed at 150 Hz
(the mic has 50 Hz hum), embedded, and the last 16 embedding frames (about 1.3 s) are the classifier's input,
exactly as the streaming detector sees them.
"""
import argparse
import glob
import os
import wave

import numpy as np
from openwakeword.utils import AudioFeatures
from scipy.signal import butter, resample_poly, sosfilt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

SR = 16000
CANVAS = int(2.5 * SR)
FRAMES = 16
HP = butter(4, 150, "hp", fs=SR, output="sos")
rng = np.random.default_rng(0)


def load(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)


def trim_speech(x, start=0, win=320):
    """Crop to the span where the high-passed signal is clearly above the noise floor.

    The filter runs over the whole clip first (its start-up transient is skipped by `start`)."""
    h = sosfilt(HP, x)
    e = np.array([np.sqrt((h[i:i + win] ** 2).mean()) for i in range(start, len(h) - win, win)])
    floor = max(300.0, 2.5 * np.median(e))
    idx = np.where(e > floor)[0]
    if len(idx) == 0:
        return None
    a = start + max(0, idx[0] * win - 1600)
    b = min(len(x), start + (idx[-1] + 1) * win + 1600)
    return x[a:b]


def background(noise_pool):
    out = np.zeros(CANVAS)
    pos = 0
    while pos < CANVAS:
        seg = noise_pool[rng.integers(len(noise_pool))]
        out[pos:pos + len(seg)] = seg[:CANVAS - pos]
        pos += len(seg)
    return out * rng.uniform(0.6, 1.4)


def canvas(speech, noise_pool, peak=None, end_gap=None):
    bg = background(noise_pool)
    if speech is not None:
        s = speech.copy()
        if rng.random() < 0.5:  # speed change
            r = rng.uniform(0.92, 1.08)
            s = resample_poly(s, 1000, int(1000 * r))
        gap = int((end_gap if end_gap is not None else rng.uniform(0.05, 0.5)) * SR)
        end = CANVAS - gap
        s = s[-(end - 1000):]
        s = s / (np.abs(s).max() + 1e-9) * (peak or rng.uniform(2500, 12000))
        bg[end - len(s):end] += s
    return np.clip(sosfilt(HP, bg), -32768, 32767).astype(np.int16)


def embed(clips, feats):
    e = feats.embed_clips(np.stack(clips), batch_size=64)
    return e[:, -FRAMES:, :].reshape(len(clips), -1)


def utterances(a):
    """Split a recording of repeated phrases into utterances using the high-passed energy."""
    h = sosfilt(HP, a)
    win = 1600
    e = np.array([np.sqrt((h[i:i + win] ** 2).mean()) for i in range(0, len(h) - win, win)])
    on = e > 1.5 * np.median(e)
    segs, start, gap, last = [], None, 0, 0
    for i, v in enumerate(on):
        if v:
            start = i if start is None else start
            gap, last = 0, i
        elif start is not None:
            gap += 1
            if gap > 5:
                segs.append((start, last + 1)); start = None
    if start is not None:
        segs.append((start, last + 1))
    return [a[max(0, s0 * win - 2400):t * win + 2400] for s0, t in segs if 3 <= t - s0 <= 25]


def windows(a, hop=1280):
    return [np.clip(a[i:i + CANVAS], -32768, 32767).astype(np.int16) for i in range(0, len(a) - CANVAS, hop)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--real-neg", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--room", default=None, nargs="*", help="folders with quiet.raw / talking.raw / phrase.raw from wake_eval.py --save")
    ap.add_argument("--holdout", default="far", help="real-clip block kept for testing (near/arm/far/none)")
    args = ap.parse_args()

    feats = AudioFeatures(inference_framework="onnx")
    real_files = sorted(glob.glob(os.path.join(args.real, "*.wav")))
    test_files = [f for f in real_files if os.path.basename(f).startswith(args.holdout)]
    train_files = [f for f in real_files if f not in test_files]
    print(len(train_files), "real train clips,", len(test_files), "real held-out clips")

    # real mic noise: the quiet end of every real clip
    noise_pool = [load(f)[int(2.4 * SR):] for f in real_files]

    def real_speech(f):
        return trim_speech(load(f), start=int(0.9 * SR))  # skip the beep at the start of each recording

    X, y, w = [], [], []
    # real positives (train)
    rp = [s for s in (real_speech(f) for f in train_files) if s is not None]
    clips = [canvas(s, noise_pool) for s in rp for _ in range(20)]
    X += list(embed(clips, feats)); y += [1] * len(clips); w += [1.0] * len(clips)
    # synthetic positives
    pos_files = sorted(glob.glob(os.path.join(args.data, "pos", "*.wav")))
    clips = [canvas(trim_speech(load(f)), noise_pool) for f in pos_files for _ in range(2)]
    X += list(embed(clips, feats)); y += [1] * len(clips); w += [0.5] * len(clips)
    # synthetic negatives (+ noise-only canvases)
    neg_files = sorted(glob.glob(os.path.join(args.data, "neg", "*.wav")))
    clips = [canvas(trim_speech(load(f)), noise_pool) for f in neg_files]
    clips += [canvas(None, noise_pool) for _ in range(300)]
    # beeps and tones (the app's own chimes must never wake it)
    def tone(hz, secs):
        t = np.arange(int(SR * secs)) / SR
        return np.sin(2 * np.pi * hz * t) * np.minimum(1, np.minimum(t, secs - t) * 200)
    clips += [canvas(tone(rng.uniform(300, 1400), rng.uniform(0.15, 1.0)), noise_pool) for _ in range(300)]
    clips += [canvas(tone(f, d), noise_pool) for f, d in ((880, 0.25), (520, 0.8)) for _ in range(60)]
    X += list(embed(clips, feats)); y += [0] * len(clips); w += [1.0] * len(clips)
    # real negatives: slide a window over real recordings of normal talking
    if args.real_neg:
        rn = []
        for f in glob.glob(os.path.join(args.real_neg, "*.wav")):
            x = sosfilt(HP, load(f))
            for i in range(0, len(x) - CANVAS, SR // 4):
                rn.append(np.clip(x[i:i + CANVAS], -32768, 32767).astype(np.int16))
        if rn:
            X += list(embed(rn, feats)); y += [0] * len(rn); w += [2.0] * len(rn)
            print(len(rn), "real negative windows")
    held = {}
    for folder in args.room or []:
        raw = {n: np.frombuffer(open(os.path.join(folder, n + ".raw"), "rb").read(), dtype=np.int16).astype(np.float64)
               for n in ("quiet", "talking", "phrase")}
        utts = utterances(raw["phrase"])
        print(folder, ":", len(utts), "utterances found; last 2 held out")
        clips = [canvas(u, noise_pool, peak=min(30000, np.abs(u).max()), end_gap=None) for u in utts[:-2] for _ in range(40)]
        X += list(embed(clips, feats)); y += [1] * len(clips); w += [2.0] * len(clips)
        held.setdefault("pos", []).extend(canvas(u, noise_pool, peak=min(30000, np.abs(u).max()), end_gap=g)
                                          for u in utts[-2:] for g in (0.05, 0.2, 0.4))
        for n in ("quiet", "talking"):
            a = sosfilt(HP, raw[n]); cut = int(0.6 * len(a))
            tr = windows(a[:cut], 4000)
            X += list(embed(tr, feats)); y += [0] * len(tr); w += [3.0] * len(tr)
            held.setdefault(n, []).extend(windows(a[cut:], 1280))
    X, y, w = np.array(X), np.array(y), np.array(w)
    print("train set:", X.shape, "positives", int(y.sum()))

    # held-out: real far clips (positives) and synthetic negatives never used for training are not available,
    # so also test on fresh noise canvases and the held-out clips at several offsets
    tp = [s for s in (real_speech(f) for f in test_files) if s is not None]
    Xt = embed([canvas(s, noise_pool, end_gap=g) for s in tp for g in (0.05, 0.2, 0.4)], feats) if tp else None

    best = None
    held_emb = {k: embed(v, feats) for k, v in held.items()}
    for name, clf in [("logreg C=0.1", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")),
                      ("mlp 64", MLPClassifier((64,), alpha=1e-2, max_iter=400, random_state=0))]:
        if name.startswith("mlp"):
            clf.fit(X, y)
        else:
            clf.fit(X, y, sample_weight=w)
        ptr = clf.predict_proba(X)[:, 1]
        msg = f"{name}: train recall {(ptr[y == 1] > 0.5).mean():.2f}, train false-positive rate {(ptr[y == 0] > 0.5).mean():.4f}"
        if Xt is not None:
            pt = clf.predict_proba(Xt)[:, 1]
            msg += f", held-out real recall {(pt > 0.5).mean():.2f} (n={len(pt)})"
        print(msg)
        for k, e in held_emb.items():
            p = clf.predict_proba(e)[:, 1]
            if k == "pos":
                print("    room held-out phrase recall at 0.5/0.9/0.98:", [round(float((p > t).mean()), 2) for t in (0.5, 0.9, 0.98)])
            else:
                sec = len(p) * 0.08
                print(f"    room held-out {k} ({sec:.0f} s): windows above 0.5/0.9/0.98:", [int((p > t).sum()) for t in (0.5, 0.9, 0.98)])
        if best is None or name.startswith("mlp"):  # the small network was clearly better on held-out room audio
            best = (name, clf)
    name, clf = best
    if isinstance(clf, LogisticRegression):
        np.savez(args.out, kind="logreg", W=clf.coef_[0], b=clf.intercept_[0], frames=FRAMES)
    else:
        np.savez(args.out, kind="mlp", W1=clf.coefs_[0], b1=clf.intercepts_[0], W2=clf.coefs_[1][:, 0],
                 b2=clf.intercepts_[1][0], frames=FRAMES)
    print("saved", args.out, "from", name)


if __name__ == "__main__":
    main()
