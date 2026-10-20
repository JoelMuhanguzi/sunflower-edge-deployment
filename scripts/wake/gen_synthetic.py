"""Generate synthetic training clips with the macOS `say` voices (16 kHz mono WAV).

positives: "hey sunflower" (a few spellings/pauses) in many voices at several speaking rates
negatives: near-misses, everyday English, and Luganda phrases the apps are used with

  python3 gen_synthetic.py --out ~/ml/wake_data
"""
import argparse
import itertools
import os
import subprocess
from multiprocessing import Pool

VOICES = ["Alex", "Aman", "Daniel", "Eddy (English (US))", "Flo (English (US))", "Fred", "Grandma (English (US))",
          "Karen", "Kathy", "Moira", "Ralph", "Reed (English (UK))", "Rishi", "Samantha", "Sandy (English (US))",
          "Shelley (English (US))", "Tessa", "Junior", "Rocko (English (US))", "Eddy (English (UK))", "Flo (English (UK))"]
RATES = [120, 150, 175, 200, 230]

POSITIVE = ["hey sunflower", "hey, sunflower", "hey sun flower", "Hey Sunflower!"]
NEGATIVE = [
    # near misses
    "sunflower", "hey", "hey sun", "hey flower", "hey summer", "hey Sandra", "sunday flowers", "hey sunbird",
    "hey Sunbird", "hey there", "hey siri", "hey google", "hey jarvis", "hello sunflower", "sunflowers are yellow",
    "the sun flower", "hey sonny", "hey someone", "okay sunflower", "hey Alexa",
    # everyday English
    "the hospital is near the market", "my name is Joel and I am happy to see you", "I am very sick please take me to the hospital",
    "good morning how are you today", "what time does the meeting start", "can you translate this for me",
    "please turn the volume down", "the weather is nice today", "let me show you the demo", "this runs on a Raspberry Pi",
    "the model translates between English and Luganda", "I think we should start now", "can everyone hear me",
    "thank you very much", "one two three four five", "what is the capital of Uganda", "I would like some water",
    "the children are going to school", "how much does this cost", "see you tomorrow", "that is a good question",
    "let us look at the results", "the paper is due next week", "my supervisor is coming", "open the door please",
    "I am going to the market", "tell me a story", "what does that word mean", "the battery is low",
    "it is very hot in here", "we need more data", "this is working very well", "wait a moment please",
    # Luganda
    "oli otya leero", "ndi mulwadde nnyo twala mu ddwaliro", "abaana bagenda ku ssomero", "webale nnyo", "nsaba amazzi",
    "oli otya nnyabo", "mwebale kujja", "tugende ku katale", "nze Joel", "osiibye otya", "kale nsiibye bulungi",
    "ddwaliro liri wa", "leero kiri kyewuunyo", "nkwagala nnyo",
]


def render(job):
    text, voice, rate, path = job
    if os.path.exists(path):
        return
    tmp = path + ".aiff"
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", tmp, text], check=False, capture_output=True)
    if os.path.exists(tmp):
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", tmp, path], check=False, capture_output=True)
        os.remove(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    jobs = []
    for kind, phrases, rates in [("pos", POSITIVE, RATES), ("neg", NEGATIVE, [150])]:
        os.makedirs(os.path.join(args.out, kind), exist_ok=True)
        voices = VOICES if kind == "pos" else VOICES[::2]  # fewer voices for the (much larger) negative set
        for (i, text), (j, voice), rate in itertools.product(enumerate(phrases), [(2 * k if kind == "neg" else k, v) for k, v in enumerate(voices)], rates):
            jobs.append((text, voice, rate, os.path.join(args.out, kind, f"{kind}_{i:02d}_{j:02d}_{rate}.wav")))
    print(len(jobs), "clips")
    with Pool(6) as p:
        p.map(render, jobs, chunksize=8)
    for kind in ("pos", "neg"):
        print(kind, len(os.listdir(os.path.join(args.out, kind))))
