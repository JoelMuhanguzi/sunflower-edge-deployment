# Part 11: Hands-free voice and the "Hey Sunflower" wake word

Two apps were built that need no touching: you say the wake word, hear a chime, speak, and hear the answer.

| App | Listens with | Answers with | Run with |
|---|---|---|---|
| `scripts/sunflower_voice_ui.py` (Sunflower Voice) | Gemma's audio encoder | Gemma, spoken by the ONNX voice | the `pipeline2` Python |
| `scripts/sunflower_translate_voice.py` (Sunflower Translate) | Moonshine streaming (int8, English) | Gemma translates to Luganda, spoken by the ONNX voice (pipeline G of [Part 10](10-moonshine-asr.md)) | the `moonshine-venv` Python |

Both show the state as an animated circle on the 480x320 screen (waiting, listening, thinking, speaking) and work without a screen: a chime means "I heard the wake word, speak now",
and three rising notes mean "loaded". Sound goes to a connected Bluetooth speaker when there is one (Part 8, setup notes).
Every turn is logged (`voice_log.jsonl`, `translate_voice_log.jsonl`) with stage times. The Translate app has completed turns correctly end to end; the Gemma-listening Voice app ran end to end on 20 and 23 October but its transcriptions were unusable (see 6).

Raw logs of the wake-word tests: [`results/wake-word-2026-10/`](../results/wake-word-2026-10/).

## 1. How one turn works

wake word, then a chime, then the question is recorded until 1 s of silence (voice-activity detector, at most 10 s; gives up after 5 s without speech),
then speech to text, then (Translate app) Gemma translates, then the voice speaks the answer sentence by sentence.
The microphone is a cheap USB mic and everything below depends on its signal, so the audio facts come first.

## 2. The microphone: hum, gain, and Auto Gain Control

- **Mains hum.** The recordings carry a very strong 50 Hz hum with harmonics at 100, 150 and 200 Hz (about 6,000 rms against speech at 1,000 to 1,500 after filtering).
  A high-pass filter is required: 150 Hz was enough for the custom wake-word model, but the recorder needed **250 Hz**; at 120 Hz Gemma was transcribing hum.
- **Capture gain and Auto Gain Control (AGC).** When first set up the mic was at gain 10 of 16 (62%). On 23 October it was found at 16 (100%, 23.8 dB) with its AGC switch on,
  and live recognition was poor: the noise floor was about twice as high (rms 1,100 to 1,900 against 600 to 700) and loud speech touched full scale.
  Both settings were changed together (gain 62%, AGC off), so the improvement cannot be attributed to one of them. The apps now set both at start
  (`set_mic_levels()` in `sunflower_voice_ui.py`): `amixer -c <card> sset Mic 10` and `sset "Auto Gain Control" off`.
- **Level after filtering.** Once the hum is removed the voice is quiet, so each recorded question is scaled to 80% of full scale (at most 8x) before it is transcribed.
- **Distance is the limit.** With this mic the voice is only reliably above the room noise within about 30 cm (see 5).

## 3. Attempt 1: a small custom wake-word model

[`scripts/wake/`](../scripts/wake/): a classifier on openWakeWord's audio embedding, trained for "hey sunflower".

- Positives: 30 real recordings of the author on the Pi's mic (10 near, 10 at arm's length, 10 across the room) plus about 420 synthetic clips from 21 macOS voices.
  Negatives: about 1,100 synthetic phrases (near misses such as "hey there", English and Luganda sentences), beeps and tones, and real recordings of the author talking. Training takes about a minute on the Mac.
- The stock "hey jarvis" model scored 0.00 to 0.08 on the live mic (a clear recording of the same phrase scored 1.0), so a custom phrase was needed.
- Results, first version, scripted live test (quiet 15 s, normal talking 20 s, the phrase said 5 times): 1 false event while quiet, 4 while talking, 6 events for 5 utterances. Too many false events.
- After adding the in-room recordings and switching to a small neural network, replaying the recorded audio gave 5 of 5 phrases and no false events (with the caveat that most of that audio was part of the training data;
  on audio held out from training there were also no false events).
- **In the live app it chimed constantly without the phrase being said.** The replay did not predict this. A likely reason is that the live mic was by then at 100% gain with AGC on (section 2), a different sound from the 62% recordings it was trained on,
  but this was not tested. The approach was set aside.

## 4. Attempt 2: wake by transcription with Moonshine

[`scripts/wake/moonshine_wake.py`](../scripts/wake/moonshine_wake.py). A voice-activity detector cuts the stream into stretches of speech (at most 3 s); each stretch is written down by Moonshine and the text is searched for
"flower", "flour", "sunflo", or "flow/flowing/flows". Silence and hum never reach Moonshine, and ordinary speech only wakes it if the transcript contains one of those, so it cannot wake by itself.
No training is needed. (The word "flow" was added after "I'm flowing" appeared for "sunflower"; it also matches "the flow of water", an accepted trade-off.)

Offline on the 30 recorded "hey sunflower" clips (author, close to the mic, 62% gain), phrase written by Moonshine:

| Model | Exact "sunflower" | Contains "flower"/"flour" | False matches on 18 other utterances |
|---|---:|---:|---:|
| tiny | 5 / 30 | not run | 0 |
| small | 16 / 30 | 23 / 30 | 0 |
| medium | 17 / 30 | 23 / 30 | 0 |

The complete detector (voice-activity detector, 3 s limit, loose match, medium model) replayed on the same clips woke on **25 of 30** and gave **0 wakes in 135 s of talking and silence** (7 stretches of speech examined).

## 5. Live results on the Pi 5 (medium Moonshine int8, author speaking)

Counts are speech stretches examined by Moonshine, not attempts: the number of times the phrase was said was not recorded, and some stretches are noise. Raw logs in `results/wake-word-2026-10/`.

| Run | Phrase | Mic setting | Stretches | Woke | What it wrote when it woke | Notes |
|---|---|---|---:|---:|---|---|
| A | "hey sunflower" | 100%, AGC on | 8 | 1 | "A hazeln flower." | misses: "Listen to me", "I'm flowing", "Peace and love", 3 empty |
| B | "hey sunflower" ×10 | 100%, AGC on | 19 | 2 | "Hands on flower", "Haven flower" | quiet stretches mostly empty; misses such as "I hate Santa Fe" |
| C | "sunflower" ×10 | 100%, AGC on | 8 | 3 | "It's not a flower", "Sunflower", "It's sunflower" | most stretches reached full scale |
| D | "sunflower" ×10 | **62%, AGC off** | 21 | 4 | "Sunflower" ×4 | the last four attempts, all close and firm, all woke |

What the saved audio showed (not stored in the repo, it is recordings of a voice): the stretches that woke had peaks of 13,000 to 22,000 after filtering; stretches with peaks of 1,800 to 6,500 came back empty or as invented sentences
("I'm going to go to the bathroom"), so a quiet or distant voice is hallucinated into other words. Re-running eight clear stretches with five clean-up variants (different gain, 300 or 400 Hz high-pass) did not recover the misses.
Dropping "hey" helps: Moonshine blurs it into the next word ("Handsome flower", "Haste and flower").

**Conclusion so far:** close to the mic (about 30 cm) and speaking firmly, saying just "Sunflower" woke reliably in run D (4 of the last 4); at a distance or quietly it fails.
This is a small test by one speaker in one room: it is not an accuracy figure.

## 6. Other findings along the way

- **Gemma hears this mic badly.** Sunflower Voice (Gemma listens) transcribed hum, not speech: replies such as "I'm sorry, I can't see what the speaker is saying" and `<|audio>Hello!`. The 120 Hz filter was too low (hum harmonics at 150 to 200 Hz remained, 96% of the recorded energy was below 300 Hz).
  After raising it to 250 Hz and normalising the level, three further turns still produced invented text ("I cannot generate content containing…", "The Samsung Galaxy Note 7 was a major recall…").
  The mic was probably still at 100% gain with AGC on at that time (section 2), which was not known then, and Gemma listening was not retried afterwards. The Moonshine route, which heard the same mic correctly, was used instead.
- **Empty replies.** Gemma sometimes returns only a control marker (`<|turn>`); these are now removed, and an empty reply is no longer stored in the conversation history (it kept later replies empty). The app says "Sorry, I did not catch that" instead.
- **A 3-turn run of the Translate app** (before the wake-word work above, logged in `results/asr-moonshine-2026-10/handsfree-translate-log-2026-10-23.jsonl`): Moonshine 1.7 to 2.5 s, translation 2.8 to 4.2 s, 5.8 to 7.5 s from the end of the question to the first sound.
- **Input limit:** a question is cut at 10 s, the pause that ends it is 1 s, and a translation is limited to 128 tokens (roughly one or two sentences).
- Operational trap: stopping the app by pattern (`pkill -f`) over `ssh` can kill the ssh session itself; stop by process id.

## 7. Open questions

1. Test the wake word with other speakers, accents and distances; measure false wakes over an hour of real room noise. Everything here is one speaker.
2. Separate the effects of gain and AGC (they were changed together).
3. A better microphone (a USB speakerphone) would likely help more than any software.
4. A combined detector (Moonshine or the custom model) and a tap on the screen as a fallback.
5. Whether the custom model would have worked at the 62% setting (it was never re-tested after the gain was corrected).
