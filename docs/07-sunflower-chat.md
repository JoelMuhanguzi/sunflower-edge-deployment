# Part 7: Sunflower Chat

[← Back to the project README](../README.md)

The first two apps translate. **Sunflower Chat** (`scripts/sunflower_chat_ui.py`) is a third touchscreen app for an ordinary conversation with Sunflower-Gemma4-E2B: ask a question by speaking or typing, read the answer as it is generated, and optionally hear it. It is a separate app with its own desktop icon, so the translators are unchanged. Only Gemma can chat; Whisper and NLLB (Part 6) can transcribe and translate but not answer, so this app uses the Gemma model of the first pipeline.

> **Status.** The model side was measured on the Pi 4 (five questions, below). The touchscreen interface was exercised automatically with a stand-in model on a Mac and then tried by hand on the Pi, where it worked and was liked; the key size and layout were not measured, and the spoken-reply path and attaching to a server started by another app have **not** been run on the Pi.

## How it works

| | |
|---|---|
| Model | The first app's Gemma server (`GemmaServer` in `sunflower_touch_ui.py`: `llama-server` holding Q4_K_M and the audio encoder). If that server is already running, Chat attaches to it; otherwise Chat starts one with a **4096-token context** (the translators use 2048). Whichever app started the server stops it on exit |
| Speak | Records 5 s (same microphone, gain and 16 kHz mono format as the other apps), transcribes with Gemma, shows the text, then answers |
| Type | A full-window on-screen keyboard (no popup windows, which do not work with touch here): letters, digits, `'`, `,` `.` `?`, the Luganda **ŋ**, shift, space, backspace, Send, Cancel |
| Reply | Streamed word by word (`stream: true`), so the first words appear long before the answer is finished. Capped at 200 tokens; the opening instruction asks for one to three sentences in the chosen language |
| Memory of the conversation | The last six messages (three questions and three answers) are sent with each question; the instruction is attached to the first of them. The server keeps its prompt cache on so earlier turns are not reprocessed |
| Voice reply | Off by default (speech costs about 10 s per sentence). When on, the first 250 characters or so, cut at a sentence end, are spoken with the English or Luganda voice; those voices load on the first spoken reply (about 30 s). Other languages load per tap |
| Language | The same five-language picker. **Changing it starts a new chat**: the instruction at the start of the conversation names the language, and the first measurements showed a language switch making the next reply slow (see below) |
| Log | One JSON line per turn in `~/ml/demo_scratch/chat_log.jsonl`: time to the first word, tokens, tokens per second, total, and for voice turns the listening time |
| Memory guard | If the server has to be started and less than 5 GB is free, it says so and asks you to close the other app. It cannot run together with Sunflower Fast |

Install the icon with `scripts/pi/install_launcher.sh` (it now installs both the Sunflower and the Sunflower Chat launchers). The app expects the same `~/ml` layout as the first app, including `vits_worker.py`.

## Measurements (Pi 4, real model, one run each)

Server start: 111 s from a cold cache (context 4096). Questions went through the same code the app uses, one conversation, in the order shown:

| Question | Language | First word | Tokens | Generation | Total | Messages sent |
|---|---|---|---|---|---|---|
| What is photosynthesis? | English | 9.6 s | 26 | 2.33 tok/s | 20.4 s | 1 |
| Can you say that in one short sentence? | English | 9.2 s | 22 | 2.30 tok/s | 18.3 s | 3 |
| Why do plants need it? | English | 8.3 s | 17 | 2.24 tok/s | 15.5 s | 5 |
| Oli otya? Nnyinza ntya okukuyamba leero? | Luganda | 25.3 s | 19 | 2.21 tok/s | 33.4 s | 5 |
| (heard from a recorded clip: "I am very sick. Please take me to the hospital") | English | 27.7 s | 9 | 2.05 tok/s | 31.6 s | 5 |

- **The follow-ups worked**: "that" and "it" were understood from the earlier answers, and the Luganda greeting was answered sensibly ("Ndi bulungi, nnyinza ntya okukuyamba leero?").
- **Generation matches the other measurements** (2.0–2.3 tokens/s against 2.45 from `llama-bench`). Streaming is what makes it usable: the first words appear after about 9 s, even though a short answer takes 15–20 s to finish.
- **Listening is the slow part of Speak mode**: the recorded clip took **45.3 s** to transcribe before the question could be answered, as in the first app.
- **A change of language was slow** (25.3 s and 27.7 s to the first word, against 8–10 s in one language). We think the cause is the instruction at the start of the conversation changing, which makes the server reread the whole history; this was **not verified**. The app therefore starts a new chat when the language changes. That change was made after these measurements and checked only with the stand-in model.
- The transcription of the recorded clip dropped the final full stop and the reply ("I'm so sorry to hear that.") did not offer help: **answer quality was not assessed**, in any language.

## Not yet established

- The quality and accuracy of the answers, especially in Luganda and Runyankole; the model was tuned for translation and the chat behaviour was not evaluated.
- Whether the keys are large enough, and how the layout reads, beyond one user's hands-on impression; the usable area is smaller than the screen because the panel does not register the outer ~25 px.
- Timing of a spoken reply, memory in use during a chat, and attaching to a server started by the first app.
- A faster listening path (the Whisper pipeline listens in about 19 s but cannot answer, and the two cannot share the Pi's memory).

## Reproduce

`scripts/sunflower_chat_ui.py` with `scripts/sunflower_touch_ui.py` (which provides the server, recording and voices), `scripts/vits_worker.py`, and the launcher template `scripts/pi/SunflowerChat.desktop.template`. Setting `CHAT_MOCK=1 CHAT_WINDOWED=1` previews the interface on a desktop machine with a canned model, which is how the layout and keyboard logic were tested. Model files are never included.
