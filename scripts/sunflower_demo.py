#!/usr/bin/env python3
"""
Sunflower-Gemma4-E2B interactive demo launcher.
Menu-driven access to text translation, speech understanding, and
text-to-speech, built on top of the already-deployed llama.cpp and
VITS/MMS-TTS pipelines.
"""

import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
LLAMA_CLI = f"{HOME}/ml/llama.cpp/build/bin/llama-cli"
MTMD_CLI = f"{HOME}/ml/llama.cpp/build/bin/llama-mtmd-cli"
TEXT_MODEL = f"{HOME}/ml/gguf/sunflower-gemma4-e2b-Q4_K_M.gguf"
MMPROJ = f"{HOME}/ml/gguf/mmproj-sunflower-gemma4-e2b-f16.gguf"
VITS_DIR = f"{HOME}/ml/vits-work/training"
VITS_VENV_PY = f"{HOME}/ml/vits-venv/bin/python"
TTS_VENV_PY = f"{HOME}/ml/tts-venv/bin/python"

MIC_DEVICE = "plughw:3,0"
SPEAKER_DEVICE = "plughw:2,0"
RECORD_SECONDS = 5

VITS_LANGUAGES = {
    "1": ("Luganda", f"{HOME}/ml/models/tts-vits-lug"),
    "2": ("English", f"{HOME}/ml/models/tts-vits-eng"),
    "3": ("Runyankole", f"{HOME}/ml/models/tts-vits-nyn"),
}

SCRATCH = f"{HOME}/ml/demo_scratch"
os.makedirs(SCRATCH, exist_ok=True)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def record_audio(out_path, seconds=RECORD_SECONDS):
    print(f"\nRecording in...")
    for n in (3, 2, 1):
        print(n)
        time.sleep(1)
    print(f"SPEAK NOW ({seconds}s)")
    result = run([
        "arecord", "-D", MIC_DEVICE, "-f", "S16_LE",
        "-r", "16000", "-c", "1", "-d", str(seconds), out_path,
    ])
    if result.returncode != 0:
        print(f"Recording failed: {result.stderr}")
        return False
    print("Done recording.")
    return True


def play_audio(path):
    result = run(["aplay", "-D", SPEAKER_DEVICE, path])
    if result.returncode != 0:
        print(f"Playback failed: {result.stderr}")


def _extract_reply(stdout, echoed_prompt):
    """llama-cli prints a banner, then '> <prompt>', the reply, a blank
    line, and a '[ Prompt: ... ]' stats line. Pull out just the reply."""
    lines = stdout.splitlines()
    prompt_line_idx = None
    for i, line in enumerate(lines):
        if line.strip() == f"> {echoed_prompt}".strip():
            prompt_line_idx = i
            break
    if prompt_line_idx is None:
        return None
    reply_lines = []
    for line in lines[prompt_line_idx + 1:]:
        if line.strip().startswith("[ Prompt:") or not line.strip():
            if reply_lines:
                break
            continue
        reply_lines.append(line.strip())
    return " ".join(reply_lines) if reply_lines else None


def llama_generate(prompt, n_predict=128, temp=0.3):
    """Text-only generation via llama-cli. Returns the model's reply text."""
    result = run([
        LLAMA_CLI, "-m", TEXT_MODEL,
        "-p", prompt, "-n", str(n_predict), "--temp", str(temp),
        "--single-turn",
    ])
    if result.returncode != 0:
        print(f"llama-cli failed:\n{result.stderr}")
        return None
    return _extract_reply(result.stdout, prompt)


def llama_audio_understand(audio_path, instruction, n_predict=128, temp=0.3):
    """Audio input via llama-mtmd-cli. Returns the model's reply text.
    All logging/chat-template noise goes to stderr; stdout holds just the
    reply (as plain text, possibly with leading/trailing blank lines)."""
    result = run([
        MTMD_CLI, "-m", TEXT_MODEL, "--mmproj", MMPROJ,
        "--audio", audio_path, "--jinja",
        "-p", instruction, "-n", str(n_predict), "--temp", str(temp),
    ])
    if result.returncode != 0:
        print(f"llama-mtmd-cli failed:\n{result.stderr}")
        return None
    reply = result.stdout.strip()
    return reply if reply else None


def choose_vits_language():
    print("\nSelect a voice:")
    for key, (name, _) in VITS_LANGUAGES.items():
        print(f"  {key}. {name} (Sunbird VITS)")
    print("  4. Other language (MMS-TTS, e.g. ach/swa/nyn/...)")
    choice = input("> ").strip()
    if choice in VITS_LANGUAGES:
        return ("vits", VITS_LANGUAGES[choice])
    if choice == "4":
        code = input("Language code (e.g. ach): ").strip()
        return ("mms", code)
    print("Invalid choice, defaulting to Luganda.")
    return ("vits", VITS_LANGUAGES["1"])


def speak_text(text, out_path):
    """Dispatch to VITS or MMS-TTS based on user choice. Returns True on success."""
    out_path = os.path.abspath(out_path)
    kind, payload = choose_vits_language()
    if kind == "vits":
        name, model_dir = payload
        print(f"Synthesizing with Sunbird VITS ({name})...")
        result = run([
            VITS_VENV_PY, "run_inference.py", model_dir, text, out_path,
        ], cwd=VITS_DIR)
        if result.returncode != 0:
            print(f"VITS synthesis failed:\n{result.stderr}")
            return False
        return True
    else:
        lang_code = payload
        print(f"Synthesizing with MMS-TTS ({lang_code})...")
        # Prefer a local copy (works offline); fall back to the Hugging Face hub.
        local = f"{HOME}/ml/models/mms-tts-{lang_code}"
        ref = local if os.path.isdir(local) else f"facebook/mms-tts-{lang_code}"
        script = f"""
import sys
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
        if result.returncode != 0 or "ok" not in result.stdout:
            print(f"MMS-TTS synthesis failed:\n{result.stderr}")
            return False
        return True


def mode_text_to_text():
    banner("TEXT -> TEXT (translate / chat)")
    print("Type your text (e.g. 'Translate to Luganda: How are you today?').")
    print("Type 'back' to return to the menu.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() == "back":
            return
        if not text:
            continue
        print("Thinking...")
        reply = llama_generate(text)
        if reply:
            print(f"Model: {reply}\n")


def mode_speech_to_text():
    banner("SPEECH -> TEXT (speak, read the answer)")
    input("Press Enter when ready to record...")
    audio_path = f"{SCRATCH}/speech_input.wav"
    if not record_audio(audio_path):
        return
    instruction = input(
        "Instruction for the model (e.g. 'Translate to Luganda', "
        "or just press Enter for 'Respond to what was said'): "
    ).strip() or "Respond to what the speaker said."
    print("Processing (this can take a while on the Pi)...")
    reply = llama_audio_understand(audio_path, instruction)
    if reply:
        print(f"\nModel: {reply}\n")


def mode_speech_to_speech():
    banner("SPEECH -> SPEECH (full voice round trip)")
    input("Press Enter when ready to record...")
    audio_path = f"{SCRATCH}/speech_input.wav"
    if not record_audio(audio_path):
        return
    instruction = input(
        "Instruction for the model (e.g. 'Translate to Luganda, "
        "respond with only the translation'): "
    ).strip() or "Respond to what the speaker said, briefly."
    print("Processing (this can take a while on the Pi)...")
    reply = llama_audio_understand(audio_path, instruction)
    if not reply:
        return
    print(f"\nModel (text): {reply}")
    out_path = f"{SCRATCH}/speech_output.wav"
    if speak_text(reply, out_path):
        print("Playing response...")
        play_audio(out_path)


def mode_text_to_speech():
    banner("TEXT -> SPEECH (type, hear it spoken)")
    text = input("Text to speak: ").strip()
    if not text:
        return
    out_path = f"{SCRATCH}/tts_output.wav"
    if speak_text(text, out_path):
        print("Playing...")
        play_audio(out_path)


def main_menu():
    while True:
        banner("Sunflower-Gemma4-E2B — Raspberry Pi Demo")
        print("1. Text -> Text (translate / chat)")
        print("2. Speech -> Text (speak, read the answer)")
        print("3. Speech -> Speech (full voice round trip)")
        print("4. Text -> Speech (type, hear it spoken)")
        print("5. Exit")
        choice = input("\nSelect a mode: ").strip()
        if choice == "1":
            mode_text_to_text()
        elif choice == "2":
            mode_speech_to_text()
        elif choice == "3":
            mode_speech_to_speech()
        elif choice == "4":
            mode_text_to_speech()
        elif choice == "5":
            print("Goodbye.")
            sys.exit(0)
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main_menu()
