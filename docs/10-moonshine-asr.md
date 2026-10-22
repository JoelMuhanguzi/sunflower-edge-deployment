# Part 10: Moonshine for English listening, and Gemma as the translator in the middle

This part tests a small English-only speech recogniser, Moonshine streaming (Useful Sensors, MIT licence), as a replacement for the
"listen" stage, and then asks whether Gemma (kept loaded, text only) should do the translation instead of NLLB.
Everything here is English speech in, Luganda speech out. Moonshine does not understand Luganda, so Luganda speech still needs
Whisper (SunflowerASR-51) or Gemma's own audio encoder.

Raw data and tables: [`results/asr-moonshine-2026-10/`](../results/asr-moonshine-2026-10/).
Code: [`scripts/asr/moonshine_test.py`](../scripts/asr/moonshine_test.py) and setups F and G in
[`scripts/fast/bench_compare.py`](../scripts/fast/bench_compare.py).

## 1. The models

| Model | Parameters (model card) | Weights as published | Notes |
|---|---:|---:|---|
| Moonshine streaming tiny | 34 M | 172 MB (fp32) | |
| Moonshine streaming small | 123 M | 539 MB | |
| Moonshine streaming medium | 245 M | 1,018 MB | model card reports average WER 6.65% on the Open ASR datasets |

The file metadata on Hugging Face says 44 M, 140 M and 266 M parameters; the card's numbers are the ones to quote, and the difference is worth a
footnote. English only, trained on about 300,000 hours; the card's benchmark WERs are from the authors, not ours.
We compared with Nemotron 3.5 ASR streaming 0.6B on paper only: it covers 40 locales, none of them Luganda or Swahili, and was not run.

**What was not exercised: streaming.** Each clip was transcribed whole with `generate()`. The Hugging Face implementation does not yet stream
efficiently (the model card says so), so the time-to-first-words benefit of a streaming recogniser is not measured here.

## 2. Speed and accuracy of Moonshine alone (Pi 5, 4 threads, three English clips of 5 s)

| Model | Time, fp32 | Time, int8 | WER, fp32 | WER, int8 | File, fp32 | File, int8 |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 0.70 s | 0.39 s | 0% | 3.3% | 172 MB | 86 MB |
| small | 2.07 s | 1.05 s | 0% | 3.3% | 539 MB | 222 MB |
| medium | 4.36 s | 2.06 s | 0% | 0% | 1,018 MB | 375 MB |

Median of three runs per clip; the CPU was not throttled (`vcgencmd get_throttled` = 0x0) and the board had the active cooler and the 5 V / 5 A supply.
On the Mac the same code takes 0.04 s (tiny), 0.11 s (small) and 0.21 s (medium) per clip in fp32.
For comparison, listening to the same kind of clip took 7.3 s with Gemma's audio encoder and 7.7 s with SunflowerASR-51 (int8) in Part 8.
The two int8 errors are the name "Joel" heard as "Jill" (tiny) and "Juro" (small): in each case one wrong word in the 10-word clip "My name is Joel, I am happy to see you". Three clips cannot rank the models on accuracy.

## 3. Quantization: exactly what was done

int8 was applied to Moonshine **at load time**, with PyTorch dynamic quantization (nothing is exported or stored as an int8 file):

```python
model = MoonshineStreamingForConditionalGeneration.from_pretrained(path).float().eval()
torch.backends.quantized.engine = "qnnpack"        # the ARM engine
model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

- Only the `nn.Linear` layers (the bulk of the parameters) get int8 weights; activations are quantized on the fly for each call. Convolutions, embeddings and
  normalisation stay in 32-bit float. That is why the int8 files are only 2.0 to 2.7 times smaller, not 4 times smaller.
- The "file, int8" column is the size of `torch.save(model.state_dict())` after quantizing. The parameter count the script prints after quantizing
  (14 M, 27 M, 36 M) only counts the layers that were left unquantized and should not be quoted as the model size.
- `torch.ao.quantization` prints a deprecation warning (PyTorch is moving this to the separate `torchao` package). It works in PyTorch 2.14.
- Moonshine's own edge runtime (Moonshine Voice) and an ONNX int8 export were **not** tried, and may be faster than this plain PyTorch path.
- Quantization is not always a speed-up on these boards: the same kind of int8 conversion made the ONNX voice 3 to 6 times *slower* (Part 6).
  Here it roughly halves the time, so it has to be measured per model.

Quantization steps made earlier in the project are in [quantization-sweep.md](quantization-sweep.md) (Gemma Q4_0 to Q8_0), Part 8 (Whisper and NLLB int8
against float32), and Part 6 (the ONNX voice).

## 4. Pipelines: Moonshine listening, with NLLB or Gemma translating (Pi 5, English clips only)

| Setup | Listen | Translate | Voice | Total | Peak memory |
|---|---:|---:|---:|---:|---:|
| D: Whisper int8 + NLLB int8 (beam 5) + ONNX voice (Part 8 run, 8 October) | 7.7 | 7.8 | 2.0 | 17.6 | not recorded |
| D with NLLB beam 1 (8 October) | 8.5 | 6.6 | 2.1 | 17.3 | not recorded |
| E: Gemma listens and translates + ONNX voice (8 October) | 7.3 | 3.7 | 2.1 | 13.2 | not recorded |
| **F**: Moonshine medium int8 + NLLB int8 (beam 5) + ONNX voice | 2.1 | 13.3 | 2.0 | 17.6 | 3.7 GB |
| F with NLLB beam 1 | 2.1 | 7.5 | 1.9 | 11.7 | 3.7 GB |
| **G**: Moonshine medium int8 + Gemma (text only) + ONNX voice | 2.1 | 3.8 | 1.9 | **7.9** | 5.8 GB |
| G with Moonshine tiny | 0.4 | 3.0 | 1.7 | **5.2** | 5.4 GB |

Median seconds. F and G: 9 runs (3 English clips, 3 repeats) after one unscored warm-up. The D and E rows are the 3 English clips of the earlier six-clip run, so each is a median of only 3 runs.
No run throttled. In G the Gemma server is started without its audio encoder (text translation only).
Table: [`results/asr-moonshine-2026-10/pipelines-english-pi5.csv`](../results/asr-moonshine-2026-10/pipelines-english-pi5.csv).
Every transcript and translation is in the `heard` and `translated` fields of the files in `results/asr-moonshine-2026-10/pipelines/`.

What it shows:
- Replacing the listening stage with Moonshine saves 5 to 7 s. With Gemma translating, the pipeline is about 2.2 times faster than D (17.6 s to 7.9 s) and about 1.7 times faster than E.
- Gemma in the middle costs memory (5.4 to 5.8 GB peak against 3.7 GB with NLLB): fine on the Pi 5, tight on the Pi 4.
- The NLLB translations in F are word for word identical to those in D, so the newer library's tokenizer warning did not change the output.

## 5. Translation quality: not yet settled

Speed was measured, quality was not scored. Examples (author, a Luganda speaker, to judge):

| English | NLLB (D, F) | Gemma (G) |
|---|---|---|
| The hospital is near the market | Eddwaliro liri kumpi n'akatale | Ekifo ekirimu eddwaliro kiri kumpi n'akatale |
| I am very sick, please take me to the hospital | Ndi mulwadde nnyo, nkusaba ontwale mu ddwaliro | Ndi mulwadde nnyo, weeyongera onnyambe mu ddwaliro |

On these two the NLLB output reads closer to the English. A fair comparison needs reference translations and a metric (chrF or BLEU) on more than three sentences.
With Moonshine tiny the misheard name ("Jill") went straight into the translation.

## 6. Caveats and open questions

- Three English clips, one speaker, quiet room. No accent test yet; the clips are the author's.
- NLLB at beam 5 took 13.3 s to translate in F but 7.8 s in D for the same sentences. The cause is not found (a guess: the PyTorch and CTranslate2 thread pools compete in one
  process). Beam 1 gives 7.5 s in F. This echoes the unexplained isolated-versus-pipeline NLLB timing in Part 8.
- The D and E rows were measured on 8 October and the F and G rows on 22 October, on the same board and setup but not interleaved.
- Memory for the earlier rows was not recorded in the per-run files.
- Moonshine is run through PyTorch fp32/int8 on the CPU; the project's other models use llama.cpp, CTranslate2 and ONNX Runtime, so the frameworks are not equal.

## 7. How to reproduce

```bash
# on the Mac: export the three models as plain folders (the Hugging Face cache stores them as links)
pip install torch "git+https://github.com/huggingface/transformers.git" scipy soundfile
python - <<'E'
from transformers import MoonshineStreamingForConditionalGeneration, AutoProcessor
for s in ["tiny", "small", "medium"]:
    r = f"moonshine-ai/moonshine-streaming-{s}"
    MoonshineStreamingForConditionalGeneration.from_pretrained(r).save_pretrained(f"moonshine_models/{s}")
    AutoProcessor.from_pretrained(r).save_pretrained(f"moonshine_models/{s}")
E
# copy moonshine_models/ to the Pi (1.7 GB; a USB stick works) into ~/ml/

# on the Pi 5: a separate environment. Use the CPU build of PyTorch, the default wheel pulls 4 GB of NVIDIA libraries and filled the Pi's disk.
python3 -m venv ~/ml/moonshine-venv
~/ml/moonshine-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/ml/moonshine-venv/bin/pip install numpy scipy soundfile ctranslate2 onnxruntime sentencepiece "git+https://github.com/huggingface/transformers.git"

python scripts/asr/moonshine_test.py --repo ~/ml/moonshine_models/{size} --int8 --clips ~/ml/bench_clips/eng_*.wav
MOONSHINE_SIZE=medium PYTHONPATH=~/ml:~/ml/pipeline2 ~/ml/moonshine-venv/bin/python scripts/fast/bench_compare.py G --repeat 3
```

## 8. Next steps

1. A larger English test set with accented speakers (the project's users), scored with word error rate and the same normalisation for every recogniser.
2. Run Whisper and Gemma on the same English set, so the comparison with Moonshine is like for like (Part 8's numbers mix English and Luganda clips).
3. Try Moonshine's own runtime and an ONNX int8 export, and use real streaming in the hands-free voice app.
4. Score Gemma's and NLLB's translations against references.
5. Fine-tuning for Ugandan-accented English, if step 1 shows a gap (Nemotron's published fine-tuning recipe covers accents).
