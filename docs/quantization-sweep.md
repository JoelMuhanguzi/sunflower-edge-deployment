# Quantization comparison: size, translation quality, and Pi speed

[← Back to the project README](../README.md)

Part 1 chose Q4_K_M because it is llama.cpp's usual default, without comparing it with anything. This page does the comparison: **eight quantization levels** of the same model, measured for size, translation quality (on a Mac) and speed and memory (on the Pi).

**Result in one line:** Q4_K_M, the level already deployed, shows no detectable quality loss against the unquantized model, has no broken outputs, and is the fastest or near-fastest on the Pi. Going up to Q5_K_M, Q6_K or Q8_0 cost 20–40% of generation speed and bought nothing measurable. Going down hurt: Q3_K_M lost quality and Q2_K broke.

## The levels

All were produced from the same F16 GGUF with `llama-quantize` (`scripts/quant-sweep/make_quants.sh`).

| Level | Size | Bits per weight (llama-quantize's report) |
|---|---:|---:|
| F16 (unquantized) | 9.27 GB | 16.00 |
| Q8_0 | 4.95 GB | 8.52 |
| Q6_K | 3.83 GB | 6.59 |
| Q5_K_M | 3.62 GB | 6.22 |
| **Q4_K_M** (deployed) | 3.42 GB | 5.88 |
| Q4_0 | 3.35 GB | 5.76 |
| Q3_K_M | 3.19 GB | 5.49 |
| Q2_K | 2.98 GB | 5.12 |

The sizes are bunched far more tightly than the names suggest: "2-bit" Q2_K is only 13% smaller than Q4_K_M, and Q4_K_M is 31% smaller than Q8_0 (F16 to Q4_K_M is 63%). A possible reason is that some parts of this model do not shrink with the headline level (for example large embedding tables); **we did not check this**.

## Translation quality (measured on a Mac)

**Task:** English to Luganda, 200 test sentences, one reference translation each. **Decoding:** temperature 0 (deterministic), one sentence per request, through `llama-server` with the model's own chat template, prompt `Translate to Luganda: <sentence>`. **Scoring:** chrF and BLEU (sacrebleu 2.6) after normalizing curly quotes and whitespace (otherwise every apostrophe the model writes as ’ would count as an error against the reference's '). The "vs F16" column is the difference in chrF with a **95% bootstrap interval** (1,000 resamples of the sentences). "Same output" is the share of sentences whose output is identical to F16's. "Broken" counts outputs that are empty or contain a leaked control token such as `<|think|>` or `<|turn>`.

| Level | chrF | BLEU | chrF vs F16 [95% interval] | Same output as F16 | Broken (of 200) |
|---|---:|---:|---:|---:|---:|
| F16 | 54.1 | 16.9 | — | — | 0 |
| Q8_0 | 54.0 | 17.4 | −0.1 [−0.5, +0.3] | 88% | 0 |
| Q6_K | 54.3 | 17.4 | +0.2 [−0.5, +1.0] | 76% | 0 |
| Q5_K_M | 54.9 | 18.0 | +0.9 [−0.3, +2.2] | 58% | 0 |
| **Q4_K_M** | 53.7 | 16.2 | −0.3 [−1.5, +0.8] | 38% | **0** |
| Q4_0 | 52.2 | 15.6 | −1.8 [−3.5, +0.1] | 26% | 4 |
| Q3_K_M | 49.7 | 13.4 | **−4.3** [−6.5, −2.3] | 13% | 6 |
| Q2_K | 28.6 | 3.6 | **−25.5** [−28.8, −22.3] | 0% | **51** |

How to read it:

- Down to **Q4_K_M**, every interval includes zero: no quality loss we can detect. Q4_0 is borderline (its interval just reaches zero) and already produces a few broken outputs. **Q3_K_M** is clearly worse; **Q2_K** collapses, with 51 of 200 outputs broken (30 contain no translation at all).
- Q5_K_M scoring *higher* than F16 is not a real gain: its interval spans zero. The differences among the top five levels are within noise.
- The "same output" column falls quickly (38% for Q4_K_M) while the score barely moves. Greedy decoding diverges easily once one token changes, so most of those differences are different but similarly good or bad wordings.
- Even the unquantized model scores only chrF 54, and it makes clear mistakes (below). Quantization is not what limits quality here.

### Five sentences, every level

Sentences 1, 41, 81, 121 and 161 of the 200 (every 40th, chosen by rule before looking). `*` marks Q4_K_M. A control token shown in code is what the model actually returned.

**1.** *The maize leaf is yellow however its colour tends to brown at the base.*
Reference: `Ekikoola kya kasooli kya kyenvu wabula langi yaakyo etera okuba eya kitaka wansi.`

| Level | Output |
|---|---|
| F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M* | `Omuti gw'emmere gwa kiragala naye nga gwa kiragala ku mutwe.` |
| Q4_0 | `<|think|>` then `Omuti gw'emmere gwa kyenvu naye nga gwa kyenvu ku mutwe.` |
| Q3_K_M | `Omutuba gwa kasooli gwa kiragala naye nga gwa kiragala ku mutwe.` |
| Q2_K | `<|turn>` (nothing else) |

**2.** *They need to guide the community about the benefits they will receive.*
Reference: `Beetaaga okuuulira abantu ku miganyulo gye bajja okufuna.`

| Level | Output |
|---|---|
| F16, Q8_0, Q4_K_M* | `Beetaaga okutendeka abantu ku mugaso gwe bajja okufuna.` |
| Q6_K, Q5_K_M | `Beetaaga okukulembera abantu ku mugaso gwe bajja okufuna.` |
| Q4_0 | `Balina okugamba abantu ku mugaso gwe balina okufuna.` |
| Q3_K_M | `Balina okugoberera abantu ku mugaso gwe balina okufuna.` |
| Q2_K | `Bateekwa okugambira abantu ku byebakufuna mu byebakola.` |

**3.** *Religious leaders should be obedient and disciplined.*
Reference: `Abakulembeze b'eddiini balina okuba abeetoowaze ate nga ba mpisa.`

| Level | Output |
|---|---|
| F16, Q8_0 | `Abakulembeze b'eddiini balina okuba abawulize era ab'obugunjufu.` |
| Q6_K, Q5_K_M | `Abakulembeze b'eddiini balina okuba abawuliriza era ab'obugunjufu.` |
| Q4_K_M* | `Abakulembeze b'eddiini balina okuba abawuliriza era ab'empisa.` |
| Q4_0 | `Abakulembeze b’eddiini balina okuba abawuliriza era abagoberera amateeka.` |
| Q3_K_M | `Abakulembeze b'eddiini balina okuba abawulirivu era abagoberera.` |
| Q2_K | `Abakulembeze b’eddiini balina okuba abawulize era abakakasa.` |

**4.** *The government will work with various organizations to develop the northern region.*
Reference: `Gavumenti ejja kukolera n'ebitongole eby'enjawulo okukulaakulanya ebitundu by'obukiikakkono.`

| Level | Output |
|---|---|
| F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M*, Q4_0 | `Gavumenti ejja kukolagana n'ebitongole eby'enjawulo okukulaakulanya ekitundu ky'obukiikaddyo.` |
| Q3_K_M | `Gavumenti ejja kukolagana n'ebitongole eby'enjawulo okukulaakulanya ekitundu ky'obukiikakkono.` |
| Q2_K | `Gavumenti egenda kukolagana n'ebitongole eby'enjawulo okukulaakulanya ekikete kya buvanjuzi.` |

**5.** *Currently, we are raising funds to build the boys' dormitory.*
Reference: `Essaawa eno tusonda ssente okuzimba ebisulo by'abalenzi.`

| Level | Output |
|---|---|
| F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M*, Q4_0 | `Mu kiseera kino, tukungaanya ssente okuzimba ekisulo ky'abasajja.` |
| Q3_K_M | `Mu kiseera kino, tukola enteekateeka y’okufuna ensimbi ez’okuzimba ekisulo ky’abasinde.` |
| Q2_K | `<|turn>` (nothing else) |

What the sample shows: the top levels mostly agree with each other, **including their mistakes**. In sentence 1 the reference says the leaf is *yellow* (`kyenvu`) while the main group says *green* (`kiragala`); in sentence 4 the main group's `obukiikaddyo` differs from the reference's `obukiikakkono` for "northern". These look like errors of the unquantized model, not of quantization. **We cannot judge Luganda ourselves, so a Luganda speaker should check them.** The occasional right answer from a worse level (Q4_0 on "yellow", Q3_K_M on "northern") is chance, not improvement.

## Speed and memory on the Raspberry Pi 4

`llama-bench`, CPU only, 4 threads, prompt of 64 tokens and 32 generated tokens, **3 repetitions per model** (mean ± standard deviation), on the rebuilt Pi (llama.cpp `a7b94df`). Peak RAM is the process's peak resident memory, which **includes the memory-mapped model file**; this is a different measure from the "~2.6 GB in use" figure in [Benchmarks](benchmarks.md), which excludes cached file pages. The CPU reached 62–69 °C and the throttling flag read zero at the start and end of each model (it was not sampled continuously).

| Level | Size | Prompt (tok/s) | Generation (tok/s) | Peak RAM |
|---|---:|---:|---:|---:|
| Q4_0 | 3.35 GB | 5.79 ± 0.02 | **2.52 ± 0.01** | 3.40 GB |
| **Q4_K_M** | 3.42 GB | **5.99 ± 0.02** | 2.45 ± 0.01 | 3.47 GB |
| Q5_K_M | 3.62 GB | 4.84 ± 0.01 | 1.97 ± 0.01 | 3.66 GB |
| Q6_K | 3.83 GB | 4.52 ± 0.01 | 1.93 ± 0.00 | 3.87 GB |
| Q8_0 | 4.95 GB | 5.84 ± 0.02 | 1.47 ± 0.01 | 4.96 GB |

- Generation speed falls steadily as the file grows. A plausible reason is that each new word needs the whole file read from memory on this board (memory-bound); **we did not test that directly**.
- Prompt speed does not follow the same pattern: Q8_0 reads prompts about as fast as Q4_K_M, while Q5_K_M and Q6_K are slower. We have no explanation.
- Q8_0 would still fit (4.96 GB of 7.6 GB), but with the 1 GB audio encoder loaded it would run tighter than the others.
- The repeat-to-repeat spread here is tiny (±0.01–0.02), much tighter than the TTS timings elsewhere in these docs. These are controlled benchmark runs on an otherwise idle device; the app on screen will be a little slower.

## Speed and memory on the Raspberry Pi 5

The same `llama-bench` measurement (CPU only, 4 threads, 64-token prompt, 32 generated tokens, 3 repetitions, mean ± standard deviation; llama.cpp `a7b94df`, the same commit as the Pi 4 rows) on a **Raspberry Pi 5** (8 GB, active cooler, 5.1 V / 5 A supply, desktop stopped, CPU below 55 °C at the start of each level, never throttled). Peak RAM is the process's peak resident memory, in the same units as the Pi 4 table. Raw data: `results/pi5-clean-2026-10-08/quant-sweep/` (`scripts/quant-sweep/bench-one-pi.sh`); both boards in `results/quant-speed.csv`.

| Level | Size | Prompt (tok/s) | Generation (tok/s) | Peak RAM | Pi 4 prompt / generation |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 3.35 GB | **68.73 ± 0.13** | **7.80 ± 0.01** | 4.75 GB | 5.79 / 2.52 |
| **Q4_K_M** | 3.42 GB | 40.10 ± 0.04 | 7.79 ± 0.01 | 4.88 GB | 5.99 / 2.45 |
| Q5_K_M | 3.62 GB | 34.35 ± 0.07 | 6.56 ± 0.01 | 5.27 GB | 4.84 / 1.97 |
| Q6_K | 3.83 GB | 35.17 ± 0.05 | 6.15 ± 0.01 | 5.69 GB | 4.52 / 1.93 |
| Q8_0 | 4.95 GB | 27.11 ± 0.07 | 4.44 ± 0.01 | 7.12 GB | 5.84 / 1.47 |

- **Generation speed again falls as the file grows**, from 7.8 to 4.4 tokens/s, and Q4_K_M generates as fast as Q4_0 (7.79 and 7.80; on the Pi 4 Q4_0 was 3% ahead). Generation speed multiplied by file size is roughly constant on each board (22–27 on the Pi 5, 7–8 on the Pi 4), which fits decoding being limited by memory reads, the explanation suggested for the Pi 4; we did not measure memory traffic.
- **The Pi 5 is 3.0–3.3× faster at generation than the Pi 4 at every level, and 4.6–11.9× faster at reading the prompt.** The prompt gain differs a lot by level: Q4_0 gains 11.9×, Q8_0 only 4.6×.
- **Q4_0 reads the prompt 1.7× faster than Q4_K_M on the Pi 5** (68.7 against 40.1 tokens/s) but not on the Pi 4 (5.8 against 6.0). A likely reason is that llama.cpp has ARM kernels for Q4_0 that use the dot-product instructions only the Pi 5 has; we did not check.
- **Peak RAM is 1.35 to 2.2 GB higher on the Pi 5 than on the Pi 4 for the same file** (Q4_K_M 4.88 against 3.47 GB), measured the same way. We do not know why. Q8_0 peaks at **7.12 GB of the 8.06 GB** the board has, so with the 0.99 GB audio encoder loaded it would not fit; Q6_K (5.69 GB) would.
- **Choosing a level is unchanged:** Q4_K_M still has the best quality-for-size in the translation comparison above, and on the Pi 5 it costs nothing in generation speed against Q4_0. Q4_0's faster prompt reading would only matter for long prompts; the apps send short ones.

## Choosing a level

| If you want | Pick |
|---|---|
| The best overall on this device | **Q4_K_M** |
| Maximum generation speed, accepting some risk | Q4_0 (+3% speed, but borderline quality and a few broken outputs) |
| Highest precision regardless of speed | Q8_0 (−40% generation speed, no measurable gain) |
| Smallest file | not Q3_K_M or Q2_K: they lose quality or break |

## Caveats

- **One language direction (English to Luganda), 200 sentences, one reference each.** Other languages, directions and speech transcription were not tested; transcription accuracy under quantization is still unmeasured.
- **Possible training overlap.** The test sentences come from Sunbird's TTS prompt sets (`Prompt-English.csv` and `Prompt-Luganda.csv` in the [SunbirdAI/vits](https://github.com/SunbirdAI/vits) repository, `test` split). We do not know whether the model saw them in training, so absolute scores may be flattering; the *comparison between levels* is still like for like. We have not checked the data's licence, so only five short sentences appear here and the full outputs are not included.
- **chrF and BLEU against a single reference** penalise acceptable wording differences.
- **Quality was measured on the Mac and speed on the Pi.** Quantization changes the stored weights, not what the processor does with them, so the quality scores should carry over, but we did not re-score on the Pi.
- **Speed is a synthetic benchmark** (64-token prompt, 32 tokens). One Pi, one SD card, one llama.cpp build.
- The Pi's Q8_0 file was run as two shards (see [rebuild](rebuild-from-scratch.md) for why); the Mac scores used the single file.

## Reproducing it

```bash
sh scripts/quant-sweep/make_quants.sh              # create the levels from F16
sh scripts/quant-sweep/run_all.sh                  # translate the test sentences with each (Mac; needs the prompt CSVs)
python3 scripts/quant-sweep/score_quants.py        # table with chrF, BLEU, intervals
sh scripts/quant-sweep/bench-quants-pi.sh          # on the Pi: speed and memory
```

`eval_quants.py` expects the two prompt CSVs under `~/ml/vits-work/training/training_files/` (they come with the VITS checkout from [Part 3](03-text-to-speech.md)).
