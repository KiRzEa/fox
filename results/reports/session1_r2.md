# Fox reproduction -- results

All cells are `value [95% CI]` in percentage points. `†` marks a gap against the baseline that the sample size cannot resolve at 95% confidence: the two methods are indistinguishable on this evidence, whatever the point estimates say.

Prompt arm: `--system-prompt llava_v1`, `--wrapper released`, `chair_v1`, `pope_v0_raw`.

### Table 1 -- POPE (nucleus sampling, p=0.9, T=1.0)

| method | adversarial Acc | adversarial F1 | adversarial yes-ratio | adversarial n |
| --- | --- | --- | --- | --- |
| `fox` | 81.40 [79.97, 82.75] | 81.85 [80.34, 83.30] | 52.5 | 3000 |
| `sampling` | 77.17 [75.63, 78.63] | 79.41 [77.85, 80.87] | 60.9 | 3000 |

Paper-reported accuracy (LLaVA-1.5), for reference:

| method | adversarial |
| --- | --- |
| `sampling` | 76.77 |
| `vcd` | 75.33 |
| `icd` | 77.90 |
| `opera` | 81.13 |
| `causalmm` | 81.27 |
| `sid` | 81.10 |
| `fox` | 81.93 |

**Gaps against `sampling`** († = not resolvable at 95%):

- POPE/adversarial: fox vs sampling: gap +4.23 pts, 95% CI [+2.92, +5.47] -- distinguishable  _[exact McNemar (discordant=423)]_


### Table 2 -- CHAIR (500 COCO val2014 images)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 49.60 [45.24, 53.97] | 500 | 14.08 [12.49, 15.71] | 3829 | 76.35 | 75.31 | 96.3 |
| `sampling` | 56.80 [52.42, 61.07] | 500 | 18.14 [16.56, 19.78] | 3599 | 76.09 | 72.40 | 102.5 |

Paper-reported CHAIR (LLaVA-1.5), for reference:

| method | CHAIR_S | CHAIR_I |
| --- | --- | --- |
| `sampling` | 57.60 | 17.18 |
| `vcd` | 54.00 | 16.02 |
| `icd` | 53.40 | 15.67 |
| `opera` | 49.00 | 13.52 |
| `causalmm` | 47.60 | 13.49 |
| `sid` | 53.40 | 15.40 |
| `fox` | 46.40 | 12.90 |

_The paper states neither CHAIR_I's denominator (total mentioned object instances) nor any interval, so the reference column cannot be given error bars._

**Gaps against `sampling`** († = not resolvable at 95%):

- CHAIR_S: fox vs sampling: gap -7.20 pts, 95% CI [-12.00, -2.40] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -4.07 pts, 95% CI [-5.79, -2.32] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_


### Table 9a -- POPE under greedy decoding (control)

_no POPE runs found._


### Table 9b -- CHAIR under greedy decoding (control)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 46.20 [41.88, 50.58] † | 500 | 13.45 [11.81, 15.21] † | 3896 | 77.82 | 76.62 | 97.3 |
| `sampling` | 49.20 [44.84, 53.57] | 500 | 13.82 [12.28, 15.39] | 3762 | 79.23 | 76.60 | 100.5 |

Paper-reported CHAIR (LLaVA-1.5), for reference:

| method | CHAIR_S | CHAIR_I |
| --- | --- | --- |
| `sampling` | 50.90 | 15.40 |
| `vcd` | 54.60 | 16.60 |
| `opera` | 49.00 | 13.52 |
| `sid` | 48.00 | 13.60 |
| `fox` | 42.60 | 12.16 |

_The paper states neither CHAIR_I's denominator (total mentioned object instances) nor any interval, so the reference column cannot be given error bars._

**Gaps against `sampling`** († = not resolvable at 95%):

- CHAIR_S: fox vs sampling: gap -3.00 pts, 95% CI [-7.40, +1.40] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -0.37 pts, 95% CI [-1.74, +0.99] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_


### Resolution budget

Smallest gap each run can resolve (two-sided 95%, 80% power):

| benchmark / split | metric | n | p | smallest resolvable gap (pts) |
| --- | --- | --- | --- | --- |
| chair/val2014_500 | CHAIR_S | 500 | 0.4620 | 8.85 |
| chair/val2014_500 | CHAIR_I | 3896 | 0.1345 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3829 | 0.1408 | 2.30 |
| pope/adversarial | accuracy | 3000 | 0.8140 | 2.73 |
| chair/val2014_500 | CHAIR_I | 3762 | 0.1382 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3599 | 0.1814 | 2.61 |

Samples **per method** needed for a given gap at p = 0.82 (`paired_rho` is the correlation between two methods' per-sample outcomes; paired runs on identical inputs are strongly correlated):

| gap (pts) | rho = 0 | rho = 0.5 | rho = 0.8 |
| --- | --- | --- | --- |
| 0.5 | 91,670 | 45,835 | 18,334 |
| 1.0 | 22,663 | 11,332 | 4,533 |
| 2.0 | 5,537 | 2,769 | 1,108 |
| 5.0 | 822 | 411 | 165 |


### Provenance audit

| run | method | bench/split | dtype | gamma | k | tau_js | seed | n | model | commit | transformers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |

CHAIR scorer: synonyms `chair_synonyms.txt` sha256 `a4c231100e8a...` (transcription-of-rohrbach-synonyms-txt), tokenizer `regex`, gt_source `both`

**Warnings**

- `fox` run with **gamma = None**: the intervention constant was not recorded.
