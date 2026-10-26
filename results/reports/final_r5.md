# Fox reproduction -- results

All cells are `value [95% CI]` in percentage points. `†` marks a gap against the baseline that the sample size cannot resolve at 95% confidence: the two methods are indistinguishable on this evidence, whatever the point estimates say.

Prompt arm: `--system-prompt llava_v1`, `--wrapper released`, `chair_v1`, `pope_v0_raw`.

The tables below are **seed 42 alone** (of 11 present), because a single run is what their within-run intervals describe. The claim this report actually makes is in **Across seeds**, further down.

### Table 1 -- POPE (nucleus sampling, p=0.9, T=1.0)

| method | random Acc | random F1 | random yes-ratio | random n | popular Acc | popular F1 | popular yes-ratio | popular n | adversarial Acc | adversarial F1 | adversarial yes-ratio | adversarial n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 89.10 [87.93, 90.17] | 88.52 [87.28, 89.72] | 45.0 | 3000 | 86.93 [85.68, 88.09] | 86.53 [85.18, 87.80] | 47.0 | 3000 | 81.40 [79.97, 82.75] | 81.85 [80.34, 83.30] | 52.5 | 3000 |
| `sampling` | 85.07 [83.75, 86.30] | 85.39 [84.05, 86.69] | 52.2 | 3000 | 82.10 [80.69, 83.43] | 82.93 [81.47, 84.32] | 54.8 | 3000 | 77.17 [75.63, 78.63] | 79.41 [77.85, 80.87] | 60.9 | 3000 |
| `vcd` | - | - | - | - | - | - | - | - | 78.83 [77.34, 80.26] † | 79.02 [77.39, 80.56] | 50.9 | 3000 |

Paper-reported accuracy (LLaVA-1.5), for reference:

| method | random | popular | adversarial |
| --- | --- | --- | --- |
| `sampling` | 85.13 | 82.53 | 76.77 |
| `vcd` | 86.84 | 80.87 | 75.33 |
| `icd` | 86.27 | 83.57 | 77.90 |
| `opera` | 89.27 | 86.80 | 81.13 |
| `causalmm` | 89.15 | 86.63 | 81.27 |
| `sid` | 88.67 | 85.37 | 81.10 |
| `fox` | 89.33 | 86.70 | 81.93 |

**Gaps against `sampling`** († = not resolvable at 95%):

- POPE/random: fox vs sampling: gap +4.03 pts, 95% CI [+2.86, +5.12] -- distinguishable  _[exact McNemar (discordant=343)]_
- POPE/popular: fox vs sampling: gap +4.83 pts, 95% CI [+3.60, +5.97] -- distinguishable  _[exact McNemar (discordant=385)]_
- POPE/adversarial: fox vs sampling: gap +4.23 pts, 95% CI [+2.92, +5.47] -- distinguishable  _[exact McNemar (discordant=423)]_
- POPE/adversarial: vcd vs sampling: gap +1.67 pts, 95% CI [+0.02, +3.30] -- NOT distinguishable at 95%  _[exact McNemar (discordant=638)]_


### Table 2 -- CHAIR (500 COCO val2014 images)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words | correct objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 49.60 [45.24, 53.97] | 500 | 14.08 [12.49, 15.71] | 3829 | 76.35 | 75.31 | 96.3 | 1191 |
| `sampling` | 56.80 [52.42, 61.07] | 500 | 18.14 [16.56, 19.78] | 3599 | 76.09 | 72.40 | 102.5 | 1187 |
| `vcd` | 41.60 [37.36, 45.97] | 500 | 14.84 [13.02, 16.67] | 3106 | 66.35 ⚠ | 70.99 | 95.3 | 1035 |

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
- CHAIR_S: vcd vs sampling: gap -15.20 pts, 95% CI [-20.40, -10.00] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: vcd vs sampling: gap -3.30 pts, 95% CI [-5.26, -1.38] -- distinguishable  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- ⚠ `vcd`: recall 66.35 against the baseline's 76.09, and 1035 objects named correctly against 1187. Its CHAIR scores are **not** evidence of better grounding: a caption that names fewer objects hallucinates fewer.


### Table 9a -- POPE under greedy decoding (control)

| method | random Acc | random F1 | random yes-ratio | random n | popular Acc | popular F1 | popular yes-ratio | popular n | adversarial Acc | adversarial F1 | adversarial yes-ratio | adversarial n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 89.53 [88.39, 90.58] † | 88.96 [87.71, 90.14] | 44.8 | 3000 | 86.90 [85.65, 88.06] | 86.51 [85.15, 87.80] | 47.1 | 3000 | 82.37 [80.96, 83.69] | 82.65 [81.15, 84.10] | 51.6 | 3000 |
| `sampling` | 89.33 [88.18, 90.39] | 89.28 [88.09, 90.42] | 49.5 | 3000 | 85.90 [84.61, 87.10] | 86.29 [84.97, 87.55] | 52.8 | 3000 | 79.03 [77.54, 80.45] | 80.89 [79.41, 82.34] | 59.7 | 3000 |

Paper-reported accuracy (LLaVA-1.5), for reference:

| method | random | popular | adversarial |
| --- | --- | --- | --- |
| `sampling` | 89.33 | 85.93 | 79.10 |
| `vcd` | 87.97 | 85.33 | 78.43 |
| `opera` | 89.27 | 86.80 | 81.13 |
| `sid` | 89.33 | 86.47 | 81.77 |
| `fox` | 89.43 | 87.07 | 82.43 |

**Gaps against `sampling`** († = not resolvable at 95%):

- POPE/random: fox vs sampling: gap +0.20 pts, 95% CI [-0.57, +0.96] -- NOT distinguishable at 95%  _[exact McNemar (discordant=142)]_
- POPE/popular: fox vs sampling: gap +1.00 pts, 95% CI [+0.14, +1.82] -- distinguishable  _[exact McNemar (discordant=174)]_
- POPE/adversarial: fox vs sampling: gap +3.33 pts, 95% CI [+2.36, +4.21] -- distinguishable  _[exact McNemar (discordant=244)]_


### Table 9b -- CHAIR under greedy decoding (control)

| method | CHAIR_S (lower better) | CHAIR_S n (captions) | CHAIR_I (lower better) | CHAIR_I n (mentions) | recall | F1 | mean words | correct objects |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fox` | 46.20 [41.88, 50.58] † | 500 | 13.45 [11.81, 15.21] † | 3896 | 77.82 | 76.62 | 97.3 | 1214 |
| `sampling` | 49.20 [44.84, 53.57] | 500 | 13.82 [12.28, 15.39] | 3762 | 79.23 | 76.60 | 100.5 | 1236 |
| `fox` | 41.40 [37.16, 45.77] | 500 | 11.91 [10.45, 13.38] † | 3914 | 76.76 | 76.91 | 95.8 | 1209 |
| `sampling` | 46.40 [42.07, 50.78] | 500 | 12.96 [11.53, 14.40] | 3782 | 78.86 | 76.98 | 98.7 | 1242 |
| `fox` | 44.60 [40.30, 48.98] | 500 | 13.15 [11.57, 14.73] † | 3849 | 77.64 | 76.80 | 97.0 | 1177 |
| `sampling` | 49.40 [45.04, 53.77] | 500 | 14.39 [12.86, 15.91] | 3711 | 79.42 | 76.74 | 99.2 | 1204 |
| `fox` | 49.40 [45.04, 53.77] † | 500 | 14.81 [13.18, 16.54] † | 4017 | 77.69 | 75.92 | 97.0 | 1247 |
| `sampling` | 49.80 [45.44, 54.17] | 500 | 14.81 [13.23, 16.48] | 3775 | 78.63 | 76.35 | 100.3 | 1262 |
| `fox` | 44.00 [39.71, 48.38] † | 500 | 12.85 [11.29, 14.43] † | 3884 | 76.51 | 76.72 | 94.6 | 1251 |
| `sampling` | 47.60 [43.26, 51.98] | 500 | 13.16 [11.71, 14.61] | 3799 | 78.78 | 77.66 | 99.5 | 1288 |

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
- CHAIR_S: fox vs sampling: gap -5.00 pts, 95% CI [-9.40, -0.60] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -1.05 pts, 95% CI [-2.37, +0.28] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -4.80 pts, 95% CI [-9.20, -0.40] -- distinguishable  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -1.24 pts, 95% CI [-2.71, +0.18] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -0.40 pts, 95% CI [-4.40, +3.60] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap +0.00 pts, 95% CI [-1.48, +1.47] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_
- CHAIR_S: fox vs sampling: gap -3.60 pts, 95% CI [-8.00, +0.80] -- NOT distinguishable at 95%  _[paired percentile bootstrap (B=10000, seed=20260101)]_
- CHAIR_I: fox vs sampling: gap -0.31 pts, 95% CI [-1.58, +0.95] -- NOT distinguishable at 95%  _[paired cluster bootstrap on ratio of sums (B=10000, seed=20260101)]_


### Across seeds

`mean gap` pairs the two methods seed by seed and puts a Student-t interval on the mean of those paired differences. That is the only interval here that sees the generation draw; every interval in the tables above is computed from a single run and quantifies item difficulty alone.

`best gap` is the best seed of each method differenced. It is reported because the paper's aggregation is ambiguous -- C.2 says "the best results averaged over 10 independent runs" without saying what "best" ranges over. If it ranges over runs, `best gap` is the comparable figure and `mean gap` is not.

| artefact | method | metric | seeds | sampling mean+-sd | method mean+-sd | mean gap [95% CI] | p | p (Holm) | best gap | paper gap | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chair/val2014_500 (greedy) | `fox` | CHAIR_S | 1 | 49.20+-0.00 | 46.20+-0.00 | n/a -- deterministic | - | - | -3.00 | -8.30 | see table above |
| chair/val2014_500 (greedy) | `fox` | CHAIR_I | 1 | 13.82+-0.00 | 13.45+-0.00 | n/a -- deterministic | - | - | -0.37 | -3.24 | see table above |
| chair/val2014_500 (greedy) | `fox as_described` | CHAIR_S | 1 | 49.20+-0.00 | 50.60+-0.00 | n/a -- deterministic | - | - | +1.40 | - | see table above |
| chair/val2014_500 (greedy) | `fox as_described` | CHAIR_I | 1 | 13.82+-0.00 | 14.37+-0.00 | n/a -- deterministic | - | - | +0.54 | - | see table above |
| chair/val2014_500 (greedy) | `fox boost_visual g=4` | CHAIR_S | 1 | 49.20+-0.00 | 42.80+-0.00 | n/a -- deterministic | - | - | -6.40 | - | see table above |
| chair/val2014_500 (greedy) | `fox boost_visual g=4` | CHAIR_I | 1 | 13.82+-0.00 | 11.52+-0.00 | n/a -- deterministic | - | - | -2.30 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=100` | CHAIR_S | 1 | 49.20+-0.00 | 49.40+-0.00 | n/a -- deterministic | - | - | +0.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=100` | CHAIR_I | 1 | 13.82+-0.00 | 13.66+-0.00 | n/a -- deterministic | - | - | -0.17 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=10000` | CHAIR_S | 1 | 49.20+-0.00 | 46.20+-0.00 | n/a -- deterministic | - | - | -3.00 | - | see table above |
| chair/val2014_500 (greedy) | `fox g=10000` | CHAIR_I | 1 | 13.82+-0.00 | 13.45+-0.00 | n/a -- deterministic | - | - | -0.37 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=bf16` | CHAIR_S | 1 | 49.20+-0.00 | 25.60+-0.00 | n/a -- deterministic | - | - | -23.60 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=bf16` | CHAIR_I | 1 | 13.82+-0.00 | 10.62+-0.00 | n/a -- deterministic | - | - | -3.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=fp32` | CHAIR_S | 1 | 49.20+-0.00 | 49.00+-0.00 | n/a -- deterministic | - | - | -0.20 | - | see table above |
| chair/val2014_500 (greedy) | `fox ld=fp32` | CHAIR_I | 1 | 13.82+-0.00 | 13.80+-0.00 | n/a -- deterministic | - | - | -0.02 | - | see table above |
| chair/val2014_500 (greedy) | `fox noise_control` | CHAIR_S | 1 | 49.20+-0.00 | 44.60+-0.00 | n/a -- deterministic | - | - | -4.60 | - | see table above |
| chair/val2014_500 (greedy) | `fox noise_control` | CHAIR_I | 1 | 13.82+-0.00 | 11.88+-0.00 | n/a -- deterministic | - | - | -1.95 | - | see table above |
| chair/val2014_500 (sampling) | `fox` | CHAIR_S | 10 | 53.72+-2.39 | 46.74+-2.12 | -6.98 [-9.25, -4.71] | 6.6e-05 | 0.00086 | -5.80 | -11.20 | **excludes paper** |
| chair/val2014_500 (sampling) | `fox` | CHAIR_I | 10 | 16.39+-0.86 | 13.72+-0.54 | -2.67 [-3.37, -1.96] | 1.3e-05 | 0.00018 | -2.02 | -4.28 | **excludes paper** |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_S | 9 | 53.84+-2.50 | 40.62+-0.94 | -13.22 [-15.05, -11.39] | 1.7e-07 | 3.2e-06 | -10.40 | - | resolved |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_I | 9 | 16.40+-0.91 | 11.62+-0.45 | -4.77 [-5.36, -4.19] | 6.9e-08 | 1.4e-06 | -3.98 | - | resolved |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_S | 3 | 54.40+-2.40 | 49.20+-0.20 | -5.20 [-11.43, +1.03] | 0.069 | 0.22 **ns** | -3.00 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_I | 3 | 16.90+-1.09 | 14.05+-0.25 | -2.85 [-5.15, -0.56] | 0.033 | 0.2 **ns** | -2.36 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_S | 3 | 54.40+-2.40 | 45.53+-1.14 | -8.87 [-17.07, -0.67] | 0.043 | 0.22 **ns** | -7.40 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_I | 3 | 16.90+-1.09 | 13.34+-0.62 | -3.56 [-7.45, +0.33] | 0.059 | 0.22 **ns** | -3.16 | - | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_S | 3 | 54.40+-2.40 | 40.53+-1.36 | -13.87 [-20.04, -7.69] | 0.011 | 0.095 **ns** | -13.00 | -3.60 | not resolved after Holm |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_I | 3 | 16.90+-1.09 | 14.56+-0.28 | -2.34 [-4.54, -0.14] | 0.045 | 0.22 **ns** | -1.83 | -1.16 | not resolved after Holm |
| chair/val2014_500_d1 (greedy) | `fox` | CHAIR_S | 1 | 46.40+-0.00 | 41.40+-0.00 | n/a -- deterministic | - | - | -5.00 | -8.30 | see table above |
| chair/val2014_500_d1 (greedy) | `fox` | CHAIR_I | 1 | 12.96+-0.00 | 11.91+-0.00 | n/a -- deterministic | - | - | -1.05 | -3.24 | see table above |
| chair/val2014_500_d2 (greedy) | `fox` | CHAIR_S | 1 | 49.40+-0.00 | 44.60+-0.00 | n/a -- deterministic | - | - | -4.80 | -8.30 | see table above |
| chair/val2014_500_d2 (greedy) | `fox` | CHAIR_I | 1 | 14.39+-0.00 | 13.15+-0.00 | n/a -- deterministic | - | - | -1.24 | -3.24 | see table above |
| chair/val2014_500_d3 (greedy) | `fox` | CHAIR_S | 1 | 49.80+-0.00 | 49.40+-0.00 | n/a -- deterministic | - | - | -0.40 | -8.30 | see table above |
| chair/val2014_500_d3 (greedy) | `fox` | CHAIR_I | 1 | 14.81+-0.00 | 14.81+-0.00 | n/a -- deterministic | - | - | +0.00 | -3.24 | see table above |
| chair/val2014_500_d4 (greedy) | `fox` | CHAIR_S | 1 | 47.60+-0.00 | 44.00+-0.00 | n/a -- deterministic | - | - | -3.60 | -8.30 | see table above |
| chair/val2014_500_d4 (greedy) | `fox` | CHAIR_I | 1 | 13.16+-0.00 | 12.85+-0.00 | n/a -- deterministic | - | - | -0.31 | -3.24 | see table above |
| pope/adversarial (greedy) | `fox` | accuracy | 1 | 79.03+-0.00 | 82.37+-0.00 | n/a -- deterministic | - | - | +3.33 | +3.33 | see table above |
| pope/adversarial (greedy) | `fox` | F1 | 1 | 80.89+-0.00 | 82.65+-0.00 | n/a -- deterministic | - | - | +1.76 | +1.69 | see table above |
| pope/adversarial (greedy) | `fox alpha=1` | accuracy | 1 | 79.03+-0.00 | 81.83+-0.00 | n/a -- deterministic | - | - | +2.80 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=1` | F1 | 1 | 80.89+-0.00 | 82.44+-0.00 | n/a -- deterministic | - | - | +1.55 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=4` | accuracy | 1 | 79.03+-0.00 | 82.37+-0.00 | n/a -- deterministic | - | - | +3.33 | - | see table above |
| pope/adversarial (greedy) | `fox alpha=4` | F1 | 1 | 80.89+-0.00 | 82.44+-0.00 | n/a -- deterministic | - | - | +1.56 | - | see table above |
| pope/adversarial (greedy) | `fox as_described` | accuracy | 1 | 79.03+-0.00 | 79.43+-0.00 | n/a -- deterministic | - | - | +0.40 | - | see table above |
| pope/adversarial (greedy) | `fox as_described` | F1 | 1 | 80.89+-0.00 | 81.13+-0.00 | n/a -- deterministic | - | - | +0.24 | - | see table above |
| pope/adversarial (greedy) | `fox boost_visual g=4` | accuracy | 1 | 79.03+-0.00 | 80.70+-0.00 | n/a -- deterministic | - | - | +1.67 | - | see table above |
| pope/adversarial (greedy) | `fox boost_visual g=4` | F1 | 1 | 80.89+-0.00 | 81.80+-0.00 | n/a -- deterministic | - | - | +0.91 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.25` | accuracy | 1 | 79.03+-0.00 | 81.60+-0.00 | n/a -- deterministic | - | - | +2.57 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.25` | F1 | 1 | 80.89+-0.00 | 82.17+-0.00 | n/a -- deterministic | - | - | +1.28 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.45` | accuracy | 1 | 79.03+-0.00 | 81.80+-0.00 | n/a -- deterministic | - | - | +2.77 | - | see table above |
| pope/adversarial (greedy) | `fox k=0.45` | F1 | 1 | 80.89+-0.00 | 82.05+-0.00 | n/a -- deterministic | - | - | +1.16 | - | see table above |
| pope/adversarial (greedy) | `fox ld=bf16` | accuracy | 1 | 79.03+-0.00 | 81.97+-0.00 | n/a -- deterministic | - | - | +2.93 | - | see table above |
| pope/adversarial (greedy) | `fox ld=bf16` | F1 | 1 | 80.89+-0.00 | 81.91+-0.00 | n/a -- deterministic | - | - | +1.03 | - | see table above |
| pope/adversarial (greedy) | `fox ld=fp32` | accuracy | 1 | 79.03+-0.00 | 79.07+-0.00 | n/a -- deterministic | - | - | +0.03 | - | see table above |
| pope/adversarial (greedy) | `fox ld=fp32` | F1 | 1 | 80.89+-0.00 | 80.92+-0.00 | n/a -- deterministic | - | - | +0.04 | - | see table above |
| pope/adversarial (greedy) | `fox noise_control` | accuracy | 1 | 79.03+-0.00 | 80.07+-0.00 | n/a -- deterministic | - | - | +1.03 | - | see table above |
| pope/adversarial (greedy) | `fox noise_control` | F1 | 1 | 80.89+-0.00 | 81.53+-0.00 | n/a -- deterministic | - | - | +0.64 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.1` | accuracy | 1 | 79.03+-0.00 | 82.20+-0.00 | n/a -- deterministic | - | - | +3.17 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.1` | F1 | 1 | 80.89+-0.00 | 82.65+-0.00 | n/a -- deterministic | - | - | +1.76 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.2` | accuracy | 1 | 79.03+-0.00 | 82.80+-0.00 | n/a -- deterministic | - | - | +3.77 | - | see table above |
| pope/adversarial (greedy) | `fox tau=0.2` | F1 | 1 | 80.89+-0.00 | 83.14+-0.00 | n/a -- deterministic | - | - | +2.25 | - | see table above |
| pope/adversarial (sampling) | `fox` | accuracy | 10 | 76.72+-0.31 | 81.76+-0.33 | +5.03 [+4.68, +5.38] | 1.2e-10 | 3.1e-09 | +4.90 | +5.16 | reproduces |
| pope/adversarial (sampling) | `fox` | F1 | 10 | 78.93+-0.28 | 82.13+-0.32 | +3.21 [+2.88, +3.53] | 3.7e-09 | 8.1e-08 | +3.18 | - | resolved |
| pope/adversarial (sampling) | `fox boost_visual g=4` | accuracy | 10 | 76.72+-0.31 | 80.46+-0.34 | +3.74 [+3.46, +4.02] | 2.1e-10 | 5.1e-09 | +3.73 | - | resolved |
| pope/adversarial (sampling) | `fox boost_visual g=4` | F1 | 10 | 78.93+-0.28 | 81.62+-0.30 | +2.69 [+2.42, +2.96] | 3e-09 | 6.8e-08 | +2.66 | - | resolved |
| pope/adversarial (sampling) | `fox ld=bf16` | accuracy | 10 | 76.72+-0.31 | 81.77+-0.28 | +5.05 [+4.69, +5.41] | 1.5e-10 | 3.8e-09 | +4.83 | - | resolved |
| pope/adversarial (sampling) | `fox ld=bf16` | F1 | 10 | 78.93+-0.28 | 81.62+-0.26 | +2.69 [+2.36, +3.03] | 2.1e-08 | 4.4e-07 | +2.54 | - | resolved |
| pope/adversarial (sampling) | `fox ld=fp32` | accuracy | 9 | 76.75+-0.31 | 78.90+-0.22 | +2.15 [+1.83, +2.47] | 3.1e-07 | 5.2e-06 | +1.90 | - | resolved |
| pope/adversarial (sampling) | `fox ld=fp32` | F1 | 9 | 78.96+-0.27 | 80.73+-0.19 | +1.77 [+1.50, +2.03] | 3.6e-07 | 5.7e-06 | +1.60 | - | resolved |
| pope/adversarial (sampling) | `fox off` | accuracy | 10 | 76.72+-0.31 | 78.93+-0.30 | +2.21 [+1.84, +2.58] | 2.8e-07 | 5.1e-06 | +2.23 | - | resolved |
| pope/adversarial (sampling) | `fox off` | F1 | 10 | 78.93+-0.28 | 80.75+-0.25 | +1.82 [+1.51, +2.14] | 3.6e-07 | 5.7e-06 | +1.83 | - | resolved |
| pope/adversarial (sampling) | `fox ulp_control` | accuracy | 10 | 76.72+-0.31 | 81.83+-0.28 | +5.11 [+4.91, +5.30] | 5.2e-13 | 1.5e-11 | +4.97 | - | resolved |
| pope/adversarial (sampling) | `fox ulp_control` | F1 | 10 | 78.93+-0.28 | 82.19+-0.28 | +3.26 [+3.04, +3.49] | 1.2e-10 | 3.2e-09 | +3.15 | - | resolved |
| pope/adversarial (sampling) | `vcd` | accuracy | 3 | 76.72+-0.42 | 78.81+-0.07 | +2.09 [+1.01, +3.17] | 0.014 | 0.11 **ns** | +1.70 | -1.44 | not resolved after Holm |
| pope/adversarial (sampling) | `vcd` | F1 | 3 | 79.04+-0.34 | 79.07+-0.04 | +0.03 [-0.91, +0.98] | 0.89 | 0.89 **ns** | -0.32 | - | not resolved after Holm |
| pope/popular (greedy) | `fox` | accuracy | 1 | 85.90+-0.00 | 86.90+-0.00 | n/a -- deterministic | - | - | +1.00 | +1.14 | see table above |
| pope/popular (greedy) | `fox` | F1 | 1 | 86.29+-0.00 | 86.51+-0.00 | n/a -- deterministic | - | - | +0.22 | +0.27 | see table above |
| pope/popular (sampling) | `fox` | accuracy | 3 | 82.57+-0.48 | 86.70+-0.35 | +4.13 [+2.63, +5.64] | 0.0071 | 0.071 **ns** | +3.87 | +4.17 | not resolved after Holm |
| pope/popular (sampling) | `fox` | F1 | 3 | 83.37+-0.41 | 86.31+-0.35 | +2.94 [+1.46, +4.41] | 0.013 | 0.11 **ns** | +2.79 | - | not resolved after Holm |
| pope/random (greedy) | `fox` | accuracy | 1 | 89.33+-0.00 | 89.53+-0.00 | n/a -- deterministic | - | - | +0.20 | +0.10 | see table above |
| pope/random (greedy) | `fox` | F1 | 1 | 89.28+-0.00 | 88.96+-0.00 | n/a -- deterministic | - | - | -0.32 | -0.47 | see table above |
| pope/random (sampling) | `fox` | accuracy | 3 | 85.12+-0.07 | 89.28+-0.15 | +4.16 [+3.86, +4.45] | 0.00026 | 0.0032 | +4.17 | +4.20 | reproduces |
| pope/random (sampling) | `fox` | F1 | 3 | 85.48+-0.10 | 88.72+-0.17 | +3.24 [+2.97, +3.52] | 0.00039 | 0.0043 | +3.24 | - | resolved |

`p (Holm)` is the Holm-Bonferroni adjustment over every row in this table; `**ns**` marks a row that does not survive it. 

Greedy rows are expected to show `sd = 0`: argmax is deterministic, so replicating a seed changes nothing and the paper's Table 9 is single-run too ("we omit averaged scores", C.7). Their intervals come from the per-run tables above, where item variance is the only variance there is.


### Equivalence to `fox`

Two one-sided tests (TOST). The null is that the arms differ by at least the margin, so **rejecting it is positive evidence that they do the same thing** -- unlike a large p from an ordinary difference test, which only means the data could not tell them apart and gets easier to obtain the noisier the measurement is.

The margin is a judgement, not a default, and is printed with every row. `smallest margin` is the tightest equivalence the data support at this alpha.

| artefact | arm | metric | seeds | diff vs fox | 90% CI | margin | TOST p | smallest margin | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_S | 9 | -6.422 | [-7.488, -5.357] | +-2 | 1.0000 | +-7.49 | not shown |
| chair/val2014_500 (sampling) | `fox boost_visual g=4` | CHAIR_I | 9 | -2.170 | [-2.442, -1.898] | +-0.5 | 1.0000 | +-2.44 | not shown |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_S | 3 | +0.733 | [-1.635, +3.102] | +-2 | 0.1294 | +-3.10 | not shown |
| chair/val2014_500 (sampling) | `fox off` | CHAIR_I | 3 | +0.158 | [-0.224, +0.541] | +-0.5 | 0.0605 | +-0.54 | not shown |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_S | 3 | -2.933 | [-5.178, -0.688] | +-2 | 0.8257 | +-5.18 | not shown |
| chair/val2014_500 (sampling) | `fox ulp_control` | CHAIR_I | 3 | -0.549 | [-1.731, +0.632] | +-0.5 | 0.5429 | +-1.73 | not shown |
| chair/val2014_500 (sampling) | `sampling` | CHAIR_S | 10 | +6.980 | [+5.142, +8.818] | +-2 | 0.9996 | +-8.82 | not shown |
| chair/val2014_500 (sampling) | `sampling` | CHAIR_I | 10 | +2.668 | [+2.097, +3.240] | +-0.5 | 1.0000 | +-3.24 | not shown |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_S | 3 | -7.933 | [-8.448, -7.418] | +-2 | 0.9996 | +-8.45 | not shown |
| chair/val2014_500 (sampling) | `vcd` | CHAIR_I | 3 | +0.671 | [+0.441, +0.900] | +-0.5 | 0.9189 | +-0.90 | not shown |
| pope/adversarial (sampling) | `fox boost_visual g=4` | accuracy | 10 | -1.293 | [-1.562, -1.025] | +-0.5 | 0.9998 | +-1.56 | not shown |
| pope/adversarial (sampling) | `fox boost_visual g=4` | F1 | 10 | -0.516 | [-0.753, -0.280] | +-0.5 | 0.5491 | +-0.75 | not shown |
| pope/adversarial (sampling) | `fox ld=bf16` | accuracy | 10 | +0.013 | [-0.244, +0.270] | +-0.5 | 0.0035 | +-0.27 | **equivalent** |
| pope/adversarial (sampling) | `fox ld=bf16` | F1 | 10 | -0.516 | [-0.754, -0.278] | +-0.5 | 0.5476 | +-0.75 | not shown |
| pope/adversarial (sampling) | `fox ld=fp32` | accuracy | 9 | -2.904 | [-3.175, -2.632] | +-0.5 | 1.0000 | +-3.17 | not shown |
| pope/adversarial (sampling) | `fox ld=fp32` | F1 | 9 | -1.458 | [-1.708, -1.208] | +-0.5 | 1.0000 | +-1.71 | not shown |
| pope/adversarial (sampling) | `fox off` | accuracy | 10 | -2.827 | [-3.044, -2.610] | +-0.5 | 1.0000 | +-3.04 | not shown |
| pope/adversarial (sampling) | `fox off` | F1 | 10 | -1.384 | [-1.588, -1.181] | +-0.5 | 1.0000 | +-1.59 | not shown |
| pope/adversarial (sampling) | `fox ulp_control` | accuracy | 10 | +0.073 | [-0.232, +0.379] | +-0.5 | 0.0154 | +-0.38 | **equivalent** |
| pope/adversarial (sampling) | `fox ulp_control` | F1 | 10 | +0.057 | [-0.237, +0.350] | +-0.5 | 0.0109 | +-0.35 | **equivalent** |
| pope/adversarial (sampling) | `sampling` | accuracy | 10 | -5.033 | [-5.316, -4.751] | +-0.5 | 1.0000 | +-5.32 | not shown |
| pope/adversarial (sampling) | `sampling` | F1 | 10 | -3.207 | [-3.472, -2.942] | +-0.5 | 1.0000 | +-3.47 | not shown |
| pope/adversarial (sampling) | `vcd` | accuracy | 3 | -2.744 | [-3.127, -2.362] | +-0.5 | 0.9983 | +-3.13 | not shown |
| pope/adversarial (sampling) | `vcd` | F1 | 3 | -2.911 | [-3.188, -2.633] | +-0.5 | 0.9992 | +-3.19 | not shown |
| pope/popular (sampling) | `sampling` | accuracy | 3 | -4.133 | [-5.156, -3.111] | +-0.5 | 0.9954 | +-5.16 | not shown |
| pope/popular (sampling) | `sampling` | F1 | 3 | -2.938 | [-3.940, -1.937] | +-0.5 | 0.9904 | +-3.94 | not shown |
| pope/random (sampling) | `sampling` | accuracy | 3 | -4.156 | [-4.353, -3.958] | +-0.5 | 0.9998 | +-4.35 | not shown |
| pope/random (sampling) | `sampling` | F1 | 3 | -3.244 | [-3.430, -3.057] | +-0.5 | 0.9997 | +-3.43 | not shown |


### Degenerate CHAIR arms

CHAIR without recall beside it is gameable, and these arms game it.

- ⚠ **`fox ld=bf16`** on chair/val2014_500 (greedy): CHAIR_S 25.60 against 49.20, which looks like a +23.60 point win -- but recall fell 79.23 -> 59.81, correct objects 1236 -> 933, caption length 100 -> 156 words. It hallucinates less because it names less. Do not read its CHAIR scores as grounding.
- ⚠ **`vcd`** on chair/val2014_500 (sampling): CHAIR_S 40.53 against 54.40, which looks like a +13.87 point win -- but recall fell 75.64 -> 66.73, correct objects 1180 -> 1041, caption length 102 -> 96 words. It hallucinates less because it names less. Do not read its CHAIR scores as grounding.


### Resolution budget

Smallest gap each run can resolve (two-sided 95%, 80% power):

| benchmark / split | metric | n | p | smallest resolvable gap (pts) |
| --- | --- | --- | --- | --- |
| chair/val2014_500 | CHAIR_S | 500 | 0.4900 | 8.83 |
| chair/val2014_500 | CHAIR_I | 3761 | 0.1380 | 2.30 |
| chair/val2014_500 | CHAIR_I | 5236 | 0.1062 | 1.75 |
| chair/val2014_500 | CHAIR_I | 3503 | 0.1188 | 2.25 |
| chair/val2014_500 | CHAIR_I | 3896 | 0.1345 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3703 | 0.1437 | 2.36 |
| chair/val2014_500 | CHAIR_I | 3749 | 0.1366 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3602 | 0.1152 | 2.19 |
| chair/val2014_500 | CHAIR_I | 3683 | 0.1423 | 2.36 |
| chair/val2014_500 | CHAIR_I | 3829 | 0.1408 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3797 | 0.1296 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3521 | 0.1252 | 2.29 |
| chair/val2014_500 | CHAIR_I | 3644 | 0.1416 | 2.37 |
| chair/val2014_500 | CHAIR_I | 3776 | 0.1377 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3782 | 0.1301 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3625 | 0.1137 | 2.17 |
| chair/val2014_500 | CHAIR_I | 3626 | 0.1376 | 2.34 |
| chair/val2014_500 | CHAIR_I | 3855 | 0.1383 | 2.28 |
| chair/val2014_500 | CHAIR_I | 3613 | 0.1174 | 2.20 |
| chair/val2014_500 | CHAIR_I | 3748 | 0.1307 | 2.26 |
| chair/val2014_500 | CHAIR_S | 447 | 0.3937 | 9.29 |
| chair/val2014_500 | CHAIR_I | 3180 | 0.1113 | 2.31 |
| chair/val2014_500 | CHAIR_I | 3842 | 0.1476 | 2.34 |
| chair/val2014_500 | CHAIR_I | 3570 | 0.1196 | 2.24 |
| chair/val2014_500 | CHAIR_I | 3837 | 0.1306 | 2.23 |
| chair/val2014_500 | CHAIR_I | 3560 | 0.1110 | 2.17 |
| chair/val2014_500 | CHAIR_I | 3824 | 0.1352 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3594 | 0.1152 | 2.19 |
| chair/val2014_500 | CHAIR_I | 3751 | 0.1330 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3559 | 0.1115 | 2.18 |
| chair/val2014_500 | CHAIR_I | 3823 | 0.1355 | 2.27 |
| chair/val2014_500 | CHAIR_I | 3589 | 0.1181 | 2.22 |
| chair/val2014_500 | CHAIR_I | 3554 | 0.1142 | 2.20 |
| chair/val2014_500 | CHAIR_S | 50 | 0.5400 | 26.03 |
| chair/val2014_500 | CHAIR_I | 428 | 0.1379 | 7.25 |
| chair/val2014_500_d1 | CHAIR_S | 500 | 0.4140 | 8.82 |
| chair/val2014_500_d1 | CHAIR_I | 3914 | 0.1191 | 2.13 |
| chair/val2014_500_d2 | CHAIR_S | 500 | 0.4460 | 8.85 |
| chair/val2014_500_d2 | CHAIR_I | 3849 | 0.1315 | 2.23 |
| chair/val2014_500_d3 | CHAIR_S | 500 | 0.4940 | 8.82 |
| chair/val2014_500_d3 | CHAIR_I | 4017 | 0.1481 | 2.29 |
| chair/val2014_500_d4 | CHAIR_S | 500 | 0.4400 | 8.84 |
| chair/val2014_500_d4 | CHAIR_I | 3884 | 0.1285 | 2.20 |
| pope/adversarial | accuracy | 3000 | 0.8237 | 2.67 |
| pope/adversarial | accuracy | 500 | 0.8160 | 6.36 |
| pope/adversarial | accuracy | 50 | 0.7400 | 20.24 |
| pope/popular | accuracy | 3000 | 0.8690 | 2.34 |
| pope/random | accuracy | 3000 | 0.8953 | 2.11 |
| chair/val2014_500 | CHAIR_I | 3762 | 0.1382 | 2.30 |
| chair/val2014_500 | CHAIR_I | 3599 | 0.1814 | 2.61 |
| chair/val2014_500 | CHAIR_I | 3486 | 0.1644 | 2.56 |
| chair/val2014_500 | CHAIR_I | 3536 | 0.1612 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3572 | 0.1632 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3562 | 0.1555 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3545 | 0.1614 | 2.52 |
| chair/val2014_500 | CHAIR_I | 3495 | 0.1508 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3658 | 0.1613 | 2.48 |
| chair/val2014_500 | CHAIR_I | 3487 | 0.1735 | 2.61 |
| chair/val2014_500 | CHAIR_I | 3534 | 0.1661 | 2.55 |
| chair/val2014_500_d1 | CHAIR_I | 3782 | 0.1296 | 2.24 |
| chair/val2014_500_d2 | CHAIR_I | 3711 | 0.1439 | 2.36 |
| chair/val2014_500_d3 | CHAIR_I | 3775 | 0.1481 | 2.36 |
| chair/val2014_500_d4 | CHAIR_I | 3799 | 0.1316 | 2.25 |
| chair/val2014_500 | CHAIR_I | 3106 | 0.1484 | 2.62 |
| chair/val2014_500 | CHAIR_I | 3164 | 0.1429 | 2.55 |
| chair/val2014_500 | CHAIR_I | 3249 | 0.1456 | 2.54 |

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
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-11fe4386` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-129590cd` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-41d5cbc1` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-7dc3dd43` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g100-9ebfabc4` | `fox` | chair/val2014_500 | fp16 | 100.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g10000-1c417645` | `fox` | chair/val2014_500 | fp16 | 10000.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-g4-4f055e8b` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-1bea1044` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-c676d8ec` | `fox` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 45 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 45 | 447 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-5c7c06df` | `fox` | chair/val2014_500 | fp16 | None | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-g4-60acf738` | `fox` | chair/val2014_500 | fp16 | 4.0 | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500__seed-997__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500 | fp16 | None | None | None | 997 | 50 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d1__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d1 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d2__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d2 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d3__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d3 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-chair__s-val2014_500_d4__seed-42__dt-fp16__v-greedy-4a534adb` | `fox` | chair/val2014_500_d4 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-2849ecbb` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-30d01489` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-51104c9a` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-7d65ec90` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-94ffb3d6` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-ec5b84b4` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-g4-77f044a1` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-k0.25-403b917a` | `fox` | pope/adversarial | fp16 | None | 0.25 | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-k0.45-a259d07c` | `fox` | pope/adversarial | fp16 | None | 0.45 | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-t0.1-4476086d` | `fox` | pope/adversarial | fp16 | None | None | 0.1 | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-t0.2-748a28e2` | `fox` | pope/adversarial | fp16 | None | None | 0.2 | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-c7d23f9e` | `fox` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-01013bf7` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-08d18927` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-93866aba` | `fox` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-g4-01a74670` | `fox` | pope/adversarial | fp16 | 4.0 | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `434b004` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-adversarial__seed-997__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/adversarial | fp16 | None | None | None | 997 | 50 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-popular__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/popular | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-42__dt-fp16__v-greedy-17d4eb8d` | `fox` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-42__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-43__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-fox__b-pope__s-random__seed-44__dt-fp16__v-sampling-2d07edcb` | `fox` | pope/random | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-45__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 45 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-46__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 46 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-47__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 47 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-48__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 48 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-49__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 49 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-50__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 50 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500__seed-51__dt-fp16__v-sampling-5c7c06df` | `sampling` | chair/val2014_500 | fp16 | None | None | None | 51 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d1__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d1 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d2__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d2 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d3__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d3 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-chair__s-val2014_500_d4__seed-42__dt-fp16__v-greedy-4a534adb` | `sampling` | chair/val2014_500_d4 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `7d52f44` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `68374d2` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-45__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 45 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-46__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 46 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-47__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 47 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-48__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 48 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-49__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 49 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-50__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 50 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-adversarial__seed-51__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/adversarial | fp16 | None | None | None | 51 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-popular__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/popular | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-42__dt-fp16__v-greedy-17d4eb8d` | `sampling` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-42__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-43__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-sampling__b-pope__s-random__seed-44__dt-fp16__v-sampling-2d07edcb` | `sampling` | pope/random | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-42__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 42 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-43__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 43 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-chair__s-val2014_500__seed-44__dt-fp16__v-sampling-5c7c06df` | `vcd` | chair/val2014_500 | fp16 | None | None | None | 44 | 500 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-42__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 42 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-43__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 43 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |
| `m-vcd__b-pope__s-adversarial__seed-44__dt-fp16__v-sampling-2d07edcb` | `vcd` | pope/adversarial | fp16 | None | None | None | 44 | 3000 | `llava-hf/llava-1.5-7b-hf` | `f587e7f` | `...on3.13/dist-packages/transformers/__init__.py` |

CHAIR scorer: synonyms `chair_synonyms.txt` sha256 `a4c231100e8a...` (transcription-of-rohrbach-synonyms-txt), tokenizer `regex`, gt_source `both`

**Warnings**

- `fox` run with **gamma = None**: the intervention constant was not recorded.
