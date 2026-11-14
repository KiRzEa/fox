# Validating the compared methods

Status: **pre-registered**. Written 2026-10-27, before any of these arms has
been run on the real model. The predictions and their readings are fixed
here so that the interpretation cannot be chosen after the numbers arrive.

## Why this document exists

Five of the six compared methods — VCD, ICD, SID, CausalMM, OPERA — were
implemented in this project from each method's own released source. Every
number-bearing line is transcribed verbatim into `fox/reference.py` and the
clean port is asserted bit-identical to it.

That proves **our code == their code**. It does not prove:

* **their code == their paper.** SID's released harness cannot run VCD at
  all; CausalMM's two-counterfactual sampler is dead code; OPERA's
  retrospection half cannot fire on POPE. A faithful port of a file inherits
  whatever the file does, including the parts that do not match the prose.
* **the mechanism is active on the real model.** A contrastive method whose
  negative branch is not actually worse still produces an accuracy number.
  So does one whose rule never changes a decision. Neither is visible in a
  table.

So each method gets arms that would **fail** if it were not doing what it
claims. They are cheap (6.8 h total, one session), they run **before** the
40 h of table rows, and they are in the plan under `target="validation"`.

The vocabulary: a `control` is a validation arm, not a tuning knob and not
an ablation of Fox. `storage.CONTROL_ARMS` lists them; `--control` selects
one; a control aimed at a method that does not understand it raises rather
than being ignored, because a run file named for a control that never
happened is the worst outcome available here.

---

## V1 — Is the negative branch actually worse?

`--control negative_only`, on `vcd`, `icd`, `sid`, `causalmm`.
POPE-adversarial, 3000 questions, seed 42. **1.8 h.**

All four methods rest on one sentence: *build a branch that reads the image
badly, then subtract it.* The arm runs that branch **alone**, with no
contrast — VCD's noised image, ICD's disturbed instruction, SID's CT2S,
CausalMM's counterfactual features — and scores it.

**Prediction: every one scores below the `sampling` baseline (76.72).**

| outcome | reading |
|---|---|
| clearly below baseline | the premise holds; subtracting it is at least coherent |
| at or above baseline | the method is subtracting a branch that is **not** worse. Whatever its gain is, it is not the mechanism the paper describes |
| far below (near chance) | the branch is uninformative rather than wrong, so the contrast is subtracting noise — a different failure, and visible in `diverged_frac` |

None of the four papers reports this number.

---

## V2 — Is the formula right?

`--alpha 0 --decoding greedy`, on `vcd`, `icd`, `sid`, `causalmm`.
POPE-adversarial, `--limit 500`, seed 42. **0.6 h.**

> **Corrected after the first run.** This arm was first specified under
> *sampling*, where identity is not achievable however correct the code is:
> the contrastive path decides on the **beta-masked** logits, which the
> baseline never applies, and the draw is stochastic. Under greedy with
> `mask_greedy` False the decision is `fused`, which at `alpha = 0` is
> `z_obs` exactly, and nothing is random.
>
> The sampling version was not wasted. It scored 76.40 / 77.00 / 77.20 /
> 78.20 against the baseline's 77.00 — the same distribution — while 14–19 %
> of individual answers differed. Same distribution, different draws, which
> is what exposed the defect below.

> **Defect found by this arm.** The per-sample seed was
> `cfg.seed * K + hash(sample_id) % K`, and `hash()` on a `str` is salted
> per interpreter unless `PYTHONHASHSEED` is set. So the same sample got a
> different seed in every process: an identical command did not reproduce
> its own tokens, a job resumed after the session wall continued on a
> different stream, and two arms at "the same seed" never saw the same draw.
> Now `methods.sample_seed`, on `zlib.crc32`. It changes the randomness of
> every FUTURE run; records already written are what they are, and the
> seed-level analysis never required the arms to share a draw, so nothing
> already concluded depends on it. The greedy determinism probe could not
> have caught this — greedy has no draw to vary.

At `cd_alpha = 0` the shared rule

```
diffs = (1 + a) * z_obs - a * z_cd
```

collapses to `z_obs` exactly, so the arm must reproduce the baseline.

**Prediction: token-identical to the archived `sampling` run at seed 42, for
all 500 samples. Not close — identical.**

Exactness is the point. A sign error, a mis-scoped branch, a perturbation
that leaks into the observational row, or a generator consumed in the wrong
order all break identity while leaving the accuracy entirely plausible. 500
samples suffice because one differing token fails it.

| outcome | reading |
|---|---|
| identical | the rule is wired correctly, including which side the coefficient is on |
| differs at all | under greedy there is no randomness left to blame: the coefficient is on the wrong term, the perturbation is reaching row 0, or the decision is not taking `fused` |

---

## V3 — Does SID's selection do any work, and is it pointing the right way?

`--control sid_keep_most` and `--control sid_random`, on `sid`.
POPE-adversarial, 3000, seed 42. **1.8 h.**

CT2S keeps the 100 vision tokens the model attends to **least**. One line
above that call in the released source, `topk(ATTENTION_RANK)` sits
commented out — that is FastV, which keeps the **most**-attended tokens to
save compute. SID inverts it deliberately: the negative branch is meant to
read the image badly on purpose.

**Prediction A (direction): as a negative branch, `keep_most` is a *better*
reader than `keep_least`, so SID-with-`keep_most` should gain less than SID
does.** If `keep_most` matches or beats plain SID, the direction in the code
is not what makes the method work.

**Prediction B (selection): keeping 100 *random* vision tokens is a worse
negative branch than `keep_most` but comparable to `keep_least`.** If
`sid_random` matches plain SID, then attention-based selection does nothing
and CT2S is simply "drop 476 of 576 vision tokens" — the same conclusion
Figure 8's random-head control would force on Fox.

These two together are the only way to tell *the ranking matters* from *the
masking matters*, and SID's paper reports neither.

---

## V4 — Is CausalMM's shuffle load-bearing?

`--control cf_constant` and `--control cf_blind`, on `causalmm`.
POPE-adversarial, `--limit 500`, seed 42. **0.3 h.**

Already established analytically and measured on a real (tiny-width,
correct-geometry) LLaVA: each CLIP attention row is a softmax over 577 keys,
so the matrix sums to 577 and its mean entry is `1/577`. A permutation
cannot change a mean, and neither bilinear resizing nor averaging over heads
does. Measured gain: mean `0.00172`, std `0.00015` — an almost uniform
downscale by ~580x. After the projector's GELU, an input scaled down 580-fold
is near-constant.

* `cf_constant` replaces the shuffled attention with exactly `1/n_keys`.
* `cf_blind` zeroes the negative branch's visual features outright.

**Prediction: all three — `shuffle`, `constant`, `blind` — give the same
answers.** On the tiny model, `negative_only` under `shuffle` and under
`blind` were already token-identical.

| outcome | reading |
|---|---|
| all three agree | the vision counterfactual is a description of something the code does not do. CausalMM is a **language-prior contrast**: arguably faithful to the paper's stated intent (modality-prior-induced hallucination) but not to the mechanism its figures show |
| `shuffle` differs from `constant` | the spread around `1/577` carries signal after all, and the claim above is wrong |
| `blind` differs from the other two | the residual 0.17 % of the image is doing real work |

---

## V5 — Which third of OPERA is working?

`--control opera_unrestricted` and `--control opera_no_penalty`, on `opera`.
POPE-adversarial, greedy, 3000. **2.2 h.**

OPERA is three mechanisms stacked, and the paper names only one:

1. a **beam search** (5 beams), where every other arm here samples;
2. a **hard five-candidate restriction** — each beam can only emit a token in
   its own top 5 by raw logit, where stock beam search takes the best
   `2 * num_beams` over `num_beams * vocab_size`;
3. the **over-trust penalty** — a bonus for attending to the image.

`opera_no_penalty` keeps 1+2, `opera_unrestricted` keeps 1+3. With plain
OPERA that is a full decomposition.

**Prediction: `opera_no_penalty` lands on top of plain OPERA.** The bonus is
added in **logit** space, and on a correct-geometry stand-in its spread
across the five candidates was `3e-4` against a logit spread of `0.1` — 300x
smaller, and it never moved the top candidate. The structural reason is that
the attention row is normalised over all keys and 576 of 586 are image
tokens, so the image share is ≈1 for every candidate and they differ only in
how much the next token attends to the handful of text keys.

| outcome | reading |
|---|---|
| `no_penalty` ≈ `opera` | the over-trust penalty is **inert on POPE**, and OPERA's published POPE number is beam search with a truncated candidate set |
| `no_penalty` < `opera` | the penalty works, and the stand-in's near-uniform attention was the reason it looked inert |
| `unrestricted` ≈ `opera` | the restriction is not what helps |
| both controls below `opera` | all three components contribute, which is the paper's implicit claim |

Every OPERA run records `penalty_spread_mean`, `logit_spread_mean` and
`n_steps_penalty_changed_top`, so the first row of this table can be read
from a single plain run before the controls are even compared.

---

## Results, round 6 session 1 (2026-10-28)

The first pass. 14 arms planned, 12 produced records: both OPERA controls
OOM'd on a T4 (a five-row prefill does not fit beside the weights; fixed by
prefilling one row and fanning the cache).

**V1 holds for all four, and the magnitudes are themselves a finding.**
Baseline 77.17.

| arm | accuracy | vs baseline |
|---|---|---|
| `vcd` negative branch | 76.10 | −1.07 |
| `icd` negative branch | 73.70 | −3.47 |
| `sid` negative branch | 56.43 | **−20.73** |
| `causalmm` negative branch | 51.97 | **−25.20** |

Every negative branch is worse, so the premise all four share survives. But
they are not worse by remotely comparable amounts: VCD's noised image barely
degrades the model, while SID's CT2S and CausalMM's counterfactual put it
near chance for a yes/no task (50 %). Those two are not "a worse reader" so
much as "almost no reader", which is consistent with the 1/577 gain measured
for CausalMM and worth carrying into how their gains are read.

**V2 ran under the wrong protocol** and is re-specified above.

**V4 is not yet interpretable.** `cf_constant` agrees with `shuffle` on
96.6 % of samples and `cf_blind` on 95.2 %. Those look high, but the arms
they must be compared against — two runs of *plain* `causalmm` differing only
in the draw — were not run, and under the old per-process seed the
same-method ceiling was around 81–86 %. The arm needs that ceiling before it
says anything; with `sample_seed` fixed, two plain runs at different seeds
now give it.

**V6 holds.** `diverged_frac` 0.038 (ICD), 0.046 (SID), 0.039 (CausalMM) on
POPE: strictly between 0 and 1, so none of the three is inert, and each
changes about one decision in 25. POPE emits one token, so this is the
floor; CHAIR should be far higher.

## Results, round 7 (2026-11-08) — the gate passes

**16 predictions held, 0 failed.** V5 is still outstanding: OPERA has not
produced a record yet.

**V2 under greedy is exact, as re-specified.** VCD, ICD and SID reproduce
the greedy baseline on all 500 samples, token for token. So the shared rule
is wired correctly in all three, including which side the coefficient sits
on — the thing the sampling version could never have shown.

CausalMM differs on **1 of 500**, and that is the eager-CLIP deviation
rather than the rule. `output_attentions=True` forces the vision tower onto
its eager path, so CausalMM's *clean* branch is not the baseline's clean
branch whatever `alpha` is (§3.13). One flipped decision in 500 puts a
number on a deviation that was previously only a bound: **0.2 % of POPE
decisions**. `check_validation.py` now reads it that way, with a 1 % ceiling
above which it would still be a fault.

**V6 across both benchmarks**, and the prediction that CHAIR would be far
higher than POPE holds:

| arm | POPE | CHAIR |
|---|---|---|
| `vcd` | 0.067 | 0.068 |
| `icd` | 0.038 | 0.027 |
| `sid` | 0.046 | 0.074 |
| `causalmm` | 0.039 | **0.306** |

CausalMM changes almost a third of its CHAIR tokens. That is consistent with
the two things already measured about it — a negative branch that is close
to image-blind, and `epsilon = 0.6`, a far tighter plausibility cutoff than
the 0.1 the others use — and it is the arm to read most carefully when the
CHAIR table arrives.

## Results, round 8/9 (2026-11-09) — V5 settles, V4 is re-specified

**17 predictions held, 0 failed.** Every arm has now run except V4's greedy
repeat, added below.

**V5: the over-trust penalty is inert on POPE, as predicted.** All three
arms land on the same number.

| arm | accuracy | vs plain OPERA |
|---|---|---|
| `opera` | 79.10 | — |
| `opera_no_penalty` | 79.03 | −0.07 |
| `opera_unrestricted` | 79.13 | +0.03 |

A 0.10-point total spread over 3000 questions. Neither of OPERA's two
unnamed components does measurable work **at this length**.

> **Corrected 2026-11-10, and the correction matters.** These three arms ran
> at `max_new_tokens=1`, this project's POPE default. At that length plain
> OPERA scores 79.10 against a greedy baseline of 79.03 — **+0.07**. The
> whole method is inert there, so a decomposition finding all three arms
> equal is a decomposition of nothing, and the sentence that stood here
> ("OPERA's POPE number is beam search") was not supported by it.
>
> At OPERA's own 10-token protocol the same implementation scores **81.13,
> +2.10** over the same baseline. For a beam search the length is not
> cosmetic — it scores whole sequences, so the first token depends on how
> many follow it (`DEVIATIONS.md` §3.13 predicted this; it is now measured).
> That +2.10 is the gain worth attributing, and the two controls are
> re-planned at 10 tokens on 1500 questions (`opera_protocol`, 5.2 h).
>
> What V5 *does* establish stands: at 1 token the penalty is numerically
> incapable of acting, by its own telemetry, and that is the row Fox's
> Table 1 quotes, because Fox re-ran its compared methods in its own
> 1-token harness. So **the OPERA number in Fox's Table 1 is beam search** —
> a claim about one table's cell, not about OPERA.

The telemetry recorded on the plain run says why, and said it before the
controls were compared: the penalty's spread across the five candidates
averages **0.067** against a logit spread of **9.099** — 136x smaller — and
over 3000 questions it moved the top candidate at **one single step**
(`n_steps_penalty_changed_top` mean 0.00033). The prediction was made from a
correct-geometry stand-in that put the ratio at 300x; the real model gives
136x, the same conclusion with a weaker margin.

The structural reason stands: the attention row is a softmax over all keys
and 576 of ~586 are image tokens, so every candidate's image share is ≈1 and
they differ only in how much the next token attends to the handful of text
keys. On POPE that next token is `Yes` or `No` after a one-token budget,
which is the least room any benchmark could give the mechanism.

**V4 was asked under the wrong protocol, the same way V2 was.** Read under
sampling, `cf_constant` agreed with `shuffle` on 96.6 % of 500 answers and
`cf_blind` on 95.2 %, which reads as "the shuffle carries signal after all".
It is not evidence of that. A multinomial draw flips whenever the logits
move at all, however little, so two arms whose distributions coincide still
disagree token by token; the accuracies were 78.40 / 77.80 / 78.00, a
0.6-point spread on n=500 against a ±3.7-point interval — indistinguishable.

So the arm is **re-specified to greedy**, where there is no draw and
"do these three produce the same text" is answerable. Three jobs, 500
samples, 0.42 h: `shuffle`, `cf_constant`, `cf_blind`, all greedy. The plain
arm is re-run because the only greedy `causalmm` on disk is V2's `alpha = 0`
row, which is the *baseline*, not CausalMM.

**Prediction, unchanged: token-identical under greedy.** `check_validation.py`
reads the sampling arms as INFO and decides only on the greedy ones.

## Results, round 10 (2026-11-10) — the plan closes, and V4 fails

**18 predictions held, 1 failed.** Every job in the plan has run.

**V4 under greedy: `cf_constant` holds, `cf_blind` does not.**

| arm | agrees with `shuffle` | accuracy |
|---|---|---|
| `shuffle` | — | 77.80 |
| `cf_constant` | 498/500 (99.6 %) | 77.80 |
| `cf_blind` | 495/500 (99.0 %) | **78.80** |

The first row is the predicted result and it is a clean one: replacing the
shuffled CLIP attention with *exactly* `1/n_keys` changes 2 answers in 500
and moves the score by nothing. A permutation cannot change a mean, the mean
is what survives the 580x downscale, and the measurement now says so on the
real model. CausalMM's vision counterfactual is a **language-prior
contrast**.

The second row is the pre-registered FAIL condition — *"`blind` differs from
the other two → the residual 0.17 % of the image is doing real work"* — and
the 7 disagreements across all three arms say which way it works:

| sample | label | `shuffle` | `cf_constant` | `cf_blind` |
|---|---|---|---|---|
| q0000035 | yes | No | **Yes** | No |
| q0000066 | no | Yes | Yes | **No** |
| q0000070 | no | Yes | Yes | **No** |
| q0000076 | no | Yes | Yes | **No** |
| q0000246 | no | Yes | Yes | **No** |
| q0000366 | no | **No** | Yes | **No** |
| q0000442 | no | Yes | Yes | **No** |

`cf_blind` is correct on 6 of the 7, and on **5 of 5** where it differs from
`shuffle` — every one of them a `no` label where the other two hallucinate a
`Yes`. That is the signature of a *stronger* contrast, not a different one: a
fully image-blind negative branch is the purest possible language prior, so
subtracting it removes more of the prior-driven `Yes` than subtracting a
branch that still carries 0.17 % of the image.

So the reading is not that the 1/n_keys argument is wrong. It is that the
argument is **99 % right and the missing 1 % runs against CausalMM**: the
shuffle leaves just enough image in the negative branch to make the contrast
weaker than the mechanism it is accidentally implementing. The method would
score higher if it did less.

**This is not yet significant and it is not being claimed as such.** Exact
McNemar on the discordant pairs:

| pair | discordant | split | p |
|---|---|---|---|
| `shuffle` vs `cf_blind` | 5 | 5–0 | 0.0625 |
| `cf_constant` vs `cf_blind` | 7 | 6–1 | 0.125 |
| `shuffle` vs `cf_constant` | 2 | 1–1 | 1.0 |

A consistent direction on 5 pairs is what it is. At the same flip rate 3000
samples would give p < 1e-4, so **the three greedy arms are re-planned on the
full split** (2.3 h, resuming from the 500 already written — `limit` is not
part of the run key, so the existing records extend rather than duplicate).
Until that lands, the honest statement is: `cf_constant` confirms the
permutation argument, and `cf_blind` suggests the residual hurts CausalMM.

**OPERA's own protocol ran.** `opera_protocol` — 10 tokens, OPERA's
`pope_eval.py` length rather than this project's 1-token POPE default — had
OOM'd on its first sample. The cause was that `opera_generate` was never
decorated `@torch.inference_mode()`, so each of the six forwards per step
retained an autograd graph pinning the KV buffers it had read; at 1 token
that is one step and it fitted, at 10 it did not (`DEVIATIONS.md` §3.13).
With the guard it ran 3000/3000 in 5.2 h at 6.2 s/sample. Grad mode changes
what is retained, not what is computed, so the 1-token records behind V5 are
unaffected.

## Results, round 11/12 (2026-11-10) — both open arms close

**18 predictions held, 1 failed.** The gate now decides V4 and V5 by **exact
McNemar on the discordant pairs** rather than by an agreement threshold. The
threshold was picked by eye at n=500 and had no sampling distribution behind
it; at n=3000 it would have called a 12-pair 9–3 split and a 46-pair 36–10
split by the same rule. Every arm here answers the same questions, so the
pairing removes question difficulty — which is most of the variance — and
settles a 0.87-point difference that an unpaired ±1.5-point interval on 3000
samples would have called noise.

### V4 at full scale: `constant` holds, `blind` fails, and the direction held

Greedy, 3000 questions, paired on identical samples:

| pair | agree | informative pairs | split | p |
|---|---|---|---|---|
| `shuffle` vs `cf_constant` | 2988/3000 (99.6 %) | 12 | 9–3 | **0.146** |
| `shuffle` vs `cf_blind` | 2954/3000 (98.5 %) | 46 | 10–36 | **1.6e-4** |
| `cf_constant` vs `cf_blind` | 2942/3000 (98.1 %) | 58 | 13–45 | **3.0e-5** |

Accuracy: `shuffle` 76.67, `cf_constant` 76.47, `cf_blind` **77.53**.

`cf_constant` is the predicted result and now has the sample size to say so:
replacing the shuffled CLIP attention with *exactly* `1/n_keys` is
indistinguishable from shuffling it. A permutation cannot change a mean, the
mean is what survives the ~580x downscale, and CausalMM's vision
counterfactual is a **language-prior contrast**.

`cf_blind` is the one failed prediction in the suite, and the 500-sample
signal that flagged it (5–0, p=0.0625) held exactly: 36–10 at p=1.6e-4. A
fully image-blind negative branch is the purest language prior, so
subtracting it removes more of the prior-driven `Yes` than subtracting a
branch that still carries 0.17 % of the image. **CausalMM would score higher
if it did less.** The 1/n_keys argument is 99 % right and the missing 1 %
runs against the method.

One number worth carrying into Table 1: greedy `causalmm` scores 76.67
against a greedy baseline of **79.03**. Under sampling it gains (+1.13); under
greedy it *loses* 2.36. The gain depends on the plausibility mask and the
stochastic draw, not on the contrast.

### V5 at the length where OPERA works: beam search, confirmed

The 1-token block is kept because it is the length Fox re-ran its compared
methods at, so it is what Fox's Table 1 cell contains. The verdict comes from
the 10-token block, 1500 questions, paired:

| | accuracy | vs baseline | agrees with `opera` | informative pairs | p |
|---|---|---|---|---|---|
| `opera` | 81.47 | +2.33 | — | — | — |
| `opera_no_penalty` | **81.47** | +2.33 | 1496/1500 | 2 (1–1) | 1.0 |
| `opera_unrestricted` | **81.47** | +2.33 | 1474/1500 | 24 (12–12) | 1.0 |

Three arms, one number. **OPERA's POPE gain is beam search.** Neither the
over-trust penalty nor the five-candidate restriction contributes anything
measurable, at the length where the method actually gains +2.33.

This is the conclusion the 1-token decomposition reached without being
entitled to it, and the correction was worth making: the evidence for it did
not exist until this arm ran. At 1 token OPERA gains +0.07 over the baseline,
so all three arms agreeing was a fact about a regime where nothing happens.

The telemetry also sharpens the mechanism, and the sharpened version is not
"inert":

| | 1 token | 10 tokens |
|---|---|---|
| penalty spread across the 5 candidates | 0.067 | 0.082 |
| raw logit spread | 9.099 | 7.889 |
| ratio | 136x | **96x** |
| steps where the penalty moved the top candidate | 0.00033/sample | **0.0913/sample** |

At 10 tokens the penalty *does* act — it changes the top candidate in about
0.9 % of steps, roughly 274 times over 3000 questions. It is not numerically
incapable of acting, as the 1-token run suggested. It is **behaviourally**
inert: of those 274 interventions, 4 reach the final answer and they split
1–1. The penalty reorders candidates the beam would have recovered from
anyway.

`opera_unrestricted` is the same story more visibly: 26 of 1500 answers
differ, 12–12. The restriction changes *which* tokens come out without
changing whether they are right.

## Results, round 13 (2026-11-12) — the diagnosis, measured directly

**These were not pre-registered.** V1–V6 ask whether *our* implementations are
the methods their papers describe. This round asks a different question — do
the paper's own published diagnostic numbers reproduce — so there is no arm to
predict and nothing to score against the gate. The gate stood at **18 held, 1
failed** when this round ran; r14 below moves it to **19 held, 2 failed**. Read what follows as measurement, not as confirmation: the analysis
was written before the run (`scripts/analyse_diagnostics.py`, committed in
`a48c207`), but the quantities below were not bet on in advance.

One forward pass per sample, no generation, 3000 POPE-adversarial questions,
greedy, 32 layers × 32 heads, 459 hallucinated (15.3 %). Labels join from the
archived greedy baseline on `sample_id`; a hallucination is a false **yes**
only (§`load_labels`). Reproduce the whole section with:

```bash
python scripts/analyse_diagnostics.py
```

### Eq. 13 publishes a real signal through a denominator whose sign is free

| | tail = 1 | tail = 32 |
|---|---|---|
| **Eq. 13 exactly as written** | **0.1310** | **0.6568** |
| paper | 0.8226 | 0.7626 |
| its denominator, Σ ΔAUC | **−2.8821** | +0.7769 |
| the magnitude it aggregates, Σ\|ΔAUC\| | 74.5528 | 24.6462 |
| so the denominator is a residual of | **3.9 %** | 3.2 % |
| heads with ΔAUC > 0 | 504/1024 | 550/1024 |
| dividing by Σ\|ΔAUC\| instead | **0.8690** | 0.6568 |
| held out (weights fitted on the other half) | **0.8735** | 0.6645 |

Eq. 13 is `s_n = Σ m_sys(n)·ΔAUC / Σ ΔAUC`. The denominator is one global
scalar, so a negative one does not distort the ranking — it **reverses** it,
and the AUC comes out at exactly `1 − AUC`. That is the whole of the
0.1310: 1 − 0.1310 = 0.8690, which is what the sign-safe denominator gives.

The sign is not a property of the mechanism. It is the residual of 1024
near-cancelling terms — 504 up, 520 down, mean ΔAUC = −0.0028 — and it comes
to under 4 % of the magnitude being summed, in both windows. Nothing in the
construction holds it positive. The paper reports 0.8226 and 0.7626, both
above 0.5, which means its residual landed positive twice; ours landed
negative at tail = 1 and positive at tail = 32, from the same 1024 heads.

**The signal itself is real, and stronger than the paper claims.** Fix the
denominator and tail = 1 gives 0.8690 in sample and **0.8735 held out** — the
weights generalise, so this is not the 1024 free parameters fitting 459
positives. Both windows also keep the paper's ordering (tail = 1 above
tail = 32) and its reading that instantaneous decision steps separate better.

So this is not a failure to reproduce. It is a reproduction of the effect
plus a defect in the scalar that reports it, and the defect is invisible as
long as the residual happens to come out positive.

### What the probe reads is *which* heads, not *how much* — and the direction is backwards

Two controls settle what is doing the work.

| | tail = 1 | tail = 32 |
|---|---|---|
| unweighted mean `m_sys`, no Eq. 12 weights at all | **0.4626** | 0.4924 |
| best single head | 0.1964 (L27H16) | 0.6241 (L27H27) |

Averaging `m_sys` over all 1024 heads predicts hallucination **at chance**.
Every bit of the 0.87 comes from the per-head ΔAUC weights. The paper's
Figure 13 caption is explicit about the direction — *"a higher score indicates
stronger reliance on system/prefix tokens"*, hallucination positive — so the
claim is that hallucinating samples put **more** mass on the system prompt.
Per layer, heads averaged within the layer:

| | |
|---|---|
| layers where hallucination **raises** `m_sys` | **11 of 32** (L0, 3, 5, 6, 9, 14, 16, 17, 29, 30, 31) |
| the five strongest layer signals | L23 **0.237** · L22 **0.299** · L26 **0.341** · L21 **0.343** · L13 **0.370** |

All five run the other way: where the signal is strongest, hallucination comes
with **less** system attention, not more. The one strong same-direction layer
is L30 (0.727). This is also why Σ ΔAUC came out negative at tail = 1 — the
negative sign is not bad luck, it is the dominant direction showing up in the
sum.

What survives is weaker and differently shaped than the paper's claim: the
*pattern* of which heads attend to the system prompt is a strong hallucination
probe (held out 0.87), while the *magnitude* of system attention is
uninformative, and in the mid-to-late layers it moves opposite to the
"pathological shortcut" story. A probe that reads a 1024-dimensional
fingerprint is not evidence of a causal shortcut; it is evidence that
hallucinating and faithful forwards are differently shaped.

`H_vis` is the half of the premise that does hold: hallucination **raises**
visual-attention entropy in **24 of 32** layers, peaking at L23 (+0.1077).
Diffuse visual attention does go with hallucination. It is just weak — the
largest per-layer effect is 0.11 of AUC.

### Figure 7's rule does not select the window the code ships

Mean |AUC − 0.5| over a window is what the rule maximises. At the released
window's own width of 14 layers:

| window | `H_vis` | `m_sys` |
|---|---|---|
| released code, L2–15 | 0.0305 | 0.0458 |
| appendix manual, L2–13 | 0.0283 | 0.0402 |
| **best 14-layer window** | **0.0571** (L11–24) | **0.1075** (L13–26) |
| every layer, L0–31 | 0.0373 | 0.0762 |

The shipped window scores **below the all-layer average on both statistics**,
and 1.9×/2.3× below the best window of equal width. The rule is stated as
keeping *"the contiguous early-to-mid window above the selection threshold"*,
and no threshold is given; at half of the peak the contiguous runs are
{12, 13}, {15, 16}, {21–24}, {26}, {30}. There is no early-to-mid run to
keep — the signal is **mid-to-late**, and the earliest qualifying run starts
at L12.

This closes a loop. Fox intervenes on `S = m_sys · H_vis` over L2–15, and
all three of those choices now have a measurement against them: `m_sys`
points the wrong way where it is strongest, `H_vis` points the right way but
weakly, and L2–15 is a below-average window by the paper's own criterion.
Three independent reasons the intervention should not work as described — and
V2 already showed it works by something else entirely, the fp16 logit cast.

### Table 13 reports an undefined quantity, and its one reproducible number is the uninformative one

| | measured | paper |
|---|---|---|
| `m_vis` | 0.1057 | ≈ 0.57 |
| `m_txt` | 0.1602 | ≈ 0.62 |
| `m_sys` | **0.7341** | ≈ 0.82 |
| sum | **1.0000** | **2.01** |

Eq. 11 defines `m_sys,tail` as attention mass on `I_sys = [0, s_img)`, bounded
in [0, 1]. The three spans partition the keys, so the three masses sum to one —
asserted in `tests/test_diagnose.py`, and measured at 1.0000 here. The
paper's three sum to 2.01, so **they are not the masses Eq. 11 defines**, and
the paper gives no other definition. Earlier drafts of ours called Table 13
arithmetically impossible; that was too strong. The right statement is that
the table reports a quantity it never defines under symbols defined elsewhere
to mean something else.

The caption offers a hint: *"the tail ROC-AUC treats system-token responses as
the positive class"*. That AUC separates **token groups**, not hallucinated
from faithful samples — can `m` tell a system token from a visual or text one.
We reproduce it exactly:

| | |
|---|---|
| sample level, heads averaged — paper's 1.00 | **1.0000** (min `m_sys` 0.7113 vs max of the other two 0.1731) |
| per (sample, layer, head) | 0.9381 |

So Table 13's one number we reproduce to the digit is the one that says
**nothing about hallucination**. It is near-tautological once `m_sys` is the
largest of three means, and the surrounding text — *"has the largest mean
score and is perfectly separated in the reported tail diagnostic, supporting
its role as the dominant prior source"* — invites it to be read as diagnostic
power. The qualitative claim it is offered for does hold, and strongly: 73 %
of attention mass sits on the system prefix.

### Figure 8, scored from the archive: the diagnosis is worth 0.49 of 5.03 points

Also not pre-registered, and worse than that — **this one was sitting in the
archive unscored.** `fox random_heads` ran ten POPE seeds and three CHAIR seeds
back in r6; `plan list` counted them under `controls`, every zip carried them,
and no document in this repo mentioned them until now. `select_heads(select=
"random")` and `mode="random_heads"` have been implemented the whole time, and
a stale comment in `plan.py` still said Figure 8 "needs a random-head
selection mode".

Figure 8 is the paper's own validation that the diagnosis does something:
*"the performance gains of Fox over a random-intervention baseline across
different intervention ratios K … consistently validate that targeting
specific risky mediators is superior to stochastic head suppression."*

At the released K (ratio 0.35 → 11 of 32 heads), POPE-adversarial, sampling,
ten seeds, paired:

| arm | accuracy | vs baseline | vs `fox` (90% CI) | TOST p (±0.5) |
|---|---|---|---|---|
| `fox` | 81.75 | **+5.03** | — | — |
| `fox random_heads` | 81.27 | **+4.54** | **−0.490** [−0.816, −0.164] | 0.478 |
| `fox off` (no intervention) | 78.92 | +2.21 | −2.827 [−3.044, −2.610] | 1.000 |
| `sampling` baseline | 76.72 | — | −5.033 | — |

**The paper's direction holds and its magnitude does not matter.** Replacing
the entire causal diagnosis with `torch.randperm` costs **0.49 of 5.03
points** — the selection is worth 10 % of the arm's gain, and 90 % survives
choosing the heads at random. The gap is distinguishable from zero (the
interval excludes it), so Figure 8 is not *wrong*; it is a true statement
about a 0.49-point effect, plotted as a validation of the causal story that
the other 4.54 points are attributed to.

That 4.54 is accounted for elsewhere in this document: `fox off` gives +2.21
from fusion alone, and `fox ulp_control` — rounding and nothing else — is
**equivalent to `fox`** at ±0.38. So the intervention's ~2.8 points are the
fp16 requantisation, and requantisation does not care which heads it is
applied to.

**On CHAIR, random heads are better, and closer to the paper.** Three matched
seeds:

| | `fox` | `fox random_heads` | paper |
|---|---|---|---|
| CHAIR_S | 48.47 | **42.27** | 46.40 |
| CHAIR_I | 13.89 | **12.11** | 12.90 |
| recall | 75.79 | **76.11** | — |
| correct objects | 1182 | 1187 | — |
| mentions | 3820 | 3790 | — |
| mean words | 96.3 | 94.6 | — |

Paired difference −6.20 CHAIR_S [−11.47, −0.93] in random's favour. This
project's own pitfall applies here and clears it: CHAIR_S falls while
**recall does not** (76.11 vs 75.79), correct objects are flat (1187 vs 1182),
and captions are the same length (94.6 vs 96.3 words). That is the opposite of
`vcd_fox_fusion`, which the report flags for buying CHAIR_S with recall
(66.35 against 76.09). Random head selection hallucinates less while naming
just as much.

So on the benchmark the paper leads with, stochastic head suppression is not
merely as good as the diagnosis — it **reproduces the paper's own reported
CHAIR numbers more closely than the diagnosis does** (42.27/12.11 against the
paper's 46.40/12.90, where our diagnosed run gives 48.47/13.89).

Two caveats, both real. The CHAIR arm has three seeds, so its interval is
wide. And the CHAIR image set is the unrecoverable part of this reproduction
(§3.8, 5.20 points of draw-to-draw drift), so the comparison **to the paper**
is weak — but the comparison between our two arms is on identical images and
identical seeds, and that is the one carrying the claim.

What is not yet answered is the axis Figure 8 actually sweeps. The paper plots
Δ against K, and this is one K. V7 below runs the sweep.

### Cost

Measured 2119.1 s for 3000 samples = **0.71 s/sample**, against the 1.2 s
estimate in `COST_S`. The estimate reasoned that eager attention's
`[1, 32, Q, Q]` fp32 materialisation would roughly double the 0.55 s SDPA
prefill; it adds 28 %, because at Q ≈ 620 the quadratic term is still small
next to the projections and the MLP. 1.7× high, in the safe direction — but
this is the second hand-reasoned multiplier in that table to be wrong by more
than half, after `BEAM_COST_S` at 2.2×. `COST_S` now carries the measurement.

## V6 — Does each rule change any decision? (no extra run)

Recorded on **every** paired run: `n_diverged` and `diverged_frac`, the
fraction of generated tokens where the contrastive rule's argmax differs
from the observational branch's.

**Prediction: strictly between 0 and 1, and higher on CHAIR than on POPE**
(one token versus ~120 gives the rule far more chances to disagree).

| outcome | reading |
|---|---|
| ≈ 0 | the method is inert — it produced a number without ever changing a decision. No accuracy figure would reveal this |
| ≈ 1 | the negative branch is uninformative and the rule is overriding the model rather than correcting it |

SID additionally records `sid_n_kept` per step, and CausalMM records
`cf_gain_mean` / `cf_gain_std`, so a run whose mechanism silently degraded
mid-way is visible in the record rather than only in the score.

---

## V7 — Does the diagnosis matter more when fewer heads are edited?

`--mode random_heads` against the default `as_implemented`, at
`k ∈ {0.10, 0.20, 0.45}`, POPE-adversarial, sampling, seeds 42/43/44, full
3000. The released `k = 0.35` already has ten seeds of each (Figure 8 above).
**16.2 h.**

This is the axis Figure 8 sweeps, and the one reading that could still rescue
the paper's account. At `k = 0.35` the diagnosis is worth 0.49 of 5.03 points
— but 11 of 32 heads is a third of them, and with a third of the heads edited
*any* selection catches some high-score heads by chance. `k = 0.10` edits
**3 heads**, where picking the right three should matter most if the ranking
means anything. `k = 0.45` is the ratio the **paper** states, versus the 0.35
the code ships.

**Prediction: the gap stays under 1 accuracy point at every K, and does not
grow as K shrinks.** Concretely: `|Δ(random − diagnosed)| < 1.0` at all three
ratios, with no monotone increase toward small K.

The reasoning is V2's result, not a guess. The intervention's ~2.8 points are
an fp16 requantisation of the attention logits (`ulp_control` ≡ `fox` at
±0.38, `ld=fp32` kills the effect entirely). Requantisation is indifferent to
*which* logits it rounds, so the arm's size should track **how many** heads
are edited, not which — falling for both arms together as K falls, while the
gap between them stays small.

| outcome | reading |
|---|---|
| gap < 1 pt at every K, flat | Figure 8 plots a real but small effect as validation of a causal story that the rest of the gain does not support. The requantisation account survives its sharpest test |
| gap > 2 pts at `k = 0.10` | **the prediction fails and the diagnosis is doing real work** that `k = 0.35` masked by editing enough heads to make selection nearly irrelevant. The requantisation account would then be incomplete, not wrong — it would explain the magnitude but not the localisation |
| both arms collapse to `fox off` at `k = 0.10` | the effect is purely the *number* of heads, and even 3-head requantisation is too little to move POPE. Consistent with the prediction and a cleaner demonstration of it |

Three seeds resolves a gap of roughly 1.0 point here, which is the resolution
the question needs — whether the gap is small or large, not its third digit.
`k = 0.35` carries ten seeds for the precise estimate.

---

## V8 — Does the intervention window matter at all?

`--intervention-layers` at `11-24` and `13-26` against the shipped `2-15`,
POPE-adversarial, sampling, seeds 42/43/44. **5.4 h.**

r13 measured the per-layer diagnostic strength the paper's own automatic rule
is defined on, and the shipped window loses on it: mean |AUC − 0.5| over
L2–15 is 0.0305 for `H_vis` and 0.0458 for `m_sys`, against all-layer averages
of 0.0373 and 0.0762. The best 14-layer windows — same width, so the budget
matches — are **L11–24** for `H_vis` (0.0571) and **L13–26** for `m_sys`
(0.1075). Table 15 already showed the paper's stated `2-13` is equivalent to
the code's `2-15` (±0.42), but both of those are inside the same weak region;
this asks the question against the region r13 says the signal is actually in.

**Prediction: both measured windows are equivalent to `2-15` at ±0.5.**

Same reasoning as V7: 14 layers of requantisation is 14 layers of
requantisation wherever they sit.

| outcome | reading |
|---|---|
| equivalent | the window is not load-bearing, and the paper's automatic rule selects a window whose choice does not matter. Three of three components of `S = m_sys · H_vis` over L2–15 then have a measurement against them |
| `11-24` or `13-26` **beats** `2-15` by > 0.5 | **the prediction fails, and constructively.** The diagnosis would be localising something real, and the released code would be shipping the wrong window — the paper's method would work *better* than the paper's implementation. That is a finding worth more than a confirmation |
| the measured windows are **worse** | layer position matters and the diagnosis points the wrong way, which would be the first evidence that the window choice is load-bearing *against* the paper |

This is the one prediction in the suite whose failure would partly rescue the
paper, which is the reason to run it.

---

## Results, round 14 (2026-11-12) — V7 holds where it was sharpest, V8 fails

**Both were pre-registered**, in `2e683d6`, before the jobs existed. V7's
prediction **splits** — the clause that carried the argument holds, the
numeric bound does not — and V8's **fails outright**. The gate moves to
**19 held, 2 failed**: V7 counts as held on its falsifier and V8 as failed,
which is the honest reading of a document whose whole point is that a
prediction written afterwards is worth nothing.

24 jobs, 3 sessions, 3 Kaggle accounts, all green. POPE-adversarial,
sampling, seeds 42/43/44, full 3000 each, paired bootstrap at 90 % over
9000 (seed, sample) pairs. Anchors on the same three seeds: `sampling`
76.72, `fox off` 78.97, `fox` (shipped L2–15) 81.56.

### V7 — the prediction splits, and the half that mattered holds

| K | heads edited | diagnosed | random | gap [90 %] |
|---|---|---|---|---|
| 0.10 | 3 | 79.40 | 79.14 | **+0.26 [−0.20, +0.71]**, p = 0.37 |
| 0.20 | 6 | 81.16 | 79.79 | +1.37 [+0.91, +1.80] |
| 0.45 | 14 | 82.28 | 81.49 | +0.79 [+0.36, +1.22] |

The prediction had two clauses. "Gap < 1 point at every K" **fails** at
`k = 0.20` on the point estimate (1.37, though its interval still contains
1.0). "Does not grow as K shrinks" **holds, decisively**: at `k = 0.10` —
three heads, where picking the right three should matter most — the gap is
**indistinguishable from zero**. The pre-registered falsifier was a gap above
2 points there; the measurement is 0.26.

What the arms do over the baseline is the clearer statement:

| K | diagnosed over baseline | random over baseline |
|---|---|---|
| 0.10 | +2.68 | +2.42 |
| 0.20 | +4.43 | +3.07 |
| 0.45 | +5.56 | +4.77 |

Both arms grow with K, and **random heads at `k = 0.45` (+4.77) beat
diagnosed heads at `k = 0.20` (+4.43)**. The effect is bought by *how many*
heads are edited, not by which — editing more of them at random buys back
everything the ranking provides, and more. That is V2's requantisation
account making a quantitative prediction and the sweep confirming it.

### V8 — the prediction fails, and the failure is the finding

| window | width | accuracy | vs L2–15 | vs `fox off` |
|---|---|---|---|---|
| **L2–15** (shipped) | 14 | 81.56 | — | **+2.59 [+2.09, +3.09]** |
| L11–24 (best for `H_vis`) | 14 | 78.49 | **−3.07 [−3.59, −2.54]** | −0.48 [−0.91, −0.06] |
| L13–26 (best for `m_sys`) | 14 | 79.31 | **−2.24 [−2.73, −1.77]** | +0.34 [−0.07, +0.77] |

Predicted equivalence at ±0.5. The measured gaps are 4× to 6× that, and both
in the worse direction. **The intervention window is load-bearing, and
"14 layers of requantisation is 14 layers of requantisation wherever they
sit" is simply wrong.**

Read the last column, because it is the sharper half. `fox off` is the arm
that keeps the dual pass and the fusion and performs **no intervention at
all**. Moving the window to L11–24 is *worse* than not intervening, and
L13–26 is **statistically indistinguishable from not intervening**
(+0.34, p = 0.18). All +2.59 of the intervention's contribution requires it
to happen **early**.

That corrects this document's own account in one direction and sharpens it in
another:

* **Against us.** V2's framing treated requantisation as position-blind. It
  is not: an edit at layer 2 passes through 30 more layers of processing, an
  edit at layer 26 through six. Depth of propagation belongs in the account,
  and r14 is what put it there;
* **Against the paper, harder.** r13 measured the per-layer diagnostic
  strength on the paper's own criterion and found it **peaks at L11–26** —
  which is exactly where intervening does nothing. So the place the
  diagnosis points and the place the intervention works are **different
  places**. Figure 7's automatic rule, applied to the signal this
  reproduction measured, selects the two worst windows tested.

The shipped L2–15 is therefore the right window for a reason the paper does
not give. Table 15 already showed the paper's stated L2–13 is equivalent to
it (±0.42) — both early, both work. The diagnostic argument for *why* those
layers is what r14 removes.

**What this does not say.** It does not say early layers are special to
*Fox*; the obvious alternative is that any logit perturbation applied early
propagates further, which would make this a depth effect of the model rather
than a property of the method. Separating those needs a position sweep at
matched width, which is r15.

---

## V9 — Is the fusion GATE doing the work the intervention is not?

`--fuse-strategy fixed` and `dynamic` against the released hybrid rule, on
POPE-adversarial (3000) **and** CHAIR (val2014_500), sampling, seeds 42/43/44.
**14.6 h.** This is the paper's own Table 14, which the released code has no
flag for; Figure 19's "always apply the intervention" arm is the `fixed` row,
so one sweep answers both.

Every other prediction in this suite interrogates the **diagnosis**. This one
interrogates the component the measurements keep pointing at instead:

| arm | over baseline | what it isolates |
|---|---|---|
| `fox` | +5.03 | everything |
| `fox random_heads` | +4.54 | the edit, with the ranking thrown away |
| `fox off` | +2.21 | the dual pass and the fusion, no intervention at all |
| `ulp_control` | ≡ `fox` (±0.38) | rounding the logits, no diagnosis |

`off` keeps 44 % of the gain while doing nothing the paper's causal framework
describes. What is left to explain Fox is the fusion rule — and the gate
inside Eq. 7 is the one piece of it nothing here has varied.

### The gate is already measured, and it almost never fires on POPE

Every archived Fox run records `d_t_mean`, `d_t_max` and `lambda_mean` per
sample, so the gate's behaviour does not have to be assumed. On the headline
arms, at `tau_js = 0.15`:

| | mean `d_t` | mean λ | samples ever reaching `d_t ≥ tau_js` |
|---|---|---|---|
| POPE-adversarial, 3000 | 0.0161 | 1.9964 | **0.2 %** |
| CHAIR, val2014_500 | 0.1285 | 1.3848 | **100 %** |

POPE emits **one** token, so its `d_t` is a single number per sample, and it
sits below the threshold on 2993 of 3000. CHAIR emits up to 512, so every
caption crosses the threshold somewhere and λ averages 1.38 rather than 2.
One axis, two completely different regimes — which is why this runs on both.

**Prediction, POPE: `fixed` ≡ `hybrid` within 0.3 points, and `dynamic`
loses most of the +5.03 — landing within 1 point of the `sampling`
baseline.**

This is close to arithmetic rather than a hunch. On POPE `fixed` can only
differ from `hybrid` on the 0.2 % of samples where the gate fires, so the two
arms are the same run for 2993 of 3000 samples. `dynamic` replaces λ = 2 with
λ = `d_t` ≈ 0.016 on the other 2993 — a 125-fold reduction — which switches
the negative branch off almost entirely and should leave something very close
to plain sampling.

**Prediction, CHAIR: `fixed` is the arm that moves, and `dynamic` moves
further.** λ already averages 1.38 there, so `fixed` is a real change (1.38 →
2.0) and `dynamic` a larger one in the other direction.

**The paper reports the opposite ordering.** Table 14 gives `fixed` as *worse*
than `dynamic` — 49.20 vs 47.68 CHAIR_S, 74.91 vs 75.83 F1 — i.e. the rule
that raises λ doing more damage than the rule that removes it. On this stack,
removing λ removes the method. One of the two accounts is wrong about this
fusion rule, and 14.6 h settles which.

| outcome | reading |
|---|---|
| POPE: `fixed` ≡ `hybrid`, `dynamic` ≈ baseline | the gate is a **floor, not a gate**: what matters is that λ is large, not that it is conditional. Eq. 7's threshold is then decoration on POPE, where the paper's headline accuracy numbers live, and `tau_js` could be set to 0 with no loss |
| CHAIR: `dynamic` beats `hybrid` | **the prediction fails and the gate is load-bearing against the paper** — amplifying on consensus would be the harmful part, and the released rule would be worse than the simpler one it replaced |
| the paper's ordering reproduces on CHAIR (`fixed` worse than `dynamic`) | **the prediction fails constructively.** Amplifying on conflict is actively harmful, which is a real mechanism and the first component of Fox to survive this audit intact |
| all three within 0.5 on both | the fusion rule does not matter either, and `off`'s +2.21 belongs to the dual pass — the plausibility mask and the second forward — rather than to any choice of λ. The strongest version of the requantisation account |

Both benchmarks also because this project has twice found them disagreeing:
random head selection loses 0.49 on POPE and **wins** 6.2 CHAIR_S points at
equal recall. A fusion rule that moved POPE and CHAIR the same way would
itself be worth reporting. CHAIR gets the standing three seeds — against a
measured 5.20-point draw-to-draw drift, three give a direction with an
interval, and ten would cost 15 h to refine a question 500 images cannot
answer more precisely anyway.

---

## V10 — Is "early" a property of Fox, or just of depth?

`--intervention-layers` at nine windows, POPE-adversarial, sampling, seeds
42/43/44. **24.8 h.** This prediction exists because **V8 failed**, and it
asks the question that failure opened rather than the one it closed.

r14 established that position is load-bearing: at matched width, L11–24 is
worse than performing no intervention at all and L13–26 is indistinguishable
from it, while the shipped L2–15 is +2.59 over it. Two readings survive, and
they are different claims:

* **depth** — any logit perturbation applied at layer 2 propagates through 30
  further layers and one applied at layer 26 through six, so earlier is
  stronger for reasons that have nothing to do with Fox;
* **selection** — something about the early layers Fox picks.

Two sweeps separate them, both at `k = 0.35`, both leaving every other
hyperparameter at the released value.

**Position, at fixed width 14** (so the number of requantised layers never
changes): starts at L0, L4, L8, L16, L18, filling in around the three windows
already measured (L2, L11, L13).

**Width, from the shipped start** L2: L2–5, L2–9, L2–23, L2–31, against the
shipped L2–15.

**Prediction: the position curve is monotone decreasing in start layer, with
no local structure, and L2–5 — four layers — already recovers more than half
of L2–15's +2.59 over `fox off`.**

Both halves follow from depth rather than from selection. If the mechanism is
that an early perturbation has more layers left to grow in, then moving the
window later can only lose, smoothly; and a *narrow* early window should
retain most of the effect, because the layers doing the work are the first
few in it.

| outcome | reading |
|---|---|
| monotone decay, L2–5 ≥ half of L2–15 | **depth, not selection.** Fox's window is "early" and that is the whole content of it; the diagnostic that selects it is unnecessary, and r13's measurement — that the diagnostic signal peaks at L11–26, where intervening does nothing — becomes the clean statement that the paper's two halves point at different layers |
| structure in the position curve (a bump away from the earliest start) | **the prediction fails and selection matters.** Some layers would be genuinely better targets than their neighbours, and the next question is whether the diagnostic predicts which — a question r13's per-layer AUCs can answer without new GPU time |
| width scales the effect, L2–5 ≪ half | the count of requantised layers matters after all, and V2's original position-blind account was closer than r14 suggested. Then L11–24's failure needs a different explanation and this audit has a loose end |
| L2–31 (every layer) beats L2–15 | the released window is not even a good early window, and nothing about the selection is doing work — the strongest version of the depth reading |

Three seeds at full 3000 resolves roughly 0.9 points here, against effects of
2 to 3; that is the resolution the question needs.

---

## Results, round 15 (2026-11-13) — the effect lives in six layers

39 jobs, 10 sessions on two Kaggle accounts, all green — six of the ten
waited in the bot's queue and were launched by cron as slots freed, which is
the first time that path has run on GPU.

**All three predictions were pre-registered** in `556b699`, before the jobs
existed. V9 holds on POPE and misses on CHAIR's direction; V10 fails on both
clauses, and the failure localises the effect better than the prediction
would have. Gate moves to **20 held, 3 failed**.

POPE-adversarial, sampling, seeds 42/43/44,
full 3000, paired bootstrap at 90 % over 9000 pairs. Anchors unchanged:
`sampling` 76.72, `fox off` 78.97, `fox` (L2–15) 81.56.

### V10a — the position curve is a cliff, not a slope

| start | window | accuracy | vs L2–15 | vs `fox off` |
|---|---|---|---|---|
| 0 | L0–13 | 81.61 | +0.06 [−0.37, +0.48] | **+2.64** |
| 2 | **L2–15** (shipped) | 81.56 | — | **+2.59** |
| 4 | L4–17 | 81.53 | −0.02 [−0.44, +0.40] | **+2.57** |
| 8 | L8–21 | 79.29 | −2.27 | +0.32, p = 0.23 |
| 11 | L11–24 | 78.49 | −3.07 | −0.48 |
| 13 | L13–26 | 79.31 | −2.24 | +0.34, p = 0.18 |
| 16 | L16–29 | 79.38 | −2.18 | +0.41, p = 0.10 |
| 18 | L18–31 | 79.43 | −2.12 | +0.47, p = 0.07 |

**The prediction fails on its shape.** It said monotone decay in start layer.
What the sweep shows is a **step**: starts at 0, 2 and 4 are mutually
indistinguishable and all deliver the full ≈ +2.6; starts at 8 and beyond
deliver nothing distinguishable from not intervening, and are flat from there
to the end of the model. There is no slope to speak of in either region.

### V10b — and the width sweep says which layers

| window | layers edited | accuracy | vs `fox off` | share of L2–15's gain |
|---|---|---|---|---|
| L2–5 | 4 | 79.91 | +0.94 | 0.36 |
| L2–9 | 8 | 81.29 | +2.32 | **0.90** |
| **L2–15** | 14 | 81.56 | +2.59 | 1.00 |
| L2–23 | 22 | 81.53 | +2.57 | 0.99 |
| L2–31 | 30 | 81.30 | +2.33 | 0.90 |

Differencing the two sweeps localises the effect to six layers:

| band | contribution | from |
|---|---|---|
| layers 2–3 | **+0.02** | L2–15 minus L4–17 |
| layers 4–5 | +0.92 | L2–5 minus the above |
| layers 6–9 | **+1.38** | L2–9 minus L2–5 |
| layers 10–15 | +0.27 | L2–15 minus L2–9 |
| layers 16–23 | −0.02 | L2–23 minus L2–15 |
| layers 24–31 | −0.24 | L2–31 minus L2–23 |

The two sweeps cross-check independently: L8–21 retains only layers 8 and 9
of the live band and scores +0.32, the size the width sweep implies for that
tail. ~~**Roughly 89 % of the intervention's contribution comes from layers
4–9.**~~ — **retracted by r16.** The differencing is exact arithmetic but not
a valid attribution: each difference is the marginal value of a layer *given
the window it is added to*, and r16 measures that marginal swinging by over a
point with the path. L4–17 excludes layers 2–3 and keeps 99 % of the effect;
L2–3 excludes 4–9 and keeps 47 %. What survives is that an early run of
layers is needed and that the effect saturates. That everything past layer 15
is inert or slightly harmful does survive.

The second clause of the prediction also fails, narrowly: L2–5 recovers
**0.36** of L2–15's gain, not the "more than half" predicted. It fails for a
locatable reason — the peak band is 6–9, and L2–5 stops one layer short of
it.

### V9 — the gate is a floor on POPE, and a liability on CHAIR

POPE-adversarial, same three seeds:

| strategy | accuracy | vs hybrid | vs `sampling` | POPE F1 |
|---|---|---|---|---|
| **hybrid** (Eq. 7, shipped) | 81.56 | — | +4.84 | 81.98 |
| `fixed` (λ = α always) | 81.91 | **+0.36 [−0.07, +0.77]**, p = 0.17 | +5.19 | 82.22 |
| `dynamic` (λ = d_t always) | 76.74 | **−4.81 [−5.47, −4.15]** | **+0.02** | 78.87 |

**Both POPE clauses hold.** `fixed` is within 0.5 of `hybrid` as predicted —
indistinguishable, in fact. `dynamic` loses more than a point as predicted,
and the magnitude is the striking part: at +0.02 over the baseline it has
**no effect at all**. Removing the consensus floor removes the method.

That is the pre-registered "floor, not a gate" outcome, and it follows from
the measurement made before the run: the gate sits in its consensus branch on
99.8 % of POPE samples, so `hybrid` is `fixed` almost everywhere, and
`dynamic` replaces λ = 2 with λ ≈ 0.016. Eq. 7's threshold — one of the
paper's two named contributions in the fusion — does nothing on the benchmark
that carries its headline accuracy numbers.

CHAIR, where the gate actually fires, with the recall guard this project
requires:

| arm | CHAIR_S | CHAIR_I | recall | distinct objects | words |
|---|---|---|---|---|---|
| `sampling` | 54.40 | 16.90 | 75.64 | 1180 | 102.1 |
| `fox off` | 49.20 | 14.05 | 78.57 | 1226 | 99.2 |
| **hybrid** | 48.47 | 13.89 | 75.79 | 1182 | 96.3 |
| `fixed` | **32.20** | **10.12** | 70.73 | 1103 | 102.8 |
| `dynamic` | 49.47 | 14.77 | 76.84 | 1199 | 99.5 |

**The paper's ordering does not reproduce — it inverts.**

| | best | middle | worst |
|---|---|---|---|
| Table 14 | hybrid 46.40 | dynamic 47.68 | **fixed 49.20** |
| measured | **fixed 32.20** | hybrid 48.47 | dynamic 49.47 |

Our `hybrid` at 48.47 against the paper's 46.40 sits inside the 5.20-point
image-set drift, so that row is consistent. `fixed` at 32.20 against the
paper's 49.20 is a **17-point** gap, far outside it. The paper ranks "always
apply the intervention" last; here it is first by sixteen points.

**The recall guard, applied honestly.** This is not free, and it is not the
trap either. `fixed` cuts CHAIR_I by 27 % relative (13.89 → 10.12) while
recall falls 6.7 % relative (75.79 → 70.73) and distinct correct objects fall
from 1182 to 1103. A favourable trade, not a clean win — and notably **not**
the "shorter caption" artefact the project has been caught by before:
`fixed`'s captions are the **longest** of any arm at 102.8 words. It writes
more and names fewer objects, rather than writing less.

One more row deserves reading: `fox off` reaches CHAIR_S 49.20 at the
**highest recall of any arm** (78.57, 1226 objects). The full method scores
48.47 at 75.79. On CHAIR the intervention buys 0.73 points of CHAIR_S for 2.8
points of recall, which is close to no trade at all.

**The CHAIR clause of the prediction fails.** It said `dynamic` would move
further than `fixed`. `dynamic` barely moves (+1.00 CHAIR_S); `fixed` moves
16 points. The reasoning was that λ already averages 1.38 on CHAIR so the
distance to α = 2 is short — true, but it treated the two directions as
symmetric in effect, and they are not.

Gate: **20 held, 3 failed.** V9 held on both POPE clauses and missed the
CHAIR direction; V10 failed on both of its clauses.

### What this settles, and what it costs each side

The pre-registered reading for "monotone decay" was *depth, not selection*.
The cliff supports the same conclusion by a sharper mechanism, so the
conclusion survives its prediction failing:

* **Position within the live band does not matter.** L0–13, L2–15 and L4–17
  are indistinguishable. Whatever the paper's automatic rule selects, any
  early start in 0–4 gives the same number;
* **Width does not matter either**, past layer 9. Fourteen layers, twenty-two
  layers and the first eight are all within 0.27 of each other;
* **So nothing in Fox's window selection is doing work.** The only thing that
  matters is whether the window covers enough early layers, and every early
  window does. (r15 wrote "layers 4–9" here; r16 retracts the specific band —
  the effect is sub-additive, so no band owns a fixed share — while the
  conclusion itself stands.)

Against the paper that is the strongest statement this suite has produced.
r13 measured the diagnostic signal — the quantity Figure 7 plots and the
automatic rule consumes — as peaking at **L11–26**. That range is now
measured to be **entirely inert**: every window starting at 8 or later is
statistically indistinguishable from performing no intervention. The
diagnosis points at layers where intervening does nothing, and the
intervention works at layers the diagnosis ranks weakest.

**What it costs us.** r14 put "depth of propagation" into the requantisation
account. The cliff does not support a propagation-depth story either: under
propagation, layer 10 should be worth a little less than layer 9, not nothing
at all, and layers 16–31 should still be worth something. A step at 9–10 with
a flat floor after it is a different phenomenon, and this document does not
have an account of it. Naming it is honest; claiming otherwise would not be.

---

## V11 — What is the cliff at layer 9?

`--intervention-layers` at cumulative windows `L2-k` for every k from 2 to 14,
POPE-adversarial, sampling, seeds 42/43/44. **27.5 h.** This is the one
question in this document that neither the paper's account nor ours answers.

r15 established that 89 % of the intervention's contribution comes from
layers 4–9, that starts at 0, 2 and 4 are interchangeable, and that starts at
8 or later are worth nothing. Neither story survives that:

* **the paper's** — the diagnostic signal peaks at L11–26, which is the inert
  half;
* **ours** — "depth of propagation" predicts a decay, and a decay does not
  look like a step with a flat floor after it.

**The design is cumulative, not per-layer, and that is the point.** A single
layer is worth roughly 0.35–0.45 points and three seeds on 3000 samples
resolve about 0.4, so a single-layer sweep would be under-powered exactly
where it must be sharp — and would return a curve of noise with the shape of
structure. Each cumulative point measures 0.9 to 2.6 points instead, and
consecutive differences recover the per-layer contributions from
well-powered measurements. k = 5, 9, 13 and 15 are already on the curve from
r15 and Table 15, so they anchor it for free.

**Prediction: the curve is flat from k = 2 to k = 3, rises roughly linearly
from k = 4 to k = 9 at 0.3–0.5 points per layer, and is flat from k = 10
onward. No single layer contributes more than 0.8 points.**

That is the shape the three points already measured imply if the band is
uniform: layers 2–3 are worth +0.02 together, layers 4–5 are worth +0.92,
layers 6–9 are worth +1.38 (0.35 each), layers 10–15 are worth +0.27 in
total.

| outcome | reading |
|---|---|
| flat, linear, flat, no layer above 0.8 | the live band is a **contiguous region**, not a privileged layer. "Which layers" then has no finer answer than "the early-middle ones", and the diagnosis cannot be rescued by pointing at a particular layer it ranks well |
| one layer contributes **> 1.0** | **the prediction fails, and interestingly.** A privileged layer would be a real localisation, and the next question is free: r13's per-layer AUCs already exist, so whether the diagnosis ranks that layer highly can be answered with no further GPU time |
| the rise continues past k = 10 | r15's width sweep was aliased by its coarse spacing (5, 9, 15), and the band is wider than six layers. The "89 % in layers 4–9" figure would need restating |
| no structure anywhere — every k within noise | three seeds are too few even cumulatively, and the honest move is to say so rather than read a curve out of it |

The fourth row is a real possibility and is why the prediction names a
resolution. If consecutive differences are all inside ±0.4 the curve will not
support per-layer claims, and this document should say the cliff is located
between k = 9 and k = 15 and no more precisely.

---

## Results, round 16 (2026-11-14) — no layer is necessary, and r15's attribution does not survive

30 of 30 cliff jobs returned (two of them partial at the session wall: L2–14
at seeds 43 and 44, 2972 and 2599 of 3000, paired on the intersection
throughout). Session 5 did not come back, so Table 1's completion is 16 of 24
jobs and is reported separately below. POPE-adversarial, sampling, seeds
42/43/44, against `fox off`.

**V11's prediction fails on both clauses, and the failure is larger than the
prediction.** It also falsifies a claim this document made in r15.

### The curve, and what each layer adds

The right statistic is the **paired step** — L2-k against L2-(k−1) on the same
samples. The two windows share almost everything, so the step is measured far
more precisely (±0.2) than either window's own interval against `off` (±0.45).

| layer k | step | 90 % CI | verdict |
|---|---|---|---|
| 3 | **+1.12** | [+0.88, +1.37] | adds |
| 4 | −0.10 | [−0.30, +0.09] | nothing |
| 5 | −0.18 | [−0.37, +0.01] | nothing |
| 6 | −0.01 | [−0.18, +0.16] | nothing |
| 7 | **+0.60** | [+0.40, +0.81] | adds |
| 8 | **+0.71** | [+0.49, +0.93] | adds |
| 9 | +0.08 | [−0.06, +0.21] | nothing |
| 10 | +0.08 | [−0.08, +0.23] | nothing |
| 11 | −0.18 | [−0.32, −0.02] | **negative** |
| 12 | **+0.43** | [+0.24, +0.62] | adds |
| 13, 14, 15 | −0.09, +0.06, −0.08 | all span 0 | nothing |

Predicted: a roughly linear rise from k = 4 to k = 9 at 0.3–0.5 per layer, and
no single layer above 0.8. Measured: layers 4, 5 and 6 add **nothing**, layer
3 adds **+1.12**, and the layers that do add are 3, 7, 8 and 12.

### But no layer is necessary, because the effect is sub-additive

| window | layers | vs `fox off` |
|---|---|---|
| L2–3 | 2, 3 only | **+1.22** |
| L4–17 | 4…17, **no 2 or 3** | **+2.57** |
| L2–15 | 2…15 | +2.59 |
| L2–6 | 2…6, **no 7 or 8** | +0.93 |
| L8–21 | 8…21, **no 2–7** | +0.32 |

Layer 3 is worth +1.12 when it is the second layer in the window and **+0.02**
when layers 4–15 are already there. L4–17 drops both of the layers the step
table makes look most valuable and keeps 99 % of the effect. Two layers alone
get 47 % of what fourteen get.

Additivity, stated as a number: L2–3 (+1.22) plus L4–17 (+2.57) would be
+3.79 if the contributions added. Measured together, L2–15 gives **+2.59** —
a shortfall of **−1.20**, almost half the total effect.

> [!warning] This retracts r15's "89 % of the effect is in layers 4–9"
> That figure came from differencing cumulative windows (L2–5, L2–9, L2–15).
> The **arithmetic was right** — a telescoping sum along one path is exact —
> but the **attribution was not**. Those differences are the marginal value of
> each layer *given the particular layers already in the window*, and r16
> shows that marginal value swings by more than a point depending on the
> path. Reading them as intrinsic per-layer importance was the error, and it
> was mine, in `f9e0916`.
>
> What survives is weaker and different: **the intervention needs a run of
> early layers beginning at or before about layer 7, and saturates once it
> has enough of them.** Which particular layers is largely interchangeable.

### What that does to the three accounts

* **The paper's localisation story is no better off.** Its diagnostic still
  peaks at L11–26, and layers 9, 10, 13, 14, 15 are measured here as adding
  nothing while 11 is slightly negative. Nothing in the diagnosis predicts
  layers 3, 7, 8;
* **Our depth story is now clearly wrong**, not merely unsupported. Depth of
  propagation predicts monotone decay in layer index. Layers 4–6 adding
  nothing while 7 and 8 add 1.31 between them is not decay;
* **Requantisation fits best of the three.** If the mechanism is fp16
  rounding of attention logits, then what matters is that *enough* early
  layers are rounded, not which — and the effect should saturate. It does,
  sharply, and the sub-additivity is what saturation looks like.

None of that is a positive account of why the live region ends around layer
7–8. This suite still does not have one, and r16 narrows the question rather
than closing it: the untested gap is now starts at 5, 6 and 7, between L4–17
which works and L8–21 which does not.

### Table 1 — 16 of 24, because session 5 did not return

VCD, ICD, SID and CausalMM on the popular and random splits: two of the three
seed-sessions came back. The eight missing jobs are one seed of each
method × split pair. Not a failure of the arm — the session never reported —
so re-releasing the same frozen plan picks up exactly those eight and nothing
else, since resume is keyed on the run file.

Gate: **20 held, 4 failed.**

---

## What this suite does **not** establish

* **That our number should match the paper's.** It cannot: the paper's run
  is not ours, the CHAIR image set is unrecoverable (§3.8), and CausalMM in
  particular runs here under the shared protocol (`top_p=0.9`, Fox's
  generation length) rather than its own (`top_p=1`, 1024 tokens). A
  mismatch there is **expected** and is not evidence of a defect.
* **That a method helps.** These arms check that each method is the thing
  its paper describes. Whether that thing improves POPE or CHAIR is what the
  `table1_methods` / `table2_methods` rows are for.
* **Anything about OPERA on CHAIR.** Its rollback is not implemented, for
  the reasons in `DEVIATIONS.md` §3.13 — including an fp16 saturation in the
  trigger that can drive every candidate to `-inf`.

## Order of operations

The validation tier sits **before** the table rows in `plan.PLAN`, so it
lands in session 1. That ordering is deliberate: running it last would risk
spending 40 h producing table rows from an implementation that one of these
arms would have shown to be wrong in the first hour.
