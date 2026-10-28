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
