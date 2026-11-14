# Every table and figure in the paper, and whether we can run it

The paper prints **15 tables and 22 figures**. This is the inventory: what each
one claims, whether this project can produce it, what it would cost, and what
has already been done. The goal set for the project is to reproduce as many as
possible — higher or lower is acceptable provided the difference is explained.

Status values:

| | |
|---|---|
| **DONE** | run and scored; numbers in `results/reports/` |
| **CHEAP** | runnable with code we already have, under ~2 GPU-hours |
| **RUNNABLE** | runnable, needs new code or real GPU time |
| **BLOCKED** | needs a model, dataset, or codebase not available here |

Backbone coverage throughout: **LLaVA-1.5 only**. InstructBLIP runs a second,
divergent copy of the intervention (§3.11) and Shikra needs a port; neither is
a configuration change. Every row below is the LLaVA-1.5 column unless stated.

---

## Main tables

### Table 1 — POPE accuracy, 3 splits x 7 methods, 3 backbones — **PARTIAL**

| | adversarial | popular | random |
|---|---|---|---|
| `sampling`, `fox` | 10 seeds | 3 seeds | 3 seeds |
| OPERA | 1 run (greedy) | 1 run | 1 run |
| VCD, ICD, SID, CausalMM | 3 seeds | **missing** | **missing** |

**OPERA is already complete**: it runs greedy, which is deterministic, so one
run per split is the whole measurement and seeds do not apply. A first pass at
this table costed it at 31 h for "two missing splits × three seeds" and both
halves of that were wrong.

What is actually missing is 4 methods × 2 splits × 3 seeds = **24 jobs, 22 h,
no new code**. Scheduled as r16. Backbones beyond LLaVA-1.5 are a separate
axis; see the loader note at the end.

### Table 2 — CHAIR, 7 methods — **PARTIAL**

`sampling` and `fox` done at 10 seeds. Measured but unresolved: the gap moves
5.20 points across five image draws, so the single-draw interval understates
it (§3.8.1).

### Table 8 — POPE **F1** on 3 splits x 3 backbones — **DONE for our rows**

F1 comes from the same runs as Table 1; no extra GPU time. Already in the
report.

### Table 9 — greedy POPE and CHAIR — **DONE**

All three POPE splits and CHAIR, both methods. Greedy is deterministic so one
run each is complete. POPE-adversarial reproduces the paper's gap exactly
(+3.33 against +3.33).

### Table 6 — efficiency, latency per token — **DONE, as a ratio**

Absolute latency cannot be compared: the paper used 8x A100, this used 2x T4.
The *ratio* can, and it reproduces:

| | sampling | fox | ratio |
|---|---|---|---|
| paper (A100) | 110 ms | 206 ms | 1.87x |
| ours (T4) | 538 ms | 1092 ms | **2.03x** |

Taken from `latency_ms`, recorded in every run record. OPERA's row is blank in
the paper for 1-token generation and would need OPERA implemented.

---

## Ablations — the part not yet touched

### Table 7 — sensitivity of α — **DONE**, all four values

Paper: α ∈ {0, 1, 2, 3} → mean POPE accuracy 81.75 / 85.33 / 85.97 / 85.98.
Measured on POPE-adversarial, greedy, seed 42:

| α | 0 | 1 | 2 (default) | 3 | 4 |
|---|---|---|---|---|---|
| ours | 79.10 | 81.83 | 82.37 | 82.37 | 82.37 |
| paper (mean of 3 splits) | 81.75 | 85.33 | 85.97 | 85.98 | — |

**The shape reproduces and then saturates.** α = 0 drops to near the baseline
as it must; the climb from 0 to 1 is most of the effect; and α ∈ {2, 3, 4} are
identical to two decimals. The paper's own 85.97 → 85.98 across α ∈ {2, 3}
says the same thing. Note "Mean Acc" there averages three POPE splits while
this is adversarial alone, so only the shape is comparable, not the level.

Worth noting: **α=0 should be the baseline**. Setting the amplification of the
interventional branch to zero leaves `z_final = z_obs`, so the only remaining
difference is the β-mask. The paper's 81.75 against a baseline mean of 81.48
is consistent with that and is a useful sanity check on our implementation.

A second note: Table 7 gives α=2 as 85.97 while Table 4 gives the same
configuration as 85.99. A 0.02 discrepancy between two tables describing one
run.

### Tables 4 and 12 — temporal-anchor / query-position ablation — **RUNNABLE**

**These are the same table printed twice** — identical five rows, identical
numbers, once in the main text as "temporal-anchor" and once in the appendix
as "query-position".

| Q | Acc | F1 | CHAIR_S | CHAIR_I |
|---|---|---|---|---|
| random token | 83.45 | 83.71 | 53.42 | 15.94 |
| mean-pool all tokens | 84.08 | 84.19 | 52.06 | 15.21 |
| x_last only | 85.02 | 84.88 | 48.63 | 13.88 |
| y_{t−1} only | 84.91 | 84.76 | 49.12 | 14.07 |
| x_last + y_{t−1} | **85.99** | 85.77 | **46.40** | **12.90** |

The bottom row is the main configuration (46.40 / 12.90 match Table 2), and
85.99 is the mean of Table 1's three splits.

Runnable, and it bears directly on a finding of ours: the released code uses
`tail_n = 1`, a single query position, not the two anchors the paper
describes. These rows would measure what that choice costs.

### Table 14 — fusion-strategy ablation — **DONE, r15. The ordering inverts**

| strategy | CHAIR_S | CHAIR_I | F1 |
|---|---|---|---|
| always fixed (λ = α) | 49.20 | 13.74 | 74.91 |
| always dynamic (λ = d_t) | 47.68 | 13.21 | 75.83 |
| hybrid, Eq. 7 | **46.40** | **12.90** | **76.57** |

High value because our `off` arm already shows the fusion alone carries +2.21
of Fox's +5.03 on POPE, and this isolates the gating rule inside it.

Implemented as `--fuse-strategy`, a `strategy` parameter on `fox.kernel.fuse`
whose default is asserted bit-equal to `reference_fuse`. Not two one-line
changes after all: the value has to reach the run key, or two fusion rules
share a file and resume from each other. It is in `OPTIONAL_VARIANT_FIELDS`,
so the 241 existing runs keep their names.

Already measurable from the archive, and it sharpens the prediction a long
way: every Fox run records `d_t` per sample, and on POPE the gate fires on
**0.2 %** of samples (mean `d_t` 0.0161, mean λ 1.9964) against **100 %** on
CHAIR (mean `d_t` 0.1285, mean λ 1.3848). So `fixed` and `hybrid` are nearly
the same run on POPE, and `dynamic` should remove the method there. The paper
reports `fixed` as the *worse* of its two non-hybrid rows, which is the
opposite ordering. See V9 in docs/VALIDATION.md.

**Measured.** On POPE the gate does nothing: `fixed` is +0.36 over `hybrid`
(p = 0.17, indistinguishable) and `dynamic` is **+0.02 over the baseline** —
removing the consensus floor removes the method entirely. On CHAIR the
paper's ordering inverts:

| | best | middle | worst |
|---|---|---|---|
| Table 14 | hybrid 46.40 | dynamic 47.68 | **fixed 49.20** |
| measured | **fixed 32.20** | hybrid 48.47 | dynamic 49.47 |

Our `hybrid` (48.47) is inside the 5.20-point image-set drift of the paper's
46.40. `fixed` at 32.20 against their 49.20 is 17 points outside it. The row
the paper ranks last is first here by sixteen points — at a real but smaller
cost: CHAIR_I falls 27 % relative while recall falls 6.7 %, and the captions
get *longer*, so it is not the shorter-caption artefact.

### Table 15 — manual vs automatic layer window — **DONE**

States the LLaVA-1.5 manual window as **L2–13** with POPE-adversarial 81.93,
and an automatic rule selecting L3–15 for 81.27.

This contradicts the released code, which runs **L2–15**. So the paper's own
appendix says the headline number came from 2–13 while the code ships 2–15 —
and 2–15 is what the automatic rule is said to select, scoring 0.66 lower.

**Measured: the two windows are equivalent**, which means the appendix's
0.66 gap does not reproduce and the question of "which produced 81.93" has no
answer — either would.

| | L2–13 (paper) | L2–15 (code) |
|---|---|---|
| POPE-adv, greedy | 82.20 | 82.37 |
| POPE-adv, sampling, 3 seeds | 81.53 | 81.56 |
| CHAIR_S, greedy | 42.80 (recall 76.09) | 46.20 (recall 77.82) |

r15/V10 explains why: both windows cover the early region that carries the
effect, so trimming two layers off the far end changes nothing. (r15 put a
figure of "89 % in layers 4–9" on this; r16 retracted it — the effect is
sub-additive, so no band owns a fixed share.) The CHAIR
difference runs the other way and comes with 1.7 points less recall.

### Table 13 — prior-source diagnostic — **DONE, r13**

Claims mean diagnostic score m_vis ≈ 0.57, m_txt ≈ 0.62, m_sys ≈ 0.82, and a
tail ROC-AUC of **1.00**.

Measured 0.1057 / 0.1602 / 0.7341, summing to 1.0000. Eq. 11 bounds each in
[0, 1] over spans that partition the keys, so the sum is forced; the paper's
2.01 means the table reports something Eq. 11 does not define, and it defines
no alternative.

The AUC of 1.00 turns out not to be a hallucination AUC at all — the caption
says it "treats system-token responses as the positive class", i.e. it asks
whether `m` can tell a system token from a visual or text one. That reproduces
exactly (1.0000 at the sample level, 0.9381 per head) and is near-tautological
once m_sys is the largest of the three. The suspicion this entry was written
with — that perfect separation is worth checking — was right for the wrong
reason: the number is sound and carries no information about hallucination.

Full numbers in `VALIDATION.md` §"Results, round 13".

### Figure 3 / 13 — ROC of the system-reliance score — **DONE, r13**

Eq. 13 as written gives **0.1310** at tail = 1 against the paper's 0.8226,
because its denominator Σ ΔAUC came out **negative** (−2.88 against Σ|ΔAUC| =
74.55, a 3.9 % residual of 1024 near-cancelling terms) and a negative global
scalar reverses the ranking. Dividing by Σ|ΔAUC| gives 0.8690 in sample and
0.8735 held out — better than the paper reports. The signal is real; the
scalar that publishes it has a sign nothing in the construction pins down.

Two controls, in the same section: unweighted mean m_sys is at **chance**
(0.4626), and per layer hallucination *raises* m_sys in only 11 of 32 layers
with all five strongest layers running the other way. The probe reads which
heads attend to the system prefix, not how much — which is not the claim.

### Figure 7 — layer-wise diagnostic strength of H_vis — **DONE, r13**

Per-layer |AUC − 0.5|, and the input to Table 15's automatic rule.

H_vis reproduces in direction: hallucination raises it in 24 of 32 layers,
peak L23 (+0.1077). The rule does **not** reproduce the shipped window. At the
released window's own width of 14 layers, L2–15 scores 0.0305 for H_vis and
0.0458 for m_sys, below the all-layer averages (0.0373 / 0.0762) and 1.9×/2.3×
below the best equal-width window (L11–24 / L13–26). There is no contiguous
early-to-mid run above threshold to keep: the signal is mid-to-late.

**r15 then swept window position at matched width, and the curve is a
cliff.** Starts at L0, L2 and L4 are mutually indistinguishable and all give
the full +2.6 over `fox off`; starts at L8 and later give nothing
distinguishable from not intervening, flat to the end of the model.
r16 then walked the window one layer at a time and found the effect is
**sub-additive**: L4–17 drops layers 2–3 and keeps 99 %, L2–3 drops 4–15 and
keeps 47 %, and the two together give less than either sum implies by 1.20
points. **No layer is necessary**; a run of early layers is, and it saturates.
Since Figure 7's signal peaks at L11–26, and layers 9–15 are measured as
adding nothing (layer 11 slightly negative), the quantity this figure plots
still does not predict where intervening helps.

**r14 first intervened on the two windows this figure's rule selects, and
they do not work.** L11–24 is
3.07 points below the shipped L2–15 and *worse than performing no
intervention at all*; L13–26 is 2.24 below and statistically identical to no
intervention. So the diagnostic signal peaks where intervening is worthless,
and the intervention works where the diagnostic signal is weakest. Those are
measurements of two different layer ranges, and the paper presents them as
one argument. r15/V10 asks whether "early" is a depth effect of the model
rather than anything Fox selects.

### Figure 8 — diagnosis-driven vs random head selection — **DONE, r6 + r14**

The paper's own validation that the diagnosis does something: Fox against
random head selection at matched budget K. The claim our finding most directly
challenges — if the effect is requantisation, random heads at the same budget
should do as well, and the paper says they do not.

**They very nearly do.** `fox random_heads` ran ten POPE seeds and three CHAIR
seeds in **r6** and went unscored until r13; the arms were in `plan list`
under `controls` the whole time. At the released ratio 0.35:

- POPE-adversarial, sampling, 10 paired seeds: random heads score **+4.54**
  against the baseline where diagnosed heads score **+5.03**. Replacing the
  diagnosis with `torch.randperm` costs **0.49 of 5.03 points** — the gap is
  real (90% CI [−0.816, −0.164] excludes zero) and it is 10% of the arm.
- CHAIR, 3 paired seeds: random heads are **better** — CHAIR_S 42.27 vs 48.47,
  CHAIR_I 12.11 vs 13.89 — at **unchanged recall** (76.11 vs 75.79, 1187 vs
  1182 correct objects, 94.6 vs 96.3 words), so this is not the
  name-less-hallucinate-less trade the report flags `vcd_fox_fusion` for. They
  also land closer to the paper's own 46.40 / 12.90 than our diagnosed run.

Figure 8 is therefore not false; it is a true statement about a sub-point
effect, plotted as validation of a causal story that the other 4.54 points do
not support.

What it has not answered is the axis the figure actually sweeps — Δ against K.
r14 ran `k ∈ {0.10, 0.20, 0.45}` × {diagnosed, random} × 3 seeds at full
3000, pre-registered as **V7**. `k = 0.10` edits three heads, where the ranking
should matter most if it matters anywhere.

### Figure 9 — k, τ_JS, β sensitivity — **DONE**, all three axes

POPE-adversarial, greedy, n = 500, seed 42. Released values are k = 0.35,
τ_JS = 0.15, β = 0.1.

| axis | values | accuracy |
|---|---|---|
| k | 0.25 / 0.45 | 81.60 / 81.80 |
| τ_JS | 0.1 / 0.2 | 82.20 / 82.80 |
| β | 0.05 / 0.2 | 82.60 / 82.60 |

Flat on every axis at this resolution — β is **identical** at both values, and
the k and τ_JS spreads are inside what 500 samples resolve. r15's V9 explains
why τ_JS looks flat: the gate it controls almost never fires on POPE, so
moving the threshold moves nothing.

### Figure 19 — the role of JSD gating — **DONE, r15, and it contradicts the figure**

Overlaps Table 14; the "always apply the intervention" arm is the
`always fixed` row, so `--fuse-strategy fixed` produces both and the sweep is
costed once. That arm is the best of the three on CHAIR and tied with the
shipped rule on POPE, so "always apply the intervention" is not the ablation
casualty the figure presents it as.

---

## Compared methods

The paper compares against five: ICD, VCD, OPERA, SID, CausalMM.

### VCD and ICD — **DONE on adversarial, and the released code cannot do it**

Both flags in the released harness are dead:

* `--use-vcd` is unreachable. `pope_eval.py:137` declares
  `--use-cd` as `action='store_true', default=True`, which cannot be set
  False from the command line, and the branch is
  `if args.use_cd: image_cd = image` / `elif args.use_vcd: ...`.
* `--use-icd` is killed by `vcd_sample.py:142`, which assigns
  `use_icd = False` unconditionally after line 121 computed it, before the
  `elif use_icd:` branch at line 226 that holds the contrastive path.

So the published code cannot reproduce its own VCD and ICD rows. Where those
numbers came from is not determinable from what was released.

Ours are implemented from the formula below rather than from that dead
branch, and run on POPE-adversarial and CHAIR at 3 seeds. The two other POPE
splits are the gap in Table 1 above.

The correct formula is nevertheless present, in that dead branch
(`vcd_sample.py:254-259`):

```python
cutoff = log(cd_beta) + min(logits.max(), logits_cd.max())
diffs  = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd
cd_logits = diffs.masked_fill(next_token_logits < cutoff, -inf)
```

with `cd_alpha = 1`, `cd_beta = 0.1`. Note it is a **subtraction** — Fox's
fusion is `z_obs + λ·z_do`, an addition. The two differ in sign and structure.

VCD and ICD share this fusion and differ only in the negative branch:

| | positive branch | negative branch |
|---|---|---|
| VCD | the image | diffusion-noised image, `noise_step = 500` |
| ICD | the image | same image, prompt carrying `"You are a confused object detector."` |

Both are implementable on the dual-branch decode loop already in
`fox/attention.py`. **Our existing `vcd` arm is not VCD** — it is VCD's noised
image inside Fox's additive fusion. It now carries the label
`vcd_fox_fusion`; `scripts/relabel_vcd_fox_fusion.py` renamed the archived
records, and real VCD with the subtractive fusion runs beside it.

### OPERA — **DONE** (was: runnable but large)

Beam search with an over-trust penalty and rollback. Needed its own decode
loop (`fox/opera.py`), and the paper evaluates it under `num_beams=5`, outside
the shared sampling protocol. Built, with the rollback deliberately left out
(`DEVIATIONS.md` §3.13). V5 then decomposed it: at OPERA's own 10-token
protocol the penalty, the restriction and plain beam search all score 81.47,
p = 1.0 — **the POPE gain is the beam search**.

### SID — **no reference implementation; ours is from the paper**

No CLI flag in the released repo. Shares the `use_cd` code path, and a
leftover author log path points at `/home/kdzlys/sid2/`, suggesting it was run
from a SID-derived checkout not published here. So there is no oracle to
assert bit-equality against, which is why V3 exists: it checks that our
selection does work and points the direction SID's paper claims.

### CausalMM — **no reference implementation; ours is from the paper**

Appears nowhere in the released repository. Implemented from its own paper and
checked by V4, which found the vision counterfactual inert (a permutation
cannot move a mean) and the image-blind variant **better** than the method as
specified — `shuffle` 76.67 vs `cf_blind` 77.53, p = 1.6e-4.

---

## Out of reach

| | why |
|---|---|
| Table 10, recent backbones (Qwen3-VL-8B, InternVL3-8B) | new models, new ports |
| Table 11, MMBench and MathVista | datasets not set up |
| Figures 5 and 18, MME | the release ships **no MME script** despite reporting MME results |
| Figure 6 and Table 5, GPT-4V judge | paid API, n=50, and the judge model is deprecated — not reproducible by anyone |
| Figures 16, 17, 21, 22 | InstructBLIP and Shikra |
| Figures 10, 11, 20 | qualitative examples, nothing to reproduce |

---

## What this adds up to

Reachable and not yet run, in descending value:

| item | why it matters | rough cost |
|---|---|---|
| ~~VCD + ICD, both benchmarks~~ | **done on POPE-adversarial and CHAIR**, from the formula, since the released code's flags are dead. The popular and random splits for VCD/ICD/SID/CausalMM are in r17 | ~~60 h~~ 22 h, in r17 |
| ~~Figure 8, random-head control~~ | **done, r6 + r14.** The gap is 0.26 pts at k=0.10 (indistinguishable from zero), 1.37 at k=0.20, 0.79 at k=0.45 — and random heads at k=0.45 beat diagnosed heads at k=0.20. The effect is bought by *how many* heads are edited | ~~22 h~~ spent |
| ~~Table 14 + Figure 19, fusion rule~~ | **done, r15.** The gate does nothing on POPE (`dynamic` is +0.02 over baseline, `fixed` ties `hybrid`) and the CHAIR ordering inverts — `fixed`, which the paper ranks last, is first by 16 points | ~~14.6 h~~ spent |
| ~~Table 15, L2–13 vs L2–15~~ | **done** — equivalent at ±0.42, and r14/V8 explains why: both are early. L11–24 and L13–26, the windows r13 measured as strongest, are 3.07 and 2.24 points **worse**, and no better than not intervening at all. r15/V10 sweeps position at matched width | done; V10 in r15 |
| Tables 4/12, query positions | measures the cost of the released `tail_n = 1`; needs cached query vectors across decode steps, which the KV cache does not keep | ~12 h |
| ~~Table 7, α = 0 and 3~~ | **done** — the shape reproduces then saturates: 79.10 / 81.83 / 82.37 / 82.37 / 82.37 for α = 0,1,2,3,4 | ~~5 h~~ spent |
| ~~Figure 9, β axis~~ | **done** — β is *identical* at 0.05 and 0.2, and r15/V9 says why τ_JS is flat too: its gate almost never fires on POPE | spent |
| Layer cliff: where the live region ends, and whether layer identity matters | not a paper artefact — a question r15 and r16 opened and nobody has answered | in r17 |
| ~~Table 13, Figure 3, Figure 7~~ | **done, r13** — 0.71 s/sample, 35 min for 3000 | ~~~1 h~~ |
