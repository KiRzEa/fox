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

Our LLaVA-1.5 rows for `sampling` and `fox` are done at 10 seeds
(adversarial) and 3 seeds (random, popular). The other five method rows are
not: see **Compared methods** below.

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

### Table 7 — sensitivity of α — **CHEAP**, 2 of 4 values done

Paper: α ∈ {0, 1, 2, 3} → mean POPE accuracy 81.75 / 85.33 / 85.97 / 85.98.
We have α ∈ {1, 2, 4}. Missing α=0 and α=3. "Mean Acc" is the average over the
three POPE splits, so completing this needs all three splits per α.

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

### Table 14 — fusion-strategy ablation — **RUNNABLE, high value**

| strategy | CHAIR_S | CHAIR_I | F1 |
|---|---|---|---|
| always fixed (λ = α) | 49.20 | 13.74 | 74.91 |
| always dynamic (λ = d_t) | 47.68 | 13.21 | 75.83 |
| hybrid, Eq. 7 | **46.40** | **12.90** | **76.57** |

Two one-line changes to `reference_fuse`. High value because our `off` arm
already shows the fusion alone carries +2.21 of Fox's +5.03 on POPE, and this
isolates the gating rule inside it.

### Table 15 — manual vs automatic layer window — **RUNNABLE**

States the LLaVA-1.5 manual window as **L2–13** with POPE-adversarial 81.93,
and an automatic rule selecting L3–15 for 81.27.

This contradicts the released code, which runs **L2–15**. So the paper's own
appendix says the headline number came from 2–13 while the code ships 2–15 —
and 2–15 is what the automatic rule is said to select, scoring 0.66 lower.
Running both windows settles which produced 81.93.

### Table 13 — prior-source diagnostic — **CHEAP, no generation**

Claims mean diagnostic score m_vis ≈ 0.57, m_txt ≈ 0.62, m_sys ≈ 0.82, and a
tail ROC-AUC of **1.00**.

Pure measurement over attention statistics — no decoding, so it costs one
forward pass per sample. A reported AUC of exactly 1.00 (perfect separation)
is the kind of claim worth checking directly.

### Figure 3 — ROC of the joint risk score, AUC = 0.818 — **CHEAP**

Same shape as Table 13: attention statistics only.

### Figure 7 — layer-wise diagnostic strength of H_vis — **CHEAP**

Per-layer |AUC − 0.5|. This is also the input to Table 15's automatic rule, so
both come from one measurement pass.

### Figure 8 — diagnosis-driven vs random head selection — **RUNNABLE, high value**

The paper's own validation that the diagnosis does something: Fox against
random head selection at matched budget K. This is the claim our finding most
directly challenges — if the effect is requantisation, random heads at the
same budget should do as well, and the paper says they do not.

### Figure 9 — k, τ_JS, β sensitivity — **PARTIAL**

We have k ∈ {0.25, 0.45} and τ_JS ∈ {0.1, 0.2} at n=500. β is untouched.

### Figure 19 — the role of JSD gating — **RUNNABLE**

Overlaps Table 14; the "always apply the intervention" arm is the
`always fixed` row.

---

## Compared methods

The paper compares against five: ICD, VCD, OPERA, SID, CausalMM.

### VCD and ICD — **RUNNABLE, and the released code cannot do it**

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
image inside Fox's additive fusion, labelled `vcd-noise-fox-fusion`, and must
be renamed once real VCD exists beside it.

### OPERA — **RUNNABLE but large**

Beam search with an over-trust penalty and rollback. Needs its own decode
loop, and the paper evaluates it under `num_beams=5`, outside the shared
sampling protocol. A separate project-sized piece of work.

### SID — **BLOCKED**

No CLI flag. Shares the `use_cd` code path, and a leftover author log path
points at `/home/kdzlys/sid2/`, suggesting it was run from a SID-derived
checkout not published here.

### CausalMM — **BLOCKED**

Appears nowhere in the released repository.

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
| VCD + ICD, both benchmarks | the paper's own comparison, and its code cannot produce it | ~60 h |
| Figure 8, random-head control | directly tests whether the diagnosis does anything | ~10 h |
| Table 14 + Figure 19, fusion rule | isolates the gate inside the component carrying 44% of the effect | ~6 h |
| Table 15, L2–13 vs L2–15 | settles which window produced the paper's 81.93 | ~3 h |
| Tables 4/12, query positions | measures the cost of the released `tail_n = 1` | ~12 h |
| Table 7, α = 0 and 3 | completes the sweep; α=0 is a sanity check on our fusion | ~5 h |
| Table 13, Figure 3, Figure 7 | diagnostics, no decoding — one forward pass per sample | ~1 h |
