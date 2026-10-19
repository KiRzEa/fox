# foxrepro — reproduction and mechanism audit of Fox (ICML 2026)

Reproduces **"Dismantling Pathological Shortcuts: A Causal Framework for
Faithful LVLM Decoding"** (Liu Yu et al., ICML 2026, PMLR v306) and audits what
its intervention actually does.

Target host: **Kaggle Notebooks, 2× Tesla T4 16 GB**.

---

## What the paper claims

Large Vision-Language Models hallucinate objects that are not in the image. The
prevailing explanation — the paper calls it the *attention intensity
assumption* — is that the model allocates too little attention to image tokens,
so prior methods amplify the visual signal (PAI) or subtract the language prior
(VCD).

The paper argues this is wrong: hallucinated outputs carry the *same* total
visual attention mass as faithful ones. It proposes instead a **dynamic
structural misalignment**, where specific attention heads ("risky mediators")
at "decision-critical steps" decouple from visual evidence and lock onto a
language prior anchored in the system prompt, forming a shortcut
`X_sys → H_R → Y_t` that bypasses visual grounding.

**Fox** is the proposed fix, in three stages:

1. **Diagnose.** At the last query position, score each head
   `S = m_sys · H_vis`, where `m_sys` is attention mass on the system prompt and
   `H_vis` is the entropy of attention renormalised over image tokens only.
2. **Intervene.** Subtract a constant `γ` from the pre-softmax attention logits
   of the top-k highest-scoring heads, in a window of middle layers.
3. **Fuse.** Run two forward passes (with and without the intervention), measure
   their Jensen-Shannon divergence `d_t`, and combine
   `z_final = z_obs + λ_t · z_do` with `λ_t = α` on consensus, `d_t` on conflict.

Reported on LLaVA-1.5: CHAIR_S 57.60 → 46.40, CHAIR_I 17.18 → 12.90,
POPE-Adversarial 76.77 → 81.93.

## What this repo found

Reading the released code and transcribing it (`foxrepro/fox/reference.py`):

**The intervention is a uniform shift across *all* key positions at *one* query
position.** Softmax is shift-invariant, and the softmax that follows is
numerically stable and accumulates in fp32 (true in every HuggingFace version
checked: 4.29.2, 4.37.2, 4.57.6, 5.17.0). So the shift cancels exactly. The only
surviving effect is the cast back to the storage dtype, which snaps the logit
row onto that dtype's ULP grid — 8.0 for fp16 at `|γ| = 10⁴`, against a real
logit spread of O(1)–O(30).

So the mechanism is a **floating-point rounding artifact**, not a causal
intervention. The effect may well be real and the numbers may well reproduce;
what does not survive is the explanation.

Supporting findings, each pinned by a test:

| Finding | Where |
|---|---|
| `γ = 10000.0` is a hardcoded local literal, never stated in the paper, and omitted from its own list of "four key hyperparameters" | `reference.py` |
| The code runs layers **2–15**, k=**0.35**, τ_JS=**0.15**; the paper says 2–13, 0.45, 0.2; a leftover author log path in a comment says 2–15 and 0.1 | `reference.py` docstring |
| A second copy of the intervention (used by InstructBLIP) **drops `m_sys`**, so Eq. 4 of the paper is not what runs there | `reference_intervene_instructblip` |
| Greedy decoding argmaxes the **unmasked** fused logits while sampling uses the β-masked ones, so Table 9 changes the decision rule, not just the sampler | `test_greedy_and_sampling_rules_differ` |
| "Truncated JSD" (Eq. 20) is not implemented; the JSD is computed over the full 32064-token vocabulary, in numpy, on the CPU, once per generated token | `reference_jsd` |
| Appendix C.2 states a two-sided t-test over "10 independent runs" but no table reports any variance | — |
| The released prompt concatenates the system message with a bare `+` (`llava.py:228`), so it reads `...user's questions.USER:` — no separator. The system message itself does not come from `conv_templates`, which the eval scripts never import, but from `eval_configs/llava-1.5_eval.yaml`. It is what `m_sys` measures attention over: drop it and only `<s>`/`USER`/`:` precede the image — image span starting at index **5** instead of ~40 — so the shortcut `X_sys → H_R → Y_t` loses its premise. `--system-prompt none` tests that claim directly | `TestSystemPrompt` |
| Section 4.2 blames the softmax's finite precision, but `torch.softmax` subtracts the row max before exponentiating, so the shift cancels before any accumulation. Storing the shifted logits in fp32 (ULP ~1e-3) barely moves the output while fp16 (ULP 8.0) changes it **~800× more** — in both softmax dtypes. The mechanism is the logit cast, nothing else | `test_the_mechanism_is_the_logit_cast_in_every_softmax_dtype` |
| At the paper's own sample sizes, several reported gaps are inside sampling noise: POPE n=3000 → ±1.37 points, CHAIR_S n=500 → ±4.35 points | `bench/stats.py` |

The last one reframes the project: the expected honest result is that Fox beats
the weak baselines significantly, while its advantage over OPERA and CausalMM is
not statistically resolvable at n=500.

Every difference between this implementation and the released code is
catalogued in [docs/DEVIATIONS.md](docs/DEVIATIONS.md), classified by whether
it can change a number: proven bit-identical, verified-equivalent
reimplementation, deviation that can move a result, present only there,
present only here, and still unverified.

## Reproduction result (session 1, round r2)

Measured on Kaggle 2x T4, LLaVA-1.5-7B fp16, seed 42, prompt byte-faithful to
the release. POPE-adversarial n=3000, CHAIR n=500 images.

**Absolute values against the paper's printed numbers:**

| artefact | baseline (ours / paper) | Fox (ours / paper) |
|---|---|---|
| POPE-adv accuracy, sampling | 77.17 / 76.77 | 81.40 / 81.93 |
| CHAIR_S, sampling | 56.80 / 57.60 | 49.60 / 46.40 |
| CHAIR_I, sampling | 18.14 / 17.18 | 14.08 / 12.90 |
| CHAIR_S, greedy | 49.20 / 50.90 | 46.20 / 42.60 |
| CHAIR_I, greedy | 13.82 / 15.40 | 13.45 / 12.16 |

Every baseline lands within 1.7 points, so the harness, the data and the
decoding protocol are right. The method's own cells are within 0.53 on POPE
and 1.2–3.6 too high on CHAIR.

**What the paper actually claims is the gap, so that is what has to reproduce:**

| artefact | our gap [95% CI] | paper's gap | verdict |
|---|---|---|---|
| POPE-adv accuracy, sampling | **+4.23** [+2.92, +5.47] | +5.16 | reproduces — CI covers it |
| CHAIR_S, sampling | **−7.20** [−12.00, −2.40] | −11.20 | reproduces — CI covers it |
| CHAIR_I, sampling | **−4.07** [−5.79, −2.32] | −4.28 | reproduces — CI covers it |
| CHAIR_S, greedy | **−3.00** [−7.40, +1.40] | −8.30 | **fails** — gap not resolvable, CI excludes −8.30 |
| CHAIR_I, greedy | **−0.37** [−1.74, +0.99] | −3.24 | **fails** — gap not resolvable, CI excludes −3.24 |

POPE and CHAIR intervals are exact McNemar and paired bootstrap respectively,
both on the same samples.

**Under sampling, Fox reproduces on all three metrics. Under greedy decoding,
its advantage disappears.** The second row of that pair is the sharp one:
CHAIR_I under greedy has n=3896 mentions and resolves 2.24 points, so
−0.37 [−1.74, +0.99] is a real null and not a power failure — it excludes the
paper's −3.24. CHAIR_S at n=500 resolves only 8.85 points, so its null is
weaker evidence.

This is exactly what the mechanism analysis predicts. `reference_fuse`
combines **unnormalised** logits with `alpha = 2`, so the fused vector carries
roughly three times the scale of the original. Under nucleus sampling that
sharpens the distribution — an implicit temperature drop. Under argmax it
changes nothing, because scaling a vector does not move its maximum. The
greedy control was built to separate "Fox grounds the model better" from "Fox
samples less randomly", and it comes out on the second side.

Note also that greedy decoding *alone* recovers most of what Fox claims to
fix: the baseline improves from CHAIR_S 56.80 to 49.20 and CHAIR_I 18.14 to
13.82 just by dropping the sampler, which is more than Fox buys under
sampling.

Still to run: POPE under greedy (Table 9a), the other two POPE splits, and the
mechanism arms (`off`, `ulp_control`, `noise_control`, logit-dtype, the gamma
sweep, `boost_visual`). Both rounds of session 1 are in
[results/runs/](results/runs/) — `r1` ran before the released prompt was read
and sits 5.13 points above the paper's POPE baseline; quote `r2`.

## The design that makes this safe

The paper's mechanism is a *numerical* effect, so a reimplementation is
dangerous in a specific way: if our numbers differed, we could not tell "the
paper is wrong" from "our port is wrong".

That is avoidable because **the method has no learned parameters** — it is
arithmetic on tensors. So:

- `foxrepro/fox/reference.py` is a **verbatim transcription** of the released
  code, lifted out of its 3370-file vendored-transformers context. It is the
  numerical oracle and is deliberately not cleaned up.
- `foxrepro/fox/kernel.py` is the clean, parameterised port.
- `tests/test_kernel_equivalence.py` asserts they are **bit-identical** on
  random tensors across fp32 / fp16 / bf16, at prefill (Q=600) and decode
  (Q=1), batched and unbatched — **with no GPU, no checkpoint and no COCO
  download**.

Once that passes, every other experiment arm is a deliberate deviation measured
against a trusted baseline.

## Three configurations people conflate

| invocation | branches | fusion | intervention | what it is |
|---|---|---|---|---|
| `--method sampling` | 1 | no | no | the baseline every table is measured against |
| `--method fox --mode off` | 2 | **yes** | no | isolates the additive fusion `z_obs + 2·z_do`, whose ~3× scale sharpens sampling on its own. **The control the paper never ran** |
| `--method fox --mode as_implemented` | 2 | yes | yes | bit-equal to the released code |

All of them share one sampling implementation (`fox.attention.fox_generate`), so the only difference between the baseline and any arm is the intervention. Two different top-p implementations would be an unfair baseline, which is the commonest silent error in this literature.

## The five arms

`--mode` selects one. They are flags, not code paths, so a run file's
provenance fully determines what was computed.

| mode | what it tests |
|---|---|
| `as_implemented` | the released behaviour, bit-equal to the original |
| `off` | fusion only, intervention disabled — **the control the paper never ran**, separating the additive fusion `z_obs + 2·z_do` from the intervention |
| `as_described` | bias restricted to image-token keys, as Appendix B describes rather than as the code does. If this works alone, the finding is "code ≠ paper", not "paper is wrong" |
| `ulp_control` | explicit `quantize_to_grid` instead of shift-and-cast. If this reproduces the gain, the mechanism is requantization, full stop |
| `noise_control` | matched-magnitude uniform noise, no grid structure. If this *also* reproduces the gain, the method is noise injection and the head selection is decorative |

Plus the **γ sweep**, which is the decisive experiment and runs entirely in
fp16 — which is why a T4 (sm_75, no bf16 *compute*) suffices:

```
γ=1e2 → fp16 ULP 0.0625      γ=1e4 → ULP 8.0   (the released value)
γ=1e3 → fp16 ULP 0.5         γ=6e4 → ULP 32.0
γ=1e5 → overflows fp16 (max 65504), excluded
```

A monotone response in **ULP** rather than in γ magnitude is the signature of
the requantization account.

---

## Quickstart (Kaggle)

Sidebar → **Accelerator = GPU T4 x2**. One T4 cannot hold LLaVA-1.5-7B in fp16
with headroom (13.16 GiB of weights against ~14.3 GiB usable); two can, sharded.
Quantization is ruled out — it would change the quantity under study.

```bash
!git clone -q https://github.com/<you>/<repo>.git /kaggle/working/fox
%cd /kaggle/working/fox
!pip install -q -e .
!python scripts/probe_host.py        # BEFORE downloading 14 GB
!bash kaggle_setup.sh --track-b
!python -m unittest discover -s tests -t .
```

Then the data, which for POPE is images only — its labels ship in the package:

```bash
!python -m foxrepro.storage --ensure
!python -m foxrepro.data.fetch_images --what pope --workers 8   # ~100 MB
!python -m foxrepro.bench.generate --method sampling --benchmark pope \
        --split random --model-id llava-hf/llava-1.5-7b-hf \
        --dtype fp16 --device-map auto --seed 42 --limit 20
```

COCO serves images individually, so the documented 6.2 GB val2014 zip is
avoidable: this fetches exactly the 500 images POPE references. It is
resumable, idempotent, and detects truncated files — a dropped connection
leaves a JPEG with a valid header that PIL will happily open and return a
partial image for, which would corrupt a run undetectably.

CHAIR additionally needs the ground-truth annotations:
`--what union --with-annotations` (~241 MB zip, the only way those two json
files are distributed).

`scripts/kaggle_notebook.py` holds the full set of copy-paste cells, in order,
through to the report. Run it as a script to print them.

### Off Kaggle

```bash
pip install -e .                                  # or: PYTHONPATH=src
pip install -r requirements-trackB.lock           # GPU work
pip install -r requirements-cpu.lock              # metrics only, no torch
python -m unittest discover -s tests -t .
```

The CPU side (`bench/evaluate`, `bench/stats`, `bench/chair`, `bench/report`)
runs on a laptop with no GPU and no torch; the bit-equivalence module skips
itself when torch is absent.

---

## Layout

```
pyproject.toml              src-layout package; declares NO torch dependency
requirements-cpu.lock       metrics only, laptop-portable
requirements-trackA.lock    fidelity env: py3.10 + torch 2.1.2 + vendored 4.29.2
requirements-trackB.lock    primary env: py3.11 + torch 2.6 + transformers 4.57.6
kaggle_setup.sh             idempotent 7-step session bootstrap
scripts/probe_host.py       host feasibility probe; run before anything
scripts/kaggle_notebook.py  copy-paste notebook cells
src/foxrepro/
  storage.py                the ONLY module with absolute paths
  fox/reference.py          verbatim transcription of the released code (oracle)
  fox/kernel.py             the clean port; proven bit-equal
  data/loaders.py           POPE / CHAIR iterators, stable sample_ids
  data/build_coco_subset.py one-time ≤1000-image tar builder
  bench/generate.py         GPU driver: method registry, resume, flushed JSONL
  bench/evaluate.py         CPU metrics, zero GPU
  bench/chair.py            Rohrbach-style CHAIR scorer
  bench/stats.py            Wilson, Newcombe, paired bootstrap, McNemar, power
  bench/report.py           paper-shaped tables with confidence intervals
  assets/                   CHAIR synonyms + POPE annotations, shipped
tests/                      174 tests
```

### Two tracks, and why they cannot be one environment

**Track B** (primary) is modern: transformers 4.57.6, torch 2.6, Python 3.11,
`llava-hf/llava-1.5-7b-hf`. Installs natively on Kaggle. The intervention is
registered as a custom attention function via `AttentionInterface`, dispatching
on `module.layer_idx` so the intervened layers take an eager path with
pre-softmax write access while the rest keep SDPA.

**Track A** (fidelity oracle) runs the released repo's own vendored
transformers 4.29.2. It exists only to cross-check a few dozen samples
token-for-token. It pins `huggingface_hub==0.16.4`, and `gradio>=5` requires
`>=0.33.5`, so the two tracks cannot co-install — keep separate venvs.

Thanks to the bit-equivalence proof, Track A shrinks from "a second full
pipeline" to "one ~50-sample oracle run".

---

## Reproducing across sessions

The paper prints 15 tables and 22 figures and this hardware cannot reach all of
them; `foxrepro/bench/plan.py` records which, why, and what to run next. **60
jobs, ~96 GPU-hours** — about three weeks of Kaggle's 30/week quota, split into
**11 sessions** of under 10 h each (the single-session wall).

Freeze the split once, locally, and commit it:

```bash
python -m foxrepro.bench.plan plan --sessions 11 --pilot   # -> scripts/run_plan.json
```

`--pilot` pins session 1 rather than cost-balancing it, because
cost-balancing would give it a baseline on one split and the method on another,
leaving nothing comparable after ten GPU hours.

**Session 1 (7.3 h) answers one question: do the paper's numbers reproduce?**
Baseline and method on each of the three main tables — POPE-adversarial
(Table 1, the largest claimed POPE gap; the other splits are inside sampling
noise), CHAIR (Table 2, where the headline "29.1% over SID" comes from), and
CHAIR greedy (Table 9, deterministic and therefore the sharpest signal at one
seed). No mechanism arms: those only become worth running once there is a
result to explain. VCD is excluded because what this repo can run is VCD's
noised branch inside Fox's fusion, not VCD proper.

Then one flag per Kaggle session — they are disjoint, so they can run at the
same time:

```bash
bash run_kaggle_fox.sh 3            # session 3
DRY=1 bash run_kaggle_fox.sh 3      # print the commands only
BUDGET=10 bash run_kaggle_fox.sh 3  # hours, default from the frozen plan
```

Each session re-packs to `/kaggle/working/fox_runs_fox_r1_s<N>.zip` after every
finished job. Download those zips into `runs_inbox/`, then one command merges
them all:

```bash
python -m foxrepro.bench.plan ingest          # reads runs_inbox/, expands the zips
python -m foxrepro.bench.evaluate      # every run file in the runs dir
python -m foxrepro.bench.report        # defaults to the faithful prompt arm
```

`ingest` is idempotent, so re-ingesting a zip you already merged adds nothing,
and it says so rather than returning a silent row of zeros.

Why it is shaped this way, mostly learned from the ACSA queue in a sibling
project:

- **The split is frozen to a file**, not recomputed per session. Otherwise
  editing the plan source silently reassigns work and sessions overlap or skip.
- **Jobs are cost-balanced** by longest-processing-time-first. Round-robin over
  60 jobs costing 0.2–2.3 h leaves one session with twice the work.
- **Reassignment is still safe.** Resume is keyed on the run file, which is
  derived from the configuration and not the session, so a job half-done by
  session 2 and later moved to session 3 is continued rather than restarted.
- **Cheap jobs run first within a session**, so a session cut early leaves more
  completed jobs behind.
- **A job that no longer fits the remaining budget is deferred, not started** —
  beginning a 2.3 h job with 40 minutes left burns the GPU for nothing.
- **Results are packed after every finished job**, so a cut session loses at
  most the job in flight.
- **A watchdog stops 15 minutes inside the budget.** A notebook killed at the
  session wall can lose its output entirely, discarding every completed job. The
  deadline is derived from `--budget-hours`, so it fires in a 10 h session; a
  fixed 11.3 h deadline never would.
- **`--calibrate` feeds measured times back.** `run` appends
  `job_id<TAB>minutes<TAB>rc` to `_job_times.tsv`; passing it to `plan`
  replaces the estimates, which are scaled from one measurement and will drift.

`ingest` merges partial files for the same job by `sample_id` union rather than
letting one overwrite the other, and appends raw lines rather than re-validating
them — the files it collects may come from sessions running an older schema,
which is exactly the case it exists to handle.

**Out of reach here, stated rather than left as a hole:** the InstructBLIP
column (it runs the repo's *second* intervention copy, which drops `m_sys` and
uses layers 4–10 — a different method needing its own implementation); the
Shikra column (no maintained HF conversion, and the released code finds its
visual span by the magic numbers `img_s==33 and img_e==290`); MME (dataset
absent, and the Fox release ships no MME script despite reporting MME results);
the GPT-4V judge (paid API); MMBench and MathVista; and the ICD / SID / OPERA /
CausalMM baselines, each of which needs its upstream repo. Roughly two thirds of
the paper's numbers are in reach, including every mechanism ablation — which is
where this reproduction's own contribution is.

## Runbook notes

**Generation and evaluation are separate programs, deliberately.**
`generate.py` does the GPU work and writes one flushed JSONL line per sample.
`evaluate.py` reads JSONL and computes metrics with no GPU. So a disconnect at
sample 317 resumes at 318, and fixing a metric bug never costs a GPU hour.

**Resume:** re-run the identical `generate.py` command. The run file is keyed on
the full configuration; completed `sample_id`s are skipped. Tested by actually
crashing the process — `os._exit`, truncated JSON lines, four successive
crashes, and a real `SIGTERM`.

**Every arm gets its own run file.** The resume key excludes `γ`, `logit_dtype`,
`mode` and `decoding` — all of which this project sweeps — so those are folded
into a variant tag in the filename. Without that, the second arm of a sweep
would read the first's records, consider them done, and silently emit a file
mixing two configurations. `tests/test_experiment_axes.py` pins this.

**Three dtypes, not one.** `--dtype` is weights/activations; `--logit-dtype` is
the pre-softmax attention logit storage — *the mechanism*; `--softmax-dtype` is
the accumulation, which HuggingFace hard-upcasts to fp32 and is why the shift
cancels. Varying `--dtype` alone moves two of them together and yields a null
result that is correct but uninformative.

**TF32 will contaminate the fp32 arm.** On Ampere+ a TF32 matmul has ~10
mantissa bits — roughly fp16 precision in the QK product.
`torch.backends.cudnn.allow_tf32` defaults to **True**.
`generate.py` calls `pin_precision_flags()` before loading any model and records
the resulting flags in every JSONL record.

**The checkpoint must be the HF-converted repo.** `llava-hf/llava-1.5-7b-hf`,
not `liuhaotian/llava-v1.5-7b`. The latter is weight-identical but stores the
language model under `model.layers.*` where transformers expects
`model.language_model.layers.*`, and ships no `preprocessor_config.json`.
`from_pretrained` does not treat that as an error: it reports the vision tower
and projector as MISSING and **initialises them randomly**, then carries on.
A model in that state writes fluent captions that have nothing to do with the
image, and every metric downstream reports plausible numbers computed from
noise. `load_llava` refuses the repo by name and, more generally, refuses any
load that leaves parameters uninitialised.

**The data step must not disagree with the run step.** `run_kaggle_fox.sh`
fetches with `--what auto --session N`, which reads the frozen plan and pulls
exactly the image sets that session's jobs need. The first real session
hardcoded `--what pope` while session 1 also contained CHAIR jobs, and all
four died within forty seconds on a missing image list. The CHAIR list now
ships in `assets/` for the same reason: deriving it needs
`instances_val2014.json`, distributed only inside a 241 MB zip.

**Check `transformers_path` in the output.** On Track A, if stock HuggingFace is
on `sys.path` instead of the vendored tree, the intervention silently does
nothing while the run reports success. Every record carries the resolved path so
that failure is detectable after the fact.

---

## Status

| | |
|---|---|
| Harness, metrics, statistics, resume | done |
| `fox/reference.py` + `fox/kernel.py`, bit-equivalence proven | done |
| POPE annotations (3000 × 3 splits, 500 images, 50/50 balanced) | shipped in `assets/` |
| Kaggle bootstrap, host probe, notebook cells | done; host measured (2x T4, py3.13, torch 2.11, transformers 5.16.1) |
| COCO images | fetched directly per-id (~100 MB for POPE) — no 6.2 GB zip |
| Reproduction plan | 60 jobs, ~96 GPU-h, 11 sessions, selected by one flag |
| Total | **335 tests** |
| Deviations from the release | [docs/DEVIATIONS.md](docs/DEVIATIONS.md) — every difference, classified by whether it can change a number, green on macOS/arm64 + torch 2.8 + transformers 4.57.6 **and** Kaggle/x86_64 + torch 2.11 + transformers 5.16.1 |
| `fox/attention.py` — `AttentionInterface` wiring + dual-branch decode loop | done, 18 tests |
| `sampling`, `fox`, `vcd` | done, 21 tests against a fake model |
| `sid` | needs the upstream SID repo; the Fox release ships only a FastV skeleton |
| `opera` | beam search with rollback; its reference is its own transformers fork |
| CHAIR synonym list | transcribed, **diff against Rohrbach's upstream before quoting CHAIR numbers externally** |
| Gradio comparison demo | after the methods |
