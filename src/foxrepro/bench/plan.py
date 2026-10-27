"""The reproduction target matrix, what is done, and what to run next.

The paper prints 15 tables and 22 figures. This module encodes which of those
this hardware can actually reproduce, tracks progress by scanning the run
files on disk, and emits the next batch of commands that fits a stated GPU-hour
budget. It exists because the work spans many Kaggle sessions and nothing else
remembers where it got to.

Honest feasibility, on 2x Tesla T4 16 GB with a 30 GPU-h weekly quota
---------------------------------------------------------------------
REPRODUCIBLE, and planned here:

=====================================  ======================================
Table 1 / 8 (POPE accuracy and F1)     LLaVA-1.5 column; same runs serve both
Table 2 (CHAIR)                        LLaVA-1.5 column
Table 9 (greedy control)               same runs with --decoding greedy
Table 6 (latency)                      nearly free; already measured per record
Table 7 (alpha ablation)               POPE sweep over alpha
Table 3 (recent backbones)             Qwen3-VL-8B and InternVL3-8B are both
                                       ~16 GB fp16 and HF-native, so they fit
                                       sharded. Needs a loader generalisation.
Figures 3, 7, 12, 13, 14               diagnostic only -- one forward pass per
                                       sample, no generation, so cheap
Figure 1a                              same data as Figure 12
Figure 8 (random vs diagnosed heads)   needs a random-head selection mode
Figure 9 (k / tau_JS / beta)           500 COCO samples, as the paper states
Figure 19 (JSD gating)                 mode=off vs always-intervene
Tables 4, 13, 14, 15                   temporal anchors, prior source, fusion
                                       rule, auto layer window
=====================================  ======================================

NOT REPRODUCIBLE here, and why -- stated so the report can say so plainly
rather than leaving a hole:

=====================================  ======================================
Table 1/2/8/9, InstructBLIP column     Possible (Vicuna-7B class) but it runs
                                       the repo's SECOND intervention copy,
                                       which drops m_sys and uses layers 4-10
                                       with k=0.4. A different method, so it
                                       needs its own implementation.
Table 1/2/8/9, Shikra column           Shikra's checkpoint has no maintained
                                       HF conversion, and the released code
                                       identifies its visual span by the magic
                                       numbers img_s==33 and img_e==290.
Figure 5 (MME)                         Needs the MME dataset; the Fox release
                                       ships no MME script at all despite
                                       reporting MME results.
Figure 6 (GPT-4V judge)                Needs a paid OpenAI API key. The
                                       protocol is in Appendix C.3 if funded.
MMBench / MathVista (appendix)         Separate datasets, not yet obtained.
Figures 16, 17                          Sensitivity on InstructBLIP / Shikra.
ICD, SID, OPERA, CausalMM baselines    Each needs its upstream repo. The Fox
                                       release has only a FastV skeleton for
                                       SID, argparse flags for OPERA, and runs
                                       "VCD" through its own fusion.
=====================================  ======================================

So roughly two thirds of the paper's numbers are in reach, the LLaVA-1.5
column of every main table among them, plus every mechanism ablation -- which
is where this reproduction's own contribution lies.

Cost model
----------
Measured on the host, not guessed: 1.315 s per POPE sample for a single-branch
method (608-token prefill, one generated token, sharded across 2x T4). POPE is
prefill-bound, so a dual-branch method runs at batch 2 and roughly doubles it.
CHAIR is decode-bound at ~100 generated tokens per caption. Diagnostic-only
passes generate nothing and cost about one prefill.

Re-measure with ``--calibrate`` once real runs exist; the estimates below are
scaled from that single number and will drift.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from foxrepro.storage import BENCHMARK_SPLITS, CHAIR_DRAWS, RunKey, paths
from foxrepro.data.loaders import DEFAULT_TEMPLATE_IDS

__all__ = [
    "Job", "PLAN", "measured_cost", "scan_progress", "next_jobs", "render_plan",
    "partition_jobs", "partition_by_priority", "pair_key",
    "ingest_runs", "archive_runs", "DUAL_BRANCH", "SPLIT_BRANCH",
    "BEAM_SEARCH",
]

#: Seconds per sample on 2x T4 with LLaVA-1.5-7B fp16 sharded. Measured, not
#: scaled: these are session 1 round r2's wall times divided by the sample
#: count (logs/_job_times_r2.tsv). The earlier figures were extrapolated from
#: one short probe and ran 2.5x high on POPE, which inflated the whole plan.
#:
#: POPE is a 625-token prefill and 1 token out, so it is prefill-bound and the
#: second branch roughly doubles it. CHAIR decodes ~120 tokens, and there the
#: two branches batch, so dual costs well under 2x.
#: ``split`` is the two-sequential-passes shape: ICD's negative branch has its
#: own token sequence, so the rows cannot share a batch and the weights are
#: read twice per token. Costed at 2x single on CHAIR rather than the batched
#: dual's 1.33x. On POPE, which is prefill-bound, batching already buys little
#: and the two shapes land in the same place.
COST_S: Dict[Tuple[str, str], float] = {
    ("pope", "single"): 0.55,     # 27.3 min / 3000
    ("pope", "dual"): 1.10,       # 55.1 min / 3000
    ("pope", "split"): 1.10,      # prefill-bound; batching saves nothing here
    ("chair", "single"): 8.2,     # 68.4 min / 500
    ("chair", "dual"): 10.9,      # 90.6 min / 500
    ("chair", "split"): 16.4,     # estimated: 2x single, no batching to share
    ("diag", "single"): 0.55,     # one prefill, no generation
    # No ("*", "beam") entry on purpose: OPERA is the one shape whose cost
    # scales with the generation length, so it is not a flat per-sample
    # number -- see BEAM_COST_S and Job.est_seconds.
}

#: OPERA's cost, split because it is the only arm where ``max_new_tokens``
#: changes the bill: one prefill at ``num_beams`` rows, then
#: ``max_new_tokens x (1 + num_attn_candidates)`` decode forwards, also at
#: ``num_beams`` rows.
#:
#: Estimated, not measured. A 625-token prefill at batch 5 is ~4.4e13 FLOPs,
#: ~1.1 s at the ~40 TFLOP/s these two T4s actually reach; a decode forward
#: is bandwidth-bound at ~13 GB of weights, ~0.04 s, and there are six per
#: step. Replace both with measurements after the first session --
#: ``logs/_job_times_*.tsv`` is where every other number here came from.
BEAM_COST_S: Dict[str, float] = {
    "prefill": 1.1,
    "forward": 0.04,
}

N_SAMPLES: Dict[Tuple[str, str], int] = {
    ("pope", "random"): 3000,
    ("pope", "popular"): 3000,
    ("pope", "adversarial"): 3000,
    ("chair", "val2014_500"): 500,
    **{("chair", d): 500 for d in CHAIR_DRAWS},
}

#: Methods that run two forward branches per token as rows of one batch.
#: ``mode="off"`` does NOT take a method out of this set: it disables the head
#: edit and keeps the second branch, which is the point of that control.
DUAL_BRANCH = frozenset({"fox", "vcd", "sid", "causalmm", "vcd_fox_fusion"})

#: Methods whose two branches cannot be batched, so the weights are read twice
#: per token. ICD disturbs the instruction, so the sequences differ in length.
SPLIT_BRANCH = frozenset({"icd"})

#: Methods that run a beam search: ``num_beams`` rows plus
#: ``num_attn_candidates`` speculative forwards per step. A different cost
#: shape from either branch count, and the only one whose cost scales with
#: ``max_new_tokens``.
BEAM_SEARCH = frozenset({"opera"})

#: ``--num_attn_candidates``: how many speculative forwards each step pays
#: for, on top of the main one.
OPERA_N_CANDIDATES = 5


@dataclass(frozen=True)
class Job:
    """One generate invocation, and which paper artefact it feeds."""

    target: str                      # e.g. "table1", "fig9"
    method: str
    benchmark: str
    split: str
    decoding: str = "sampling"
    mode: str = "as_implemented"
    system_prompt_id: str = "llava_v1"
    wrapper_id: str = "released"
    prompt_template_id: Optional[str] = None   # None -> faithful default
    logit_dtype: str = "native"
    gamma: Optional[float] = None
    k: Optional[float] = None
    tau_js: Optional[float] = None
    alpha: Optional[float] = None
    seed: int = 42
    limit: Optional[int] = None
    #: Overrides the per-benchmark default. OPERA is the only user: its own
    #: harness generates 10 tokens on POPE where ours generates 1, and with
    #: beam search that is not a cosmetic difference -- the search optimises
    #: the whole sequence, so the first token it picks depends on how many
    #: follow. It is in VARIANT_FIELDS, so the run file reflects it.
    max_new_tokens: Optional[int] = None
    #: A validation arm rather than a table row; see storage.CONTROL_ARMS.
    control: Optional[str] = None
    note: str = ""

    def template_id(self) -> str:
        """The faithful default unless the job overrides it.

        ``pope_v0_raw`` sends the POPE question verbatim, which is what the
        release does; ``pope_v1`` appends the "single word or phrase"
        instruction and is an ablation.
        """
        if self.prompt_template_id:
            return self.prompt_template_id
        return DEFAULT_TEMPLATE_IDS[self.benchmark]

    def job_id(self) -> str:
        """Stable, readable identifier used in the frozen plan and the
        calibration TSV. Derived from the fields that distinguish jobs, so it
        survives plan edits as long as the job itself is unchanged."""
        bits = [self.target, self.method, self.benchmark, self.split]
        if self.decoding != "sampling":
            bits.append(self.decoding)
        if self.method == "fox" and self.mode != "as_implemented":
            bits.append(self.mode)
        if self.control is not None:
            bits.append(self.control)
        if self.logit_dtype != "native":
            bits.append(f"ld{self.logit_dtype}")
        if self.system_prompt_id != "llava_v1":
            bits.append(f"sp{self.system_prompt_id}")
        if self.wrapper_id != "released":
            bits.append(f"w{self.wrapper_id}")
        if self.prompt_template_id:
            bits.append(self.prompt_template_id)
        for nm, v in (("g", self.gamma), ("k", self.k),
                      ("t", self.tau_js), ("a", self.alpha)):
            if v is not None:
                bits.append(f"{nm}{v:g}")
        if self.seed != 42:
            # Without this, ten seeds of one job share a job_id: the frozen
            # plan keeps one of them, progress tracking reports nine as
            # pending forever, and --calibrate sums ten runs into one
            # estimate. 42 is left bare so the ids session 1 already wrote to
            # logs/_job_times_r*.tsv still match.
            bits.append(f"s{self.seed}")
        if self.limit:
            bits.append(f"n{self.limit}")
        return ":".join(bits)

    @property
    def n_samples(self) -> int:
        try:
            n = N_SAMPLES[(self.benchmark, self.split)]
        except KeyError:
            # Defaulting to 0 here made a new split look free: the estimate
            # was zero, the session packer placed it anywhere, and the budget
            # it was supposed to respect no longer meant anything.
            raise KeyError(
                f"no sample count for ({self.benchmark!r}, {self.split!r}); "
                f"add it to N_SAMPLES. Known: {sorted(N_SAMPLES)}"
            ) from None
        return min(n, self.limit) if self.limit else n

    @property
    def branches(self) -> str:
        if self.control == "negative_only":
            # The whole point of that arm: no pair, so no second branch.
            return "single"
        if self.method in BEAM_SEARCH:
            return "beam"
        if self.method in SPLIT_BRANCH:
            return "split"
        return "dual" if self.method in DUAL_BRANCH else "single"

    def est_seconds(self) -> float:
        if self.branches == "beam":
            n_fwd = 1 + OPERA_N_CANDIDATES
            per_sample = (
                BEAM_COST_S["prefill"]
                + self.n_new_tokens() * n_fwd * BEAM_COST_S["forward"]
            )
            return self.n_samples * per_sample
        return self.n_samples * COST_S.get(
            (self.benchmark, self.branches), COST_S[("pope", "single")]
        )

    def n_new_tokens(self) -> int:
        """Generation length: the benchmark default unless overridden."""
        if self.max_new_tokens is not None:
            return int(self.max_new_tokens)
        return 1 if self.benchmark == "pope" else 512

    def run_key(self) -> RunKey:
        """Must match what ``generate.py`` would build, or progress is wrong."""
        variant = {
            "gamma": self.gamma, "k": self.k, "tau_js": self.tau_js,
            "alpha": self.alpha, "beta": None,
            "max_new_tokens": self.n_new_tokens(),
            "prompt_template_id": self.template_id(),
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "decoding": self.decoding, "top_p": 0.9, "temperature": 1.0,
            "intervention_layers": None,
            "logit_dtype": self.logit_dtype, "softmax_dtype": "fp32",
            "mode": self.mode, "system_prompt_id": self.system_prompt_id,
            "wrapper_id": self.wrapper_id,
        }
        # Omitted when unset, so a run without a control keeps exactly the
        # file name it had before the field existed -- see
        # storage.OPTIONAL_VARIANT_FIELDS.
        if self.control is not None:
            variant["control"] = self.control
        return RunKey(
            method=self.method, benchmark=self.benchmark, split=self.split,
            seed=self.seed, dtype="fp16", variant_fields=variant,
        )

    def command(self) -> str:
        parts = [
            "python -m foxrepro.bench.generate",
            f"--method {self.method}",
            f"--benchmark {self.benchmark}",
            f"--split {self.split}",
            "--model-id llava-hf/llava-1.5-7b-hf",
            "--dtype fp16 --device-map auto",
            f"--seed {self.seed}",
            f"--decoding {self.decoding}",
            f"--system-prompt {self.system_prompt_id}",
            f"--wrapper {self.wrapper_id}",
            f"--prompt-template-id {self.template_id()}",
        ]
        if self.method == "fox":
            parts.append(f"--mode {self.mode}")
            parts.append(f"--logit-dtype {self.logit_dtype}")
        for flag, val in (("--gamma", self.gamma), ("--k", self.k),
                          ("--tau-js", self.tau_js), ("--alpha", self.alpha)):
            if val is not None:
                parts.append(f"{flag} {val:g}")
        if self.control is not None:
            parts.append(f"--control {self.control}")
        if self.max_new_tokens is not None:
            parts.append(f"--max-new-tokens {self.max_new_tokens}")
        if self.limit:
            parts.append(f"--limit {self.limit}")
        return " ".join(parts)


def _pope_splits() -> Sequence[str]:
    return BENCHMARK_SPLITS["pope"]


#: Seeds for the arms that are actually stochastic.
#:
#: The paper's Tables 1 and 2 are "the best results averaged over 10
#: independent runs" (C.2, verbatim), with significance from "a two-sided
#: t-test (p < 0.05)" -- and no table anywhere reports a standard deviation,
#: so that test cannot be checked. One seed per approach here would reproduce
#: the same weakness: an interval computed from a single run captures only
#: which ITEMS are hard, never which GENERATION was drawn, and is therefore
#: too narrow by an unknown amount.
#:
#: Ten everywhere a comparison is reported under sampling, so the mean gap has
#: a real t-interval and "best of 10" can also be matched on its own terms.
#: Three where an arm is informative but not load-bearing. All start at 42,
#: the seed session 1 already spent, so none of its work is orphaned.
SEEDS_FULL = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)
SEEDS_BREADTH = (42, 43, 44)

#: Kept as an alias: ``SEEDS_HEADLINE`` was the name before the replication
#: was widened past the two headline arms.
SEEDS_HEADLINE = SEEDS_FULL

#: Arms replicated ten times on POPE-adversarial under sampling. Each is a
#: separate *approach*, so each needs its own interval rather than borrowing
#: fox's:
#:
#: ``sampling``       the baseline every gap is measured against
#: ``fox``            the method as released
#: ``off``            the fusion with the intervention disabled -- isolates
#:                    how much of the gain is the two-branch fusion alone
#: ``ulp_control``    the requantisation applied with no shift at all -- if
#:                    this matches ``fox``, the mechanism is the fp16 cast
#:
#: The remaining arms (``as_described``, ``noise_control``, ``boost_visual``,
#: the logit-dtype arms, the gamma sweep) are run under greedy instead, where
#: argmax is deterministic and one run is the complete answer. For the
#: mechanism question -- does the shift change the output at all -- that is
#: both cheaper and a sharper instrument than ten noisy sampling runs.
POPE_SEEDED_MODES = ("off", "ulp_control")


def _build_plan() -> List[Job]:
    """The plan, in priority order rather than cost order.

    Tiers, and why they are in this sequence:

    1. **The greedy control.** The paper offers it as its own strongest
       evidence: "These improvements persist without any sampling variance,
       suggesting that the gains are not attributable to temperature tuning or
       favorable sampling randomness, but stem from the proposed structural
       intervention" (C.7). Session 1 already found CHAIR greedy gives a null.
       POPE greedy is the other half and has never been run. Greedy is
       deterministic and the paper's Table 9 is explicitly single-run ("we
       omit averaged scores"), so this comparison needs no seeds on either
       side -- the cheapest decisive evidence in the whole plan.
    2. **Ten seeds on the two arms compared directly to Tables 1 and 2.**
    3. **Ten seeds on the two controls that decide the mechanism**, under the
       paper's own decoding, so their intervals are comparable to fox's.
    4. **The remaining mechanism arms, under greedy**, one run each.
    5. **Breadth**: the other POPE splits, the prompt and system-prompt arms,
       and the sweeps.
    6. **The VCD-noise arm**, last and at three seeds, because it is not VCD
       proper (see docs/DEVIATIONS.md) and so cannot be put against the
       paper's VCD row whatever its interval.
    """
    jobs: List[Job] = []

    # ---- Tier 1: the greedy control, 1 seed, deterministic --------------- #
    # CHAIR greedy ran in session 1 and stays in the plan: the plan is the
    # record of what the project ran, and dropping a finished job would orphan
    # its run file from progress tracking.
    for method in ("sampling", "fox"):
        jobs.append(Job("table9", method, "chair", "val2014_500",
                        decoding="greedy",
                        note="greedy CHAIR -- done, session 1 r2"))
    for split in _pope_splits():
        for method in ("sampling", "fox"):
            jobs.append(Job("table9", method, "pope", split,
                            decoding="greedy",
                            note="greedy POPE -- the paper's own control"))

    # Greedy is only deterministic if nothing else in the path is random.
    # Asserting that costs two short runs and converts the assumption this
    # whole tier rests on into a measurement.
    for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
        jobs.append(Job("determinism", "fox", bench, split, decoding="greedy",
                        seed=997, limit=50,
                        note="same inputs, different seed: outputs must match"))

    # ---- Tier 2: ten seeds, the arms the paper prints -------------------- #
    for seed in SEEDS_FULL:
        for method in ("sampling", "fox"):
            jobs.append(Job("table1", method, "pope", "adversarial",
                            seed=seed, note="POPE accuracy (T1) and F1 (T8)"))
    # CHAIR_S has n=500 captions and resolves only 8.85 points on its own.
    # That is ITEM variance, which averaging seeds over the same 500 images
    # cannot shrink -- only more images would, and that would leave the
    # paper's protocol. What the seeds do shrink is the generation component
    # of the PAIRED gap, where the item difficulty largely cancels. CHAIR is
    # also where the abstract's headline lives ("29.1% over SID" is CHAIR_S).
    for seed in SEEDS_FULL:
        for method in ("sampling", "fox"):
            jobs.append(Job("table2", method, "chair", "val2014_500",
                            seed=seed, note="CHAIR_S / CHAIR_I"))

    # ---- Tier 3: ten seeds on the controls that decide the mechanism ----- #
    for mode in POPE_SEEDED_MODES:
        for seed in SEEDS_FULL:
            jobs.append(Job("controls", "fox", "pope", "adversarial",
                            mode=mode, seed=seed,
                            note=f"mode={mode}, sampling, seeded"))
    # Three on CHAIR: enough for a direction with an interval, where ten would
    # cost 30 h to refine a question the greedy arms already answer.
    for mode in POPE_SEEDED_MODES:
        for seed in SEEDS_BREADTH:
            jobs.append(Job("controls", "fox", "chair", "val2014_500",
                            mode=mode, seed=seed,
                            note=f"mode={mode}, sampling, seeded"))

    # ---- Tier 4: the rest of the mechanism audit, under greedy ----------- #
    for mode in ("as_described", "noise_control"):
        for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
            jobs.append(Job("controls", "fox", bench, split, mode=mode,
                            decoding="greedy", note=f"mode={mode}, greedy"))
    # boost_visual is this repo's own arm: the intervention the paper argues
    # from and never performs. Small delta, because it is an attention bias
    # meant to be felt, not a precision artifact.
    for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
        jobs.append(Job("boost_visual", "fox", bench, split,
                        mode="boost_visual", gamma=4.0, decoding="greedy",
                        note="amplify visual attention, delta=4"))
    for ld in ("fp32", "bf16"):
        for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
            jobs.append(Job("dtype_arms", "fox", bench, split, logit_dtype=ld,
                            decoding="greedy",
                            note=f"logit_dtype={ld}, greedy"))
    # fp16 ULP at 1e2 and 1e4 is 0.0625 and 8.0, a 128x span. A response that
    # tracks ULP rather than gamma is the signature of the requantisation
    # account. 1e5 overflows fp16 and is out.
    for gamma in (1e2, 1e4):
        jobs.append(Job("gamma_sweep", "fox", "chair", "val2014_500",
                        gamma=gamma, decoding="greedy",
                        note=f"gamma={gamma:g}, greedy"))

    # ---- Tier 5: breadth ------------------------------------------------- #
    for seed in SEEDS_BREADTH:
        for split in ("random", "popular"):
            for method in ("sampling", "fox"):
                jobs.append(Job("table1", method, "pope", split, seed=seed,
                                note="POPE accuracy (T1) and F1 (T8)"))
    # Does the paper's shortcut survive without a system prompt to anchor it?
    for method in ("sampling", "fox"):
        jobs.append(Job("system_prompt", method, "pope", "adversarial",
                        system_prompt_id="none", decoding="greedy",
                        note="tests the X_sys anchor claim directly"))
    # The ways session 1 round r1's prompt differed from the release. r1 ran
    # all of them at once, which measured their sum (POPE-adversarial baseline
    # 81.90 against the paper's 76.77). These isolate them.
    for label, over in (
        ("newline", {"wrapper_id": "newline"}),
        ("single_word", {"prompt_template_id": "pope_v1"}),
        ("spaced_sysprompt", {"system_prompt_id": "llava_v1_spaced"}),
    ):
        jobs.append(Job("prompt_ablation", "sampling", "pope", "adversarial",
                        decoding="greedy", note=f"faithful + {label}", **over))
    for alpha in (1.0, 4.0):        # 2.0 is the default, already covered
        jobs.append(Job("table7", "fox", "pope", "adversarial", alpha=alpha,
                        decoding="greedy", note=f"alpha={alpha:g}"))
    for k in (0.25, 0.45):          # 0.35 is released; 0.45 is the PAPER's
        jobs.append(Job("fig9", "fox", "pope", "adversarial", k=k, limit=500,
                        decoding="greedy", note=f"k={k:g}, 500 samples"))
    for tau in (0.1, 0.2):          # 0.15 is released; 0.2 is the PAPER's
        jobs.append(Job("fig9", "fox", "pope", "adversarial", tau_js=tau,
                        limit=500, decoding="greedy",
                        note=f"tau_js={tau:g}, 500 samples"))

    # ---- Round 4 ---------------------------------------------------------- #
    # Three gaps the first three rounds left, each tied to a claim that is
    # currently weaker than the claim it supports.

    # A. How much of the CHAIR discrepancy is just WHICH 500 images?
    #    char_eval.py takes os.listdir of an unpublished directory, shuffles
    #    it and keeps 500, so the paper's image set is not recoverable even in
    #    principle. Our CHAIR baseline sits 3.88 points from theirs while our
    #    POPE baseline -- on POPE's fixed official files -- sits 0.05 from it.
    #    Four more independent draws turn "probably the image set" into a
    #    measured spread. Greedy, so one run per draw is the whole answer.
    for split in CHAIR_DRAWS:
        for method in ("sampling", "fox"):
            jobs.append(Job("chair_draws", method, "chair", split,
                            decoding="greedy",
                            note=f"image-set sensitivity, {split}"))

    # D. The dtype arms decide the mechanism, and they exist only as single
    #    greedy runs. Under sampling with ten seeds they become comparable to
    #    the fox and ulp_control intervals rather than a separate kind of
    #    evidence. fp32 is the one that matters: it removes the rounding while
    #    keeping the shift, and under greedy it took the whole effect with it.
    for ld in ("fp32", "bf16"):
        for seed in SEEDS_FULL:
            jobs.append(Job("dtype_arms", "fox", "pope", "adversarial",
                            logit_dtype=ld, seed=seed,
                            note=f"logit_dtype={ld}, sampling, seeded"))

    # F. boost_visual beat fox on CHAIR_I but not on CHAIR_S, on one run each.
    #    "The intervention the paper argues from works better than the one it
    #    ships" is the strongest claim in the project and currently rests on
    #    a single greedy pair.
    for seed in SEEDS_FULL:
        for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
            jobs.append(Job("boost_visual", "fox", bench, split,
                            mode="boost_visual", gamma=4.0, seed=seed,
                            note="amplify visual attention, delta=4, sampling"))

    # ---- Round 6, gate: do the ported methods work before we trust them? --- #
    # ---- Validation tier: is each ported method doing what its paper says?
    #
    # Bit-equality against a transcribed reference proves "our code == their
    # code". It does NOT prove "their code == the description in the paper",
    # and it proves nothing about whether the mechanism is active on the real
    # model. Five of the six compared methods here were implemented from
    # source by this project, so that gap has to be closed by measurement.
    #
    # Every job below is cheap, and every one is FALSIFIABLE -- each has a
    # stated outcome that, if it does not happen, means the method is not
    # doing what it claims.

    # V1. The premise all four contrastive methods share: "the negative
    #     branch is a WORSE model, so subtracting it helps". Run each
    #     negative branch alone, with no contrast. PREDICTION: every one of
    #     these must score BELOW the sampling baseline. One that does not is
    #     a branch whose subtraction cannot be doing what the paper says.
    #     None of the four papers reports this number.
    for method in ("vcd", "icd", "sid", "causalmm"):
        jobs.append(Job("validation", method, "pope", "adversarial",
                        control="negative_only",
                        note="the negative branch alone; must be worse than baseline"))

    # V2. The formula. At alpha = 0, (1 + a) z_obs - a z_cd collapses to
    #     z_obs, so the output must be TOKEN-IDENTICAL to the sampling
    #     baseline at the same seed -- not close, identical. A sign error, a
    #     mis-scoped branch or a leaked perturbation all break exactness
    #     while leaving the accuracy plausible. 500 samples is plenty: one
    #     differing token fails it.
    for method in ("vcd", "icd", "sid", "causalmm"):
        jobs.append(Job("validation", method, "pope", "adversarial",
                        alpha=0.0, limit=500,
                        note="alpha=0 must reproduce the baseline token for token"))

    # V3. SID's direction and selection. The released source has
    #     `topk(ATTENTION_RANK)` commented out one line above the call it
    #     uses -- that is FastV, which keeps the MOST-attended tokens.
    #     PREDICTION: keeping the least-attended must build a worse reader
    #     than keeping the most-attended. And if keeping 100 RANDOM tokens
    #     builds an equally bad one, then CT2S's selection does no work and
    #     the method is "drop 476 vision tokens" -- the same control Figure 8
    #     applies to Fox's head scoring.
    for control in ("sid_keep_most", "sid_random"):
        jobs.append(Job("validation", "sid", "pope", "adversarial",
                        control=control,
                        note=f"CT2S direction control: {control}"))

    # V4. CausalMM's gain is 1/n_keys by construction: each attention row is
    #     a softmax, so the matrix sums to n_rows and a permutation cannot
    #     change a mean. Measured at 0.00172 against 1/577 = 0.00173, which
    #     makes the negative branch close to image-blind.
    #     PREDICTION: if `constant` (exactly 1/n_keys) and `blind` (features
    #     zeroed) give the same answers as `shuffle`, the vision
    #     counterfactual is a description of something the code does not do,
    #     and CausalMM is a language-prior contrast.
    for control in ("cf_constant", "cf_blind"):
        jobs.append(Job("validation", "causalmm", "pope", "adversarial",
                        control=control, limit=500,
                        note=f"is the shuffle load-bearing? {control}"))

    # V5. OPERA is three things at once: a beam search, a hard five-candidate
    #     restriction, and an attention bonus. These two rows separate them --
    #     whatever the restriction alone buys is `opera_no_penalty` minus a
    #     plain beam search, and whatever the penalty alone buys is
    #     `opera_unrestricted` minus the same.
    #     PREDICTION, from the diagnostic the ported run already records: the
    #     bonus is added in LOGIT space and its spread across the five
    #     candidates is tiny next to their logit spread, so `opera_no_penalty`
    #     should land on top of plain OPERA. If it does, OPERA's POPE number
    #     is beam search with a truncated candidate set.
    for control in ("opera_unrestricted", "opera_no_penalty"):
        jobs.append(Job("validation", "opera", "pope", "adversarial",
                        decoding="greedy", control=control,
                        note=f"OPERA decomposition: {control}"))


    # ---- Round 6: the compared methods, as themselves --------------------- #
    # Tables 1, 2 and 8 put Fox against Sampling, VCD, ICD, OPERA, CausalMM and
    # SID, and the paper re-ran all of them rather than quoting prior numbers
    # (its VCD row differs from VCD's own paper by up to 5.55 points and flips
    # sign on one split). So these rows are reproducible in principle, and
    # until they exist the only thing this project can say about Table 1 is
    # whether OUR fox beats OUR sampling.
    #
    # Three of the six are implemented and verified against their released
    # source (tests/test_kernel_equivalence.py, tests/test_attention_wiring.py):
    #
    #   vcd       Leng et al. 2024   negative = diffusion-noised image
    #   icd       Wang et al. 2024   negative = disturbed instruction
    #   sid       Huo et al. 2025    negative = the model's own worst read
    #   causalmm  Zhou et al. 2025   negative = shuffled CLIP attention
    #
    # All four share ONE decoding rule, (1 + a) z_obs - a z_cd with the beta
    # cutoff -- CausalMM calls the coefficients gamma and epsilon, but the
    # formula is the same term for term. They differ only in how the negative
    # branch is built, so a bug in the rule would move all four together and
    # a bug in one branch would move one. That is worth more than any single
    # row, and it is not stated in any of the four papers.
    #
    # Breadth seeds, not ten. The seed-level interval on POPE-adversarial is
    # about +/-0.6 points at n=3, which is well inside the 4-10 point gaps
    # these rows are being read for, and ten seeds for three methods on both
    # benchmarks would cost 80 h to refine a comparison against a number we
    # cannot pair with anyway (the paper's own run is not ours).
    #
    # Their own `target`, not table1/table2, so they form their own pair_key
    # groups. Sharing the headline target would graft them onto Tier 2's
    # groups and hand them Tier 2's priority -- and would put five methods in
    # one indivisible group, which on CHAIR is 7.9 h of a 9 h session. The
    # baseline they are read against is `sampling` at the same seed, which is
    # already archived for 42/43/44, so nothing here needs it in-session.
    # Grouping the three TOGETHER is the useful pairing: they share the
    # decoding rule, so a bug in it moves all three and a bug in one negative
    # branch moves one.
    for seed in SEEDS_BREADTH:
        for method in ("vcd", "icd", "sid", "causalmm"):
            jobs.append(Job("table1_methods", method, "pope", "adversarial",
                            seed=seed,
                            note="compared baseline, own decoding rule"))
            jobs.append(Job("table2_methods", method, "chair", "val2014_500",
                            seed=seed,
                            note="compared baseline, own decoding rule"))

    # The released harnesses argmax the UNMASKED fused logits whenever
    # --sample-greedy is set, and SID ships with it True -- so SID's published
    # POPE numbers are greedy whatever --sample says. One greedy run per
    # method makes that comparable instead of assumed, and costs one seed.
    for method in ("vcd", "icd", "sid"):
        for bench, split in (("pope", "adversarial"), ("chair", "val2014_500")):
            jobs.append(Job("table9_methods", method, bench, split,
                            decoding="greedy",
                            note="released --sample-greedy default"))
    # ---- OPERA: POPE only, and that is the method's own boundary --------- #
    # Beam search, so deterministic -- one run per split is the whole arm, and
    # the paper agrees: it prints the identical OPERA numbers in Table 1
    # (sampling) and Table 9 (greedy), because there is no sampling variant to
    # print. Hence `decoding="greedy"` and no seeds.
    #
    # No CHAIR row. OPERA's second mechanism, retrospection-reallocation,
    # cannot fire below 16 response tokens (two independent gates; see
    # docs/DEVIATIONS.md Sec 3.13), so POPE at 10 tokens exercises exactly the
    # half of OPERA that the published POPE numbers came from, and CHAIR at
    # 512 would need the rollback this project has audited but not yet built.
    # `opera_generate` refuses the long length rather than quietly running
    # OPERA with half its method missing.
    #
    # TWO lengths, because the three forks disagree and for a beam search the
    # disagreement is load-bearing: it scores whole sequences, so the first
    # token depends on how many follow. Fox's pope_eval.py passes 1; OPERA's
    # and SID's pass 10. Fox re-ran its compared rows in its own harness, so
    # the OPERA number in Fox's Table 1 is a 1-token run -- that is the row
    # that belongs next to ours, and the 10-token run is the one that belongs
    # next to OPERA's own paper. For every other compared method the length
    # only changes how much text the parser reads, so they get one arm each.
    for split in _pope_splits():
        jobs.append(Job("table1_methods", "opera", "pope", split,
                        decoding="greedy",
                        note="beam-5 + over-trust penalty, Fox's 1-token harness"))
    # OPERA's own protocol. One split: this answers "does the length change
    # the answer for a beam search", and adversarial is where the gaps live.
    jobs.append(Job("opera_protocol", "opera", "pope", "adversarial",
                    decoding="greedy", max_new_tokens=10,
                    note="OPERA's own 10-token harness; still rollback-free"))

    # CausalMM is not in that round: its harness never argmaxes the unmasked
    # logits, so there is no greedy default to reproduce. It does run
    # top_p=1 instead of 0.9, which is a protocol difference rather than a
    # decoding one and is recorded in DEVIATIONS 3.13 rather than run as an
    # arm -- changing top_p for one method would make its row the only one
    # not comparable to the others.

    # ---- Tier 6: the VCD-noise arm --------------------------------------- #
    # Not VCD: VCD's noised branch inside FOX's additive fusion. It ran in
    # rounds 1-5 under the name `vcd`, which now means VCD proper, so the
    # archived files are relabelled on read by
    # evaluate.LEGACY_METHOD_NAMES. Kept in the plan because its three seeds
    # exist and because the pair (vcd, vcd_fox_fusion) isolates the fusion
    # rule at a fixed negative branch -- the one thing neither paper ran.
    for seed in SEEDS_BREADTH:
        jobs.append(Job("table1_methods", "vcd_fox_fusion", "pope",
                        "adversarial", seed=seed,
                        note="VCD's noise in Fox's fusion"))
        jobs.append(Job("table2_methods", "vcd_fox_fusion", "chair",
                        "val2014_500", seed=seed,
                        note="VCD's noise in Fox's fusion"))

    return jobs


#: The full plan, ordered by priority: main tables, then the mechanism audit,
#: then the sensitivity sweeps.
PLAN: List[Job] = _build_plan()



# --------------------------------------------------------------------------- #
# sharding across independent sessions
# --------------------------------------------------------------------------- #
#: Session 1 when ``--pilot`` is set, pinned rather than cost-balanced.
#:
#: It answers exactly one question: **do the paper's numbers reproduce?**
#: Nothing about mechanism, because that only becomes worth asking once there
#: is a result to explain. If Fox does not beat the baseline here, every
#: control downstream is measuring a difference that does not exist.
#:
#: Cost-balancing optimises the wrong thing for this. It would give session 1 a
#: baseline on one split and the method on another, and ten GPU hours later
#: nothing could be compared.
#:
#: Six jobs, baseline and method on each of the paper's three main tables:
#:
#:   Table 1  POPE-adversarial   the hardest split, largest claimed POPE gain
#:                               (76.77 -> 81.93); the random and popular gaps
#:                               are inside sampling noise anyway
#:   Table 2  CHAIR              where the headline lives: 57.60 -> 46.40, and
#:                               the abstract's "29.1% over SID" is a CHAIR
#:                               number. 500 captions also cost less than 3000
#:                               POPE questions
#:   Table 9  CHAIR greedy       the paper's own de-randomisation. Greedy is
#:                               deterministic, so with one seed it is a
#:                               SHARPER reproducibility signal than sampling,
#:                               which carries its own variance
#:
#: VCD is deliberately absent. What this repo can run is VCD's noised branch
#: inside Fox's fusion (see docs/DEVIATIONS.md), not VCD proper, so its row
#: says nothing about whether the paper reproduces.
PILOT_SESSIONS: Tuple[Tuple[Dict[str, object], ...], ...] = (
    (
        {"target": "table1", "method": "sampling", "split": "adversarial"},
        {"target": "table1", "method": "fox", "split": "adversarial"},
        {"target": "table2", "method": "sampling", "benchmark": "chair"},
        {"target": "table2", "method": "fox", "benchmark": "chair"},
        {"target": "table9", "method": "sampling", "benchmark": "chair"},
        {"target": "table9", "method": "fox", "benchmark": "chair"},
    ),
)

#: Defaults applied under each session's selectors, so they stay readable.
_PILOT_DEFAULTS = (
    {"benchmark": "pope", "split": "adversarial", "decoding": "sampling",
     "mode": "as_implemented", "logit_dtype": "native"},
)


def _pilot_defaults_for(spec: Dict[str, object], base: Dict[str, object]
                        ) -> Dict[str, object]:
    """Merge a selector over the session defaults, letting it override them.

    A CHAIR selector has to drop the POPE split, and a Table 9 selector has to
    set greedy decoding, so the defaults cannot simply be unioned.
    """
    full = dict(base)
    full.update(spec)
    if full.get("benchmark") == "chair":
        full["split"] = "val2014_500"
    if spec.get("target") == "table9":
        full["decoding"] = "greedy"
    return full


def _matches(job: Job, spec: Dict[str, object]) -> bool:
    return all(getattr(job, k) == v for k, v in spec.items())


def pilot_jobs(jobs: Iterable[Job] = ()) -> List[List[Job]]:
    """The pinned opening sessions, as a list of job lists."""
    jobs = list(jobs) or PLAN
    out: List[List[Job]] = []
    for specs, defaults in zip(PILOT_SESSIONS, _PILOT_DEFAULTS):
        picked: List[Job] = []
        for spec in specs:
            full = _pilot_defaults_for(spec, defaults)
            for j in jobs:
                if _matches(j, full) and j not in picked:
                    picked.append(j)
                    break
            else:
                raise LookupError(f"pilot job not in the plan: {full}")
        out.append(picked)
    return out


def partition_jobs(n_sessions: int, jobs: Iterable[Job] = (),
                   pilot: bool = False) -> List[List[Job]]:
    """Split the plan into ``n_sessions`` cost-balanced, disjoint groups.

    Longest-processing-time-first: sort by estimated cost descending and give
    each job to whichever session is currently lightest. That is the standard
    greedy makespan heuristic and it matters here because a naive round-robin
    over 53 jobs whose costs range from 0.2 h to 2.3 h can leave one session
    with twice the work of another.

    **Reassignment is safe.** Editing the plan can move a job between
    sessions, but every job resumes from its own run file, whose name is
    derived from the configuration and not from the session. So a job half-done
    by session 2 and later reassigned to session 3 is continued by session 3
    rather than restarted, and nothing is lost or duplicated. This is why
    balance was preferred over a stable hash-based assignment.
    """
    if n_sessions < 1:
        raise ValueError("n_sessions must be >= 1")
    jobs = list(jobs) or PLAN

    pinned: List[List[Job]] = []
    if pilot:
        pinned = pilot_jobs(jobs)
        if n_sessions <= len(pinned):
            raise ValueError(
                f"--pilot pins {len(pinned)} sessions; need more than that"
            )
        chosen = {id(j) for g in pinned for j in g}
        jobs = [j for j in jobs if id(j) not in chosen]
        n_sessions -= len(pinned)

    buckets: List[List[Job]] = [[] for _ in range(n_sessions)]
    loads = [0.0] * n_sessions
    for job in sorted(jobs, key=lambda j: -j.est_seconds()):
        i = min(range(n_sessions), key=lambda x: loads[x])
        buckets[i].append(job)
        loads[i] += job.est_seconds()
    # Within a session, run the cheap jobs first: a session that dies early
    # then leaves more completed jobs behind than one that began with a 2.3 h
    # job and got cut at 90%.
    rest = [sorted(b, key=lambda j: j.est_seconds()) for b in buckets]
    return [sorted(g, key=lambda j: j.est_seconds()) for g in pinned] + rest


def pair_key(job: Job) -> Tuple:
    """Jobs that must land in the same session: everything but the method.

    A session that holds a baseline without its method, or the reverse, has
    produced nothing comparable. Cost-balancing does not know that, so the
    units it balances are these groups rather than individual jobs.

    Seed is part of the key. Comparing seed 43's fox against seed 42's
    baseline would mix the generation draw into the gap, which is exactly the
    thing the seed replication exists to measure.
    """
    return (job.target, job.benchmark, job.split, job.decoding, job.seed,
            job.mode, job.logit_dtype, job.system_prompt_id, job.wrapper_id,
            job.prompt_template_id, job.gamma, job.k, job.tau_js, job.alpha,
            job.limit)


def partition_by_priority(
    budget_hours: float, jobs: Iterable[Job] = (),
    runs_root: Optional[Path] = None, skip_done: bool = True,
) -> List[List[Job]]:
    """Fill sessions in **plan order** until each reaches ``budget_hours``.

    The alternative, :func:`partition_jobs`, minimises makespan and so is the
    right choice when every job will be run. It is the wrong choice for a plan
    whose first tier is the decisive experiment and whose last is a sweep:
    longest-processing-time-first scatters the tiers across every session, so
    no prefix of the sessions answers anything. Here session 1 is the highest
    priority work that fits, session 2 the next, and stopping after any
    session leaves a coherent result.

    Already-finished jobs are dropped by default, so the sessions returned are
    the work that remains rather than the work the plan ever described.

    Args:
        budget_hours: per-session wall budget.
        jobs: defaults to :data:`PLAN`.
        runs_root: where to look for completed runs.
        skip_done: drop jobs whose run file is already complete.

    Returns:
        A list of sessions, each a list of jobs, in priority order.
    """
    if budget_hours <= 0:
        raise ValueError("budget_hours must be > 0")
    jobs = list(jobs) or PLAN

    if skip_done:
        rows = scan_progress(jobs, runs_root=runs_root)["rows"]
        finished = {r["file"] for r in rows if r["complete"]}
        jobs = [j for j in jobs if j.run_key().filename() not in finished]

    # Group while preserving first-appearance order.
    groups: Dict[Tuple, List[Job]] = {}
    for job in jobs:
        groups.setdefault(pair_key(job), []).append(job)

    # First-fit, in plan order: each group goes into the earliest session it
    # fits in. Closing a session as soon as the next group overflows it, which
    # is the obvious alternative, left three hours idle whenever a 2.65 h CHAIR
    # pair followed a nearly full session -- first-fit backfills those with
    # later groups that do fit, while early groups still take early sessions,
    # so priority survives.
    budget_s = budget_hours * 3600
    sessions: List[List[Job]] = []
    loads: List[float] = []
    for group in groups.values():
        cost = sum(j.est_seconds() for j in group)
        for i, load in enumerate(loads):
            if load + cost <= budget_s:
                sessions[i].extend(group)
                loads[i] = load + cost
                break
        else:
            # A single group over budget cannot be split without splitting a
            # comparison, so it gets a session of its own and the printed
            # estimate says so rather than the overrun being silent.
            sessions.append(list(group))
            loads.append(cost)
    return sessions


#: Provenance fields that must agree before two sets of records may share a
#: run file. Not the full :data:`PROVENANCE_FIELDS`: a session legitimately
#: runs at a newer commit, so ``commit_hash`` is allowed to differ. What may
#: not differ is the machine the numbers came off.
#:
#: Round 4 is why this exists. Two CHAIR jobs ran on a Kaggle notebook with no
#: accelerator attached and produced ~50 samples each on CPU before the wall.
#: Deleting those records from the live runs directory did not help, because
#: the session zips still held them: the next ingest merged them straight back
#: in beside the GPU records that replaced them, by sample_id union, and
#: produced two 500-record files that were 10% CPU. Greedy decoding is
#: deterministic on one device; it is NOT device-independent, since fp16
#: matmul on CPU and on CUDA accumulate differently. Those files were a
#: corrupted run wearing the shape of a finished one.
MERGE_COMPATIBLE_FIELDS = (
    "n_gpus", "gpu_name", "torch_version", "compute_capability",
    "transformers_version", "dtype", "decoding", "seed", "split",
    "prompt_template_id", "system_prompt_id", "wrapper_id",
    "mode", "logit_dtype", "softmax_dtype",
)


def _compatible_records(existing, incoming):
    """Split ``incoming`` into records that may join ``existing``, and those
    that may not.

    Returns ``(accepted, reasons)``. ``reasons`` lists the distinct FIELDS
    that disagreed, so the report can name what was wrong; the count of
    refused records is ``len(incoming) - len(accepted)``, which is not the
    same number and was reported as such once.
    """
    if not existing or not incoming:
        return incoming, []
    ref = existing[0]
    accepted, reasons = [], set()
    for rec in incoming:
        bad = [f for f in MERGE_COMPATIBLE_FIELDS
               if f in ref and f in rec and ref[f] != rec[f]]
        if bad:
            for f in bad:
                reasons.add(f"{f}: {ref.get(f)!r} vs {rec.get(f)!r}")
        else:
            accepted.append(rec)
    return accepted, sorted(reasons)


def ingest_runs(
    sources: Sequence[Path] = (), dest: Optional[Path] = None,
    dry_run: bool = False, inbox: Optional[Path] = None,
) -> Dict[str, object]:
    """Merge run files produced by independent sessions into one directory.

    Each session writes to its own ``/kaggle/working``, so the files have to be
    collected before ``evaluate``/``report`` can see them all. Filenames encode
    the full configuration, so files from different sessions never collide by
    accident -- but a plan edit can move a job between sessions, and then two
    sessions may hold partial files for the SAME job. Those are merged by
    sample_id union rather than one overwriting the other, since both halves
    are real work.
    """
    from foxrepro.bench.generate import scan_run_file

    def _records(path: Path) -> List[Dict[str, object]]:
        out: List[Dict[str, object]] = []
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A trailing partial line from a hard kill; scan_run_file
                    # repairs those in place, so skipping here is safe.
                    continue
        return out

    dest = dest or paths().runs_root
    dest.mkdir(parents=True, exist_ok=True)

    # With no sources, read the inbox: the conventional place to drop the zips
    # downloaded from each Kaggle session, so collecting five sessions is one
    # command rather than five unzips.
    # A parameter, not a constant: otherwise every test of this function
    # depends on whatever happens to be sitting in the repo's inbox.
    inbox = inbox or (paths().repo_root / "runs_inbox")
    sources = [Path(x) for x in sources] or ([inbox] if inbox.is_dir() else [])
    if not sources:
        return {
            "dest": str(dest), "copied": 0, "merged": 0, "skipped": 0,
            "files": [],
            "note": f"nothing to ingest: pass directories or zips, or drop the "
                    f"session zips in {inbox}",
        }

    # Expand any .zip in place so the directory walk below sees plain files.
    # pack_runs.sh zips with -j (flat), so every member is at the archive root.
    staged: List[Path] = []
    tmpdirs: List[tempfile.TemporaryDirectory] = []
    for src in sources:
        if src.is_file() and src.suffix == ".zip":
            td = tempfile.TemporaryDirectory()
            tmpdirs.append(td)
            with zipfile.ZipFile(src) as z:
                for member in z.namelist():
                    if member.endswith(".jsonl") and "/" not in member.strip("/"):
                        z.extract(member, td.name)
            staged.append(Path(td.name))
        elif src.is_dir():
            # A folder of zips, a folder of jsonl, or both. rglob rather than
            # glob because /kaggle/input holds one SUBDIRECTORY per attached
            # dataset, and that is how previous sessions' results come back
            # into a fresh one.
            zips = sorted(src.rglob("*.zip"))
            for zp in zips:
                td = tempfile.TemporaryDirectory()
                tmpdirs.append(td)
                with zipfile.ZipFile(zp) as z:
                    for member in z.namelist():
                        if member.endswith(".jsonl"):
                            z.extract(member, td.name)
                staged.append(Path(td.name))
            if any(src.glob("*.jsonl")):
                staged.append(src)
        else:
            staged.append(src)
    sources = staged
    report: Dict[str, object] = {
        "dest": str(dest), "copied": 0, "merged": 0, "skipped": 0,
        "rejected": 0, "files": []
    }
    for src_dir in sources:
        # rglob: a zip may have nested members even though pack_runs.sh is flat.
        for src in sorted(Path(src_dir).rglob("*.jsonl")):
            target = dest / src.name
            if not target.exists():
                if not dry_run:
                    target.write_bytes(src.read_bytes())
                report["copied"] = int(report["copied"]) + 1  # type: ignore[arg-type]
                report["files"].append(f"copied {src.name}")  # type: ignore[union-attr]
                continue
            scan_run_file(target, repair=True)   # repair a partial tail first
            existing = _records(target)
            have = {r.get("sample_id") for r in existing}
            incoming = [
                r for r in _records(src) if r.get("sample_id") not in have
            ]
            before = len(incoming)
            incoming, reasons = _compatible_records(existing, incoming)
            n_refused = before - len(incoming)
            if n_refused:
                report["rejected"] = int(report.get("rejected", 0)) + n_refused
                report["files"].append(  # type: ignore[union-attr]
                    f"REFUSED {n_refused} record(s) from {src.name}, "
                    f"incompatible with what is already there: "
                    + "; ".join(reasons[:3])
                )
            if not incoming:
                report["skipped"] = int(report["skipped"]) + 1  # type: ignore[arg-type]
                continue
            if not dry_run:
                # Append raw lines rather than going through JsonlWriter.
                # These records were validated when generate.py wrote them;
                # re-validating here would refuse files produced by a session
                # running an older schema, which is precisely the case this
                # function exists to handle.
                import os as _os

                with target.open("a") as fh:
                    for rec in incoming:
                        fh.write(json.dumps(rec, sort_keys=True) + "\n")
                    fh.flush()
                    _os.fsync(fh.fileno())
            report["merged"] = int(report["merged"]) + 1  # type: ignore[arg-type]
            report["files"].append(  # type: ignore[union-attr]
                f"merged {len(incoming)} records into {src.name}"
            )

    for td in tmpdirs:
        td.cleanup()

    if not (
        int(report["copied"]) + int(report["merged"])  # type: ignore[arg-type]
        + int(report["skipped"])  # type: ignore[arg-type]
    ):
        # "I ran ingest and nothing happened" is the obvious next question, so
        # answer it in the result rather than returning a silent row of zeros.
        report["note"] = (
            f"no run files found. Looked in: "
            f"{', '.join(str(x) for x in sources) or '(nothing)'}. "
            f"Drop the session zips in {inbox} and re-run, or pass paths "
            f"explicitly."
        )
    return report


def measured_cost(runs_root: Optional[Path] = None) -> Dict[str, float]:
    """Re-derive seconds-per-sample from the run files that exist.

    The constants in :data:`COST_S` are scaled from one measurement and will
    drift. This reports what the completed runs actually cost, per
    (benchmark, branches), so the estimates can be corrected.
    """
    runs_root = runs_root or paths().runs_root
    buckets: Dict[Tuple[str, str], List[float]] = {}
    if not runs_root.is_dir():
        return {}
    for f in sorted(runs_root.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                lat = rec.get("latency_ms")
                if not isinstance(lat, (int, float)):
                    continue
                bench = rec.get("benchmark", "?")
                m = rec.get("method")
                branches = (
                    "beam" if m in BEAM_SEARCH
                    else "split" if m in SPLIT_BRANCH
                    else "dual" if m in DUAL_BRANCH
                    else "single"
                )
                buckets.setdefault((bench, branches), []).append(lat / 1000.0)
    return {
        f"{b}/{br}": round(sum(v) / len(v), 3)
        for (b, br), v in sorted(buckets.items())
    }


def scan_progress(
    jobs: Iterable[Job] = (), runs_root: Optional[Path] = None
) -> Dict[str, object]:
    """How far each job got, by counting records in its own run file.

    A job is complete when its file holds as many records as the benchmark has
    samples. Partial files are reported with their remaining count, since
    re-running the identical command resumes rather than restarting.
    """
    from foxrepro.bench.generate import scan_run_file

    runs_root = runs_root or paths().runs_root
    jobs = list(jobs) or PLAN
    rows: List[Dict[str, object]] = []
    for job in jobs:
        key = job.run_key()
        path = runs_root / key.filename()
        done = 0
        if path.is_file():
            try:
                done = len(scan_run_file(path, key=key).done_ids)
            except (OSError, ValueError, json.JSONDecodeError):
                # A corrupt or mismatched file counts as no progress. Caught
                # narrowly on purpose: a broad `except Exception` here once hid
                # an AttributeError and made every job report 0 done, which is
                # the exact "plan loses track of itself" failure this module
                # is supposed to prevent.
                done = 0
        total = job.n_samples
        rows.append({
            "target": job.target, "method": job.method,
            "benchmark": job.benchmark, "split": job.split,
            "note": job.note, "done": done, "total": total,
            "complete": total > 0 and done >= total,
            "remaining_s": max(total - done, 0) * (
                job.est_seconds() / total if total else 0
            ),
            "file": key.filename(),
        })
    return {
        "rows": rows,
        "n_jobs": len(rows),
        "n_complete": sum(1 for r in rows if r["complete"]),
        "remaining_hours": round(
            sum(float(r["remaining_s"]) for r in rows) / 3600, 1
        ),
        "measured_cost_s_per_sample": measured_cost(runs_root),
    }


def next_jobs(
    budget_hours: float, jobs: Iterable[Job] = (),
    runs_root: Optional[Path] = None,
) -> List[Dict[str, object]]:
    """The next incomplete jobs that fit ``budget_hours``, in plan order.

    Plan order is priority order, so this fills a session from the top. A job
    larger than the whole budget is still emitted alone with a warning: it
    resumes, so a partial session is progress rather than waste.
    """
    prog = scan_progress(jobs, runs_root)
    chosen: List[Dict[str, object]] = []
    spent = 0.0
    all_jobs = list(jobs) or PLAN
    for row, job in zip(prog["rows"], all_jobs):  # type: ignore[arg-type]
        if row["complete"]:
            continue
        hours = float(row["remaining_s"]) / 3600
        if chosen and spent + hours > budget_hours:
            continue
        chosen.append({
            "command": job.command(), "target": job.target,
            "note": job.note, "est_hours": round(hours, 2),
            "resumes_from": row["done"],
            "exceeds_budget": hours > budget_hours,
        })
        spent += hours
    return chosen


def render_plan(budget_hours: float = 9.0, runs_root: Optional[Path] = None) -> str:
    """A session brief: progress, measured cost, and the commands to paste."""
    prog = scan_progress(runs_root=runs_root)
    lines: List[str] = []
    lines.append("# Fox reproduction -- session plan\n")
    lines.append(
        f"{prog['n_complete']}/{prog['n_jobs']} jobs complete, "
        f"~{prog['remaining_hours']} GPU-hours remaining "
        f"(Kaggle quota is 30/week).\n"
    )

    measured = prog["measured_cost_s_per_sample"]
    if measured:
        lines.append("## Measured cost, seconds per sample\n")
        for k, v in measured.items():  # type: ignore[union-attr]
            est = COST_S.get(tuple(k.split("/")), None)  # type: ignore[arg-type]
            drift = f" (estimate {est})" if est else ""
            lines.append(f"- `{k}`: {v}{drift}")
        lines.append("")

    lines.append("## By target\n")
    lines.append("| target | done | total | remaining h |")
    lines.append("|---|---|---|---|")
    agg: Dict[str, List[float]] = {}
    for r in prog["rows"]:  # type: ignore[union-attr]
        a = agg.setdefault(str(r["target"]), [0, 0, 0.0])
        a[0] += 1 if r["complete"] else 0
        a[1] += 1
        a[2] += float(r["remaining_s"]) / 3600
    for target, (done, total, hours) in agg.items():
        lines.append(f"| {target} | {int(done)} | {int(total)} | {hours:.1f} |")

    lines.append(f"\n## Next, within {budget_hours} GPU-hours\n")
    nxt = next_jobs(budget_hours, runs_root=runs_root)
    if not nxt:
        lines.append("Nothing left in the plan. ")
        lines.append("Run `fox-report` and read the confidence intervals.")
    for i, j in enumerate(nxt, 1):
        warn = "  **exceeds the whole budget; it resumes, so run it anyway**" if j["exceeds_budget"] else ""
        resume = f", resuming from {j['resumes_from']}" if j["resumes_from"] else ""
        lines.append(
            f"{i}. `{j['target']}` -- {j['note']} "
            f"(~{j['est_hours']}h{resume}){warn}\n"
            f"   ```bash\n   !{j['command']}\n   ```"
        )
    return "\n".join(lines)


def archive_runs(
    runs_root: Optional[Path] = None, dest: Optional[Path] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    """Copy the live run files into the committed archive.

    The inverse of the ingest the bootstrap does. Merging is by ``sample_id``
    union, so archiving a job that was half done in one session and finished
    in the next adds the new records rather than replacing the file -- both
    halves are real GPU time.
    """
    runs_root = runs_root or paths().runs_root
    dest = dest or (paths().repo_root / "results" / "runs")
    if not runs_root.is_dir():
        return {"dest": str(dest), "copied": 0, "merged": 0, "skipped": 0,
                "files": [], "note": f"no live run directory at {runs_root}"}
    return ingest_runs([runs_root], dest, dry_run=dry_run, inbox=dest)


def freeze_plan(
    n_sessions: Optional[int], budget_hours: float, plan_path: Path,
    calibrate: Optional[Path] = None, round_tag: str = "r1",
    pilot: bool = False, order: str = "priority",
    runs_root: Optional[Path] = None, skip_done: bool = True,
) -> Dict[str, object]:
    """Partition the plan and write it to disk.

    The partition is **frozen to a file** rather than recomputed per session.
    That is the difference between sessions that agree on who does what and
    sessions that silently overlap or skip work after the plan source is
    edited. Run this once, locally, commit the file, and every session reads
    the same assignment.

    ``calibrate`` is a TSV of ``job_id<TAB>minutes<TAB>returncode`` appended by
    :func:`run_session`. Feeding it back replaces the estimates, which are
    scaled from a single measurement and will drift.

    Rows with a non-zero return code are discarded: a job that died in 6
    seconds on a missing file measures the failure, not the work. Rows for the
    same job are **summed**, because a job cut by the session wall and resumed
    appears once per invocation and the cost of running it from scratch is the
    total. Taking the last row instead -- which is what a plain dict
    assignment does -- would let one failed retry after a good run convince
    the planner that a 93-minute job takes 6 seconds, and the session that
    believed it would overcommit by hours.
    """
    measured: Dict[str, float] = {}
    if calibrate and Path(calibrate).is_file():
        for line in Path(calibrate).read_text().splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                minutes = float(parts[1])
            except ValueError:
                continue
            if len(parts) >= 3:
                try:
                    if int(parts[2]) != 0:
                        continue
                except ValueError:
                    continue
            measured[parts[0]] = measured.get(parts[0], 0.0) + minutes * 60.0

    jobs = PLAN
    if order == "priority":
        groups = partition_by_priority(
            budget_hours, jobs, runs_root=runs_root, skip_done=skip_done)
    elif order == "balanced":
        if n_sessions is None:
            raise ValueError("--sessions is required with --order balanced")
        groups = partition_jobs(n_sessions, jobs, pilot=pilot)
    else:
        raise ValueError("order must be 'priority' or 'balanced'")
    payload: Dict[str, object] = {
        "round": round_tag,
        "order": order,
        "n_sessions": len(groups),
        "budget_hours": budget_hours,
        "n_calibrated": len(measured),
        "jobs": {},
        "sessions": [],
    }
    for gi, group in enumerate(groups, 1):
        ids = []
        for job in group:
            jid = job.job_id()
            est = measured.get(jid, job.est_seconds())
            payload["jobs"][jid] = {  # type: ignore[index]
                "id": jid, "target": job.target, "note": job.note,
                "command": job.command(), "est_s": round(est, 1),
                "run_file": job.run_key().filename(),
                "n_samples": job.n_samples,
                "measured": jid in measured,
            }
            ids.append(jid)
        payload["sessions"].append({  # type: ignore[union-attr]
            "session": gi, "jobs": ids,
            "est_hours": round(
                sum(float(payload["jobs"][i]["est_s"]) for i in ids) / 3600, 2  # type: ignore[index]
            ),
        })
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2))
    return payload


def run_session(
    session: int, plan_path: Path, *, budget_hours: Optional[float] = None,
    dry_run: bool = False, runs_root: Optional[Path] = None,
    pack: bool = True, hard_deadline_hours: Optional[float] = None,
) -> int:
    """Execute one session's jobs, in budget, resumably.

    Four behaviours borrowed from the ACSA queue, each earned the hard way:

    * **A job whose estimate no longer fits the remaining budget is skipped,
      not started.** Beginning a 2.3 h job with 40 minutes left burns the GPU
      for nothing, because the next session resumes it from the same place
      either way.
    * **Results are packed after every finished job**, so a cut session loses
      at most the job in flight.
    * **Per-job logs** go to a logs directory; the console keeps only the
      timeline.
    * **A watchdog stops cleanly before the session wall.** A notebook killed
      by the platform may lose its output entirely, throwing away every
      completed job. It defaults to 15 minutes inside ``budget_hours``,
      which is enough to finish the last pack and exit -- a fixed deadline
      would either overshoot a short session or waste a long one.
    """
    import subprocess
    import threading

    p = paths()
    plan = json.loads(Path(plan_path).read_text())
    try:
        ids = plan["sessions"][session - 1]["jobs"]
    except (IndexError, KeyError):
        print(f"[plan] session {session} is not in {plan_path}", file=sys.stderr)
        return 2

    tag = f"fox_{plan.get('round', 'r1')}_s{session}"
    if dry_run:
        for jid in ids:
            j = plan["jobs"][jid]
            print(f"# {jid}  est {j['est_s'] / 60:.0f} min"
                  f"{'  (measured)' if j.get('measured') else ''}")
            print(j["command"])
        return 0

    runs_root = runs_root or p.runs_root
    logs = p.logs_dir
    logs.mkdir(parents=True, exist_ok=True)
    times = p.logs_dir / "_job_times.tsv"
    budget_h = budget_hours or plan.get("budget_hours", 10.0)
    budget = budget_h * 3600
    if hard_deadline_hours is None:
        # Leave room for the final pack; never less than half the budget.
        hard_deadline_hours = max(budget_h - 0.25, budget_h * 0.5)
    t0 = time.time()
    stop = threading.Event()
    procs: set = set()

    def watchdog() -> None:
        hard = hard_deadline_hours * 3600
        while not stop.wait(10):
            if time.time() - t0 > hard:
                print(f"[{time.strftime('%H:%M:%S')}] HARD DEADLINE "
                      f"({hard_deadline_hours:.1f} h): stopping so the notebook "
                      f"can finish and keep its output", flush=True)
                stop.set()
                for pr in list(procs):
                    try:
                        pr.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                return

    threading.Thread(target=watchdog, daemon=True).start()
    ran = skipped = failed = 0
    try:
        for jid in ids:
            if stop.is_set():
                print("[plan] deadline reached; remaining jobs left for the "
                      "next session")
                break
            job = plan["jobs"][jid]
            run_file = runs_root / job["run_file"]
            if run_file.is_file():
                from foxrepro.bench.generate import scan_run_file
                try:
                    done = len(scan_run_file(run_file).done_ids)
                except Exception:  # noqa: BLE001
                    done = 0
                if done >= job["n_samples"] > 0:
                    print(f"[{time.strftime('%H:%M:%S')}] SKIP {jid} (complete)")
                    skipped += 1
                    continue

            left = budget - (time.time() - t0)
            if job["est_s"] > left:
                print(f"[{time.strftime('%H:%M:%S')}] DEFER {jid} "
                      f"(needs {job['est_s'] / 60:.0f} min, "
                      f"{left / 60:.0f} min left in budget)")
                skipped += 1
                continue

            print(f"[{time.strftime('%H:%M:%S')}] START {jid} "
                  f"(~{job['est_s'] / 60:.0f} min)", flush=True)
            start = time.time()
            log_path = logs / (jid.replace("/", "_").replace(":", "_") + ".log")
            with log_path.open("w") as log:
                pr = subprocess.Popen(
                    job["command"], shell=True, cwd=str(p.repo_root),
                    stdout=log, stderr=subprocess.STDOUT,
                )
                procs.add(pr)
                rc = pr.wait()
                procs.discard(pr)
            minutes = (time.time() - start) / 60
            with times.open("a") as fh:
                fh.write(f"{jid}\t{minutes:.1f}\t{rc}\n")
            if rc == 0:
                print(f"[{time.strftime('%H:%M:%S')}] DONE  {jid} "
                      f"({minutes:.1f} min)", flush=True)
                ran += 1
            else:
                # Exit code 7 is generate.py's "stopped early, resume me".
                kind = "STOPPED" if rc == 7 else "FAIL"
                print(f"[{time.strftime('%H:%M:%S')}] {kind} {jid} "
                      f"(rc={rc}, {minutes:.1f} min) -- tail of {log_path}:",
                      flush=True)
                tail = log_path.read_text().splitlines()[-8:]
                print("\n".join("    " + x for x in tail), flush=True)
                failed += 0 if rc == 7 else 1
            if pack:
                subprocess.run(["bash", "scripts/pack_runs.sh", tag],
                               cwd=str(p.repo_root),
                               stdout=subprocess.DEVNULL)
    finally:
        stop.set()

    print(f"\n[plan] session {session}: {ran} ran, {skipped} skipped/deferred, "
          f"{failed} failed, {(time.time() - t0) / 3600:.2f} h elapsed")
    print(f"[plan] pack tag: {tag}")
    return 1 if failed else 0


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    default_plan = p.repo_root / "scripts" / "run_plan.json"
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd")

    a = sub.add_parser("list", help="inventory: done / remaining per target")
    a.add_argument("--runs-dir", type=Path, default=None)
    a.add_argument("--budget-hours", type=float, default=9.0)
    a.add_argument("--json", action="store_true")

    a = sub.add_parser("plan", help="freeze the session split to a JSON file")
    a.add_argument(
        "--order", choices=("priority", "balanced"), default="priority",
        help="priority (default): fill sessions in plan order until each "
             "reaches --budget-hours, so session 1 is the decisive work and "
             "stopping after any session still leaves a coherent result. "
             "balanced: minimise makespan over --sessions, which is the right "
             "choice only when every job will be run.")
    a.add_argument("--sessions", type=int, default=None,
                   help="required with --order balanced; ignored otherwise")
    a.add_argument("--include-done", action="store_true",
                   help="keep finished jobs in the session list")
    a.add_argument("--budget-hours", type=float, default=10.0,
                   help="per session; Kaggle caps a session at 12 h")
    a.add_argument("--calibrate", type=Path, default=None,
                   help="ONE round's _job_times.tsv, written by `run`, to "
                        "replace the estimates. Rows for the same job are "
                        "summed (a job cut by the wall and resumed appears "
                        "once per invocation), so do not concatenate rounds: "
                        "keep logs/_job_times_r1.tsv, _r2.tsv and pass the "
                        "latest one.")
    a.add_argument("--round", default="r1", help="tag for the pack filenames")
    a.add_argument("--pilot", action="store_true",
                   help="pin session 1 to the jobs that make it conclusive "
                        "on its own: baseline and fox on POPE-adversarial, "
                        "and on CHAIR under both sampling and greedy. "
                        "Cost-balancing alone would spread those across "
                        "sessions -- a baseline in one and the method in "
                        "another -- leaving nothing comparable after ten "
                        "GPU hours.")
    a.add_argument("--out", type=Path, default=default_plan)

    a = sub.add_parser("run", help="execute one session (on Kaggle)")
    a.add_argument("--session", type=int, required=True)
    a.add_argument("--plan", type=Path, default=default_plan)
    a.add_argument("--budget-hours", type=float, default=None)
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--no-pack", action="store_true")
    a.add_argument("--runs-dir", type=Path, default=None)
    a.add_argument("--tag", action="store_true",
                   help="print the pack tag for this session and exit")

    a = sub.add_parser("ingest", help="merge run files from several sessions")
    a.add_argument("dirs", type=Path, nargs="*",
               help="directories or .zip files; with none, reads "
                    "./runs_inbox (where the downloaded session "
                    "zips belong)")
    a.add_argument("--runs-dir", type=Path, default=None)
    a.add_argument("--dry-run", action="store_true")

    a = sub.add_parser(
        "archive", help="copy the live runs into results/runs for committing")
    a.add_argument("--runs-dir", type=Path, default=None)
    a.add_argument("--dest", type=Path, default=None)
    a.add_argument("--dry-run", action="store_true")

    a = sub.add_parser("calibrate", help="measured seconds per sample")
    a.add_argument("--runs-dir", type=Path, default=None)

    args = ap.parse_args(argv)
    cmd = args.cmd or "list"

    if cmd == "archive":
        print(json.dumps(
            archive_runs(args.runs_dir, args.dest, args.dry_run), indent=2))
        return 0
    if cmd == "calibrate":
        print(json.dumps(measured_cost(args.runs_dir), indent=2))
        return 0
    if cmd == "ingest":
        print(json.dumps(ingest_runs(args.dirs, args.runs_dir, args.dry_run),
                         indent=2))
        return 0
    if cmd == "plan":
        payload = freeze_plan(args.sessions, args.budget_hours, args.out,
                              args.calibrate, args.round, pilot=args.pilot,
                              order=args.order,
                              skip_done=not args.include_done)
        print(f"# Wrote {args.out}  (round {payload['round']}, "
              f"{payload['n_calibrated']} calibrated estimates)\n")
        print("| session | jobs | est GPU-h |")
        print("|---|---|---|")
        over = []
        for sess in payload["sessions"]:  # type: ignore[union-attr]
            flag = ""
            if float(sess["est_hours"]) > args.budget_hours:
                flag = " **over budget**"
                over.append(int(sess["session"]))
            print(f"| {sess['session']} | {len(sess['jobs'])} | "
                  f"{sess['est_hours']}{flag} |")
        total = sum(float(x["est_hours"]) for x in payload["sessions"])  # type: ignore[union-attr]
        n = payload["n_sessions"]
        print(f"\nTotal ~{total:.0f} GPU-h over {n} sessions "
              f"at {args.budget_hours} h each.")
        if over:
            need = -(-int(total) // int(args.budget_hours)) + 1
            print(
                f"\n[warn] session(s) {over} exceed the {args.budget_hours} h "
                f"budget. Their last job will be DEFERRED rather than started, "
                f"which is safe but leaves work over. Use --sessions {need} to "
                f"fit, or run the leftovers as an extra pass: a second "
                f"`bash run_kaggle_fox.sh <N>` on the same session picks up "
                f"exactly what was deferred."
            )
        print(f"\nOn Kaggle:  bash run_kaggle_fox.sh <N>   "
              f"(N = 1..{payload['n_sessions']})")
        return 0
    if cmd == "run":
        if args.tag:
            plan = json.loads(Path(args.plan).read_text())
            print(f"fox_{plan.get('round', 'r1')}_s{args.session}")
            return 0
        return run_session(
            args.session, args.plan, budget_hours=args.budget_hours,
            dry_run=args.dry_run, runs_root=args.runs_dir,
            pack=not args.no_pack,
        )
    if getattr(args, "json", False):
        print(json.dumps(scan_progress(runs_root=args.runs_dir), indent=2,
                         default=str))
        return 0
    print(render_plan(args.budget_hours, args.runs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
