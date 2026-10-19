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

from foxrepro.storage import BENCHMARK_SPLITS, RunKey, paths

__all__ = [
    "Job", "PLAN", "measured_cost", "scan_progress", "next_jobs", "render_plan",
    "partition_jobs", "ingest_runs", "archive_runs",
]

#: Seconds per sample, measured on 2x T4 with LLaVA-1.5-7B fp16 sharded.
#: POPE: 608-token prefill, 1 token out. The dual-branch figure is the
#: single-branch one doubled, because POPE is prefill-bound and the second
#: branch doubles the prefill compute.
COST_S: Dict[Tuple[str, str], float] = {
    ("pope", "single"): 1.4,      # 1.315 measured, +5% for the system prompt
    ("pope", "dual"): 2.8,
    ("chair", "single"): 6.0,     # ~100 decode tokens at ~48 ms/token
    ("chair", "dual"): 7.8,       # decode batches, so less than 2x
    ("diag", "single"): 1.4,      # one prefill, no generation
}

N_SAMPLES: Dict[Tuple[str, str], int] = {
    ("pope", "random"): 3000,
    ("pope", "popular"): 3000,
    ("pope", "adversarial"): 3000,
    ("chair", "val2014_500"): 500,
}

#: Methods that run two forward branches per token.
DUAL_BRANCH = frozenset({"fox", "vcd"})


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
    note: str = ""

    def template_id(self) -> str:
        """The faithful default unless the job overrides it.

        ``pope_v0_raw`` sends the POPE question verbatim, which is what the
        release does; ``pope_v1`` appends the "single word or phrase"
        instruction and is an ablation.
        """
        if self.prompt_template_id:
            return self.prompt_template_id
        return "pope_v0_raw" if self.benchmark == "pope" else "chair_v1"

    def job_id(self) -> str:
        """Stable, readable identifier used in the frozen plan and the
        calibration TSV. Derived from the fields that distinguish jobs, so it
        survives plan edits as long as the job itself is unchanged."""
        bits = [self.target, self.method, self.benchmark, self.split]
        if self.decoding != "sampling":
            bits.append(self.decoding)
        if self.method == "fox" and self.mode != "as_implemented":
            bits.append(self.mode)
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
        if self.limit:
            bits.append(f"n{self.limit}")
        return ":".join(bits)

    @property
    def n_samples(self) -> int:
        n = N_SAMPLES.get((self.benchmark, self.split), 0)
        return min(n, self.limit) if self.limit else n

    @property
    def branches(self) -> str:
        return "dual" if self.method in DUAL_BRANCH and self.mode != "off" else (
            "dual" if self.method in DUAL_BRANCH else "single"
        )

    def est_seconds(self) -> float:
        return self.n_samples * COST_S.get(
            (self.benchmark, self.branches), COST_S[("pope", "single")]
        )

    def run_key(self) -> RunKey:
        """Must match what ``generate.py`` would build, or progress is wrong."""
        variant = {
            "gamma": self.gamma, "k": self.k, "tau_js": self.tau_js,
            "alpha": self.alpha, "beta": None,
            "max_new_tokens": 1 if self.benchmark == "pope" else 512,
            "prompt_template_id": self.template_id(),
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "decoding": self.decoding, "top_p": 0.9, "temperature": 1.0,
            "intervention_layers": None,
            "logit_dtype": self.logit_dtype, "softmax_dtype": "fp32",
            "mode": self.mode, "system_prompt_id": self.system_prompt_id,
            "wrapper_id": self.wrapper_id,
        }
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
        if self.limit:
            parts.append(f"--limit {self.limit}")
        return " ".join(parts)


def _pope_splits() -> Sequence[str]:
    return BENCHMARK_SPLITS["pope"]


def _build_plan() -> List[Job]:
    jobs: List[Job] = []

    # --- Table 1 / 8: POPE accuracy and F1. Same runs feed both. ----------- #
    for method in ("sampling", "vcd", "fox"):
        for split in _pope_splits():
            jobs.append(Job("table1", method, "pope", split,
                            note="POPE accuracy (T1) and F1 (T8)"))

    # --- Table 2: CHAIR ---------------------------------------------------- #
    for method in ("sampling", "vcd", "fox"):
        jobs.append(Job("table2", method, "chair", "val2014_500",
                        note="CHAIR_S / CHAIR_I"))

    # --- Table 9: the greedy control --------------------------------------- #
    for method in ("sampling", "vcd", "fox"):
        for split in _pope_splits():
            jobs.append(Job("table9", method, "pope", split,
                            decoding="greedy", note="greedy POPE"))
        jobs.append(Job("table9", method, "chair", "val2014_500",
                        decoding="greedy", note="greedy CHAIR"))

    # --- The mechanism audit: this repo's own contribution ----------------- #
    # Gamma sweep. fp16 ULP at these magnitudes is 0.0625 / 0.5 / 8 / 32, a
    # 512x span, so a response that tracks ULP rather than gamma is the
    # signature of the requantization account. 1e5 overflows fp16 and is out.
    for gamma in (1e2, 1e3, 1e4, 6e4):
        jobs.append(Job("gamma_sweep", "fox", "pope", "adversarial",
                        gamma=gamma, note=f"gamma={gamma:g}"))
        jobs.append(Job("gamma_sweep", "fox", "chair", "val2014_500",
                        gamma=gamma, note=f"gamma={gamma:g}, finer metric"))

    # Control arms, on the hardest split and on CHAIR where the gain is largest.
    # boost_visual is ours, not the paper's: it runs the intervention the
    # paper argues from and never performs. Small gammas, because this is
    # an attention bias meant to be felt, not a precision artifact.
    for gamma in (2.0, 4.0, 8.0):
        jobs.append(Job("boost_visual", "fox", "chair", "val2014_500",
                        mode="boost_visual", gamma=gamma,
                        note=f"amplify visual attention, delta={gamma:g}"))
    jobs.append(Job("boost_visual", "fox", "pope", "adversarial",
                    mode="boost_visual", gamma=4.0,
                    note="amplify visual attention, delta=4"))

    for mode in ("off", "as_described", "ulp_control", "noise_control"):
        jobs.append(Job("controls", "fox", "pope", "adversarial", mode=mode,
                        note=f"mode={mode}"))
        jobs.append(Job("controls", "fox", "chair", "val2014_500", mode=mode,
                        note=f"mode={mode}"))

    # Logit storage dtype: the axis the mechanism actually lives on.
    for ld in ("fp32", "bf16"):
        jobs.append(Job("dtype_arms", "fox", "pope", "adversarial",
                        logit_dtype=ld, note=f"logit_dtype={ld}"))
        jobs.append(Job("dtype_arms", "fox", "chair", "val2014_500",
                        logit_dtype=ld, note=f"logit_dtype={ld}"))

    # Does the paper's shortcut survive without a system prompt to anchor it?
    for method in ("sampling", "fox"):
        jobs.append(Job("system_prompt", method, "pope", "adversarial",
                        system_prompt_id="none",
                        note="tests the X_sys anchor claim directly"))

    # --- Prompt ablation: the ways session 1's prompt differed from the
    # release. It ran with all of them present at once, which measured their
    # sum (POPE-adversarial baseline 81.90 against the paper's 76.77). These
    # isolate each one against the faithful baseline.
    for label, over in (
        ("newline", {"wrapper_id": "newline"}),
        ("single_word", {"prompt_template_id": "pope_v1"}),
        ("spaced_sysprompt", {"system_prompt_id": "llava_v1_spaced"}),
    ):
        jobs.append(Job("prompt_ablation", "sampling", "pope", "adversarial",
                        note=f"faithful + {label}", **over))

    # --- Table 7: alpha ablation ------------------------------------------- #
    for alpha in (1.0, 3.0, 4.0):   # 2.0 is the default, already covered
        jobs.append(Job("table7", "fox", "pope", "adversarial", alpha=alpha,
                        note=f"alpha={alpha:g}"))

    # --- Figure 9: hyperparameter sensitivity, 500 samples as in the paper -- #
    for k in (0.25, 0.45):          # 0.35 is the released default
        jobs.append(Job("fig9", "fox", "pope", "adversarial", k=k, limit=500,
                        note=f"k={k:g}, 500 samples"))
    for tau in (0.1, 0.2):          # 0.15 is the released default
        jobs.append(Job("fig9", "fox", "pope", "adversarial", tau_js=tau,
                        limit=500, note=f"tau_js={tau:g}, 500 samples"))

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
        "dest": str(dest), "copied": 0, "merged": 0, "skipped": 0, "files": []
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
            have = {r.get("sample_id") for r in _records(target)}
            incoming = [
                r for r in _records(src) if r.get("sample_id") not in have
            ]
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
                branches = (
                    "dual"
                    if rec.get("method") in DUAL_BRANCH and rec.get("mode") != "off"
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
    n_sessions: int, budget_hours: float, plan_path: Path,
    calibrate: Optional[Path] = None, round_tag: str = "r1",
    pilot: bool = False,
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
    groups = partition_jobs(n_sessions, jobs, pilot=pilot)
    payload: Dict[str, object] = {
        "round": round_tag,
        "n_sessions": n_sessions,
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
    a.add_argument("--sessions", type=int, required=True)
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
                              args.calibrate, args.round, pilot=args.pilot)
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
        print(f"\nTotal ~{total:.0f} GPU-h over {args.sessions} sessions "
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
              f"(N = 1..{args.sessions})")
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
