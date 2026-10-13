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
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from foxrepro.storage import BENCHMARK_SPLITS, RunKey, paths

__all__ = [
    "Job", "PLAN", "measured_cost", "scan_progress", "next_jobs", "render_plan",
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
    logit_dtype: str = "native"
    gamma: Optional[float] = None
    k: Optional[float] = None
    tau_js: Optional[float] = None
    alpha: Optional[float] = None
    seed: int = 42
    limit: Optional[int] = None
    note: str = ""

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
            "prompt_template_id": (
                "pope_v1" if self.benchmark == "pope" else "chair_v1"
            ),
            "model_id": "llava-hf/llava-1.5-7b-hf",
            "decoding": self.decoding, "top_p": 0.9, "temperature": 1.0,
            "intervention_layers": None,
            "logit_dtype": self.logit_dtype, "softmax_dtype": "fp32",
            "mode": self.mode, "system_prompt_id": self.system_prompt_id,
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
                done = len(scan_run_file(path, key=key).sample_ids)
            except Exception:  # noqa: BLE001 - a corrupt file is "not done"
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


def _main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-hours", type=float, default=9.0,
                    help="Kaggle caps a session at 12 h; 9 leaves headroom")
    ap.add_argument("--runs-dir", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="report measured seconds-per-sample and stop")
    args = ap.parse_args(argv)

    if args.calibrate:
        print(json.dumps(measured_cost(args.runs_dir), indent=2))
        return 0
    if args.json:
        print(json.dumps(scan_progress(runs_root=args.runs_dir), indent=2,
                         default=str))
        return 0
    print(render_plan(args.budget_hours, args.runs_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
