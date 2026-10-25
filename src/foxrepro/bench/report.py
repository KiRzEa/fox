"""Render result tables shaped like the paper's, with intervals attached.

Three tables, matching the paper:

* **Table 1** -- POPE accuracy (and F1) per split, under nucleus sampling.
* **Table 2** -- CHAIR_S / CHAIR_I.
* **Table 9** -- the greedy-decoding control.

Each cell carries a 95% interval, every metric carries its denominator, and a
gap against the baseline that the data cannot resolve is flagged with a dagger
(:data:`FLAG`). Where per-sample outcome files exist for both methods, the
comparison is **paired** (McNemar for POPE accuracy, cluster bootstrap for
CHAIR_I), which is tighter and correct for same-input runs; otherwise it falls
back to the independent Newcombe interval and says so.

The report also prints:

* a **resolution budget** -- the smallest gap each benchmark's n can resolve,
  and the n that would be needed for the gaps the paper reports;
* a **provenance audit** -- model id, commit, dtype, gamma, the resolved
  ``transformers`` path (a stock path means a Fox run may have silently done
  nothing) and the CHAIR synonym-list checksum.

Markdown only; no plotting dependencies.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


from foxrepro.bench.stats import (  # noqa: E402
    seed_level_comparison,
    Comparison, compare_proportions, mcnemar_exact, paired_bootstrap_proportion,
    paired_bootstrap_ratio, power_table, resolvable_gap,
)
from foxrepro.storage import paths  # noqa: E402

__all__ = ["FLAG", "load_results", "render_report", "PAPER_REFERENCE"]

#: Marker appended to a cell whose gap against the baseline is not resolvable.
FLAG = "†"

#: Values as printed in the paper for LLaVA-1.5, for the "paper" column.
#: Table 1 prints accuracy only; Table 9 prints accuracy and F1.
PAPER_REFERENCE: Dict[str, Dict[str, Dict[str, float]]] = {
    "table1_pope_accuracy": {
        "sampling": {"random": 85.13, "popular": 82.53, "adversarial": 76.77},
        "vcd": {"random": 86.84, "popular": 80.87, "adversarial": 75.33},
        "icd": {"random": 86.27, "popular": 83.57, "adversarial": 77.90},
        "opera": {"random": 89.27, "popular": 86.80, "adversarial": 81.13},
        "causalmm": {"random": 89.15, "popular": 86.63, "adversarial": 81.27},
        "sid": {"random": 88.67, "popular": 85.37, "adversarial": 81.10},
        "fox": {"random": 89.33, "popular": 86.70, "adversarial": 81.93},
    },
    "table2_chair": {
        "sampling": {"chair_s": 57.60, "chair_i": 17.18},
        "vcd": {"chair_s": 54.00, "chair_i": 16.02},
        "icd": {"chair_s": 53.40, "chair_i": 15.67},
        "opera": {"chair_s": 49.00, "chair_i": 13.52},
        "causalmm": {"chair_s": 47.60, "chair_i": 13.49},
        "sid": {"chair_s": 53.40, "chair_i": 15.40},
        "fox": {"chair_s": 46.40, "chair_i": 12.90},
    },
    "table9_greedy": {
        "sampling": {"random_acc": 89.33, "random_f1": 89.28, "popular_acc": 85.93,
                     "popular_f1": 86.33, "adversarial_acc": 79.10,
                     "adversarial_f1": 80.95, "chair_s": 50.90, "chair_i": 15.40},
        "vcd": {"random_acc": 87.97, "random_f1": 87.84, "popular_acc": 85.33,
                "popular_f1": 85.67, "adversarial_acc": 78.43,
                "adversarial_f1": 80.26, "chair_s": 54.60, "chair_i": 16.60},
        "opera": {"random_acc": 89.27, "random_f1": 88.72, "popular_acc": 86.80,
                  "popular_f1": 86.59, "adversarial_acc": 81.13,
                  "adversarial_f1": 81.87, "chair_s": 49.00, "chair_i": 13.52},
        "sid": {"random_acc": 89.33, "random_f1": 89.10, "popular_acc": 86.47,
                "popular_f1": 86.25, "adversarial_acc": 81.77,
                "adversarial_f1": 82.31, "chair_s": 48.00, "chair_i": 13.60},
        "fox": {"random_acc": 89.43, "random_f1": 88.81, "popular_acc": 87.07,
                "popular_f1": 86.60, "adversarial_acc": 82.43,
                "adversarial_f1": 82.64, "chair_s": 42.60, "chair_i": 12.16},
    },
}

POPE_SPLIT_ORDER = ("random", "popular", "adversarial")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """One evaluated run: its metrics document plus per-sample outcomes."""

    metrics_path: Path
    doc: Dict[str, Any]
    outcomes: List[Dict[str, Any]]

    @property
    def prov(self) -> Dict[str, Any]:
        return self.doc["provenance"]

    @property
    def met(self) -> Dict[str, Any]:
        return self.doc["metrics"]

    @property
    def method(self) -> str:
        return str(self.prov["method"])

    @property
    def benchmark(self) -> str:
        return str(self.prov["benchmark"])

    @property
    def split(self) -> str:
        return str(self.prov["split"])

    @property
    def decoding(self) -> str:
        return str(self.prov.get("decoding") or "sampling")

    @property
    def seed(self) -> int:
        return int(self.prov["seed"])

    @property
    def n_samples(self) -> int:
        """How many items this run covers.

        Part of the grouping key, because a run over 50 samples is not a
        replicate of one over 500 -- averaging the two would silently mix a
        short probe into a headline arm. The determinism jobs (50 samples,
        seed 997) and the fig9 sweeps (500 POPE questions against the usual
        3000) are both this shape.
        """
        m = self.met
        if self.benchmark == "chair":
            return int(m["totals"]["chair_s_denominator"])
        return int(m["n_samples"])

    @property
    def arm(self) -> str:
        """The approach, not just the method name.

        Every control is ``method="fox"`` with a different ``mode``, and the
        dtype and gamma arms differ only in a provenance field. Keying on
        ``method`` alone would collapse all of them into one bucket, so each
        comparison needs the whole arm identity. The label is built to read
        like the flags that produced it.
        """
        bits = [self.method]
        mode = self.prov.get("mode")
        if self.method == "fox" and mode and mode != "as_implemented":
            bits.append(str(mode))
        ld = self.prov.get("logit_dtype")
        if ld and ld != "native":
            bits.append(f"ld={ld}")
        sd = self.prov.get("softmax_dtype")
        if sd and sd != "fp32":
            bits.append(f"sm={sd}")
        for name, key in (("g", "gamma"), ("k", "k"), ("tau", "tau_js"),
                          ("alpha", "alpha"), ("beta", "beta")):
            v = self.prov.get(key)
            if v is not None:
                bits.append(f"{name}={float(v):g}")
        return " ".join(bits)

    @property
    def prompt_arm(self) -> Dict[str, str]:
        """The prompt fields that make two runs of one job incomparable.

        Session 1 was run twice -- once before the released prompt was read
        and once after -- so the archive legitimately holds two results for
        the same (benchmark, split, method). They differ only here.
        """
        return {
            "system_prompt_id": str(self.prov.get("system_prompt_id") or "?"),
            "wrapper_id": str(self.prov.get("wrapper_id") or "?"),
            "prompt_template_id": str(self.prov.get("prompt_template_id") or "?"),
        }

    def arm_label(self) -> str:
        a = self.prompt_arm
        return (f"{a['system_prompt_id']}/{a['wrapper_id']}"
                f"/{a['prompt_template_id']}")


def load_results(
    reports_dir: Path,
    decoding: Optional[str] = None,
    dtype: Optional[str] = None,
    seed: Optional[int] = None,
    system_prompt_id: Optional[str] = None,
    wrapper_id: Optional[str] = None,
    prompt_template_ids: Optional[Sequence[str]] = None,
) -> List[RunResult]:
    """Load every ``*.metrics.json`` in ``reports_dir`` (plus its outcomes).

    Args:
        reports_dir: where ``evaluate.py`` wrote its output.
        decoding: keep only this decoding mode (``"sampling"`` / ``"greedy"``).
        dtype: keep only this dtype.
        seed: keep only this seed.
        system_prompt_id: keep only this system-prompt arm.
        wrapper_id: keep only this turn wrapper.
        prompt_template_ids: keep only these prompt templates. A sequence
            because one report spans both benchmarks, which use different
            templates -- pass the faithful id for each.

    Returns:
        Matching :class:`RunResult` objects.
    """
    out: List[RunResult] = []
    for path in sorted(Path(reports_dir).glob("*.metrics.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        prov = doc["provenance"]
        if decoding and str(prov.get("decoding") or "sampling") != decoding:
            continue
        if dtype and str(prov.get("dtype")) != dtype:
            continue
        if seed is not None and int(prov.get("seed", -1)) != seed:
            continue
        if (system_prompt_id is not None
                and str(prov.get("system_prompt_id") or "?") != system_prompt_id):
            continue
        if (wrapper_id is not None
                and str(prov.get("wrapper_id") or "?") != wrapper_id):
            continue
        if (prompt_template_ids is not None
                and str(prov.get("prompt_template_id") or "?")
                not in prompt_template_ids):
            continue
        outcomes_path = Path(str(path).replace(".metrics.json", ".outcomes.jsonl"))
        rows: List[Dict[str, Any]] = []
        if outcomes_path.is_file():
            rows = [
                json.loads(line)
                for line in outcomes_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        out.append(RunResult(path, doc, rows))
    return out


#: Metrics the across-seed section analyses, per benchmark:
#: ``(key in metrics, label, higher_is_better)``.
SEED_METRICS: Dict[str, Tuple[Tuple[str, str, bool], ...]] = {
    "pope": (("accuracy", "accuracy", True), ("f1", "F1", True)),
    "chair": (("chair_s", "CHAIR_S", False), ("chair_i", "CHAIR_I", False)),
}


def _metric_point(res: "RunResult", key: str) -> float:
    """The point estimate, in percentage points."""
    if res.benchmark == "chair":
        return 100.0 * float(res.met["totals"][key])
    return 100.0 * float(res.met[key]["point"])


def group_by_seed(
    results: Sequence[RunResult],
) -> Dict[Tuple[str, str, str, str], Dict[int, RunResult]]:
    """``(benchmark, split, decoding, arm) -> {seed: result}``.

    The seed is a first-class axis here rather than a duplicate to be
    rejected, because replicating a run across seeds is the only way to see
    the generation variance at all.

    The fourth element is :attr:`RunResult.arm`, not the method: ``fox
    --mode off`` and ``fox --mode ulp_control`` are different approaches that
    share a method name, and keying on the name would merge them.
    """
    out: Dict[Tuple[str, str, str, str], Dict[int, RunResult]] = {}
    sizes: Dict[Tuple[str, str, str, str], int] = {}
    for r in results:
        key = (r.benchmark, r.split, r.decoding, r.arm)
        # A 50-sample probe is not a replicate of a 500-sample run. Keeping
        # the largest size per arm drops the short ones rather than averaging
        # them in, which is what pooled the seed-997 determinism job into the
        # headline greedy arm and moved its CHAIR_S by 4 points.
        n = r.n_samples
        if key not in sizes or n > sizes[key]:
            sizes[key] = n
            out[key] = {}
        elif n < sizes[key]:
            continue
        bucket = out[key]
        if r.seed in bucket:
            raise ValueError(
                f"two metrics files for {key} seed {r.seed}: "
                f"{bucket[r.seed].metrics_path.name} and {r.metrics_path.name}"
            )
        bucket[r.seed] = r
    return out


def reference_seed(results: Sequence[RunResult]) -> Optional[int]:
    """The seed the per-run tables are built from: the lowest present.

    Those tables show one run each, with the within-run interval. Picking the
    lowest seed makes the choice deterministic and reproducible rather than
    dependent on filesystem order, and it is stated in the report. The
    across-seed section is where the statistical claim is made.
    """
    seeds = {r.seed for r in results}
    return min(seeds) if seeds else None


def bare_arms(results: Sequence[RunResult]) -> List[RunResult]:
    """Only the arms the paper also prints: a method with no modifiers.

    The paper-comparison tables put our cell next to theirs, so they may hold
    only arms that have a counterpart there. ``fox --mode ulp_control`` and
    ``fox --logit-dtype bf16`` are this repo's own, and they appear in the
    across-seed section instead, where each gets its own row and interval.
    """
    return [r for r in results if " " not in r.arm]


def _index(results: Sequence[RunResult]) -> Dict[Tuple[str, str, str], RunResult]:
    """``(benchmark, split, arm) -> result``; later duplicates win loudly."""
    index: Dict[Tuple[str, str, str], RunResult] = {}
    for r in results:
        key = (r.benchmark, r.split, r.arm)
        if key in index:
            prev = index[key]
            differing = {
                f: (prev.prov.get(f), r.prov.get(f))
                for f in sorted(set(prev.prov) | set(r.prov))
                if prev.prov.get(f) != r.prov.get(f)
            }
            # Reporting "pass --dtype" when the runs differ by prompt sends the
            # reader after a filter that cannot separate them. Name the fields.
            detail = "; ".join(f"{f}: {a!r} vs {b!r}"
                               for f, (a, b) in differing.items()) or "nothing"
            raise ValueError(
                f"two metrics files for {key}: {prev.metrics_path.name} "
                f"and {r.metrics_path.name}. They differ in {detail}. "
                "Filter with --system-prompt / --wrapper / --prompt-template, "
                "or --dtype / --seed / --decoding, or move the stale one out "
                "of the reports directory."
            )
        index[key] = r
    return index


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #
def _pope_accuracy_comparison(a: RunResult, b: RunResult) -> Comparison:
    """Compare two POPE runs, paired when both have per-sample outcomes."""
    if a.outcomes and b.outcomes:
        by_a = {r["sample_id"]: int(r["correct"]) for r in a.outcomes}
        by_b = {r["sample_id"]: int(r["correct"]) for r in b.outcomes}
        shared = sorted(set(by_a) & set(by_b))
        if len(shared) >= 30:
            return mcnemar_exact(
                [by_a[s] for s in shared], [by_b[s] for s in shared],
                label_a=a.method, label_b=b.method,
            )
    acc_a, acc_b = a.met["accuracy"], b.met["accuracy"]
    return compare_proportions(
        int(round(acc_a["successes"])), int(acc_a["n"]),
        int(round(acc_b["successes"])), int(acc_b["n"]),
        label_a=a.method, label_b=b.method,
    )


def _chair_s_comparison(a: RunResult, b: RunResult) -> Comparison:
    """Compare CHAIR_S, paired over the same images when possible."""
    if a.outcomes and b.outcomes:
        by_a = {r["sample_id"]: int(bool(r["has_hallucination"])) for r in a.outcomes}
        by_b = {r["sample_id"]: int(bool(r["has_hallucination"])) for r in b.outcomes}
        shared = sorted(set(by_a) & set(by_b))
        if len(shared) >= 30:
            return paired_bootstrap_proportion(
                [by_a[s] for s in shared], [by_b[s] for s in shared],
                label_a=a.method, label_b=b.method,
            )
    ta, tb = a.met["totals"], b.met["totals"]
    return compare_proportions(
        int(ta["chair_s_numerator"]), int(ta["chair_s_denominator"]),
        int(tb["chair_s_numerator"]), int(tb["chair_s_denominator"]),
        label_a=a.method, label_b=b.method,
    )


def _chair_i_comparison(a: RunResult, b: RunResult) -> Comparison:
    """Compare CHAIR_I; cluster bootstrap when per-caption counts are available."""
    if a.outcomes and b.outcomes:
        by_a = {r["sample_id"]: (r["n_hallucinated"], r["n_mentions"]) for r in a.outcomes}
        by_b = {r["sample_id"]: (r["n_hallucinated"], r["n_mentions"]) for r in b.outcomes}
        shared = sorted(set(by_a) & set(by_b))
        if len(shared) >= 30:
            return paired_bootstrap_ratio(
                [by_a[s][0] for s in shared], [by_a[s][1] for s in shared],
                [by_b[s][0] for s in shared], [by_b[s][1] for s in shared],
                label_a=a.method, label_b=b.method,
            )
    ta, tb = a.met["totals"], b.met["totals"]
    return compare_proportions(
        int(ta["chair_i_numerator"]), int(ta["chair_i_denominator"]),
        int(tb["chair_i_numerator"]), int(tb["chair_i_denominator"]),
        label_a=a.method, label_b=b.method,
    )


def _cell(point: float, lo: float, hi: float, flagged: bool) -> str:
    """Format one table cell as ``value [lo, hi]`` in percentage points."""
    if point != point:  # nan
        return "n/a"
    text = f"{100 * point:.2f} [{100 * lo:.2f}, {100 * hi:.2f}]"
    return text + (f" {FLAG}" if flagged else "")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def render_pope_table(
    results: Sequence[RunResult], baseline: str, title: str, paper_key: Optional[str]
) -> str:
    """Render the POPE table (paper Table 1 / the POPE half of Table 9)."""
    index = _index(bare_arms([r for r in results if r.benchmark == "pope"]))
    methods = sorted({m for (_, _, m) in index})
    if not methods:
        return f"### {title}\n\n_no POPE runs found._\n"
    splits = [s for s in POPE_SPLIT_ORDER if any((("pope", s, m) in index) for m in methods)]

    header = ["method"]
    for s in splits:
        header += [f"{s} Acc", f"{s} F1", f"{s} yes-ratio", f"{s} n"]
    lines = ["### " + title, "", "| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    comparisons: List[str] = []

    for method in methods:
        row = [f"`{method}`"]
        for split in splits:
            res = index.get(("pope", split, method))
            if res is None:
                row += ["-", "-", "-", "-"]
                continue
            acc = res.met["accuracy"]
            f1 = res.met["f1"]
            yes = res.met["yes_ratio"]
            flagged = False
            base = index.get(("pope", split, baseline))
            if base is not None and method != baseline:
                cmp_ = _pope_accuracy_comparison(res, base)
                flagged = not cmp_.significant
                comparisons.append(f"- POPE/{split}: {cmp_.verdict()}  _[{cmp_.method}]_")
            row.append(_cell(acc["point"], acc["ci_lo"], acc["ci_hi"], flagged))
            row.append(_cell(f1["point"], f1["ci_lo"], f1["ci_hi"], False))
            row.append(f"{100 * yes['point']:.1f}")
            row.append(str(acc["n"]))
        lines.append("| " + " | ".join(row) + " |")

    if paper_key:
        lines += ["", "Paper-reported accuracy (LLaVA-1.5), for reference:", "",
                  "| method | " + " | ".join(splits) + " |",
                  "| " + " | ".join("---" for _ in range(len(splits) + 1)) + " |"]
        ref = PAPER_REFERENCE[paper_key]
        for method, values in ref.items():
            cells = [
                f"{values.get(s, values.get(s + '_acc', float('nan'))):.2f}"
                for s in splits
            ]
            lines.append(f"| `{method}` | " + " | ".join(cells) + " |")

    if comparisons:
        lines += ["", f"**Gaps against `{baseline}`** ({FLAG} = not resolvable at 95%):", ""]
        lines += comparisons
    return "\n".join(lines) + "\n"


def render_chair_table(
    results: Sequence[RunResult], baseline: str, title: str, paper_key: Optional[str]
) -> str:
    """Render the CHAIR table (paper Table 2 / the CHAIR half of Table 9)."""
    rows = [r for r in results if r.benchmark == "chair"]
    index = _index(rows)
    methods = sorted({m for (_, _, m) in index})
    if not methods:
        return f"### {title}\n\n_no CHAIR runs found._\n"

    header = ["method", "CHAIR_S (lower better)", "CHAIR_S n (captions)",
              "CHAIR_I (lower better)", "CHAIR_I n (mentions)",
              "recall", "F1", "mean words"]
    lines = ["### " + title, "", "| " + " | ".join(header) + " |",
             "| " + " | ".join("---" for _ in header) + " |"]
    comparisons: List[str] = []

    for (bench, split, method), res in sorted(index.items()):
        totals = res.met["totals"]
        cs = res.met["chair_s_ci"]
        ci = res.met["chair_i_ci_cluster_bootstrap"]
        flag_s = flag_i = False
        base = index.get(("chair", split, baseline))
        if base is not None and method != baseline:
            cmp_s = _chair_s_comparison(res, base)
            cmp_i = _chair_i_comparison(res, base)
            flag_s, flag_i = not cmp_s.significant, not cmp_i.significant
            comparisons.append(f"- CHAIR_S: {cmp_s.verdict()}  _[{cmp_s.method}]_")
            comparisons.append(f"- CHAIR_I: {cmp_i.verdict()}  _[{cmp_i.method}]_")
        lines.append("| " + " | ".join([
            f"`{method}`",
            _cell(cs["point"], cs["ci_lo"], cs["ci_hi"], flag_s),
            str(totals["chair_s_denominator"]),
            _cell(ci["point"], ci["ci_lo"], ci["ci_hi"], flag_i),
            str(totals["chair_i_denominator"]),
            f"{100 * totals['recall']:.2f}",
            f"{100 * totals['f1']:.2f}",
            f"{totals['mean_caption_words']:.1f}",
        ]) + " |")

    if paper_key:
        ref = PAPER_REFERENCE[paper_key]
        lines += ["", "Paper-reported CHAIR (LLaVA-1.5), for reference:", "",
                  "| method | CHAIR_S | CHAIR_I |", "| --- | --- | --- |"]
        for method, values in ref.items():
            lines.append(
                f"| `{method}` | {values['chair_s']:.2f} | {values['chair_i']:.2f} |"
            )
        lines.append("")
        lines.append(
            "_The paper states neither CHAIR_I's denominator (total mentioned "
            "object instances) nor any interval, so the reference column cannot "
            "be given error bars._"
        )

    if comparisons:
        lines += ["", f"**Gaps against `{baseline}`** ({FLAG} = not resolvable at 95%):", ""]
        lines += comparisons
    return "\n".join(lines) + "\n"


def render_resolution_budget(results: Sequence[RunResult]) -> str:
    """What the sample sizes in use can and cannot resolve."""
    lines = ["### Resolution budget", "",
             "Smallest gap each run can resolve (two-sided 95%, 80% power):", "",
             "| benchmark / split | metric | n | p | smallest resolvable gap (pts) |",
             "| --- | --- | --- | --- | --- |"]
    seen: set[Tuple[str, str, int]] = set()
    for res in results:
        if res.benchmark == "pope":
            acc = res.met["accuracy"]
            entries = [("accuracy", int(acc["n"]), float(acc["point"]))]
        else:
            totals = res.met["totals"]
            entries = [
                ("CHAIR_S", int(totals["chair_s_denominator"]), float(totals["chair_s"])),
                ("CHAIR_I", int(totals["chair_i_denominator"]), float(totals["chair_i"])),
            ]
        for metric, n, p in entries:
            key = (f"{res.benchmark}/{res.split}", metric, n)
            if key in seen or n == 0 or p != p:
                continue
            seen.add(key)
            if p <= 0.0 or p >= 1.0:
                cell = "n/a (degenerate p)"
            else:
                cell = f"{100 * resolvable_gap(n, p):.2f}"
            lines.append(
                f"| {res.benchmark}/{res.split} | {metric} | {n} | {p:.4f} | {cell} |"
            )

    lines += ["", "Samples **per method** needed for a given gap at p = 0.82 "
              "(`paired_rho` is the correlation between two methods' per-sample "
              "outcomes; paired runs on identical inputs are strongly correlated):",
              "", "| gap (pts) | rho = 0 | rho = 0.5 | rho = 0.8 |",
              "| --- | --- | --- | --- |"]
    rows = power_table(0.82, gaps=(0.005, 0.01, 0.02, 0.05), paired_rho=(0.0, 0.5, 0.8))
    by_gap: Dict[float, Dict[float, int]] = {}
    for row in rows:
        by_gap.setdefault(row["gap"], {})[row["paired_rho"]] = int(row["required_n_per_method"])
    for gap, cells in by_gap.items():
        lines.append(
            f"| {100 * gap:.1f} | {cells[0.0]:,} | {cells[0.5]:,} | {cells[0.8]:,} |"
        )
    return "\n".join(lines) + "\n"


def render_provenance_audit(results: Sequence[RunResult]) -> str:
    """Per-run provenance, with the checks that catch a silently wrong run."""
    lines = ["### Provenance audit", "",
             "| run | method | bench/split | dtype | gamma | k | tau_js | seed | "
             "n | model | commit | transformers |",
             "| " + " | ".join("---" for _ in range(12)) + " |"]
    warnings: List[str] = []
    for res in sorted(results, key=lambda r: r.metrics_path.name):
        prov = res.prov
        tpath = str(prov.get("transformers_path", "?"))
        short = tpath if len(tpath) < 48 else "..." + tpath[-45:]
        lines.append("| " + " | ".join([
            f"`{res.metrics_path.name.replace('.metrics.json', '')}`",
            f"`{res.method}`", f"{res.benchmark}/{res.split}",
            str(prov.get("dtype")), str(prov.get("gamma")), str(prov.get("k")),
            str(prov.get("tau_js")), str(prov.get("seed")),
            str(prov.get("n_records")), f"`{prov.get('model_id')}`",
            f"`{prov.get('commit_hash')}`", f"`{short}`",
        ]) + " |")
        if res.method == "fox" and prov.get("gamma") is None:
            warnings.append(f"- `{res.method}` run with **gamma = None**: the "
                            "intervention constant was not recorded.")
        if res.method in ("fox", "vcd", "sid") and "site-packages/transformers" in tpath:
            warnings.append(
                f"- `{res.metrics_path.name}`: transformers resolves to "
                f"`{tpath}`, i.e. **stock HuggingFace**, not the vendored "
                "4.29.2 the method patches. The run may have silently done nothing."
            )
        if str(prov.get("commit_hash", "")).endswith("-dirty"):
            warnings.append(f"- `{res.metrics_path.name}`: generated from a dirty "
                            "working tree; the commit hash is not reproducible.")
        if int(prov.get("n_duplicate_sample_ids", 0) or 0):
            warnings.append(f"- `{res.metrics_path.name}`: "
                            f"{prov['n_duplicate_sample_ids']} duplicate sample_ids "
                            "were skipped when reading the run.")
    chair_cfgs = {
        json.dumps(r.met["scorer"], sort_keys=True)
        for r in results if r.benchmark == "chair"
    }
    if len(chair_cfgs) > 1:
        warnings.append("- CHAIR runs were scored with **different scorer "
                        "configurations**; their CHAIR numbers are not comparable.")
    for r in results:
        if r.benchmark == "chair":
            cfg = r.met["scorer"]
            lines += ["", f"CHAIR scorer: synonyms `{Path(cfg['synonyms_path']).name}` "
                      f"sha256 `{cfg['synonyms_sha256'][:12]}...` "
                      f"({cfg['synonyms_provenance']}), tokenizer `{cfg['tokenizer']}`, "
                      f"gt_source `{cfg['gt_source']}`"]
            break
    if warnings:
        lines += ["", "**Warnings**", ""] + sorted(set(warnings))
    return "\n".join(lines) + "\n"


#: ``(benchmark, decoding) -> (PAPER_REFERENCE key, metric -> field)``.
#: The paper's gap is computed from its own printed numbers, fox minus
#: baseline, so nothing is typed in twice.
_PAPER_FIELDS: Dict[Tuple[str, str], Tuple[str, Dict[str, str]]] = {
    ("pope", "sampling"): ("table1_pope_accuracy", {"accuracy": "{split}"}),
    ("chair", "sampling"): ("table2_chair", {"chair_s": "chair_s",
                                             "chair_i": "chair_i"}),
    ("pope", "greedy"): ("table9_greedy", {"accuracy": "{split}_acc",
                                           "f1": "{split}_f1"}),
    ("chair", "greedy"): ("table9_greedy", {"chair_s": "chair_s",
                                            "chair_i": "chair_i"}),
}


def paper_gap(benchmark: str, split: str, decoding: str, metric: str,
              baseline: str, method: str) -> Optional[float]:
    """The paper's own gap for this cell, or None if it does not print one."""
    entry = _PAPER_FIELDS.get((benchmark, decoding))
    if entry is None:
        return None
    table_key, fields = entry
    field_tmpl = fields.get(metric)
    if field_tmpl is None:
        return None
    field = field_tmpl.format(split=split)
    ref = PAPER_REFERENCE.get(table_key, {})
    try:
        return float(ref[method][field]) - float(ref[baseline][field])
    except (KeyError, TypeError):
        return None


def render_seed_section(results: Sequence[RunResult], baseline: str,
                        confidence: float = 0.95) -> str:
    """The across-seed analysis: the report's actual statistical claim.

    Every interval in the per-run tables is computed from one generation. It
    quantifies which items were hard and nothing else. With
    ``do_sample=True`` the gap also moves with the draw, and that component is
    invisible to any single run, so a one-seed interval is too narrow by an
    unknown amount. This section pairs the two methods seed by seed and puts a
    Student-t interval on the mean of those paired gaps.

    Two comparisons against the paper are printed, because the paper's
    aggregation is ambiguous. C.2 says "the best results averaged over 10
    independent runs" without saying what "best" ranges over:

    * if it ranges over configurations, the comparable figure here is the
      **mean gap**;
    * if it ranges over the ten runs, the comparable figure is the
      **best-of gap**, which flatters the method and is reported separately
      rather than silently.

    Args:
        results: evaluated runs, one prompt arm, any number of seeds.
        baseline: the method every gap is measured against.
        confidence: coverage for the t-interval.

    Returns:
        A Markdown section.
    """
    grouped = group_by_seed(results)
    lines = [
        "### Across seeds",
        "",
        "`mean gap` pairs the two methods seed by seed and puts a Student-t "
        "interval on the mean of those paired differences. That is the only "
        "interval here that sees the generation draw; every interval in the "
        "tables above is computed from a single run and quantifies item "
        "difficulty alone.",
        "",
        "`best gap` is the best seed of each method differenced. It is "
        "reported because the paper's aggregation is ambiguous -- C.2 says "
        "\"the best results averaged over 10 independent runs\" without "
        "saying what \"best\" ranges over. If it ranges over runs, `best "
        "gap` is the comparable figure and `mean gap` is not.",
        "",
    ]

    header = ["artefact", "method", "metric", "seeds",
              f"{baseline} mean+-sd", "method mean+-sd",
              "mean gap [95% CI]", "p", "best gap", "paper gap", "verdict"]
    lines += ["| " + " | ".join(header) + " |",
              "| " + " | ".join("---" for _ in header) + " |"]

    any_row = False
    for (bench, split, decoding, arm) in sorted(grouped):
        if arm == baseline:
            continue
        base_runs = grouped.get((bench, split, decoding, baseline))
        if not base_runs:
            continue
        runs = grouped[(bench, split, decoding, arm)]
        for key, label, higher in SEED_METRICS.get(bench, ()):
            try:
                cmp_ = seed_level_comparison(
                    {sd: _metric_point(r, key) for sd, r in base_runs.items()},
                    {sd: _metric_point(r, key) for sd, r in runs.items()},
                    label_a=baseline, label_b=arm,
                    confidence=confidence, higher_is_better=higher,
                )
            except ValueError:
                continue
            any_row = True
            # Only a bare method appears in the paper. "fox off" and
            # "fox ld=bf16" are this repo's own arms, so there is no paper
            # value to compare them to and the column is left empty rather
            # than filled with fox's.
            pg = (paper_gap(bench, split, decoding, key, baseline, arm)
                  if " " not in arm else None)
            if cmp_.n_seeds < 2:
                pval = "-"
                if decoding == "greedy":
                    # Nothing to replicate: argmax is deterministic, so the
                    # per-run interval above is the whole uncertainty and one
                    # run is the complete answer rather than a missing one.
                    ci = "n/a -- deterministic"
                    verdict = "see table above"
                else:
                    ci = "**one seed -- no interval**"
                    verdict = "**not assessable**"
            else:
                ci = (f"{cmp_.mean_gap:+.2f} "
                      f"[{cmp_.lo:+.2f}, {cmp_.hi:+.2f}]")
                pval = f"{cmp_.p_value:.2g}"
                if pg is None:
                    verdict = ("resolved" if cmp_.significant
                               else "not resolved")
                elif not cmp_.significant:
                    verdict = "**no effect resolved**"
                elif cmp_.covers(pg):
                    verdict = "reproduces"
                else:
                    verdict = "**excludes paper**"
            lines.append("| " + " | ".join([
                f"{bench}/{split} ({decoding})",
                f"`{arm}`",
                label,
                str(cmp_.n_seeds),
                f"{cmp_.mean_a:.2f}+-{cmp_.sd_a:.2f}",
                f"{cmp_.mean_b:.2f}+-{cmp_.sd_b:.2f}",
                ci,
                pval,
                f"{cmp_.best_gap:+.2f}",
                f"{pg:+.2f}" if pg is not None else "-",
                verdict,
            ]) + " |")

    if not any_row:
        return ("### Across seeds\n\n_no method has a baseline on the same "
                "(benchmark, split, decoding) to pair against._\n")
    lines += [
        "",
        "Greedy rows are expected to show `sd = 0`: argmax is deterministic, "
        "so replicating a seed changes nothing and the paper's Table 9 is "
        "single-run too (\"we omit averaged scores\", C.7). Their intervals "
        "come from the per-run tables above, where item variance is the only "
        "variance there is.",
    ]
    return "\n".join(lines) + "\n"


def _arm_note(system_prompt_id: Optional[str], wrapper_id: Optional[str],
              prompt_template_ids: Optional[Sequence[str]]) -> Optional[str]:
    """One line saying which prompt arm the tables below are built from.

    A report that silently tabulated whichever of two archived arms happened
    to load first would be worse than one that refused to build, so the arm
    is stated in the output rather than left to the reader's assumption.
    """
    bits = []
    if system_prompt_id:
        bits.append(f"`--system-prompt {system_prompt_id}`")
    if wrapper_id:
        bits.append(f"`--wrapper {wrapper_id}`")
    if prompt_template_ids:
        bits.append("`" + "`, `".join(sorted(prompt_template_ids)) + "`")
    if not bits:
        return None
    return "Prompt arm: " + ", ".join(bits) + "."


def render_report(
    results: Sequence[RunResult], baseline: str = "sampling",
    include_paper: bool = True, arm_note: Optional[str] = None,
) -> str:
    """Compose the whole Markdown report.

    Args:
        results: evaluated runs, already filtered to one prompt arm.
        baseline: the method every gap is measured against.
        include_paper: add the paper's printed values for comparison.
        arm_note: one line naming the arm, from :func:`_arm_note`.
    """
    # The per-run tables show one generation each, so they must show the
    # same one for every method or their gaps mix seeds. The across-seed
    # section is where several generations are actually combined.
    ref_seed = reference_seed(results)
    one_seed = [r for r in results if r.seed == ref_seed]
    sampling = [r for r in one_seed if r.decoding == "sampling"]
    greedy = [r for r in one_seed if r.decoding == "greedy"]
    n_seeds = len({r.seed for r in results})
    parts = [
        "# Fox reproduction -- results",
        "",
        "All cells are `value [95% CI]` in percentage points. "
        f"`{FLAG}` marks a gap against the baseline that the sample size "
        "cannot resolve at 95% confidence: the two methods are "
        "indistinguishable on this evidence, whatever the point estimates say.",
        "",
    ]
    if arm_note:
        parts.extend([arm_note, ""])
    if n_seeds > 1:
        parts.extend([
            f"The tables below are **seed {ref_seed} alone** "
            f"(of {n_seeds} present), because a single run is what their "
            "within-run intervals describe. The claim this report actually "
            "makes is in **Across seeds**, further down.",
            "",
        ])
    if sampling:
        parts.append(render_pope_table(
            sampling, baseline, "Table 1 -- POPE (nucleus sampling, p=0.9, T=1.0)",
            "table1_pope_accuracy" if include_paper else None))
        parts.append("")
        parts.append(render_chair_table(
            sampling, baseline, "Table 2 -- CHAIR (500 COCO val2014 images)",
            "table2_chair" if include_paper else None))
        parts.append("")
    if greedy:
        parts.append(render_pope_table(
            greedy, baseline, "Table 9a -- POPE under greedy decoding (control)",
            "table9_greedy" if include_paper else None))
        parts.append("")
        parts.append(render_chair_table(
            greedy, baseline, "Table 9b -- CHAIR under greedy decoding (control)",
            "table9_greedy" if include_paper else None))
        parts.append("")
    parts.append(render_seed_section(results, baseline))
    parts.append("")
    parts.append(render_resolution_budget(results))
    parts.append("")
    parts.append(render_provenance_audit(results))
    return "\n".join(parts)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(
        description="Render Markdown result tables with confidence intervals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--reports-dir", type=Path, default=p.reports_dir)
    ap.add_argument("--out", type=Path, default=None, help="default: stdout")
    ap.add_argument("--baseline", default="sampling")
    ap.add_argument("--decoding", choices=("sampling", "greedy", "all"), default="all")
    ap.add_argument("--dtype", default=None, help="restrict to one dtype")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-paper-reference", action="store_true")
    ap.add_argument(
        "--prompt-arm", choices=("faithful", "any"), default="faithful",
        help="'faithful' (default) keeps only the runs whose prompt is "
             "byte-identical to the release, which are the only ones "
             "comparable to the paper. 'any' keeps everything and will "
             "refuse to build a table if two runs of one job survive -- use "
             "it with the explicit filters below to report a specific arm.")
    ap.add_argument("--system-prompt", default=None,
                    help="restrict to one system-prompt arm (e.g. none)")
    ap.add_argument("--wrapper", default=None,
                    help="restrict to one turn wrapper (e.g. newline)")
    ap.add_argument("--prompt-template", action="append", default=None,
                    metavar="ID",
                    help="restrict to these prompt templates; repeatable, "
                         "since one report spans both benchmarks")
    args = ap.parse_args(argv)

    sysp, wrap, tmpl = args.system_prompt, args.wrapper, args.prompt_template
    if args.prompt_arm == "faithful":
        # The defaults in bench.methods and data.loaders, which DEVIATIONS
        # 3.5 shows are the released ones. Hardcoding them here would drift
        # the moment either default is corrected again.
        from foxrepro.bench.methods import (
            DEFAULT_SYSTEM_PROMPT_ID, DEFAULT_WRAPPER_ID,
        )
        from foxrepro.data.loaders import DEFAULT_TEMPLATE_IDS

        sysp = sysp or DEFAULT_SYSTEM_PROMPT_ID
        wrap = wrap or DEFAULT_WRAPPER_ID
        tmpl = tmpl or sorted(DEFAULT_TEMPLATE_IDS.values())

    results = load_results(
        args.reports_dir,
        decoding=None if args.decoding == "all" else args.decoding,
        dtype=args.dtype, seed=args.seed,
        system_prompt_id=sysp, wrapper_id=wrap, prompt_template_ids=tmpl,
    )
    if not results:
        # "run evaluate.py first" is wrong, and wastes the reader's time, when
        # the files are there and the filter removed them all.
        present = load_results(args.reports_dir)
        if not present:
            print(f"no metrics files in {args.reports_dir}; "
                  "run evaluate.py first", file=sys.stderr)
            return 1
        arms = sorted({r.arm_label() for r in present})
        print(
            f"{len(present)} metrics file(s) in {args.reports_dir}, but none "
            f"match the requested prompt arm "
            f"({_arm_note(sysp, wrap, tmpl) or 'no filter'}).\n"
            f"Arms present (system_prompt/wrapper/template): "
            + ", ".join(arms)
            + "\nPass --prompt-arm any, or name one of the above with "
              "--system-prompt / --wrapper / --prompt-template.",
            file=sys.stderr,
        )
        return 1
    text = render_report(results, baseline=args.baseline,
                         include_paper=not args.no_paper_reference,
                         arm_note=_arm_note(sysp, wrap, tmpl))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[report] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
