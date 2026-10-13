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


def load_results(
    reports_dir: Path,
    decoding: Optional[str] = None,
    dtype: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[RunResult]:
    """Load every ``*.metrics.json`` in ``reports_dir`` (plus its outcomes).

    Args:
        reports_dir: where ``evaluate.py`` wrote its output.
        decoding: keep only this decoding mode (``"sampling"`` / ``"greedy"``).
        dtype: keep only this dtype.
        seed: keep only this seed.

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


def _index(results: Sequence[RunResult]) -> Dict[Tuple[str, str, str], RunResult]:
    """``(benchmark, split, method) -> result``; later duplicates win loudly."""
    index: Dict[Tuple[str, str, str], RunResult] = {}
    for r in results:
        key = (r.benchmark, r.split, r.method)
        if key in index:
            raise ValueError(
                f"two metrics files for {key}: {index[key].metrics_path.name} "
                f"and {r.metrics_path.name}. Filter by --dtype/--seed/--decoding, "
                "or move the stale one out of the reports directory."
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
    index = _index([r for r in results if r.benchmark == "pope"])
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


def render_report(
    results: Sequence[RunResult], baseline: str = "sampling",
    include_paper: bool = True,
) -> str:
    """Compose the whole Markdown report."""
    sampling = [r for r in results if r.decoding == "sampling"]
    greedy = [r for r in results if r.decoding == "greedy"]
    parts = [
        "# Fox reproduction -- results",
        "",
        "All cells are `value [95% CI]` in percentage points. "
        f"`{FLAG}` marks a gap against the baseline that the sample size "
        "cannot resolve at 95% confidence: the two methods are "
        "indistinguishable on this evidence, whatever the point estimates say.",
        "",
    ]
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
    args = ap.parse_args(argv)

    results = load_results(
        args.reports_dir,
        decoding=None if args.decoding == "all" else args.decoding,
        dtype=args.dtype, seed=args.seed,
    )
    if not results:
        print(f"no metrics files in {args.reports_dir}; run evaluate.py first",
              file=sys.stderr)
        return 1
    text = render_report(results, baseline=args.baseline,
                         include_paper=not args.no_paper_reference)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"[report] wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
