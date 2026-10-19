"""CPU-only metrics from generation JSONL. No GPU, no torch, no model.

Decision 1 of the project: generation and evaluation are separate programs.
This one reads the JSONL that ``bench/generate.py`` wrote and produces a
metrics JSON per run, plus a per-sample outcome file that
``bench/report.py`` uses for *paired* comparisons.

Metrics
-------
POPE (binary object existence, positive class = "yes"):
    accuracy, precision, recall, F1, and **yes-ratio** -- the fraction of
    "yes" answers. The yes-ratio is not decoration: a method can raise POPE
    accuracy simply by answering "no" more often, because the splits are
    balanced and the hard questions are the "yes" ones. Accuracy without the
    yes-ratio cannot distinguish better grounding from a shifted threshold.

CHAIR (open-ended captioning):
    CHAIR_S (captions with >= 1 hallucinated object / captions) and CHAIR_I
    (hallucinated object mentions / all object mentions), plus the
    completeness guards from the paper (recall, precision, F1) so that a drop
    in CHAIR achieved by saying less is visible.

Every metric is reported with its **numerator and denominator**. CHAIR_I's
denominator -- the total number of mentioned object instances -- is not stated
in the paper and cannot be recovered from a published CHAIR_I value, yet
nothing can be said about significance without it.

Interval notes
--------------
* POPE accuracy / yes-ratio and CHAIR_S are proportions over independent
  samples: Wilson intervals apply directly.
* CHAIR_I's trials (object mentions) are **clustered within captions**, so a
  Wilson interval over mentions is anti-conservative. Both are reported: the
  Wilson interval (comparable to what others publish) and a cluster bootstrap
  over captions (the honest one). The report prefers the bootstrap.
* F1 is a ratio of ratios, so it gets a bootstrap interval instead.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


from foxrepro.bench.chair import CHAIRScorer, CaptionScore  # noqa: E402
from foxrepro.bench.stats import (  # noqa: E402
    bootstrap_ci, wilson_interval,
)
from foxrepro.storage import PROVENANCE_FIELDS, VARIANT_FIELDS, paths  # noqa: E402

__all__ = [
    "parse_yes_no", "parse_yes_no_released", "POPE_PARSERS",
    "DEFAULT_POPE_PARSER", "read_run", "evaluate_pope", "evaluate_chair",
    "evaluate_run_file", "PopeOutcome",
]

_WORD_RE = re.compile(r"[a-z]+")


# --------------------------------------------------------------------------- #
# answer parsing
# --------------------------------------------------------------------------- #
def parse_yes_no(text: str) -> Optional[str]:
    """Map a POPE answer to ``"yes"`` / ``"no"`` / ``None`` (unparsable).

    POPE is generated with ``max_new_tokens = 1``, so in practice the output is
    literally ``"Yes"`` or ``"No"``. The parser still handles longer answers,
    because the greedy control and any prompt change can produce them:

    1. the first alphabetic word decides, when it is "yes" or "no";
    2. otherwise the first occurrence of either word anywhere decides;
    3. otherwise the answer is unparsable and is **counted and reported**,
       never silently dropped.

    Args:
        text: raw model output.

    Returns:
        ``"yes"``, ``"no"`` or ``None``.
    """
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return None
    if words[0] in ("yes", "no"):
        return words[0]
    for w in words:
        if w in ("yes", "no"):
            return w
    return None


#: ``pope_eval.py:233``, verbatim. The released scorer has no unparsable
#: class: it looks for a negation token anywhere in the answer and calls
#: everything else "yes".
RELEASED_NEG_WORDS = ("No", "not", "no", "NO")


def parse_yes_no_released(text: str) -> str:
    """The released POPE rule (``pope_eval.py:232-242``), transcribed.

    Not a cleaned-up version of :func:`parse_yes_no`: a different function.
    The release strips ``.`` and ``,``, splits on a single space, and answers
    "no" if any token is in ``RELEASED_NEG_WORDS`` or ends in ``n't``. Every
    other answer -- including ``""``, ``"There"`` and ``"Maybe"`` -- is "yes".

    With ``max_new_tokens = 1`` and the "single word or phrase" suffix the two
    parsers agree on every sample (measured: 3000/3000 on both session-1 POPE
    runs). Without that suffix the first token is often neither "Yes" nor
    "No", and then they diverge: the release scores such an answer as a
    positive, this module's strict parser calls it unparsable. Since the
    paper's numbers came out of the released rule, that rule is the default.

    Args:
        text: raw model output.

    Returns:
        ``"yes"`` or ``"no"`` -- never ``None``.
    """
    line = (text or "").replace(".", "").replace(",", "")
    words = line.split(" ")
    if any(w in RELEASED_NEG_WORDS for w in words):
        return "no"
    if any(w.endswith("n't") for w in words):
        return "no"
    return "yes"


#: ``--pope-parser``. ``released`` reproduces the paper; ``strict`` is this
#: module's own rule, which reports unparsable answers instead of scoring them.
POPE_PARSERS = {
    "released": parse_yes_no_released,
    "strict": parse_yes_no,
}
DEFAULT_POPE_PARSER = "released"


# --------------------------------------------------------------------------- #
# reading runs
# --------------------------------------------------------------------------- #
def read_run(path: Path, strict: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Read a run JSONL, de-duplicating by ``sample_id``.

    Resume can in principle append a ``sample_id`` twice (for example if a
    file was concatenated by hand). The **first** occurrence wins, and the
    duplicate count is reported.

    Args:
        path: run file.
        strict: raise if the records disagree on the run-level provenance
            fields, or if a provenance field is missing.

    Returns:
        ``(records, meta)``. ``meta`` carries the shared provenance, the
        duplicate count and any trailing-partial-line byte count.
    """
    records: List[Dict[str, Any]] = []
    seen: Dict[str, int] = {}
    n_dupes = 0
    partial_bytes = 0
    with open(path, "rb") as fh:
        for raw in fh:
            if not raw.endswith(b"\n"):
                partial_bytes = len(raw)
                break
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            if sid in seen:
                n_dupes += 1
                continue
            seen[sid] = len(records)
            records.append(rec)
    if not records:
        raise ValueError(f"{path} contains no records")

    if strict:
        missing = [f for f in PROVENANCE_FIELDS if f not in records[0]]
        if missing:
            raise ValueError(f"{path}: records missing provenance fields {missing}")

    shared_keys = (
        "method", "benchmark", "split", "seed", "dtype", "model_id",
        "commit_hash", "transformers_path", "prompt_template_id",
        "max_new_tokens", *VARIANT_FIELDS,
    )
    meta: Dict[str, Any] = {"run_file": str(path)}
    for key in dict.fromkeys(shared_keys):
        values = {json.dumps(r.get(key), sort_keys=True) for r in records}
        if len(values) == 1:
            meta[key] = records[0].get(key)
        elif strict:
            raise ValueError(
                f"{path}: field {key!r} is not constant across records "
                f"({len(values)} distinct values); this file mixes runs."
            )
        else:
            meta[key] = sorted(json.loads(v) for v in values)  # type: ignore[arg-type]
    meta["n_records"] = len(records)
    meta["n_duplicate_sample_ids"] = n_dupes
    meta["trailing_partial_bytes"] = partial_bytes
    lat = [float(r["latency_ms"]) for r in records if r.get("latency_ms") is not None]
    gen = [int(r["n_gen_tokens"]) for r in records if r.get("n_gen_tokens") is not None]
    meta["latency_ms_mean"] = sum(lat) / len(lat) if lat else None
    meta["latency_ms_median"] = sorted(lat)[len(lat) // 2] if lat else None
    meta["n_gen_tokens_mean"] = sum(gen) / len(gen) if gen else None
    return records, meta


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
@dataclass
class PopeOutcome:
    """Per-sample POPE outcome; the unit a paired comparison resamples."""

    sample_id: str
    image_id: int
    label: str
    pred: Optional[str]
    correct: int
    said_yes: int

    def as_dict(self) -> Dict[str, Any]:
        """JSON-ready outcome row."""
        return {
            "sample_id": self.sample_id, "image_id": self.image_id,
            "label": self.label, "pred": self.pred,
            "correct": self.correct, "said_yes": self.said_yes,
        }


def _f1_from_codes(codes: Sequence[int]) -> float:
    """F1 from per-sample confusion codes (0=TN, 1=TP, 2=FP, 3=FN)."""
    counts = Counter(codes)
    tp, fp, fn = counts[1], counts[2], counts[3]
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def evaluate_pope(
    records: Sequence[Dict[str, Any]],
    labels: Optional[Dict[str, str]] = None,
    unparsed_policy: str = "negative",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
    pope_parser: str = DEFAULT_POPE_PARSER,
) -> Tuple[Dict[str, Any], List[PopeOutcome]]:
    """Compute POPE metrics.

    Args:
        records: run records for one POPE split.
        labels: ``sample_id -> "yes"/"no"`` ground truth. Optional: the
            generator stores ``label`` in each record, and that copy is used
            when present. Supplying ``labels`` re-checks it and raises on
            disagreement, which catches a mismatched POPE file.
        unparsed_policy: ``"negative"`` treats an unparsable answer as "no"
            (what the common POPE scripts do, since they test ``"no" in
            text``); ``"exclude"`` drops it from every denominator. Either way
            the count is reported. Irrelevant under the ``released`` parser,
            which never returns ``None``.
        pope_parser: ``"released"`` (default, :func:`parse_yes_no_released`)
            or ``"strict"`` (:func:`parse_yes_no`). Recorded in the metrics,
            because the two rules can disagree and a number is meaningless
            without knowing which produced it.
        confidence: interval coverage.
        n_boot: bootstrap replicates for the F1 interval.
        seed: bootstrap seed.

    Returns:
        ``(metrics, outcomes)``.
    """
    if unparsed_policy not in ("negative", "exclude"):
        raise ValueError("unparsed_policy must be 'negative' or 'exclude'")
    try:
        parse = POPE_PARSERS[pope_parser]
    except KeyError:
        raise ValueError(
            f"unknown pope_parser {pope_parser!r}; known: {sorted(POPE_PARSERS)}"
        ) from None

    outcomes: List[PopeOutcome] = []
    n_unparsed = 0
    for rec in records:
        sid = rec["sample_id"]
        label = rec.get("label")
        if labels is not None:
            truth = labels.get(sid)
            if truth is None:
                raise KeyError(f"no ground-truth label for {sid}")
            if label is not None and truth != label:
                raise ValueError(
                    f"{sid}: record label {label!r} != annotation label {truth!r}; "
                    "the run and the POPE file disagree."
                )
            label = truth
        if label not in ("yes", "no"):
            raise ValueError(
                f"{sid}: no usable ground-truth label (got {label!r}). Pass "
                "--pope-dir so labels can be joined from the annotation file."
            )
        pred = parse(rec.get("output", ""))
        if pred is None:
            n_unparsed += 1
            if unparsed_policy == "exclude":
                continue
            pred_eff = "no"
        else:
            pred_eff = pred
        outcomes.append(
            PopeOutcome(
                sample_id=sid, image_id=int(rec.get("image_id", -1)), label=label,
                pred=pred, correct=int(pred_eff == label),
                said_yes=int(pred_eff == "yes"),
            )
        )

    n = len(outcomes)
    tp = sum(1 for o in outcomes if o.label == "yes" and o.said_yes)
    fp = sum(1 for o in outcomes if o.label == "no" and o.said_yes)
    fn = sum(1 for o in outcomes if o.label == "yes" and not o.said_yes)
    tn = sum(1 for o in outcomes if o.label == "no" and not o.said_yes)
    n_correct = tp + tn
    n_pred_yes = tp + fp
    n_gt_yes = tp + fn

    codes = [
        1 if (o.label == "yes" and o.said_yes) else
        2 if (o.label == "no" and o.said_yes) else
        3 if (o.label == "yes" and not o.said_yes) else 0
        for o in outcomes
    ]
    f1_point = _f1_from_codes(codes)
    f1_lo, f1_hi = bootstrap_ci(codes, _f1_from_codes, n_boot=n_boot,
                                confidence=confidence, seed=seed)

    acc = wilson_interval(n_correct, n, confidence, "pope_accuracy")
    prec = wilson_interval(tp, n_pred_yes, confidence, "pope_precision")
    rec_ = wilson_interval(tp, n_gt_yes, confidence, "pope_recall")
    yes = wilson_interval(n_pred_yes, n, confidence, "pope_yes_ratio")

    metrics: Dict[str, Any] = {
        "benchmark": "pope",
        "n_samples": n,
        "n_unparsed": n_unparsed,
        "unparsed_policy": unparsed_policy,
        "pope_parser": pope_parser,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "denominators": {
            "accuracy": n,
            "precision": n_pred_yes,
            "recall": n_gt_yes,
            "f1": n,
            "yes_ratio": n,
            "n_ground_truth_yes": n_gt_yes,
            "n_ground_truth_no": tn + fp,
        },
        "accuracy": acc.as_dict(),
        "precision": prec.as_dict(),
        "recall": rec_.as_dict(),
        "yes_ratio": yes.as_dict(),
        "f1": {
            "name": "pope_f1", "point": f1_point, "ci_lo": f1_lo, "ci_hi": f1_hi,
            "n": n, "confidence": confidence,
            "interval_method": f"percentile bootstrap (B={n_boot}, seed={seed})",
        },
        "interval_method": "Wilson score (except F1)",
    }
    return metrics, outcomes


def load_pope_labels(pope_dir: Path, split: str) -> Dict[str, str]:
    """``sample_id -> label`` for one POPE split (cross-check / repair)."""
    from foxrepro.data.loaders import load_pope  # local import keeps module import cheap

    # image paths are unused here (require_images=False); only labels are read
    samples = load_pope(pope_dir, Path("."), split, require_images=False)
    return {s.sample_id: str(s.label) for s in samples}


# --------------------------------------------------------------------------- #
# CHAIR
# --------------------------------------------------------------------------- #
def evaluate_chair(
    records: Sequence[Dict[str, Any]],
    scorer: CHAIRScorer,
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
) -> Tuple[Dict[str, Any], List[CaptionScore]]:
    """Compute CHAIR metrics for one run.

    Args:
        records: run records for the CHAIR split.
        scorer: configured :class:`~bench.chair.CHAIRScorer`.
        confidence: interval coverage.
        n_boot: replicates for the CHAIR_I cluster bootstrap.
        seed: bootstrap seed.

    Returns:
        ``(metrics, per_caption_scores)``.
    """
    scores = scorer.score_many(
        (r["sample_id"], int(r["image_id"]), str(r.get("output", ""))) for r in records
    )
    totals = scorer.totals(scores)

    chair_s = wilson_interval(
        totals.n_captions_with_hallucination, totals.n_captions, confidence, "chair_s"
    )
    chair_i_wilson = wilson_interval(
        totals.n_hallucinated_mentions, totals.n_mentions, confidence, "chair_i"
    )

    # cluster bootstrap over captions: the correct interval for a ratio of
    # sums whose trials are grouped inside captions
    num = [float(s.n_hallucinated) for s in scores]
    den = [float(s.n_mentions) for s in scores]
    idx_values = list(range(len(scores)))

    def _ratio(resampled: Sequence[int]) -> float:
        d = sum(den[i] for i in resampled)
        return sum(num[i] for i in resampled) / d if d else float("nan")

    ci_lo, ci_hi = bootstrap_ci(idx_values, _ratio, n_boot=n_boot,
                                confidence=confidence, seed=seed)

    metrics: Dict[str, Any] = {
        "benchmark": "chair",
        "n_samples": totals.n_captions,
        "denominators": {
            "chair_s": totals.n_captions,
            "chair_i": totals.n_mentions,
            "chair_i_note": (
                "total mentioned object instances, duplicates included; not "
                "stated in the paper, required for any interval"
            ),
            "recall": totals.n_gt_objects,
            "precision": totals.n_distinct_mentions,
        },
        "totals": totals.as_dict(),
        "chair_s_ci": chair_s.as_dict(),
        "chair_i_ci_wilson": chair_i_wilson.as_dict(),
        "chair_i_ci_cluster_bootstrap": {
            "name": "chair_i", "point": totals.chair_i, "ci_lo": ci_lo,
            "ci_hi": ci_hi, "n": totals.n_mentions, "confidence": confidence,
            "interval_method": (
                f"cluster bootstrap over captions (B={n_boot}, seed={seed}); "
                "preferred over Wilson because mentions are clustered in captions"
            ),
        },
        "scorer": scorer.config(),
    }
    return metrics, scores


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def evaluate_run_file(
    path: Path,
    scorer_factory: Optional[Callable[[], CHAIRScorer]] = None,
    pope_dir: Optional[Path] = None,
    unparsed_policy: str = "negative",
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 20260101,
    strict: bool = True,
    pope_parser: str = DEFAULT_POPE_PARSER,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate one run file.

    Args:
        path: run JSONL.
        scorer_factory: zero-arg factory for a :class:`CHAIRScorer`; required
            for CHAIR runs, built lazily so POPE evaluation needs no COCO
            annotations at all.
        pope_dir: POPE annotation directory, for cross-checking labels.
        unparsed_policy: see :func:`evaluate_pope`.
        confidence: interval coverage.
        n_boot: bootstrap replicates.
        seed: bootstrap seed.
        strict: see :func:`read_run`.
        pope_parser: see :func:`evaluate_pope`.

    Returns:
        ``(result, outcome_rows)`` where ``result`` is the full metrics
        document and ``outcome_rows`` are per-sample rows for paired tests.
    """
    records, meta = read_run(Path(path), strict=strict)
    benchmark = meta["benchmark"]
    if benchmark == "pope":
        labels = (
            load_pope_labels(pope_dir, str(meta["split"])) if pope_dir else None
        )
        metrics, outcomes = evaluate_pope(
            records, labels=labels, unparsed_policy=unparsed_policy,
            confidence=confidence, n_boot=n_boot, seed=seed,
            pope_parser=pope_parser,
        )
        rows = [o.as_dict() for o in outcomes]
    elif benchmark == "chair":
        if scorer_factory is None:
            raise ValueError("CHAIR evaluation needs a scorer_factory")
        metrics, scores = evaluate_chair(
            records, scorer_factory(), confidence=confidence, n_boot=n_boot, seed=seed
        )
        rows = [s.as_dict() for s in scores]
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")

    result = {
        "schema": "fox-metrics/1",
        "provenance": meta,
        "metrics": metrics,
        "eval_config": {
            "confidence": confidence, "n_boot": n_boot, "bootstrap_seed": seed,
            "unparsed_policy": unparsed_policy,
            "pope_parser": pope_parser,
        },
    }
    return result, rows


def _format_markdown(result: Dict[str, Any]) -> str:
    """Single-run metrics as a short Markdown block (for eyeballing a run)."""
    prov, met = result["provenance"], result["metrics"]
    head = (
        f"### {prov['method']} / {prov['benchmark']} / {prov['split']} "
        f"(seed {prov['seed']}, {prov['dtype']}, gamma={prov.get('gamma')})\n\n"
        f"- run file: `{Path(prov['run_file']).name}`\n"
        f"- model: `{prov.get('model_id')}`  commit: `{prov.get('commit_hash')}`\n"
        f"- transformers: `{prov.get('transformers_path')}`\n"
        f"- records: {prov['n_records']} (duplicates skipped: "
        f"{prov['n_duplicate_sample_ids']})\n\n"
    )
    lines = ["| metric | value | 95% CI | numerator | denominator |",
             "| --- | --- | --- | --- | --- |"]
    if met["benchmark"] == "pope":
        for name in ("accuracy", "precision", "recall", "yes_ratio"):
            m = met[name]
            lines.append(
                f"| {name} | {100 * m['point']:.2f} | "
                f"[{100 * m['ci_lo']:.2f}, {100 * m['ci_hi']:.2f}] | "
                f"{m['successes']:.0f} | {m['n']} |"
            )
        f1 = met["f1"]
        lines.append(
            f"| f1 | {100 * f1['point']:.2f} | "
            f"[{100 * f1['ci_lo']:.2f}, {100 * f1['ci_hi']:.2f}] | - | {f1['n']} |"
        )
        lines.append(
            f"| unparsed | {met['n_unparsed']} | - | - | {met['n_samples']} |"
        )
    else:
        t = met["totals"]
        cs, ci = met["chair_s_ci"], met["chair_i_ci_cluster_bootstrap"]
        lines.append(
            f"| CHAIR_S | {100 * cs['point']:.2f} | "
            f"[{100 * cs['ci_lo']:.2f}, {100 * cs['ci_hi']:.2f}] | "
            f"{t['chair_s_numerator']} | {t['chair_s_denominator']} |"
        )
        lines.append(
            f"| CHAIR_I | {100 * ci['point']:.2f} | "
            f"[{100 * ci['ci_lo']:.2f}, {100 * ci['ci_hi']:.2f}] | "
            f"{t['chair_i_numerator']} | {t['chair_i_denominator']} |"
        )
        for name in ("recall", "precision", "f1"):
            lines.append(f"| {name} | {100 * t[name]:.2f} | - | - | "
                         f"{t.get(name + '_denominator', '-')} |")
        lines.append(f"| mean caption words | {t['mean_caption_words']:.1f} | - | - | - |")
    return head + "\n".join(lines) + "\n"


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(
        description="Compute POPE / CHAIR metrics from generation JSONL (no GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("runs", nargs="*", help="run JSONL files or globs")
    ap.add_argument("--runs-dir", type=Path, default=p.runs_root,
                    help="used when no explicit run files are given")
    ap.add_argument("--out-dir", type=Path, default=p.reports_dir)
    ap.add_argument("--format", choices=("json", "markdown", "both"), default="both")
    ap.add_argument(
        "--pope-parser", choices=tuple(POPE_PARSERS), default=DEFAULT_POPE_PARSER,
        help="POPE answer rule. 'released' transcribes pope_eval.py (any "
             "answer without a negation token counts as yes); 'strict' uses "
             "this module's parser, which reports unparsable answers.")
    ap.add_argument("--unparsed-policy", choices=("negative", "exclude"),
                    default="negative")
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--pope-dir", type=Path, default=None,
                    help="cross-check POPE labels against the annotation files")
    ap.add_argument("--synonyms", type=Path, default=p.chair_synonyms)
    # Prefers the 500-image subset that build_coco_subset.py packs, and falls
    # back to the full file that `fetch_images --with-annotations` leaves.
    ap.add_argument("--instances", type=Path,
                    default=p.coco_annotation("instances_val2014"))
    ap.add_argument("--captions", type=Path,
                    default=p.coco_annotation("captions_val2014"))
    ap.add_argument("--gt-source", choices=("both", "instances", "captions"),
                    default="both")
    ap.add_argument("--tokenizer", choices=("regex", "nltk"), default="regex")
    ap.add_argument("--no-strict", action="store_true",
                    help="tolerate runs whose provenance fields are not constant")
    args = ap.parse_args(argv)

    targets: List[Path] = []
    for pattern in args.runs:
        targets.extend(Path(x) for x in sorted(glob.glob(pattern)))
    if not targets:
        targets = sorted(Path(args.runs_dir).glob("*.jsonl"))
    if not targets:
        print(f"no run files found (looked in {args.runs_dir})", file=sys.stderr)
        return 1

    def scorer_factory() -> CHAIRScorer:
        return CHAIRScorer(
            synonyms_path=args.synonyms, instances_json=args.instances,
            captions_json=args.captions if args.gt_source != "instances" else None,
            tokenizer=args.tokenizer, gt_source=args.gt_source,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exit_code = 0
    for target in targets:
        try:
            result, rows = evaluate_run_file(
                target, scorer_factory=scorer_factory, pope_dir=args.pope_dir,
                unparsed_policy=args.unparsed_policy, confidence=args.confidence,
                pope_parser=args.pope_parser,
                n_boot=args.n_boot, seed=args.seed, strict=not args.no_strict,
            )
        except Exception as exc:
            print(f"[evaluate] FAILED {target.name}: {exc}", file=sys.stderr)
            exit_code = 1
            continue
        stem = target.name[: -len(".jsonl")]
        if args.format in ("json", "both"):
            out = args.out_dir / f"{stem}.metrics.json"
            out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
            outcomes = args.out_dir / f"{stem}.outcomes.jsonl"
            with open(outcomes, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"[evaluate] wrote {out.name} and {outcomes.name}")
        if args.format in ("markdown", "both"):
            print()
            print(_format_markdown(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(_main())
