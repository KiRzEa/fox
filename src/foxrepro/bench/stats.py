"""Uncertainty quantification for hallucination-benchmark proportions.

Why this module exists
----------------------
The headline numbers in the Fox paper are proportions over finite samples:
POPE accuracy over n = 3000 questions per split, CHAIR_S over n = 500 captions,
CHAIR_I over the total number of *mentioned object instances*. Differences of
0.5-2 points between methods are reported as improvements, but a proportion
measured on n = 3000 carries a 95% interval of roughly +-1.4 points, and one
measured on n = 500 roughly +-4.4 points. This module makes that explicit so
the reproduction can state which reported gaps are resolvable at all.

Everything here is CPU-only and pure standard library except for an optional
numpy fast path in the bootstrap. The normal quantile comes from
``statistics.NormalDist``, so scipy is not required.

Interval choices
----------------
* Single proportion: **Wilson score interval** (Wilson, 1927), the inversion of
  the score test. Count form used here, transcribed from the standard
  statement of the interval::

      p in [ (ns + z^2/2) / (n + z^2)
             +- (z / (n + z^2)) * sqrt( ns*nf/n + z^2/4 ) ]

  with ``ns`` successes, ``nf = n - ns`` failures and ``z`` the two-sided
  normal quantile. The implementation is checked in ``tests/test_stats.py``
  both against published numeric examples and against a numerical inversion of
  the score equation, so it does not rely on the algebra being remembered
  correctly.
* Difference of two independent proportions: **Newcombe's hybrid score
  interval** (Newcombe, 1998, "method 10"), built from the two Wilson
  intervals. This is the companion of the Wilson interval for a difference and
  behaves well at extreme proportions, unlike the Wald difference interval.
* Two methods evaluated on the *same* samples are not independent. Use
  :func:`paired_bootstrap_proportion`, :func:`paired_bootstrap_ratio` or
  :func:`mcnemar_exact`; they are materially tighter than the independent
  interval because the shared per-sample difficulty cancels.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:  # optional fast path; the module is fully functional without it
    import numpy as _np
except Exception:  # pragma: no cover - exercised only on numpy-less machines
    _np = None  # type: ignore[assignment]

__all__ = [
    "z_for_confidence",
    "Proportion",
    "wilson_interval",
    "wilson_halfwidth",
    "Comparison",
    "newcombe_diff_interval",
    "two_proportion_ztest",
    "compare_proportions",
    "paired_bootstrap_proportion",
    "paired_bootstrap_ratio",
    "bootstrap_ci",
    "mcnemar_exact",
    "required_n_for_gap",
    "resolvable_gap",
    "power_table",
]

_ND = NormalDist()


def z_for_confidence(confidence: float = 0.95) -> float:
    """Two-sided normal quantile for a confidence level.

    ``z_for_confidence(0.95) == 1.959963...``

    Args:
        confidence: coverage in (0, 1), e.g. ``0.95``.

    Returns:
        The value ``z`` with ``P(-z <= Z <= z) == confidence``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0,1), got {confidence}")
    return _ND.inv_cdf(0.5 + confidence / 2.0)


# --------------------------------------------------------------------------- #
# single proportion
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Proportion:
    """A proportion metric with its interval and, crucially, its denominator.

    Attributes:
        name: metric name, e.g. ``"pope_accuracy"`` or ``"chair_i"``.
        successes: numerator count.
        n: denominator count. Reported everywhere; CHAIR_I's denominator (the
            total number of mentioned object instances) is not stated in the
            paper and cannot be reconstructed after the fact.
        point: ``successes / n`` (``nan`` when ``n == 0``).
        lo: lower Wilson bound.
        hi: upper Wilson bound.
        confidence: coverage of ``[lo, hi]``.
    """

    name: str
    successes: float
    n: int
    point: float
    lo: float
    hi: float
    confidence: float

    @property
    def halfwidth(self) -> float:
        """Half the interval width, in proportion units."""
        return (self.hi - self.lo) / 2.0

    def as_dict(self) -> Dict[str, float | str | int]:
        """JSON-ready mapping including the denominator."""
        return {
            "name": self.name,
            "successes": self.successes,
            "n": self.n,
            "point": self.point,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "confidence": self.confidence,
        }

    def pct(self, digits: int = 2) -> str:
        """Render as ``"81.93 [80.52, 83.25]"`` in percentage points."""
        if self.n == 0:
            return "n/a"
        return (
            f"{100 * self.point:.{digits}f} "
            f"[{100 * self.lo:.{digits}f}, {100 * self.hi:.{digits}f}]"
        )


def wilson_interval(
    successes: float,
    n: int,
    confidence: float = 0.95,
    name: str = "proportion",
) -> Proportion:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of successes ``ns``; may be fractional only for
            display purposes (counts should be integers in real use).
        n: number of trials. ``n == 0`` yields a degenerate ``[0, 1]`` interval
            with ``point = nan`` rather than raising, so report code can print
            an empty cell.
        confidence: coverage, default 0.95.
        name: metric label carried into the result.

    Returns:
        A :class:`Proportion`. Bounds are clipped to ``[0, 1]``; for
        ``successes == 0`` the lower bound is exactly 0 and for
        ``successes == n`` the upper bound is exactly 1.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0 <= successes <= max(n, successes):
        raise ValueError(f"successes={successes} outside [0, n={n}]")
    if n == 0:
        return Proportion(name, successes, 0, float("nan"), 0.0, 1.0, confidence)

    z = z_for_confidence(confidence)
    z2 = z * z
    ns = float(successes)
    nf = float(n) - ns
    denom = n + z2
    center = (ns + z2 / 2.0) / denom
    # sqrt(ns*nf/n + z^2/4) -- the count form, numerically stable at the edges.
    spread = (z / denom) * math.sqrt(ns * nf / n + z2 / 4.0)
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    if ns == 0:
        lo = 0.0
    if nf == 0:
        hi = 1.0
    return Proportion(name, ns, int(n), ns / n, lo, hi, confidence)


def wilson_halfwidth(p: float, n: int, confidence: float = 0.95) -> float:
    """Half-width of the Wilson interval at proportion ``p`` and size ``n``.

    Handy for planning: ``wilson_halfwidth(0.82, 3000)`` is about 0.0137, i.e.
    the +-1.4 points that POPE splits afford.
    """
    return wilson_interval(round(p * n), n, confidence).halfwidth


# --------------------------------------------------------------------------- #
# two independent proportions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Comparison:
    """Result of comparing two proportions.

    Attributes:
        label_a / label_b: method names.
        p_a / p_b: point estimates.
        diff: ``p_a - p_b``.
        lo / hi: confidence interval for ``diff``.
        confidence: coverage.
        significant: ``True`` iff the interval excludes 0, i.e. the gap is
            distinguishable at this confidence level.
        p_value: two-sided p-value of the accompanying test (``None`` for
            bootstrap-only comparisons that do not produce one).
        method: which procedure produced the interval.
        n_a / n_b: denominators, always reported.
    """

    label_a: str
    label_b: str
    p_a: float
    p_b: float
    diff: float
    lo: float
    hi: float
    confidence: float
    significant: bool
    p_value: Optional[float]
    method: str
    n_a: int
    n_b: int

    def verdict(self) -> str:
        """One-line human summary."""
        if not self.significant:
            return (
                f"{self.label_a} vs {self.label_b}: gap {100 * self.diff:+.2f} pts, "
                f"95% CI [{100 * self.lo:+.2f}, {100 * self.hi:+.2f}] -- "
                f"NOT distinguishable at {100 * self.confidence:.0f}%"
            )
        return (
            f"{self.label_a} vs {self.label_b}: gap {100 * self.diff:+.2f} pts, "
            f"95% CI [{100 * self.lo:+.2f}, {100 * self.hi:+.2f}] -- distinguishable"
        )

    def as_dict(self) -> Dict[str, object]:
        """JSON-ready mapping."""
        return {
            "label_a": self.label_a, "label_b": self.label_b,
            "p_a": self.p_a, "p_b": self.p_b, "n_a": self.n_a, "n_b": self.n_b,
            "diff": self.diff, "ci_lo": self.lo, "ci_hi": self.hi,
            "confidence": self.confidence, "significant": self.significant,
            "p_value": self.p_value, "method": self.method,
        }


def newcombe_diff_interval(
    k_a: int, n_a: int, k_b: int, n_b: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """Newcombe hybrid-score interval for the difference of two proportions.

    Built from the two Wilson intervals ``[l1, u1]`` and ``[l2, u2]`` as::

        lower = d - sqrt((p1 - l1)^2 + (u2 - p2)^2)
        upper = d + sqrt((u1 - p1)^2 + (p2 - l2)^2)

    with ``d = p1 - p2`` (Newcombe 1998, method 10).

    Returns:
        ``(lower, upper)`` clipped to ``[-1, 1]``.
    """
    a = wilson_interval(k_a, n_a, confidence)
    b = wilson_interval(k_b, n_b, confidence)
    d = a.point - b.point
    lower = d - math.sqrt((a.point - a.lo) ** 2 + (b.hi - b.point) ** 2)
    upper = d + math.sqrt((a.hi - a.point) ** 2 + (b.point - b.lo) ** 2)
    return max(-1.0, lower), min(1.0, upper)


def two_proportion_ztest(k_a: int, n_a: int, k_b: int, n_b: int) -> float:
    """Two-sided p-value of the pooled two-proportion z-test.

    Returns ``1.0`` when both proportions are identical or a denominator is 0.
    """
    if n_a == 0 or n_b == 0:
        return 1.0
    p_a, p_b = k_a / n_a, k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 1.0
    z = (p_a - p_b) / se
    return 2.0 * (1.0 - _ND.cdf(abs(z)))


def compare_proportions(
    k_a: int, n_a: int, k_b: int, n_b: int,
    label_a: str = "A", label_b: str = "B",
    confidence: float = 0.95,
) -> Comparison:
    """Compare two proportions treated as **independent** samples.

    Use this when the two methods were not run on the same inputs (or when you
    want the conservative answer). For same-input runs prefer
    :func:`paired_bootstrap_proportion` or :func:`mcnemar_exact`.
    """
    lo, hi = newcombe_diff_interval(k_a, n_a, k_b, n_b, confidence)
    p_a = k_a / n_a if n_a else float("nan")
    p_b = k_b / n_b if n_b else float("nan")
    return Comparison(
        label_a=label_a, label_b=label_b, p_a=p_a, p_b=p_b, diff=p_a - p_b,
        lo=lo, hi=hi, confidence=confidence,
        significant=not (lo <= 0.0 <= hi),
        p_value=two_proportion_ztest(k_a, n_a, k_b, n_b),
        method="newcombe-hybrid-score (independent)", n_a=n_a, n_b=n_b,
    )


# --------------------------------------------------------------------------- #
# paired comparisons on the same samples
# --------------------------------------------------------------------------- #
def _rng_indices(rng: random.Random, n: int, reps: int) -> List[List[int]]:
    return [[rng.randrange(n) for _ in range(n)] for _ in range(reps)]


def paired_bootstrap_proportion(
    outcomes_a: Sequence[float],
    outcomes_b: Sequence[float],
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 20260101,
    label_a: str = "A",
    label_b: str = "B",
) -> Comparison:
    """Paired bootstrap over samples for a mean/proportion metric.

    ``outcomes_a[i]`` and ``outcomes_b[i]`` must describe the **same** input
    sample ``i`` (same image, same question). Each bootstrap replicate
    resamples sample indices with replacement and recomputes both means from
    the same resampled index set, so per-sample difficulty cancels. The
    interval is the percentile interval of the difference of means.

    Suitable for POPE accuracy (outcomes in {0,1}) and CHAIR_S (whether a
    caption contains a hallucination). For a ratio of sums such as CHAIR_I use
    :func:`paired_bootstrap_ratio`.

    Returns:
        A :class:`Comparison` whose ``p_value`` is the bootstrap two-sided
        achieved significance level ``2 * min(P(d* <= 0), P(d* >= 0))``.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("paired bootstrap needs aligned, equal-length outcome vectors")
    n = len(outcomes_a)
    if n == 0:
        raise ValueError("no samples to bootstrap")
    if n_boot < 100:
        raise ValueError("n_boot < 100 gives meaningless percentiles")

    alpha = (1.0 - confidence) / 2.0
    mean_a = sum(outcomes_a) / n
    mean_b = sum(outcomes_b) / n
    diff = mean_a - mean_b

    if _np is not None:
        rng = _np.random.default_rng(seed)
        arr_a = _np.asarray(outcomes_a, dtype=float)
        arr_b = _np.asarray(outcomes_b, dtype=float)
        idx = rng.integers(0, n, size=(n_boot, n))
        diffs = arr_a[idx].mean(axis=1) - arr_b[idx].mean(axis=1)
        lo = float(_np.quantile(diffs, alpha))
        hi = float(_np.quantile(diffs, 1 - alpha))
        n_le = int((diffs <= 0).sum())
        n_ge = int((diffs >= 0).sum())
    else:  # pragma: no cover - numpy-less fallback
        rng_py = random.Random(seed)
        diffs_list: List[float] = []
        for _ in range(n_boot):
            s_a = s_b = 0.0
            for _ in range(n):
                j = rng_py.randrange(n)
                s_a += outcomes_a[j]
                s_b += outcomes_b[j]
            diffs_list.append((s_a - s_b) / n)
        diffs_list.sort()
        lo = diffs_list[int(alpha * (n_boot - 1))]
        hi = diffs_list[int((1 - alpha) * (n_boot - 1))]
        n_le = sum(1 for d in diffs_list if d <= 0)
        n_ge = sum(1 for d in diffs_list if d >= 0)

    p_value = min(1.0, 2.0 * min(n_le, n_ge) / n_boot)
    return Comparison(
        label_a=label_a, label_b=label_b, p_a=mean_a, p_b=mean_b, diff=diff,
        lo=lo, hi=hi, confidence=confidence, significant=not (lo <= 0.0 <= hi),
        p_value=p_value, method=f"paired percentile bootstrap (B={n_boot}, seed={seed})",
        n_a=n, n_b=n,
    )


def paired_bootstrap_ratio(
    num_a: Sequence[float], den_a: Sequence[float],
    num_b: Sequence[float], den_b: Sequence[float],
    n_boot: int = 10000, confidence: float = 0.95, seed: int = 20260101,
    label_a: str = "A", label_b: str = "B",
) -> Comparison:
    """Paired bootstrap for a **ratio of sums**, e.g. CHAIR_I.

    CHAIR_I is ``sum(hallucinated mentions) / sum(all mentions)`` over captions,
    so the denominator is itself random and differs between methods. Resampling
    the caption indices and recomputing both ratios from the resampled
    numerator/denominator pairs is the correct cluster bootstrap; treating the
    mentions as independent trials would understate the uncertainty, because
    mentions within one caption are correlated.

    Args:
        num_a, den_a: per-sample numerator/denominator for method A.
        num_b, den_b: same for method B, aligned on the same sample order.
    """
    lengths = {len(num_a), len(den_a), len(num_b), len(den_b)}
    if len(lengths) != 1:
        raise ValueError("all four vectors must be aligned and equal length")
    n = len(num_a)
    if n == 0:
        raise ValueError("no samples to bootstrap")
    alpha = (1.0 - confidence) / 2.0

    def ratio(num: Sequence[float], den: Sequence[float]) -> float:
        d = float(sum(den))
        return float(sum(num)) / d if d else float("nan")

    point_a, point_b = ratio(num_a, den_a), ratio(num_b, den_b)

    if _np is not None:
        rng = _np.random.default_rng(seed)
        na, da = _np.asarray(num_a, float), _np.asarray(den_a, float)
        nb, db = _np.asarray(num_b, float), _np.asarray(den_b, float)
        idx = rng.integers(0, n, size=(n_boot, n))
        with _np.errstate(invalid="ignore", divide="ignore"):
            r_a = na[idx].sum(axis=1) / da[idx].sum(axis=1)
            r_b = nb[idx].sum(axis=1) / db[idx].sum(axis=1)
        diffs = r_a - r_b
        diffs = diffs[~_np.isnan(diffs)]
        lo = float(_np.quantile(diffs, alpha))
        hi = float(_np.quantile(diffs, 1 - alpha))
        n_le = int((diffs <= 0).sum())
        n_ge = int((diffs >= 0).sum())
        b_eff = int(diffs.size)
    else:  # pragma: no cover
        rng_py = random.Random(seed)
        diffs_list = []
        for _ in range(n_boot):
            sna = sda = snb = sdb = 0.0
            for _ in range(n):
                j = rng_py.randrange(n)
                sna += num_a[j]; sda += den_a[j]; snb += num_b[j]; sdb += den_b[j]
            if sda and sdb:
                diffs_list.append(sna / sda - snb / sdb)
        diffs_list.sort()
        b_eff = len(diffs_list)
        lo = diffs_list[int(alpha * (b_eff - 1))]
        hi = diffs_list[int((1 - alpha) * (b_eff - 1))]
        n_le = sum(1 for d in diffs_list if d <= 0)
        n_ge = sum(1 for d in diffs_list if d >= 0)

    p_value = min(1.0, 2.0 * min(n_le, n_ge) / max(b_eff, 1))
    return Comparison(
        label_a=label_a, label_b=label_b, p_a=point_a, p_b=point_b,
        diff=point_a - point_b, lo=lo, hi=hi, confidence=confidence,
        significant=not (lo <= 0.0 <= hi), p_value=p_value,
        method=f"paired cluster bootstrap on ratio of sums (B={b_eff}, seed={seed})",
        n_a=int(sum(den_a)), n_b=int(sum(den_b)),
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float],
    n_boot: int = 10000,
    confidence: float = 0.95,
    seed: int = 20260101,
) -> Tuple[float, float]:
    """One-sample percentile bootstrap CI for an arbitrary statistic.

    Needed for metrics that are not proportions and so have no Wilson
    interval -- F1 above all, which is a ratio of ratios.

    Args:
        values: per-sample quantities the statistic is computed from. For F1
            these are per-sample codes the ``statistic`` callable knows how to
            aggregate (e.g. 0=TN, 1=TP, 2=FP, 3=FN).
        statistic: maps a resampled sequence to a scalar.
        n_boot: number of replicates.
        confidence: coverage.
        seed: RNG seed, so a report is reproducible.

    Returns:
        ``(lo, hi)`` percentile bounds.
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    alpha = (1.0 - confidence) / 2.0
    rng = random.Random(seed)
    reps: List[float] = []
    for _ in range(n_boot):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        reps.append(statistic(resample))
    reps.sort()
    return reps[int(alpha * (n_boot - 1))], reps[int((1 - alpha) * (n_boot - 1))]


def mcnemar_exact(
    outcomes_a: Sequence[int], outcomes_b: Sequence[int],
    confidence: float = 0.95, label_a: str = "A", label_b: str = "B",
) -> Comparison:
    """Exact McNemar test for two binary-outcome methods on the same samples.

    Only the discordant pairs carry information: ``b`` = A correct & B wrong,
    ``c`` = A wrong & B correct. Under the null, ``b ~ Binomial(b + c, 0.5)``.
    The reported interval is the Wilson interval for ``b / (b + c)`` mapped back
    onto the accuracy difference scale via ``diff = (b - c) / n``.

    This is the sharpest test available for paired POPE accuracy and needs no
    resampling, so it is a good cross-check on the bootstrap.
    """
    if len(outcomes_a) != len(outcomes_b):
        raise ValueError("McNemar needs aligned outcome vectors")
    n = len(outcomes_a)
    if n == 0:
        raise ValueError("no samples")
    b = sum(1 for x, y in zip(outcomes_a, outcomes_b) if x and not y)
    c = sum(1 for x, y in zip(outcomes_a, outcomes_b) if y and not x)
    disc = b + c
    if disc == 0:
        p_value = 1.0
    else:
        # two-sided exact binomial p-value with p = 1/2
        k = min(b, c)
        tail = sum(math.comb(disc, i) for i in range(0, k + 1)) / (2.0 ** disc)
        p_value = min(1.0, 2.0 * tail)
    w = wilson_interval(b, disc, confidence) if disc else None
    if w is None:
        lo = hi = 0.0
    else:
        # diff = (b - c)/n = (2*b/disc - 1) * disc/n
        lo = (2 * w.lo - 1) * disc / n
        hi = (2 * w.hi - 1) * disc / n
    mean_a = sum(outcomes_a) / n
    mean_b = sum(outcomes_b) / n
    return Comparison(
        label_a=label_a, label_b=label_b, p_a=mean_a, p_b=mean_b,
        diff=(b - c) / n, lo=lo, hi=hi, confidence=confidence,
        significant=p_value < (1.0 - confidence), p_value=p_value,
        method=f"exact McNemar (discordant={disc})", n_a=n, n_b=n,
    )


# --------------------------------------------------------------------------- #
# planning / power
# --------------------------------------------------------------------------- #
def required_n_for_gap(
    p_baseline: float, d: float, confidence: float = 0.95, power: float = 0.80,
    paired_rho: float = 0.0,
) -> int:
    """Samples **per method** needed to resolve a gap of size ``d``.

    Normal approximation for two proportions::

        n = (z_{1-a/2} * sqrt(2*pbar*(1-pbar)) + z_{power} * sqrt(p1*q1 + p2*q2))^2 / d^2

    Args:
        p_baseline: baseline proportion, e.g. 0.8177 for POPE-Adversarial SID.
        d: absolute gap to detect in proportion units (0.005 = 0.5 points).
        confidence: 1 - type-I error rate (two-sided).
        power: desired 1 - type-II error rate.
        paired_rho: correlation between the two methods' per-sample outcomes.
            For paired runs on identical inputs this is typically 0.5-0.9 and
            shrinks the requirement by ``(1 - paired_rho)``; leave at 0 for the
            independent (conservative) answer.

    Returns:
        Required ``n`` per method, rounded up.

    Example:
        Resolving the 0.16-point POPE-Adversarial gap Fox reports over SID
        (81.93 vs 81.77) needs on the order of a million questions per method
        in the unpaired design -- which is the point of the function.
    """
    if not 0.0 <= p_baseline <= 1.0:
        raise ValueError("p_baseline must be a proportion")
    if d <= 0:
        raise ValueError("d must be > 0")
    if not 0.0 <= paired_rho < 1.0:
        raise ValueError("paired_rho must be in [0,1)")
    p1 = p_baseline
    p2 = min(1.0, max(0.0, p_baseline + d))
    pbar = (p1 + p2) / 2.0
    z_a = z_for_confidence(confidence)
    z_b = _ND.inv_cdf(power)
    num = (
        z_a * math.sqrt(2.0 * pbar * (1.0 - pbar))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n = num / (d * d)
    n *= (1.0 - paired_rho)
    return int(math.ceil(n))


def resolvable_gap(
    n: int, p_baseline: float, confidence: float = 0.95, power: float = 0.80,
    paired_rho: float = 0.0, tol: float = 1e-6,
) -> float:
    """Smallest gap resolvable with ``n`` samples per method (inverse of above).

    Found by bisection on :func:`required_n_for_gap`, which is monotone in ``d``.
    """
    lo, hi = tol, 1.0 - tol
    if required_n_for_gap(p_baseline, hi, confidence, power, paired_rho) > n:
        return hi
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if required_n_for_gap(p_baseline, mid, confidence, power, paired_rho) > n:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return hi


def power_table(
    p_baseline: float, gaps: Sequence[float] = (0.005, 0.01, 0.02, 0.05),
    confidence: float = 0.95, power: float = 0.80,
    paired_rho: Sequence[float] = (0.0, 0.5, 0.8),
) -> List[Dict[str, float]]:
    """Rows of ``{gap, paired_rho, required_n}`` for a planning table.

    Used by ``bench/report.py`` to print, next to each benchmark, the n that
    would have been needed to resolve the gaps the paper reports.
    """
    rows: List[Dict[str, float]] = []
    for d in gaps:
        for rho in paired_rho:
            rows.append({
                "gap": d,
                "paired_rho": rho,
                "required_n_per_method": required_n_for_gap(
                    p_baseline, d, confidence, power, rho
                ),
            })
    return rows


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: quick interval / planning answers without writing a script."""
    import argparse

    ap = argparse.ArgumentParser(description="Wilson intervals and sample-size planning.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ci = sub.add_parser("ci", help="Wilson interval for k/n")
    ci.add_argument("k", type=int)
    ci.add_argument("n", type=int)
    ci.add_argument("--confidence", type=float, default=0.95)

    cmp_ = sub.add_parser("compare", help="independent comparison of k1/n1 vs k2/n2")
    for arg in ("k1", "n1", "k2", "n2"):
        cmp_.add_argument(arg, type=int)
    cmp_.add_argument("--confidence", type=float, default=0.95)

    pw = sub.add_parser("power", help="n needed to resolve a gap")
    pw.add_argument("--p", type=float, required=True, help="baseline proportion")
    pw.add_argument("--gap", type=float, required=True, help="absolute gap (0.01 = 1 point)")
    pw.add_argument("--rho", type=float, default=0.0)
    pw.add_argument("--power", type=float, default=0.80)

    rg = sub.add_parser("resolvable", help="smallest gap resolvable at a given n")
    rg.add_argument("--n", type=int, required=True)
    rg.add_argument("--p", type=float, required=True)
    rg.add_argument("--rho", type=float, default=0.0)

    args = ap.parse_args(argv)
    if args.cmd == "ci":
        print(wilson_interval(args.k, args.n, args.confidence).pct())
    elif args.cmd == "compare":
        print(compare_proportions(args.k1, args.n1, args.k2, args.n2,
                                  confidence=args.confidence).verdict())
    elif args.cmd == "power":
        n = required_n_for_gap(args.p, args.gap, power=args.power, paired_rho=args.rho)
        print(f"n per method = {n} (baseline p={args.p}, gap={args.gap}, rho={args.rho})")
    elif args.cmd == "resolvable":
        d = resolvable_gap(args.n, args.p, paired_rho=args.rho)
        print(f"smallest resolvable gap at n={args.n}: {100 * d:.3f} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
