"""Tests for :mod:`bench.stats`.

The Wilson interval is checked three independent ways, because an interval
implemented from memory is exactly the kind of bug that silently changes every
conclusion in the project:

1. **Published numeric examples.** The 95% Wilson bounds tabulated in S.
   Wallis, "Inferential statistics" (UCL Survey of English Usage /
   corplingstats, table of attributive-AJP probabilities in ICE-GB):

       p = 0.0452, n = 155    -> [0.0220, 0.0903]
       p = 0.0526, n = 2944   -> [0.0451, 0.0613]
       p = 0.0789, n = 37305  -> [0.0762, 0.0817]

2. **Numerical inversion of the score test**, which is the *definition* of the
   interval: the bounds are the roots of (p_hat - p)^2 = z^2 p(1-p)/n. Found
   here by stdlib bisection, so it shares no algebra with the implementation.

3. **The algebraically equivalent p_hat form** of the interval, written out
   separately from the count form used in the implementation.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


from foxrepro.bench.stats import (  # noqa: E402
    compare_proportions,
    mcnemar_exact,
    newcombe_diff_interval,
    paired_bootstrap_proportion,
    paired_bootstrap_ratio,
    power_table,
    required_n_for_gap,
    resolvable_gap,
    two_proportion_ztest,
    wilson_halfwidth,
    wilson_interval,
    z_for_confidence,
)


def _score_bounds_by_bisection(k: int, n: int, z: float) -> tuple[float, float]:
    """Invert the score test numerically: roots of (p_hat-p)^2 - z^2 p(1-p)/n."""
    p_hat = k / n

    def f(p: float) -> float:
        return (p_hat - p) ** 2 - z * z * p * (1.0 - p) / n

    def bisect(lo: float, hi: float) -> float:
        # f is positive outside the interval and negative inside it.
        for _ in range(300):
            mid = (lo + hi) / 2.0
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    lower = bisect(0.0, p_hat) if p_hat > 0 else 0.0
    upper = bisect(1.0, p_hat) if p_hat < 1 else 1.0
    return lower, upper


def _wilson_phat_form(k: int, n: int, z: float) -> tuple[float, float]:
    """Independent transcription of the p_hat form of the Wilson interval."""
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    spread = (z / (2.0 * n)) * math.sqrt(4.0 * n * p_hat * (1.0 - p_hat) + z * z) / denom
    return centre - spread, centre + spread


class TestZQuantile(unittest.TestCase):
    def test_95_percent(self) -> None:
        self.assertAlmostEqual(z_for_confidence(0.95), 1.959963985, places=7)

    def test_99_percent(self) -> None:
        self.assertAlmostEqual(z_for_confidence(0.99), 2.575829304, places=7)

    def test_rejects_bad_confidence(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                z_for_confidence(bad)


class TestWilsonAgainstPublishedExamples(unittest.TestCase):
    """Reference values from the UCL/corplingstats Wilson table (95%)."""

    CASES = [
        # (successes, n, published_lower, published_upper)
        (7, 155, 0.0220, 0.0903),
        (155, 2944, 0.0451, 0.0613),
        (2943, 37305, 0.0762, 0.0817),
    ]

    def test_published_bounds(self) -> None:
        for k, n, lo_ref, hi_ref in self.CASES:
            with self.subTest(k=k, n=n):
                got = wilson_interval(k, n, 0.95)
                self.assertAlmostEqual(got.lo, lo_ref, delta=5e-5)
                self.assertAlmostEqual(got.hi, hi_ref, delta=5e-5)


class TestWilsonSelfConsistency(unittest.TestCase):
    GRID = [
        (0, 10), (1, 10), (5, 10), (10, 10),
        (0, 1), (1, 1),
        (3, 500), (232, 500), (250, 500), (497, 500),
        (2458, 3000), (2680, 3000), (1500, 3000),
        (1, 100000), (99999, 100000),
    ]

    def test_matches_score_test_inversion(self) -> None:
        z = z_for_confidence(0.95)
        for k, n in self.GRID:
            with self.subTest(k=k, n=n):
                got = wilson_interval(k, n, 0.95)
                lo_ref, hi_ref = _score_bounds_by_bisection(k, n, z)
                self.assertAlmostEqual(got.lo, max(0.0, lo_ref), places=9)
                self.assertAlmostEqual(got.hi, min(1.0, hi_ref), places=9)

    def test_matches_phat_form(self) -> None:
        for conf in (0.90, 0.95, 0.99):
            z = z_for_confidence(conf)
            for k, n in self.GRID:
                with self.subTest(k=k, n=n, conf=conf):
                    got = wilson_interval(k, n, conf)
                    lo_ref, hi_ref = _wilson_phat_form(k, n, z)
                    self.assertAlmostEqual(got.lo, max(0.0, lo_ref), places=9)
                    self.assertAlmostEqual(got.hi, min(1.0, hi_ref), places=9)

    def test_symmetry_under_complement(self) -> None:
        """The interval for k/n mirrors the interval for (n-k)/n."""
        for k, n in self.GRID:
            a = wilson_interval(k, n)
            b = wilson_interval(n - k, n)
            self.assertAlmostEqual(a.lo, 1.0 - b.hi, places=12)
            self.assertAlmostEqual(a.hi, 1.0 - b.lo, places=12)

    def test_edges_are_exact(self) -> None:
        self.assertEqual(wilson_interval(0, 50).lo, 0.0)
        self.assertEqual(wilson_interval(50, 50).hi, 1.0)
        self.assertLess(wilson_interval(0, 50).hi, 0.08)
        self.assertGreater(wilson_interval(50, 50).lo, 0.92)

    def test_point_estimate_inside_interval(self) -> None:
        for k, n in self.GRID:
            got = wilson_interval(k, n)
            self.assertLessEqual(got.lo, got.point + 1e-12)
            self.assertGreaterEqual(got.hi, got.point - 1e-12)

    def test_width_shrinks_with_n(self) -> None:
        widths = [wilson_interval(round(0.8 * n), n).halfwidth for n in (100, 500, 3000, 20000)]
        self.assertEqual(widths, sorted(widths, reverse=True))

    def test_zero_denominator_is_degenerate_not_fatal(self) -> None:
        got = wilson_interval(0, 0)
        self.assertTrue(math.isnan(got.point))
        self.assertEqual((got.lo, got.hi), (0.0, 1.0))

    def test_rejects_impossible_counts(self) -> None:
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)
        with self.assertRaises(ValueError):
            wilson_interval(-1, 10)


class TestBenchmarkScaleSanity(unittest.TestCase):
    """The project's headline claim: POPE gives ~+-1.4 pts, CHAIR_S ~+-4.4 pts."""

    def test_pope_halfwidth_at_n_3000(self) -> None:
        hw = wilson_halfwidth(0.82, 3000)
        self.assertAlmostEqual(100 * hw, 1.37, delta=0.10)

    def test_chair_s_halfwidth_at_n_500(self) -> None:
        hw = wilson_halfwidth(0.50, 500)
        self.assertAlmostEqual(100 * hw, 4.38, delta=0.15)

    def test_pct_rendering(self) -> None:
        text = wilson_interval(2458, 3000, name="pope_accuracy").pct()
        self.assertTrue(text.startswith("81.93 ["), text)

    def test_as_dict_reports_denominator(self) -> None:
        d = wilson_interval(64, 497, name="chair_i").as_dict()
        self.assertEqual(d["n"], 497)
        self.assertEqual(d["successes"], 64.0)
        self.assertEqual(d["name"], "chair_i")


class TestTwoProportionComparison(unittest.TestCase):
    def test_newcombe_interval_contains_difference(self) -> None:
        lo, hi = newcombe_diff_interval(2458, 3000, 2453, 3000)
        d = 2458 / 3000 - 2453 / 3000
        self.assertLess(lo, d)
        self.assertGreater(hi, d)

    def test_paper_pope_adversarial_gap_is_not_resolvable(self) -> None:
        """Fox 81.93 vs SID 81.10 on POPE-Adversarial, n = 3000 each."""
        cmp_ = compare_proportions(
            round(0.8193 * 3000), 3000, round(0.8110 * 3000), 3000,
            label_a="Fox", label_b="SID",
        )
        self.assertFalse(cmp_.significant)
        self.assertIn("NOT distinguishable", cmp_.verdict())

    def test_large_gap_is_resolvable(self) -> None:
        """Fox 81.93 vs plain sampling 76.77 on POPE-Adversarial."""
        cmp_ = compare_proportions(
            round(0.8193 * 3000), 3000, round(0.7677 * 3000), 3000,
            label_a="Fox", label_b="Sampling",
        )
        self.assertTrue(cmp_.significant)
        self.assertLess(cmp_.p_value or 1.0, 1e-6)

    def test_chair_s_gap_of_7_points_at_n_500(self) -> None:
        """CHAIR_S 46.40 vs 53.40 (Fox vs SID) -- borderline at n = 500."""
        cmp_ = compare_proportions(232, 500, 267, 500, label_a="Fox", label_b="SID")
        self.assertAlmostEqual(100 * cmp_.diff, -7.0, places=6)
        # Newcombe interval half-width here is ~6.2 points, so a 7-point gap
        # only just clears zero; assert the bounds rather than a verdict.
        self.assertLess(cmp_.hi, 0.0)

    def test_identical_proportions_give_p_one(self) -> None:
        self.assertEqual(two_proportion_ztest(100, 200, 100, 200), 1.0)

    def test_ztest_matches_known_value(self) -> None:
        """z = 2 corresponds to p ~= 0.0455 two-sided."""
        # 60/100 vs 40/100: pooled p = 0.5, se = sqrt(.25*(.02)) = 0.0707,
        # z = 0.2/0.0707 = 2.828 -> p = 0.00468
        p = two_proportion_ztest(60, 100, 40, 100)
        self.assertAlmostEqual(p, 0.00468, delta=5e-5)

    def test_zero_denominator(self) -> None:
        self.assertEqual(two_proportion_ztest(0, 0, 1, 10), 1.0)


class TestPairedBootstrap(unittest.TestCase):
    def test_paired_is_tighter_than_independent_when_correlated(self) -> None:
        """Highly correlated methods: the paired interval must be narrower."""
        n = 1000
        a = [1 if i % 10 else 0 for i in range(n)]          # 90% correct
        b = list(a)
        for i in range(0, n, 50):                            # B loses 20 cases
            b[i] = 0
        paired = paired_bootstrap_proportion(a, b, n_boot=2000, seed=7)
        indep = compare_proportions(sum(a), n, sum(b), n)
        self.assertLess(paired.hi - paired.lo, indep.hi - indep.lo)
        self.assertAlmostEqual(paired.diff, sum(a) / n - sum(b) / n, places=12)

    def test_identical_outcomes_give_zero_width_interval(self) -> None:
        a = [1, 0, 1, 1, 0] * 40
        res = paired_bootstrap_proportion(a, list(a), n_boot=500, seed=1)
        self.assertEqual(res.diff, 0.0)
        self.assertAlmostEqual(res.lo, 0.0, places=12)
        self.assertAlmostEqual(res.hi, 0.0, places=12)
        self.assertFalse(res.significant)

    def test_deterministic_for_fixed_seed(self) -> None:
        a = [i % 3 == 0 for i in range(300)]
        b = [i % 4 == 0 for i in range(300)]
        r1 = paired_bootstrap_proportion(a, b, n_boot=1000, seed=42)
        r2 = paired_bootstrap_proportion(a, b, n_boot=1000, seed=42)
        self.assertEqual((r1.lo, r1.hi, r1.p_value), (r2.lo, r2.hi, r2.p_value))

    def test_rejects_misaligned_vectors(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_proportion([1, 0], [1, 0, 1], n_boot=200)

    def test_rejects_tiny_bootstrap(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_proportion([1, 0], [0, 1], n_boot=10)

    def test_ratio_bootstrap_reports_mention_denominators(self) -> None:
        """CHAIR_I style: numerator and denominator both vary per caption."""
        n = 400
        num_a = [1 if i % 8 == 0 else 0 for i in range(n)]
        den_a = [5] * n
        num_b = [1 if i % 4 == 0 else 0 for i in range(n)]
        den_b = [5] * n
        res = paired_bootstrap_ratio(num_a, den_a, num_b, den_b, n_boot=2000, seed=3)
        self.assertEqual(res.n_a, 2000)   # total mentioned instances, not captions
        self.assertEqual(res.n_b, 2000)
        self.assertAlmostEqual(res.p_a, sum(num_a) / 2000, places=12)
        self.assertTrue(res.significant)

    def test_ratio_bootstrap_rejects_misaligned(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_ratio([1], [2], [1, 1], [2, 2], n_boot=200)


class TestMcNemar(unittest.TestCase):
    def test_known_discordant_counts(self) -> None:
        """b=10, c=0 -> two-sided exact p = 2 * 0.5^10 = 0.001953125."""
        a = [1] * 10 + [1] * 90
        b = [0] * 10 + [1] * 90
        res = mcnemar_exact(a, b)
        self.assertAlmostEqual(res.p_value or 0.0, 2.0 * 0.5 ** 10, places=12)
        self.assertTrue(res.significant)

    def test_balanced_discordance_is_not_significant(self) -> None:
        a = [1] * 20 + [0] * 20 + [1] * 60
        b = [0] * 20 + [1] * 20 + [1] * 60
        res = mcnemar_exact(a, b)
        self.assertAlmostEqual(res.diff, 0.0, places=12)
        self.assertFalse(res.significant)

    def test_no_discordance(self) -> None:
        a = [1, 1, 0, 0] * 25
        res = mcnemar_exact(a, list(a))
        self.assertEqual(res.p_value, 1.0)
        self.assertFalse(res.significant)

    def test_is_tighter_than_independent_comparison(self) -> None:
        n = 3000
        a = [1 if i % 10 else 0 for i in range(n)]
        b = list(a)
        for i in range(0, n, 25):
            b[i] = 0
        mc = mcnemar_exact(a, b)
        ind = compare_proportions(sum(a), n, sum(b), n)
        self.assertLess(mc.hi - mc.lo, ind.hi - ind.lo)


class TestPowerHelpers(unittest.TestCase):
    def test_required_n_matches_textbook_formula(self) -> None:
        """p1=0.5, p2=0.6, alpha=0.05, power=0.80 -> ~388-408 per arm.

        The standard worked answer for this classic example is 388 per group
        with the pooled-variance form used here.
        """
        n = required_n_for_gap(0.5, 0.10, confidence=0.95, power=0.80)
        self.assertGreaterEqual(n, 380)
        self.assertLessEqual(n, 420)

    def test_tiny_gaps_need_absurd_n(self) -> None:
        """The 0.16-point POPE gap Fox reports over SID (81.93 vs 81.77)."""
        n = required_n_for_gap(0.8177, 0.0016)
        self.assertGreater(n, 500_000)

    def test_monotone_in_gap(self) -> None:
        ns = [required_n_for_gap(0.8, d) for d in (0.05, 0.02, 0.01, 0.005)]
        self.assertEqual(ns, sorted(ns))

    def test_pairing_reduces_requirement(self) -> None:
        base = required_n_for_gap(0.8, 0.01, paired_rho=0.0)
        paired = required_n_for_gap(0.8, 0.01, paired_rho=0.8)
        self.assertAlmostEqual(paired / base, 0.2, delta=0.02)

    def test_resolvable_gap_round_trips(self) -> None:
        for n in (500, 3000, 20000):
            d = resolvable_gap(n, 0.8)
            self.assertLessEqual(required_n_for_gap(0.8, d), n + 1)

    def test_resolvable_gap_at_pope_scale(self) -> None:
        """n=3000 per method resolves gaps of roughly 2.5-3 points, not 0.5."""
        d = resolvable_gap(3000, 0.82)
        self.assertGreater(100 * d, 1.5)
        self.assertLess(100 * d, 4.0)

    def test_power_table_shape(self) -> None:
        rows = power_table(0.8, gaps=(0.01, 0.02), paired_rho=(0.0, 0.5))
        self.assertEqual(len(rows), 4)
        self.assertEqual(set(rows[0]), {"gap", "paired_rho", "required_n_per_method"})

    def test_rejects_bad_inputs(self) -> None:
        with self.assertRaises(ValueError):
            required_n_for_gap(1.5, 0.01)
        with self.assertRaises(ValueError):
            required_n_for_gap(0.8, 0.0)
        with self.assertRaises(ValueError):
            required_n_for_gap(0.8, 0.01, paired_rho=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
