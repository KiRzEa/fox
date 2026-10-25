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


class TestStudentT(unittest.TestCase):
    """stats.py is stdlib-only, so the t distribution is implemented here.

    Checked against published tables rather than against scipy, since scipy
    is deliberately not a dependency of the CPU metrics path.
    """

    TABLE_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                9: 2.262, 10: 2.228, 19: 2.093, 29: 2.045, 99: 1.984}

    def test_two_sided_95_quantiles_match_the_table(self) -> None:
        from foxrepro.bench.stats import t_for_confidence

        for df, want in self.TABLE_95.items():
            with self.subTest(df=df):
                self.assertAlmostEqual(t_for_confidence(0.95, df), want, places=3)

    def test_99_percent_quantiles_match_the_table(self) -> None:
        from foxrepro.bench.stats import t_for_confidence

        for df, want in {1: 63.657, 4: 4.604, 9: 3.250, 29: 2.756}.items():
            with self.subTest(df=df):
                self.assertAlmostEqual(t_for_confidence(0.99, df), want, places=3)

    def test_large_df_approaches_the_normal(self) -> None:
        from foxrepro.bench.stats import t_for_confidence, z_for_confidence

        self.assertAlmostEqual(
            t_for_confidence(0.95, 1e7), z_for_confidence(0.95), places=4
        )

    def test_the_small_sample_penalty_is_the_whole_point(self) -> None:
        """At 3 seeds the multiplier is more than twice the normal one.

        Using z where t belongs is how a three-seed interval ends up looking
        as tight as a thirty-seed one.
        """
        from foxrepro.bench.stats import t_for_confidence, z_for_confidence

        self.assertGreater(t_for_confidence(0.95, 2), 2 * z_for_confidence(0.95))

    def test_sf_and_quantile_are_inverses(self) -> None:
        from foxrepro.bench.stats import student_t_sf, t_for_confidence

        for df in (1, 2, 4, 9, 30):
            for conf in (0.80, 0.95, 0.99):
                with self.subTest(df=df, conf=conf):
                    t = t_for_confidence(conf, df)
                    self.assertAlmostEqual(student_t_sf(t, df), 1 - conf, places=6)

    def test_sf_is_symmetric_and_bounded(self) -> None:
        from foxrepro.bench.stats import student_t_sf

        self.assertAlmostEqual(student_t_sf(1.5, 7), student_t_sf(-1.5, 7), places=12)
        self.assertEqual(student_t_sf(0.0, 7), 1.0)
        self.assertEqual(student_t_sf(float("inf"), 7), 0.0)

    def test_bad_arguments_are_rejected(self) -> None:
        from foxrepro.bench.stats import student_t_sf, t_for_confidence

        with self.assertRaises(ValueError):
            student_t_sf(1.0, 0)
        with self.assertRaises(ValueError):
            t_for_confidence(1.0, 5)


class TestSeedLevelComparison(unittest.TestCase):
    """The interval a single-seed run cannot produce.

    Every other interval in this module quantifies which ITEMS were sampled.
    With do_sample=True there is a second source -- which GENERATION was
    drawn -- and it is invisible to any one run. The plan replicates the
    headline arms over ten seeds so this function has something to measure.
    """

    def test_a_single_seed_reports_no_interval_rather_than_a_narrow_one(self) -> None:
        from foxrepro.bench.stats import seed_level_comparison

        c = seed_level_comparison({42: 77.17}, {42: 81.40})
        self.assertEqual(c.n_seeds, 1)
        self.assertEqual(c.df, 0)
        self.assertEqual(c.lo, -math.inf)
        self.assertEqual(c.hi, math.inf)
        self.assertIsNone(c.p_value)
        self.assertFalse(c.significant)
        self.assertAlmostEqual(c.mean_gap, 4.23, places=10)

    def test_pairing_is_by_seed(self) -> None:
        """Seed 43's fox against seed 42's baseline is not a paired gap."""
        from foxrepro.bench.stats import seed_level_comparison

        c = seed_level_comparison({42: 10.0, 43: 20.0}, {43: 25.0, 44: 99.0})
        self.assertEqual(c.seeds, (43,))
        self.assertEqual(c.gaps, (5.0,))

    def test_no_shared_seed_raises_and_says_which_are_present(self) -> None:
        from foxrepro.bench.stats import seed_level_comparison

        with self.assertRaises(ValueError) as ctx:
            seed_level_comparison({42: 1.0}, {43: 2.0}, "base", "fox")
        self.assertIn("42", str(ctx.exception))
        self.assertIn("43", str(ctx.exception))

    def test_a_consistent_gap_is_significant_and_a_noisy_one_is_not(self) -> None:
        from foxrepro.bench.stats import seed_level_comparison

        seeds = list(range(42, 47))
        tight = seed_level_comparison(
            {s: 77.0 + 0.1 * i for i, s in enumerate(seeds)},
            {s: 81.0 + 0.1 * i for i, s in enumerate(seeds)},
        )
        self.assertAlmostEqual(tight.mean_gap, 4.0, places=10)
        self.assertAlmostEqual(tight.sd_gap, 0.0, places=12)
        self.assertTrue(tight.significant)

        noisy = seed_level_comparison(
            {s: 77.0 for s in seeds},
            {s: 77.0 + g for s, g in zip(seeds, (-6.0, 8.0, -4.0, 7.0, -3.0))},
        )
        self.assertAlmostEqual(noisy.mean_gap, 0.4, places=10)
        self.assertFalse(noisy.significant)
        self.assertGreater(noisy.p_value, 0.05)

    def test_the_interval_uses_t_not_z(self) -> None:
        """Three seeds, so the multiplier must be 4.303 and not 1.960."""
        from foxrepro.bench.stats import seed_level_comparison

        vals_a = {42: 0.0, 43: 0.0, 44: 0.0}
        vals_b = {42: 1.0, 43: 2.0, 44: 3.0}      # gaps 1,2,3 -> mean 2, sd 1
        c = seed_level_comparison(vals_a, vals_b)
        se = 1.0 / math.sqrt(3)
        self.assertAlmostEqual(c.hi - c.mean_gap, 4.3027 * se, places=3)

    def test_best_of_respects_the_metric_direction(self) -> None:
        """CHAIR is lower-is-better; accuracy is not.

        This exists because the paper reports "the best results averaged over
        10 independent runs" (C.2) without saying what "best" ranges over. If
        it ranges over the runs, the comparable number from this side is a
        best-of too, not a mean -- so both are reported and labelled.
        """
        from foxrepro.bench.stats import seed_level_comparison

        acc = seed_level_comparison({42: 70.0, 43: 72.0}, {42: 80.0, 43: 85.0},
                                    higher_is_better=True)
        self.assertEqual((acc.best_a, acc.best_b), (72.0, 85.0))
        self.assertAlmostEqual(acc.best_gap, 13.0, places=10)
        self.assertAlmostEqual(acc.mean_gap, 11.5, places=10)

        chair = seed_level_comparison({42: 57.0, 43: 59.0}, {42: 49.0, 43: 46.0},
                                      higher_is_better=False)
        self.assertEqual((chair.best_a, chair.best_b), (57.0, 46.0))
        self.assertAlmostEqual(chair.best_gap, -11.0, places=10)
        self.assertAlmostEqual(chair.mean_gap, -10.5, places=10)
        # best-of flatters the method more than the mean does, in both cases
        self.assertLess(chair.best_gap, chair.mean_gap)

    def test_identical_gaps_every_seed_is_reported_without_a_fake_p(self) -> None:
        """Greedy decoding is deterministic, so this is the expected shape."""
        from foxrepro.bench.stats import seed_level_comparison

        c = seed_level_comparison({42: 1.0, 43: 1.0}, {42: 3.0, 43: 3.0})
        self.assertEqual(c.sd_gap, 0.0)
        self.assertEqual(c.lo, c.hi)
        self.assertEqual(c.p_value, 0.0)
        self.assertEqual(c.t_stat, math.inf)

    def test_covers_answers_the_paper_question_without_overclaiming(self) -> None:
        from foxrepro.bench.stats import seed_level_comparison

        c = seed_level_comparison(
            {s: 0.0 for s in range(42, 47)},
            {s: g for s, g in zip(range(42, 47), (4.0, 4.2, 4.4, 4.1, 4.3))},
        )
        self.assertTrue(c.covers(4.2))
        self.assertFalse(c.covers(5.16))

    def test_as_dict_is_json_ready(self) -> None:
        import json

        from foxrepro.bench.stats import seed_level_comparison

        c = seed_level_comparison({42: 1.0, 43: 2.0}, {42: 3.0, 43: 5.0})
        json.dumps(c.as_dict())


class TestEquivalence(unittest.TestCase):
    """A large p from a difference test is not evidence that two arms agree.

    It is the probability of the data IF they agree, and it rises as the
    measurement gets noisier -- two seeds give a larger p than ten. The claim
    this project makes about ulp_control is the positive one, that it does the
    same thing as fox, so it needs a test whose null is "they differ".
    """

    def test_a_tight_zero_difference_is_equivalent(self) -> None:
        from foxrepro.bench.stats import tost_equivalence

        a = {s: 80.0 for s in range(42, 52)}
        b = {s: 80.0 + d for s, d in zip(range(42, 52),
                                         (0.1, -0.1, 0.05, 0, -0.05,
                                          0.1, -0.1, 0, 0.05, -0.05))}
        e = tost_equivalence(a, b, margin=0.5, label_a="fox", label_b="ulp")
        self.assertTrue(e.equivalent)
        self.assertLess(e.p_value, 0.05)
        self.assertIn("equivalent", e.verdict())

    def test_noise_makes_equivalence_harder_not_easier(self) -> None:
        """The property that distinguishes TOST from reading a large p.

        Same mean difference of zero, twenty times the spread: the ordinary test
        still fails to reject, but TOST correctly refuses to call it
        equivalent.
        """
        from foxrepro.bench.stats import seed_level_comparison, tost_equivalence

        seeds = range(42, 52)
        small = (0.1, -0.1, 0.05, 0, -0.05, 0.1, -0.1, 0, 0.05, -0.05)
        a = {s: 80.0 for s in seeds}
        tight = {s: 80.0 + d for s, d in zip(seeds, small)}
        loose = {s: 80.0 + 20 * d for s, d in zip(seeds, small)}

        for arm in (tight, loose):
            self.assertGreater(
                seed_level_comparison(a, arm).p_value, 0.05,
                "the difference test should fail to reject either way")
        self.assertTrue(tost_equivalence(a, tight, 0.5).equivalent)
        self.assertFalse(tost_equivalence(a, loose, 0.5).equivalent)

    def test_a_real_difference_is_not_equivalent(self) -> None:
        from foxrepro.bench.stats import tost_equivalence

        a = {s: 80.0 for s in range(42, 52)}
        b = {s: 83.0 + 0.01 * i for i, s in enumerate(range(42, 52))}
        e = tost_equivalence(a, b, margin=0.5)
        self.assertFalse(e.equivalent)
        self.assertGreater(e.p_value, 0.05)
        self.assertIn("NOT shown equivalent", e.verdict())

    def test_a_wider_margin_is_easier_to_satisfy(self) -> None:
        from foxrepro.bench.stats import tost_equivalence

        a = {s: 80.0 for s in range(42, 52)}
        b = {s: 80.4 + 0.01 * i for i, s in enumerate(range(42, 52))}
        self.assertFalse(tost_equivalence(a, b, 0.25).equivalent)
        self.assertTrue(tost_equivalence(a, b, 1.0).equivalent)

    def test_the_interval_is_the_one_tost_corresponds_to(self) -> None:
        """TOST at alpha matches a 1-2*alpha interval, not 1-alpha."""
        from foxrepro.bench.stats import t_for_confidence, tost_equivalence
        import math

        a = {s: 80.0 for s in range(42, 52)}
        b = {s: 80.0 + d for s, d in zip(range(42, 52),
                                         (0.3, -0.2, 0.1, 0.0, -0.1,
                                          0.2, -0.3, 0.1, 0.0, -0.1))}
        e = tost_equivalence(a, b, 0.5, alpha=0.05)
        se = e.sd_diff / math.sqrt(e.n)
        self.assertAlmostEqual(e.ci_hi - e.mean_diff,
                               t_for_confidence(0.90, e.df) * se, places=9)

    def test_one_replicate_gives_no_verdict(self) -> None:
        from foxrepro.bench.stats import tost_equivalence

        e = tost_equivalence({42: 80.0}, {42: 80.0}, 0.5)
        self.assertFalse(e.equivalent)
        self.assertIsNone(e.p_value)
        self.assertEqual(e.ci_lo, -math.inf)

    def test_a_nonpositive_margin_is_rejected(self) -> None:
        from foxrepro.bench.stats import tost_equivalence

        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                tost_equivalence({42: 1.0, 43: 1.0}, {42: 1.0, 43: 1.0}, bad)


class TestHolm(unittest.TestCase):
    """This report makes dozens of comparisons and criticises the paper for
    an unverifiable significance claim. It cannot make an uncorrected one."""

    def test_matches_the_worked_example(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        # p sorted: .01 x4, .02 x3, .03 x2, .04 x1
        self.assertEqual(
            [round(x, 10) for x in holm_adjust([0.01, 0.02, 0.03, 0.04])],
            [0.04, 0.06, 0.06, 0.06],
        )

    def test_output_order_matches_input_order(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        # sorted: 0.01 at rank 0 -> 2*0.01 = 0.02; 0.04 at rank 1 -> 1*0.04,
        # kept monotone at max(0.02, 0.04) = 0.04.
        self.assertEqual([round(x, 10) for x in holm_adjust([0.04, 0.01])],
                         [0.04, 0.02])

    def test_adjusted_values_are_monotone_and_capped(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        raw = [0.001, 0.2, 0.4, 0.9, 0.02]
        adj = holm_adjust(raw)
        ranked = [a for _, a in sorted(zip(raw, adj))]
        self.assertEqual(ranked, sorted(ranked))
        self.assertTrue(all(a <= 1.0 for a in adj))

    def test_it_never_makes_a_p_value_smaller(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        raw = [0.001, 0.049, 0.05, 0.3]
        for r, a in zip(raw, holm_adjust(raw)):
            self.assertGreaterEqual(a, r)

    def test_none_passes_through_and_is_not_counted(self) -> None:
        """A one-seed row has no p-value and must not inflate the family."""
        from foxrepro.bench.stats import holm_adjust

        with_none = holm_adjust([0.01, None, 0.02])
        without = holm_adjust([0.01, 0.02])
        self.assertIsNone(with_none[1])
        self.assertEqual([with_none[0], with_none[2]], without)

    def test_a_single_comparison_is_unchanged(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        self.assertEqual(holm_adjust([0.03]), [0.03])

    def test_an_empty_family_is_not_an_error(self) -> None:
        from foxrepro.bench.stats import holm_adjust

        self.assertEqual(holm_adjust([]), [])
        self.assertEqual(holm_adjust([None, None]), [None, None])
