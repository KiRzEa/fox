"""The validation arms: do the ported methods do what their papers say?

Bit-equality against a transcribed reference (``test_kernel_equivalence``)
proves *our code == their code*. It does not prove *their code == the paper*,
and it proves nothing about whether a mechanism is active on a real model.
Five of the six compared methods here were implemented from source by this
project, so that gap is closed by measurement -- and these are the tests that
make sure the measuring arms themselves are wired correctly, before any GPU
time is spent on them.

The arms, and the prediction each one makes:

``negative_only``      every contrastive method says its negative branch is a
                       WORSE model. Run it alone; it must score below the
                       baseline.
``alpha = 0``          the formula collapses to ``z_obs``, so the output must
                       be token-identical to the baseline.
``sid_keep_most``      SID keeps the LEAST attended tokens; FastV (one line
                       up, commented out) keeps the most. Least must be worse.
``sid_random``         if 100 random tokens are as bad, the selection does
                       nothing.
``cf_constant``        CausalMM's gain is 1/n_keys by construction; if a flat
``cf_blind``           1/n_keys, or zero, gives the same answers, the vision
                       counterfactual is not what is doing the work.
``opera_*``            OPERA is a beam search + a candidate restriction + a
                       penalty; these separate the three.
"""

from __future__ import annotations

import glob
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

from foxrepro.bench.generate import GenConfig, get_method  # noqa: E402
from foxrepro.storage import CONTROL_ARMS  # noqa: E402

from tests.test_methods import _Base  # noqa: E402


def cfg(method: str, **over):
    base = dict(
        method=method, benchmark="pope", split="adversarial", seed=42,
        dtype="fp16", model_id="m", prompt_template_id="pope_v0_raw",
        max_new_tokens=1,
    )
    base.update(over)
    return GenConfig(**base)


class TestTheArchiveStaysReachable(unittest.TestCase):
    """Adding ``control`` must not rename a single existing run file.

    ``control`` is in the run key, and the run key is a hash of a JSON
    payload, so including ``"control": null`` everywhere would change every
    hash this project has ever produced and orphan the whole of
    results/runs/. It is therefore omitted when unset, and this is the test
    that says so about the real files rather than about the intention.
    """

    def test_every_archived_name_is_still_produced(self) -> None:
        files = sorted(glob.glob("results/runs/*.jsonl"))
        if not files:
            self.skipTest("archive not present in this checkout")
        self.assertGreater(len(files), 100)
        import json
        import os

        from foxrepro.storage import (
            OPTIONAL_VARIANT_FIELDS, VARIANT_FIELDS, RunKey,
        )

        checked = 0
        for path in files:
            with open(path, "rb") as fh:
                rec = json.loads(fh.readline())
            if any(f not in rec for f in VARIANT_FIELDS):
                continue
            # Mirrors GenConfig._variant_fields: the optional ones are in the
            # payload only when set, which is what keeps a control-free run's
            # name byte-identical to what it was before `control` existed.
            fields = {f: rec.get(f) for f in VARIANT_FIELDS}
            fields.update({f: rec[f] for f in OPTIONAL_VARIANT_FIELDS
                           if rec.get(f) is not None})
            key = RunKey(
                method=rec["method"], benchmark=rec["benchmark"],
                split=rec["split"], seed=rec["seed"], dtype=rec["dtype"],
                variant_fields=fields,
            )
            with self.subTest(run=os.path.basename(path)):
                self.assertEqual(key.filename(), os.path.basename(path))
            checked += 1
        self.assertGreater(checked, 100, "almost nothing was actually checked")

    def test_a_control_run_gets_its_own_file(self) -> None:
        plain = cfg("vcd").run_key().filename()
        for control in sorted(CONTROL_ARMS):
            with self.subTest(control=control):
                method = "opera" if control.startswith("opera") else "vcd"
                kw = {"decoding": "greedy"} if method == "opera" else {}
                name = cfg(method, control=control, **kw).run_key().filename()
                self.assertNotEqual(name, plain)

    def test_an_unknown_control_is_refused(self) -> None:
        with self.assertRaises(ValueError) as cm:
            cfg("vcd", control="make_it_better")
        self.assertIn("unknown control", str(cm.exception))


class TestControlsApplyToTheRightMethods(unittest.TestCase):
    """A control aimed at the wrong method must raise, not be ignored.

    Silently ignoring it would produce a run file named for a control that
    did not happen -- the worst possible outcome for a validation arm.
    """

    CASES = (
        ("sid", "sid_keep_most", "sid_select", "highest"),
        ("sid", "sid_random", "sid_select", "random"),
        ("sid", None, "sid_select", "lowest"),
        ("causalmm", "cf_constant", "cf_edit", "constant"),
        ("causalmm", "cf_blind", "cf_edit", "blind"),
        ("causalmm", None, "cf_edit", "shuffle"),
    )

    def test_each_control_resolves_to_its_setting(self) -> None:
        for method, control, key, want in self.CASES:
            with self.subTest(method=method, control=control):
                got = get_method(method)(cfg(method, control=control))._record()
                self.assertEqual(got[key], want)
                if control is not None:
                    self.assertEqual(got["control"], control)

    def test_opera_controls_decompose_the_method(self) -> None:
        want = {
            None: (True, 1.0),
            "opera_unrestricted": (False, 1.0),
            "opera_no_penalty": (True, 0.0),
        }
        for control, (restrict, weight) in want.items():
            with self.subTest(control=control):
                m = get_method("opera")(
                    cfg("opera", decoding="greedy", control=control))
                self.assertEqual(m.restrict, restrict)
                self.assertEqual(m.weight, weight)

    def test_a_control_for_another_method_raises(self) -> None:
        for method, control in (("sid", "cf_blind"),
                                ("causalmm", "sid_random"),
                                ("opera", "cf_constant"),
                                ("vcd", "sid_keep_most")):
            with self.subTest(method=method, control=control):
                kw = {"decoding": "greedy"} if method == "opera" else {}
                with self.assertRaises(ValueError) as cm:
                    get_method(method)(
                        cfg(method, control=control, **kw))._record()
                self.assertIn("does not apply", str(cm.exception))

    def test_negative_only_applies_to_all_four(self) -> None:
        for method in ("vcd", "icd", "sid", "causalmm"):
            with self.subTest(method=method):
                got = get_method(method)(
                    cfg(method, control="negative_only"))._record()
                self.assertEqual(got["control"], "negative_only")


class TestNegativeOnlyRunsOneBranch(_Base):
    """The arm must drop the contrast, not merely relabel it."""

    def test_vcd_negative_only_is_single_branch_on_the_noised_image(self) -> None:
        self.run_method("vcd", control="negative_only")
        self.assertEqual(set(self.model.rows_seen), {1})
        self.assertEqual(set(self.model.pixel_rows_seen), {1})

    def test_icd_negative_only_runs_one_pass_on_the_disturbed_prompt(self) -> None:
        m, _ = self.run_method("icd", control="negative_only")
        self.assertEqual(set(self.model.rows_seen), {1})
        self.assertEqual(len(self.processor.seen_text), 2)
        # the clean prompt is built by _prepare and then replaced; the pass
        # that actually ran is the disturbed one
        self.assertIn("confused object detector", self.processor.seen_text[-1])

    def test_the_contrast_is_gone_not_hidden(self) -> None:
        for method in ("vcd", "icd"):
            with self.subTest(method=method):
                _, out = self.run_method(method, control="negative_only")
                self.assertNotIn("n_diverged", out.extra)
                self.assertEqual(out.extra["control"], "negative_only")

    def test_the_paired_arm_still_records_divergence(self) -> None:
        """The diagnostic that says whether a method is doing anything.

        Zero divergence means the contrastive rule never changed a decision,
        which no accuracy number would reveal.
        """
        for method in ("vcd", "icd", "sid"):
            with self.subTest(method=method):
                _, out = self.run_method(method)
                self.assertIn("n_diverged", out.extra)
                self.assertIn("diverged_frac", out.extra)
                self.assertLessEqual(out.extra["diverged_frac"], 1.0)


class TestCausalMMControlsChangeTheGain(unittest.TestCase):
    """``constant`` and ``blind`` pin the magnitude the shuffle happens to have."""

    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_tiny_llava import tiny_llava

        cls.model = tiny_llava()

    def gain(self, edit: str) -> dict:
        from foxrepro.fox.attention import causalmm_vision

        px = torch.randn(1, 3, 336, 336).repeat(2, 1, 1, 1)
        stats: dict = {}
        with causalmm_vision(self.model, rows=(1,), edit=edit,
                             generator=torch.Generator().manual_seed(5),
                             stats=stats):
            with torch.no_grad():
                feats = self.model.model.get_image_features(pixel_values=px)
        return stats, feats

    def test_shuffle_already_has_a_gain_of_one_over_n_keys(self) -> None:
        stats, _ = self.gain("shuffle")
        self.assertAlmostEqual(stats["cf_gain_mean"], 1.0 / 577, places=4)

    def test_constant_pins_that_magnitude_exactly(self) -> None:
        stats, _ = self.gain("constant")
        self.assertAlmostEqual(stats["cf_gain_mean"], 1.0 / 577, places=6)
        # and removes the spread the shuffle had
        self.assertLess(stats["cf_gain_std"], 1e-6)

    def test_blind_removes_the_image_entirely(self) -> None:
        stats, feats = self.gain("blind")
        self.assertEqual(stats["cf_gain_mean"], 0.0)
        # row 1 is then the projector's response to a zero input: constant
        # across visual positions, and different from row 0
        self.assertFalse(torch.equal(feats[0], feats[1]))
        self.assertTrue(torch.allclose(feats[1], feats[1][0].expand_as(feats[1]),
                                       atol=1e-5))


class TestSidSelectionControls(unittest.TestCase):
    def test_the_three_rules_keep_different_sets(self) -> None:
        from foxrepro.fox.kernel import ct2s_keep

        lo, n_img, n_keys = 2, 16, 24
        probs = torch.zeros(1, 1, n_keys, n_keys)
        probs[0, 0, -1, lo:lo + n_img] = torch.arange(1.0, n_img + 1)
        got = {}
        for select in ("lowest", "highest", "random"):
            got[select] = sorted(ct2s_keep(
                probs, lo, lo + n_img - 1, rank=4, select=select,
                generator=torch.Generator().manual_seed(1)).tolist())
        self.assertEqual(got["lowest"], [lo, lo + 1, lo + 2, lo + 3])
        self.assertEqual(got["highest"],
                         [lo + n_img - 4, lo + n_img - 3, lo + n_img - 2,
                          lo + n_img - 1])
        self.assertNotEqual(got["random"], got["lowest"])
        self.assertNotEqual(got["random"], got["highest"])

    def test_random_is_seeded(self) -> None:
        from foxrepro.fox.kernel import ct2s_keep

        probs = torch.rand(1, 1, 24, 24)
        a = ct2s_keep(probs, 2, 17, rank=4, select="random",
                      generator=torch.Generator().manual_seed(1))
        b = ct2s_keep(probs, 2, 17, rank=4, select="random",
                      generator=torch.Generator().manual_seed(1))
        c = ct2s_keep(probs, 2, 17, rank=4, select="random",
                      generator=torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_an_unknown_rule_raises(self) -> None:
        from foxrepro.fox.kernel import ct2s_keep

        with self.assertRaises(ValueError):
            ct2s_keep(torch.rand(1, 1, 8, 8), 1, 4, rank=2, select="best")


class TestOperaControlsOnTheRealSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from foxrepro.fox import attention as A

        from tests.test_tiny_llava import tiny_llava

        A.register()
        cls.model = tiny_llava()
        cls.model.set_attn_implementation("fox")

    def run_opera(self, **kw):
        from foxrepro.fox.opera import opera_generate

        from tests.test_opera import prompt

        with torch.no_grad():
            return opera_generate(self.model, prompt(), max_new_tokens=3, **kw)

    def test_dropping_the_restriction_widens_the_reachable_set(self) -> None:
        """Without it, a token outside each beam's top five can be emitted."""
        restricted = self.run_opera(restrict_candidates=True)
        wide = self.run_opera(restrict_candidates=False)
        self.assertTrue(restricted["restrict_candidates"])
        self.assertFalse(wide["restrict_candidates"])
        # both are valid runs; whether they differ is the measurement
        self.assertEqual(len(restricted["token_ids"]),
                         len(wide["token_ids"]))

    def test_zero_weight_removes_the_penalty_but_not_the_restriction(self) -> None:
        got = self.run_opera(penalty_weights=0.0)
        self.assertEqual(got["penalty_weights"], 0.0)
        self.assertTrue(got["restrict_candidates"])

    def test_the_three_settings_are_distinguishable_in_the_record(self) -> None:
        seen = set()
        for restrict, weight in ((True, 1.0), (False, 1.0), (True, 0.0)):
            got = self.run_opera(restrict_candidates=restrict,
                                 penalty_weights=weight)
            seen.add((got["restrict_candidates"], got["penalty_weights"]))
        self.assertEqual(len(seen), 3)


if __name__ == "__main__":
    unittest.main()
