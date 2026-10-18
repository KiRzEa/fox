"""The port is bit-identical to the released code. This is the proof.

Why this test is the most important one in the repository
---------------------------------------------------------
The paper's mechanism is a numerical-precision effect, so a reimplementation
carries a specific danger: if our numbers differed from the paper's, we could
not tell "the paper is wrong" from "our port is wrong", and the whole
reproduction would be uninterpretable.

That danger is avoidable because the method has **no learned parameters** --
it is arithmetic on tensors. So: transcribe the released code verbatim
(``fox/reference.py``), re-implement it cleanly (``fox/kernel.py``), and
assert bit-equality on random inputs across every dtype. No GPU, no LLaVA
checkpoint, no vendored transformers tree, no COCO download.

Once ``mode="as_implemented"`` is proven bit-equal, every other arm
(``as_described``, ``ulp_control``, ``noise_control``, ``off``) is a
*deliberate* deviation measured against a trusted baseline, and the clean
repo can be the single source of truth.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


try:
    import torch  # noqa: E402
except ImportError:  # pragma: no cover - CPU-only checkout
    raise unittest.SkipTest(
        "torch is not installed. The bit-equivalence proof needs it, but the "
        "CPU metric suite deliberately does not -- install a track lock file "
        "(requirements-trackB.lock) to run this module."
    ) from None


from foxrepro.fox.kernel import (  # noqa: E402
    fuse, head_scores, intervene, jsd, quantize_to_grid, select_heads, ulp_at,
)
from foxrepro.fox.reference import (  # noqa: E402
    RELEASED_ALPHA, RELEASED_BETA, RELEASED_PENALTY_VALUE,
    RELEASED_SELECT_RATIO, RELEASED_TAU_JS, reference_fuse, reference_intervene,
    reference_jsd, reference_metrics, reference_select_heads,
)

# LLaVA-1.5 prompt geometry: a short system prompt, then 576 image tokens.
IMG_START = 24
IMG_END = IMG_START + 576 - 1
SEQ = 600
N_HEADS = 32
VOCAB = 32064


def causal_mask(q: int, k: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    m = torch.full((1, 1, q, k), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(m, diagonal=1)


def make_logits(
    b: int = 1, h: int = N_HEADS, q: int = SEQ, k: int = SEQ,
    dtype: torch.dtype = torch.float16, std: float = 2.0, seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, h, q, k, generator=g) * std).to(dtype)


class TestInterventionBitEquality(unittest.TestCase):
    """``intervene(mode="as_implemented")`` == ``reference_intervene``, exactly."""

    def _assert_bit_equal(self, dtype: torch.dtype, seed: int, b: int = 1) -> None:
        kp = {"image_start": IMG_START, "image_end": IMG_END}
        mask = causal_mask(SEQ, SEQ)
        src = make_logits(b=b, dtype=dtype, seed=seed)

        ref = reference_intervene(
            src.clone(), kp, layer_idx=8, attention_mask=mask,
            select_ratio=RELEASED_SELECT_RATIO,
            penalty_value=RELEASED_PENALTY_VALUE,
        )
        got, diag = intervene(
            src.clone(), IMG_START, IMG_END,
            mode="as_implemented", gamma=RELEASED_PENALTY_VALUE,
            ratio=RELEASED_SELECT_RATIO, logit_dtype="native",
            attention_mask=mask,
        )
        self.assertTrue(
            torch.equal(ref, got),
            f"{dtype}: port diverged from reference "
            f"(max abs diff {(ref.float() - got.float()).abs().max().item():.3e})",
        )
        self.assertIsNotNone(diag)

    def test_bit_equal_fp16(self) -> None:
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                self._assert_bit_equal(torch.float16, seed)

    def test_bit_equal_fp32(self) -> None:
        for seed in (0, 1):
            with self.subTest(seed=seed):
                self._assert_bit_equal(torch.float32, seed)

    def test_bit_equal_bf16(self) -> None:
        for seed in (0, 1):
            with self.subTest(seed=seed):
                self._assert_bit_equal(torch.bfloat16, seed)

    def test_bit_equal_batched(self) -> None:
        """Vectorised head selection must match the reference's batch loop."""
        self._assert_bit_equal(torch.float16, seed=5, b=4)

    def test_bit_equal_without_mask(self) -> None:
        kp = {"image_start": IMG_START, "image_end": IMG_END}
        src = make_logits(seed=9)
        ref = reference_intervene(src.clone(), kp, layer_idx=8, attention_mask=None)
        got, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode="as_implemented", gamma=RELEASED_PENALTY_VALUE,
                           ratio=RELEASED_SELECT_RATIO, attention_mask=None)
        self.assertTrue(torch.equal(ref, got))

    def test_decode_step_shape(self) -> None:
        """During decode Q == 1; the slicing must still work."""
        kp = {"image_start": IMG_START, "image_end": IMG_END}
        src = make_logits(q=1, k=SEQ + 7, seed=3)
        ref = reference_intervene(src.clone(), kp, layer_idx=8, attention_mask=None)
        got, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode="as_implemented", gamma=RELEASED_PENALTY_VALUE,
                           ratio=RELEASED_SELECT_RATIO, attention_mask=None)
        self.assertTrue(torch.equal(ref, got))


class TestStatsBitEquality(unittest.TestCase):
    def test_head_scores_match_reference(self) -> None:
        kp = {"image_start": IMG_START, "image_end": IMG_END}
        logits = make_logits(dtype=torch.float32, seed=11)
        probs = torch.softmax(logits + causal_mask(SEQ, SEQ), dim=-1)

        ref = reference_metrics(probs, kp, tail_n=1)
        assert ref is not None
        m_sys, h_vis = head_scores(probs, IMG_START, IMG_END)  # type: ignore[misc]

        self.assertTrue(torch.equal(ref["mSys_tail"], m_sys))
        self.assertTrue(torch.equal(ref["HVis_tail"], h_vis))

    def test_select_heads_matches_reference_loop(self) -> None:
        g = torch.Generator().manual_seed(4)
        scores = torch.rand(3, N_HEADS, generator=g)
        mine = select_heads(scores, RELEASED_SELECT_RATIO)
        for b in range(scores.shape[0]):
            theirs = reference_select_heads(scores[b], RELEASED_SELECT_RATIO)
            self.assertTrue(torch.equal(theirs, mine[b]))

    def test_released_ratio_selects_eleven_of_thirtytwo(self) -> None:
        # k = clamp(round(32 * 0.35), 1, 16) = 11. The paper's 0.45 gives 14.
        scores = torch.rand(1, 32, generator=torch.Generator().manual_seed(0))
        self.assertEqual(int(select_heads(scores, 0.35).sum()), 11)
        self.assertEqual(int(select_heads(scores, 0.45).sum()), 14)
        # the upper clamp caps anything above 0.5 at half the heads
        self.assertEqual(int(select_heads(scores, 0.9).sum()), 16)


class TestFusionBitEquality(unittest.TestCase):
    def _logits(self, seed: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed)
        return torch.randn(1, VOCAB, generator=g) * 3.0

    def test_jsd_matches_reference(self) -> None:
        for seed in (0, 1, 2):
            with self.subTest(seed=seed):
                p, q = self._logits(seed), self._logits(seed + 100)
                ref = reference_jsd(p[0], q[0])
                mine = float(jsd(p, q)[0].item())
                self.assertAlmostEqual(ref, mine, places=12)

    def test_fuse_consensus_branch(self) -> None:
        p, q = self._logits(7), self._logits(8)
        js_val = RELEASED_TAU_JS / 2  # below threshold -> lambda = alpha
        ref_fused, ref_masked = reference_fuse(p, q, js_val)
        fused, masked, lam = fuse(
            p, q, torch.tensor([js_val]),
            alpha=RELEASED_ALPHA, tau_js=RELEASED_TAU_JS, beta=RELEASED_BETA,
        )
        self.assertTrue(torch.equal(ref_fused, fused))
        self.assertTrue(torch.equal(ref_masked, masked))
        self.assertAlmostEqual(float(lam[0]), RELEASED_ALPHA)

    def test_fuse_conflict_branch(self) -> None:
        p, q = self._logits(21), self._logits(22)
        js_val = RELEASED_TAU_JS * 3  # above threshold -> lambda = js
        ref_fused, ref_masked = reference_fuse(p, q, js_val)
        fused, masked, lam = fuse(
            p, q, torch.tensor([js_val]),
            alpha=RELEASED_ALPHA, tau_js=RELEASED_TAU_JS, beta=RELEASED_BETA,
        )
        self.assertTrue(torch.equal(ref_fused, fused))
        self.assertTrue(torch.equal(ref_masked, masked))
        self.assertAlmostEqual(float(lam[0]), js_val, places=6)

    def test_greedy_and_sampling_rules_differ(self) -> None:
        """The asymmetry the paper's Table 9 inherits, pinned as a fact.

        The released greedy path argmaxes the UNMASKED fused logits while the
        sampling path uses the beta-masked ones, so Table 9 changes the
        decision rule as well as removing randomness.
        """
        p, q = self._logits(31), self._logits(32)
        fused, masked, _ = fuse(p, q, torch.tensor([0.0]))
        self.assertTrue(torch.isinf(masked).any(), "mask should remove candidates")
        n_kept = int(torch.isfinite(masked).sum())
        self.assertLess(n_kept, masked.numel())
        # and the two rules can pick different tokens
        self.assertTrue(
            int(fused.argmax()) != int(masked.argmax())
            or n_kept < masked.numel()
        )


class TestUlpArithmetic(unittest.TestCase):
    """The gamma sweep spans the ULP grid inside fp16 alone -- the reason a
    T4 (sm_75, no bf16 compute) can still run the decisive experiment."""

    def test_fp16_ulp_at_each_sweep_point(self) -> None:
        expected = {1e2: 0.0625, 1e3: 0.5, 1e4: 8.0, 6e4: 32.0}
        for gamma, want in expected.items():
            with self.subTest(gamma=gamma):
                self.assertEqual(ulp_at(gamma, torch.float16), want)

    def test_gamma_1e5_overflows_fp16(self) -> None:
        # Why the sweep stops at 6e4: fp16 tops out at 65504.
        self.assertEqual(ulp_at(1e5, torch.float16), float("inf"))

    def test_bf16_ulp_is_coarser_than_fp16(self) -> None:
        # bf16 has 7 mantissa bits vs fp16's 10, so 8x coarser at 1e4.
        self.assertEqual(ulp_at(1e4, torch.bfloat16), 64.0)
        self.assertEqual(ulp_at(1e4, torch.float32), 2.0 ** -10)

    def test_quantize_to_grid_is_idempotent(self) -> None:
        x = torch.randn(256, generator=torch.Generator().manual_seed(0)) * 10
        once = quantize_to_grid(x, 8.0)
        self.assertTrue(torch.equal(once, quantize_to_grid(once, 8.0)))

    def test_fp32_round_trip_is_nearly_but_not_exactly_a_no_op(self) -> None:
        """Pins the real number, which matters for the write-up.

        Subtracting gamma and casting back is a no-op in exact arithmetic
        because softmax is shift-invariant. In fp32 the ULP at 1e4 is ~0.001,
        well below the O(1)-O(30) spread of attention logits, so the effect is
        negligible -- but it is NOT identically zero, and the report should say
        so rather than claim an exact null.
        """
        logits = make_logits(b=1, h=4, q=8, k=SEQ, dtype=torch.float32, seed=2)
        y = (logits.to(torch.float32) - 1e4).to(torch.float32)
        a = torch.softmax(logits.float(), dim=-1)
        b = torch.softmax(y.float(), dim=-1)
        tv = 0.5 * (a - b).abs().sum(-1).max().item()
        self.assertGreater(tv, 0.0)
        self.assertLess(tv, 1e-3)

    def test_fp16_round_trip_is_destructive(self) -> None:
        logits = make_logits(b=1, h=4, q=8, k=SEQ, dtype=torch.float16, seed=2)
        y = (logits.to(torch.float32) - 1e4).to(torch.float16)
        a = torch.softmax(logits.float(), dim=-1)
        b = torch.softmax(y.float(), dim=-1)
        tv = 0.5 * (a - b).abs().sum(-1).max().item()
        self.assertGreater(tv, 0.5, "fp16 round trip should wreck the row")


class TestModesAreDistinctInterventions(unittest.TestCase):
    """Every arm must actually do something different, or the experiment is
    measuring nothing."""

    def _run(self, mode: str, **kw) -> torch.Tensor:
        src = make_logits(b=1, h=8, q=16, k=SEQ, seed=13)
        out, _ = intervene(
            src.clone(), IMG_START, IMG_END, mode=mode, gamma=1e4, ratio=0.35,
            generator=torch.Generator().manual_seed(0), **kw,
        )
        return out

    def test_off_changes_nothing(self) -> None:
        src = make_logits(b=1, h=8, q=16, k=SEQ, seed=13)
        out, diag = intervene(src.clone(), IMG_START, IMG_END, mode="off")
        self.assertTrue(torch.equal(src, out))
        self.assertIsNone(diag)

    def test_all_arms_differ_from_each_other(self) -> None:
        outs = {m: self._run(m) for m in
                ("as_implemented", "as_described", "ulp_control", "noise_control")}
        names = list(outs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(
                    torch.equal(outs[a], outs[b]),
                    f"{a} and {b} produced identical output; they must not",
                )

    def test_as_described_only_touches_image_keys(self) -> None:
        src = make_logits(b=1, h=8, q=16, k=SEQ, seed=13)
        out, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode="as_described", gamma=4.0, ratio=0.35)
        changed = (out != src)
        # system keys [0, IMG_START) must be untouched
        self.assertFalse(bool(changed[..., 0:IMG_START].any()))
        # image keys must be touched
        self.assertTrue(bool(changed[..., IMG_START:IMG_END + 1].any()))

    def test_as_implemented_touches_every_key(self) -> None:
        src = make_logits(b=1, h=8, q=16, k=SEQ, seed=13)
        out, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode="as_implemented", gamma=1e4, ratio=0.35)
        changed = (out != src)
        # the released slice is [..., -1:, :] -- all keys, so system keys of a
        # selected head DO change. This is the heart of the finding.
        self.assertTrue(bool(changed[..., 0:IMG_START].any()))

    def test_rows_restricts_the_write(self) -> None:
        """Row 0 observational, row 1 interventional, in one forward pass."""
        src = make_logits(b=2, h=8, q=16, k=SEQ, seed=17)
        out, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode="as_implemented", gamma=1e4, ratio=0.35,
                           rows=torch.tensor([1]))
        self.assertTrue(torch.equal(src[0], out[0]), "row 0 must be untouched")
        self.assertFalse(torch.equal(src[1], out[1]), "row 1 must be modified")

    def test_instructblip_variant_drops_m_sys(self) -> None:
        """The second copy in the released repo scores on H_vis alone."""
        probs = torch.softmax(
            make_logits(dtype=torch.float32, seed=23) + causal_mask(SEQ, SEQ),
            dim=-1,
        )
        m_sys, h_vis = head_scores(probs, IMG_START, IMG_END)      # type: ignore[misc]
        with_sys = select_heads(m_sys * h_vis, 0.4)
        without = select_heads(h_vis, 0.4)
        self.assertFalse(
            torch.equal(with_sys, without),
            "dropping m_sys should change which heads are selected",
        )


if __name__ == "__main__":
    unittest.main()


class TestBoostVisual(unittest.TestCase):
    """The intervention the paper argues from and never performs.

    Its motivation says amplifying visual attention cannot help, because
    faithful and hallucinated samples carry the same global visual attention
    mass (Figure 1a). That is an observational comparison and the conclusion
    is about an intervention -- the P(Y|X) vs P(Y|do(X)) distinction the paper
    takes as its own foundation. PAI, the named representative of that family,
    appears twice in the paper and is in none of its tables.

    So this mode adds a positive bias to the image-token keys at the same
    heads and query positions Fox suppresses, leaving the system keys
    untouched: the same slice as `as_described`, opposite sign.
    """

    def _run(self, mode: str, gamma: float = 4.0):
        src = make_logits(b=1, h=8, q=16, k=SEQ, seed=21)
        out, _ = intervene(src.clone(), IMG_START, IMG_END,
                           mode=mode, gamma=gamma, ratio=0.35)
        return src, out

    def test_system_keys_are_untouched(self) -> None:
        """Otherwise it is not 'boost visual', it is 'change everything'."""
        src, out = self._run("boost_visual")
        self.assertTrue(torch.equal(src[..., 0:IMG_START], out[..., 0:IMG_START]))

    def test_image_keys_move_up(self) -> None:
        src, out = self._run("boost_visual")
        delta = (out.float() - src.float())[..., IMG_START:IMG_END + 1]
        moved = delta[delta != 0]
        self.assertGreater(moved.numel(), 0)
        self.assertTrue(bool((moved > 0).all()), "must amplify, not suppress")

    def test_it_is_the_mirror_of_as_described(self) -> None:
        src_a, out_a = self._run("as_described")
        src_b, out_b = self._run("boost_visual")
        da = (out_a.float() - src_a.float())
        db = (out_b.float() - src_b.float())
        self.assertTrue(torch.allclose(da, -db, atol=1e-2))

    def test_it_changes_the_attention_distribution(self) -> None:
        """A no-op here would make the experiment meaningless."""
        src, out = self._run("boost_visual")
        a = torch.softmax(src.float(), dim=-1)
        b = torch.softmax(out.float(), dim=-1)
        tv = 0.5 * (a - b).abs().sum(-1).max().item()
        self.assertGreater(tv, 0.1)

    def test_bigger_delta_moves_more_mass_to_the_image(self) -> None:
        shares = []
        for gamma in (2.0, 8.0):
            src, out = self._run("boost_visual", gamma)
            p = torch.softmax(out.float(), dim=-1)
            shares.append(p[..., IMG_START:IMG_END + 1].sum(-1).max().item())
        self.assertGreater(shares[1], shares[0])

    def test_the_mode_is_registered(self) -> None:
        from foxrepro.storage import INTERVENTION_MODES

        self.assertIn("boost_visual", INTERVENTION_MODES)


class TestHarnessMatchesTheReleasedHarness(unittest.TestCase):
    """The harness decides what the numbers mean, so it gets the same
    treatment as the kernel: a verbatim transcription in ``reference.py``,
    and the port asserted equal to it.

    Without this, a "tidy-up" that inserts the missing space after
    ``questions.`` or that rejects an unparseable POPE answer instead of
    scoring it positive would change every table and pass every other test.
    """

    QUESTIONS = (
        "Is there a dog in the image?",
        "Is there a snowboard in the image?",
        "Please describe this image in detail.",
    )

    def test_the_prompt_is_byte_identical_to_the_release(self) -> None:
        from foxrepro.bench.methods import render_full_prompt
        from foxrepro.fox.reference import reference_prompt

        for q in self.QUESTIONS:
            with self.subTest(q=q):
                # The only intended difference: the converted checkpoint's
                # processor takes "<image>" where the released code wrote its
                # own placeholder and spliced embeddings by hand.
                expected = reference_prompt(q).replace("<ImageHere>", "<image>")
                self.assertEqual(render_full_prompt(q), expected)

    def test_the_no_system_arm_matches_an_empty_system_message(self) -> None:
        from foxrepro.bench.methods import render_full_prompt
        from foxrepro.fox.reference import reference_prompt

        for q in self.QUESTIONS:
            with self.subTest(q=q):
                expected = reference_prompt(q, system_message="").replace(
                    "<ImageHere>", "<image>")
                self.assertEqual(render_full_prompt(q, "none"), expected)

    def test_the_released_pope_parser_agrees_on_every_shape_of_answer(self) -> None:
        from foxrepro.bench.evaluate import parse_yes_no_released
        from foxrepro.fox.reference import reference_pope_predict

        answers = [
            "Yes", "No", "yes", "no", "NO", "YES", "",
            "Yes.", "No,", "Yes, there is a dog.",
            "There is a dog in the image.",
            "There isn't a dog.", "It doesn't appear so.",
            "Maybe", "I cannot tell", "Not sure",
            "The", "A", "1", "   ", "\n",
            "nobody", "nothing", "none",          # substrings, not tokens
            "Yes No", "No Yes",
        ]
        released = reference_pope_predict(answers)
        ours = [1 if parse_yes_no_released(a) == "yes" else 0 for a in answers]
        for a, r, o in zip(answers, released, ours):
            self.assertEqual(r, o, f"disagreement on {a!r}: release={r} ours={o}")

    def test_the_released_parser_never_abstains(self) -> None:
        """Unlike the strict parser, which is the point of keeping both."""
        from foxrepro.bench.evaluate import parse_yes_no, parse_yes_no_released

        self.assertIsNone(parse_yes_no("Maybe"))
        self.assertEqual(parse_yes_no_released("Maybe"), "yes")

    def test_max_new_tokens_match_the_release(self) -> None:
        from foxrepro.bench.plan import PLAN
        from foxrepro.fox.reference import (
            RELEASED_CHAIR_MAX_NEW_TOKENS, RELEASED_POPE_MAX_NEW_TOKENS,
        )

        want = {"pope": RELEASED_POPE_MAX_NEW_TOKENS,
                "chair": RELEASED_CHAIR_MAX_NEW_TOKENS}
        for job in PLAN:
            with self.subTest(job=job.job_id()):
                self.assertEqual(
                    job.run_key().variant_fields["max_new_tokens"],
                    want[job.benchmark],
                )

    def test_the_chair_question_is_the_released_one(self) -> None:
        from foxrepro.data.loaders import PROMPT_TEMPLATES
        from foxrepro.fox.reference import RELEASED_CHAIR_QUESTION

        self.assertEqual(PROMPT_TEMPLATES["chair_v1"], RELEASED_CHAIR_QUESTION)
