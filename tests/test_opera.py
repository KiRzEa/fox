"""OPERA's beam search, against the tiny-but-real LLaVA.

Three things need pinning and none of them can be checked by reading:

* the beam bookkeeping is ours, because ``transformers``'
  ``BeamSearchScorer`` is deprecated and removed in v4.62.0 and this project
  must keep running on one environment. So it is asserted *equivalent* to the
  library's for as long as the library still has one;
* the penalty must be added to RAW LOGITS, before ``log_softmax``, and only
  to the five candidates. An earlier reading of this code had it as a bonus
  on the log-probs over the full vocabulary, which is a different method;
* the speculative forwards must leave the search's own KV cache exactly as
  they found it. On transformers 4.29 the cache was an immutable tuple and
  OPERA could discard the extended copy; ``DynamicCache.update`` mutates in
  place, so a missing ``crop`` would corrupt the search silently -- it would
  still produce fluent text.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

from foxrepro.fox import attention as A  # noqa: E402
from foxrepro.fox.opera import (  # noqa: E402
    REJECT_FLOOR, BeamHypotheses, BeamScorer, opera_generate,
    rollback_is_reachable,
)

from tests.test_tiny_llava import IMAGE_TOKEN_ID, N_IMG, tiny_llava  # noqa: E402


def prompt(pre: int = 6, post: int = 4):
    ids = torch.arange(1, pre + N_IMG + post + 1, dtype=torch.long) % 60
    ids = ids.unsqueeze(0)
    ids[0, pre:pre + N_IMG] = IMAGE_TOKEN_ID
    torch.manual_seed(4)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pixel_values": torch.randn(1, 3, 336, 336),
    }


class _ModelBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        A.register()
        cls.model = tiny_llava()
        cls.model.set_attn_implementation("fox")

    def setUp(self) -> None:
        self.inputs = prompt()

    def run_opera(self, **kw):
        with torch.no_grad():
            return opera_generate(self.model, self.inputs, **kw)


class TestBeamBookkeepingMatchesTransformers(unittest.TestCase):
    """Ours against the library's, on the settings OPERA actually uses.

    OPERA passes ``generation_config.length_penalty`` and
    ``.early_stopping`` and sets neither, so it runs at 1.0 and False.
    """

    N, PROMPT = 4, 7

    def _library(self):
        try:
            from transformers.generation.beam_search import BeamSearchScorer
        except ImportError:  # pragma: no cover - removed in v4.62
            self.skipTest("BeamSearchScorer has been removed from transformers")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return BeamSearchScorer(
                batch_size=1, num_beams=self.N, device=torch.device("cpu"),
                length_penalty=1.0, do_early_stopping=False,
                num_beam_hyps_to_keep=1, max_length=self.PROMPT + 6,
            )

    def test_process_agrees_step_for_step(self) -> None:
        eos = [2]
        mine = BeamScorer(self.N, length_penalty=1.0, early_stopping=False,
                          max_length=self.PROMPT + 6)
        theirs = self._library()
        g = torch.Generator().manual_seed(11)
        ids = torch.randint(3, 50, (self.N, self.PROMPT), generator=g)
        for step in range(5):
            with self.subTest(step=step):
                # Descending scores, as a topk would produce, and an eos
                # planted where `process` must bank rather than extend.
                scores = torch.sort(
                    torch.randn(1, 2 * self.N, generator=g),
                    descending=True).values
                tokens = torch.randint(3, 50, (1, 2 * self.N), generator=g)
                if step % 2 == 1:
                    tokens[0, 1] = eos[0]
                idx = torch.randint(0, self.N, (1, 2 * self.N), generator=g)
                a = mine.process(
                    ids, scores, tokens, idx, pad_token_id=0,
                    eos_token_id=eos, decoder_prompt_len=self.PROMPT)
                b = theirs.process(
                    ids, scores, tokens, idx, pad_token_id=0,
                    eos_token_id=eos, decoder_prompt_len=self.PROMPT)
                for key in ("next_beam_scores", "next_beam_tokens",
                            "next_beam_indices"):
                    self.assertTrue(
                        torch.equal(a[key], b[key]),
                        f"{key} differs at step {step}: {a[key]} vs {b[key]}",
                    )
                self.assertEqual(mine.is_done, bool(theirs.is_done))
                ids = torch.cat(
                    [ids[a["next_beam_indices"]],
                     a["next_beam_tokens"].unsqueeze(-1)], dim=-1)

    def test_hypotheses_rank_and_evict_the_same(self) -> None:
        try:
            from transformers.generation.beam_search import (
                BeamHypotheses as TheirHyps,
            )
        except ImportError:  # pragma: no cover
            self.skipTest("BeamHypotheses has been removed from transformers")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = TheirHyps(num_beams=3, length_penalty=1.0,
                              early_stopping=False)
        mine = BeamHypotheses(3, 1.0, False)
        g = torch.Generator().manual_seed(2)
        for i in range(8):
            hyp = torch.randint(0, 50, (6 + i,), generator=g)
            score = float(torch.randn(1, generator=g))
            mine.add(hyp, score, generated_len=6 + i)
            theirs.add(hyp, score, generated_len=6 + i)
            with self.subTest(i=i):
                self.assertEqual(len(mine), len(theirs))
                self.assertAlmostEqual(mine.worst_score, theirs.worst_score,
                                       places=6)
                # theirs stores (score, hyp, beam_indices); ours stores
                # (score, hyp), since nothing here consumes beam_indices
                self.assertEqual(
                    sorted(round(s, 6) for s, _ in mine.beams),
                    sorted(round(b[0], 6) for b in theirs.beams),
                )
                self.assertEqual(mine.is_done(0.5, 6 + i),
                                 theirs.is_done(0.5, 6 + i))


class TestOperaDecodingRule(_ModelBase):
    """The penalty is in logit space, on five candidates, before log_softmax.

    The capture is stubbed so the attention is ours to choose; everything
    else -- topk, the floor, log_softmax, the beam walk, the cache -- is the
    real code path.
    """

    def _candidates(self):
        """The five candidate tokens the first step will consider, per beam."""
        with torch.no_grad():
            out = self.model(
                input_ids=self.inputs["input_ids"],
                attention_mask=self.inputs["attention_mask"],
                pixel_values=self.inputs["pixel_values"],
            )
        return torch.topk(out.logits[:, -1, :], 5, dim=-1, sorted=True)

    def _stub_capture(self, favour: int, strength: float = 1.0):
        """Give candidate ``favour`` all of the image attention, others none.

        Returns a cleanup-registered patch. The row is built in the key frame
        the real capture uses, so the span slice and the row normalisation in
        ``opera_generate`` are still exercised.
        """
        calls = {"n": 0}
        img_lo, img_hi = 6, 6 + N_IMG - 1

        def fake(module, query, key, attention_mask, scaling):  # noqa: ANN001
            n_keys = key.shape[-2]
            row = torch.zeros(query.shape[0], n_keys, dtype=query.dtype)
            c = calls["n"] % 5
            # all mass on the image span for the favoured candidate, all mass
            # outside it for the others
            if c == favour:
                row[:, img_lo:img_hi + 1] = strength
            else:
                row[:, :img_lo] = strength
            calls["n"] += 1
            A.FOX.opera_attn = row

        original = A._opera_capture_attn
        A._opera_capture_attn = fake
        self.addCleanup(lambda: setattr(A, "_opera_capture_attn", original))
        return calls

    def test_the_penalty_can_flip_the_chosen_token(self) -> None:
        """With enough weight on candidate 4, candidate 4 wins.

        Candidate 0 is the raw-logit argmax by construction, so a different
        first token can only come from the penalty.
        """
        scores, tokens = self._candidates()
        self._stub_capture(favour=4)
        got = self.run_opera(max_new_tokens=1, penalty_weights=50.0)
        self.assertEqual(got["token_ids"][0], int(tokens[0, 4]))

    def test_without_the_penalty_the_raw_argmax_wins(self) -> None:
        scores, tokens = self._candidates()
        self._stub_capture(favour=4)
        got = self.run_opera(max_new_tokens=1, penalty_weights=0.0)
        self.assertEqual(got["token_ids"][0], int(tokens[0, 0]))

    def test_the_weight_needed_is_a_logit_gap_not_a_logprob_gap(self) -> None:
        """Pins the SPACE the penalty lives in, which is the thing I had wrong.

        Candidate 4's raw logit is below candidate 0's by ``gap``. The bonus
        is 1.0 for the favoured candidate and 0.0 for the others, so a weight
        just over ``gap`` must flip it and a weight just under must not. If
        the penalty were applied after ``log_softmax`` over the five, the
        threshold would be the log-prob gap instead, which is a different
        number.
        """
        scores, tokens = self._candidates()
        gap = float(scores[0, 0] - scores[0, 4])
        self.assertGreater(gap, 0.0)
        for weight, expect in ((gap * 0.5, 0), (gap * 2.0, 4)):
            with self.subTest(weight=round(weight, 4)):
                self._stub_capture(favour=4)
                got = self.run_opera(max_new_tokens=1, penalty_weights=weight)
                self.assertEqual(got["token_ids"][0],
                                 int(tokens[0, expect]))

    def test_only_the_five_candidates_are_reachable(self) -> None:
        """The rest of the vocabulary is discarded before log_softmax."""
        _, tokens = self._candidates()
        allowed = set(tokens[0].tolist())
        self._stub_capture(favour=0)
        got = self.run_opera(max_new_tokens=1, penalty_weights=1.0)
        self.assertIn(got["token_ids"][0], allowed)

    def test_the_floor_is_finite(self) -> None:
        """-999, not -inf, and that is load-bearing.

        After log_softmax the discarded tokens sit at a finite value, so when
        process banks an eos candidate and walks on, a discarded token can be
        promoted into a live beam. With -inf it could not.
        """
        self.assertEqual(REJECT_FLOOR, -999.0)
        vec = torch.full((1, 32000), REJECT_FLOOR)
        vec[0, :5] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
        lsm = torch.log_softmax(vec, dim=-1)
        self.assertTrue(bool(torch.isfinite(lsm).all()))
        self.assertLess(float(lsm[0, 100]), -900.0)


class TestSpeculativeForwardsLeaveNoTrace(_ModelBase):
    def test_the_cache_grows_by_exactly_one_per_step(self) -> None:
        """Six forwards per step, one of which is allowed to extend the cache.

        Without the crop the cache would grow by six and the search would be
        conditioning on five tokens it never emitted.
        """
        seen = []
        original = A._opera_capture_attn

        def spy(module, query, key, attention_mask, scaling):  # noqa: ANN001
            seen.append(int(key.shape[-2]))
            original(module, query, key, attention_mask, scaling)

        A._opera_capture_attn = spy
        self.addCleanup(lambda: setattr(A, "_opera_capture_attn", original))
        got = self.run_opera(max_new_tokens=3)
        n_prompt = got["n_prompt_tokens"]
        # Exactly 5 per step: the main forward does not capture, because its
        # last row belongs to no candidate.
        self.assertEqual(len(seen), 15)
        for step in range(3):
            window = seen[step * 5:(step + 1) * 5]
            self.assertEqual(window, [n_prompt + step + 1] * 5,
                             f"step {step} saw {window}")

    def test_the_run_is_deterministic(self) -> None:
        a = self.run_opera(max_new_tokens=3)
        b = self.run_opera(max_new_tokens=3)
        self.assertEqual(a["token_ids"], b["token_ids"])

    def test_the_forward_count_is_recorded(self) -> None:
        got = self.run_opera(max_new_tokens=1, num_attn_candidates=3)
        self.assertEqual(got["n_forwards_per_step"], 4)


class TestOperaGuards(_ModelBase):
    def test_one_beam_is_refused(self) -> None:
        with self.assertRaises(ValueError) as cm:
            self.run_opera(max_new_tokens=1, num_beams=1)
        self.assertIn("num_beams >= 2", str(cm.exception))

    def test_zero_candidates_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.run_opera(max_new_tokens=1, num_attn_candidates=0)

    def test_a_length_that_would_need_rollback_is_refused(self) -> None:
        """Running long without rollback is OPERA with half its method gone."""
        with self.assertRaises(ValueError) as cm:
            self.run_opera(max_new_tokens=512)
        self.assertIn("rollback", str(cm.exception))

    def test_that_refusal_can_be_overridden_explicitly(self) -> None:
        got = self.run_opera(max_new_tokens=16,
                             allow_unreachable_rollback=True)
        self.assertGreater(got["n_gen_tokens"], 0)

    def test_the_reachability_boundary(self) -> None:
        """Both gates are off at <= 15 tokens and on above it."""
        for n in (1, 10, 15):
            self.assertFalse(rollback_is_reachable(n), n)
        for n in (16, 64, 512):
            self.assertTrue(rollback_is_reachable(n), n)

    def test_a_missing_capture_raises(self) -> None:
        """A model not loaded through our attention has no penalty at all."""
        original = A._opera_capture_attn
        A._opera_capture_attn = lambda *a, **k: None
        self.addCleanup(lambda: setattr(A, "_opera_capture_attn", original))
        with self.assertRaises(RuntimeError) as cm:
            self.run_opera(max_new_tokens=1)
        self.assertIn("attn_implementation='fox'", str(cm.exception))


class TestAttentionCaptureIsMaxNotMean(_ModelBase):
    """SID averages over heads, OPERA max-pools. Different statistic."""

    def test_the_captured_row_is_the_head_max(self) -> None:
        import torch.nn.functional as F

        q = torch.randn(2, 4, 7, 8)
        k = torch.randn(2, 4, 7, 8)
        mask = torch.triu(torch.full((1, 1, 7, 7), -1e9), diagonal=1)
        scaling = 8 ** -0.5

        class Mod:
            num_key_value_groups = 1
            head_dim = 8

        A.FOX.opera_attn = None
        A._opera_capture_attn(Mod(), q, k, mask, scaling)
        got = A.FOX.opera_attn

        logits = torch.matmul(q[:, :, -1:, :], k.transpose(2, 3)) * scaling
        logits = logits + mask[..., -1:, :]
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
        self.assertTrue(torch.equal(got, probs.max(dim=1).values[:, 0, :]))
        self.assertFalse(torch.equal(got, probs.mean(dim=1)[:, 0, :]))

    def test_the_row_is_not_normalised_at_capture(self) -> None:
        """A max over heads does not sum to 1; the division happens later.

        The release normalises after assembling the attention square, so
        doing it here would be a different number.
        """
        q = torch.randn(2, 4, 7, 8)
        k = torch.randn(2, 4, 7, 8)

        class Mod:
            num_key_value_groups = 1
            head_dim = 8

        A.FOX.opera_attn = None
        A._opera_capture_attn(Mod(), q, k, None, 8 ** -0.5)
        row = A.FOX.opera_attn
        self.assertIsNotNone(row)
        # max over 4 heads of 4 distributions: strictly greater than 1
        self.assertTrue(bool((row.sum(-1) > 1.0).all()))


if __name__ == "__main__":
    unittest.main()
