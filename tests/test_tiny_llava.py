"""The compared methods against a real -- but tiny -- LLaVA.

``tests/test_methods.py`` uses a stub model, which is enough to pin the
decoding rule and the branch structure but cannot exercise anything that
touches the vision tower or the projector. CausalMM lives entirely there, so
it needs a model that really has one.

The fixture is a genuine ``LlavaForConditionalGeneration`` built from config
-- no download, no checkpoint -- with CLIP's geometry kept EXACTLY right
(image_size 336, patch_size 14, so 576 visual tokens, which
``locate_image_span`` requires) and every width shrunk to toy size. So the
shapes, the dtypes, the hidden-state indexing and the projector are the real
ones; only the parameter count is not.

What this module is for:

* the reconstruction of ``get_image_features`` inside ``causalmm_vision``
  duplicates six lines of transformers, because the stock method returns only
  the projected features and the counterfactual needs the pre-projector ones.
  A transformers change could make that copy drift, so it is asserted
  bit-identical against the stock method on the shared code path;
* the counterfactual must land on the negative row and ONLY the negative row;
* the gain CausalMM applies is ~1/577 by construction, which is a property of
  the method worth failing on if it ever stops being true.
"""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only checkout
    raise unittest.SkipTest("torch is not installed") from None

try:
    from transformers import (  # noqa: F401
        CLIPVisionConfig, LlamaConfig, LlavaConfig,
        LlavaForConditionalGeneration,
    )
except ImportError:  # pragma: no cover - metrics-only environment
    raise unittest.SkipTest("transformers is not installed") from None

from foxrepro.fox.attention import causalmm_vision  # noqa: E402

N_IMG = 576
IMAGE_TOKEN_ID = 63


def tiny_llava(seed: int = 0) -> "LlavaForConditionalGeneration":
    """Real class, real geometry, toy widths."""
    vision = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=2, image_size=336, patch_size=14,
    )
    text = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
        max_position_embeddings=2048,
    )
    cfg = LlavaConfig(
        vision_config=vision, text_config=text,
        image_token_index=IMAGE_TOKEN_ID, vision_feature_layer=-2,
        vision_feature_select_strategy="default",
    )
    torch.manual_seed(seed)
    model = LlavaForConditionalGeneration(cfg).eval()
    assert (336 // 14) ** 2 == N_IMG
    return model


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tiny_llava()

    def setUp(self) -> None:
        torch.manual_seed(1)
        one = torch.randn(1, 3, 336, 336)
        # Both rows the SAME image: that is what fox_generate builds for
        # CausalMM, and the wrapper refuses anything else.
        self.px = one.repeat(2, 1, 1, 1)

    def features(self, **kw):
        with torch.no_grad():
            return self.model.model.get_image_features(pixel_values=self.px, **kw)


class TestGetImageFeaturesReconstruction(_Base):
    """The six duplicated lines must stay equal to the stock method."""

    def test_reconstruction_is_bit_identical_with_attentions_off(self) -> None:
        inner = self.model.model
        stock = self.features()
        with torch.no_grad():
            outs = inner.vision_tower(self.px[:1], output_hidden_states=True)
            selected = outs.hidden_states[
                self.model.config.vision_feature_layer][:, 1:]
            mine = inner.multi_modal_projector(selected)[0]
        self.assertTrue(
            torch.equal(mine, stock[0]),
            "causalmm_vision's copy of get_image_features has drifted from "
            "transformers'. Re-read LlavaModel.get_image_features.",
        )

    def test_the_stock_method_is_restored_afterwards(self) -> None:
        """And leaves no instance attribute shadowing the class method.

        Bound methods are created fresh on each attribute access, so identity
        is the wrong test; what matters is that the wrapper is gone and
        nothing is left in the instance dict for a later arm to run through.
        """
        inner = self.model.model
        self.assertNotIn("get_image_features", vars(inner))
        with causalmm_vision(self.model, rows=(1,)):
            self.assertEqual(inner.get_image_features.__name__, "wrapped")
        self.assertNotIn("get_image_features", vars(inner))
        self.assertEqual(
            inner.get_image_features.__func__,
            type(inner).get_image_features,
        )

    def test_the_vision_tower_is_put_back_on_sdpa(self) -> None:
        """Eager is needed for one call, not for the rest of the run."""
        tower = self.model.model.vision_tower
        before = tower.config._attn_implementation
        with causalmm_vision(self.model, rows=(1,)):
            self.features()
        self.assertEqual(tower.config._attn_implementation, before)


class TestCausalMMCounterfactual(_Base):
    def wrapped(self, **kw):
        stats: dict = {}
        with causalmm_vision(
            self.model, rows=(1,),
            generator=torch.Generator().manual_seed(5), stats=stats, **kw
        ):
            return self.features(), stats

    def test_only_the_negative_row_changes(self) -> None:
        stock = self.features()
        got, _ = self.wrapped()
        self.assertFalse(torch.equal(got[0], got[1]))
        # row 0 differs from stock only by eager-vs-sdpa in CLIP, which is
        # the recorded deviation; it must not differ by the counterfactual
        clean_drift = (stock[0] - got[0]).abs().max().item()
        cf_drift = (stock[1] - got[1]).abs().max().item()
        self.assertLess(clean_drift, 1e-5)
        self.assertGreater(cf_drift, 1e-4)
        self.assertGreater(cf_drift, clean_drift * 100)

    def test_the_gain_is_one_over_the_number_of_keys(self) -> None:
        """A structural property, not a fitted one.

        Each attention row sums to 1 over 577 keys, so the whole matrix sums
        to 577 and its mean entry is 1/577 -- and a permutation cannot change
        that. Resizing and averaging over heads preserve the mean. So
        CausalMM's counterfactual multiplies the visual features by a gain of
        order 0.0017: the negative branch is much closer to "no image" than
        to "a shuffled image". If this assertion ever fails, the method being
        run is not the one that produced the published numbers.
        """
        _, stats = self.wrapped()
        self.assertAlmostEqual(stats["cf_gain_mean"], 1.0 / 577, places=4)
        self.assertLess(stats["cf_gain_std"], stats["cf_gain_mean"])

    def test_differing_rows_are_refused(self) -> None:
        self.px = torch.randn(2, 3, 336, 336)
        with self.assertRaises(ValueError) as cm:
            self.wrapped()
        self.assertIn("share one image", str(cm.exception))

    def test_an_anyres_call_is_refused(self) -> None:
        with self.assertRaises(ValueError) as cm:
            with causalmm_vision(self.model, rows=(1,)):
                self.features(image_sizes=torch.tensor([[336, 336]]))
        self.assertIn("LLaVA-1.5", str(cm.exception))

    def test_a_row_outside_the_batch_is_refused(self) -> None:
        with self.assertRaises(ValueError) as cm:
            with causalmm_vision(self.model, rows=(5,)):
                self.features()
        self.assertIn("batch has 2 rows", str(cm.exception))

    def test_the_other_three_edits_run_and_differ(self) -> None:
        """Unreachable from the release, but the paper reports them."""
        seen = {}
        for edit in ("shuffle", "random", "uniform", "reversed"):
            with self.subTest(edit=edit):
                got, _ = self.wrapped(edit=edit)
                seen[edit] = got[1]
        for a in seen:
            for b in seen:
                if a < b:
                    self.assertFalse(torch.equal(seen[a], seen[b]),
                                     f"{a} and {b} produced the same features")

    def test_an_unknown_edit_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.wrapped(edit="nonsense")


if __name__ == "__main__":
    unittest.main()
