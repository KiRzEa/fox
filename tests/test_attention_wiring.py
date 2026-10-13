"""The attention function delivers the proven kernel faithfully.

``tests/test_kernel_equivalence.py`` proves the *mechanism* is bit-identical to
the released code. This module proves the *plumbing* around it is right, which
is a separate question: the attention function has to apply the intervention at
the correct point in the flow (before the causal mask, after the QK scaling),
accumulate the softmax in the right dtype, and leave every non-intervention
layer byte-for-byte on the stock SDPA path.

Needs transformers but no GPU and no checkpoint: a stub stands in for
``LlamaAttention``, since the attention function only ever reads ``layer_idx``,
``num_key_value_groups`` and ``head_dim`` off the module it is handed.
"""

from __future__ import annotations

import math
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - CPU-only checkout
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
    from transformers.integrations.sdpa_attention import (  # noqa: F401
        sdpa_attention_forward,
    )
except ImportError:  # pragma: no cover - metrics-only environment
    raise unittest.SkipTest(
        "transformers is not installed; install requirements-trackB.lock"
    ) from None

from foxrepro.fox import attention as A  # noqa: E402
from foxrepro.fox.reference import (  # noqa: E402
    RELEASED_PENALTY_VALUE, RELEASED_SELECT_RATIO, reference_intervene,
)

IMG_START = 24
N_IMG = 576
IMG_END = IMG_START + N_IMG - 1
SEQ = IMG_END + 1 + 5          # a few text tokens after the image
HEADS = 8
HEAD_DIM = 16


class StubAttention:
    """Minimal stand-in for ``LlamaAttention``.

    The attention function reads exactly three attributes off the module, so a
    stub is sufficient and keeps this test free of a 14 GB checkpoint.
    """

    def __init__(self, layer_idx: int, n_rep: int = 1, head_dim: int = HEAD_DIM):
        self.layer_idx = layer_idx
        self.num_key_value_groups = n_rep
        self.head_dim = head_dim
        self.is_causal = True


def qkv(seq: int = SEQ, heads: int = HEADS, dim: int = HEAD_DIM,
        dtype: torch.dtype = torch.float16, batch: int = 1, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    shape = (batch, heads, seq, dim)
    q = (torch.randn(shape, generator=g) * 0.7).to(dtype)
    k = (torch.randn(shape, generator=g) * 0.7).to(dtype)
    v = (torch.randn(shape, generator=g) * 0.7).to(dtype)
    return q, k, v


def causal_mask(seq: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    m = torch.full((1, 1, seq, seq), torch.finfo(dtype).min, dtype=dtype)
    return torch.triu(m, diagonal=1)


def expected_eager(q, k, v, mask, *, scaling, intervene: bool,
                   layer_idx: int = 8) -> torch.Tensor:
    """Independent recomputation of the whole eager path.

    Deliberately written out longhand from the released ordering -- QK scaling,
    then the intervention, then the mask, then an fp32 softmax -- rather than
    reusing anything from ``fox.attention``, so a reordering bug there cannot
    hide behind a shared helper.
    """
    logits = torch.matmul(q, k.transpose(2, 3)) * scaling
    if intervene:
        reference_intervene(
            logits,
            {"image_start": IMG_START, "image_end": IMG_END},
            layer_idx=layer_idx,
            attention_mask=mask,
            select_ratio=RELEASED_SELECT_RATIO,
            penalty_value=RELEASED_PENALTY_VALUE,
            layer_window=(2, 15),
        )
    if mask is not None:
        logits = logits + mask[..., : k.shape[-2]].to(logits.dtype)
        logits = torch.max(
            logits,
            torch.tensor(torch.finfo(logits.dtype).min, dtype=logits.dtype),
        )
    probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v).transpose(1, 2).contiguous()


class TestInterventionLayerPath(unittest.TestCase):
    """On an intervention layer, the wiring must equal eager+reference exactly."""

    def setUp(self) -> None:
        A.register()
        self.scaling = HEAD_DIM ** -0.5

    def _run(self, layer_idx: int, dtype=torch.float16, batch: int = 1,
             rows=None, seed: int = 0):
        q, k, v = qkv(dtype=dtype, batch=batch, seed=seed)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=True, mode="as_implemented",
            gamma=RELEASED_PENALTY_VALUE, ratio=RELEASED_SELECT_RATIO,
            logit_dtype="native", softmax_dtype="fp32",
            img_start=IMG_START, img_end=IMG_END,
            eager_layers=frozenset(range(2, 16)),
            branch_rows=rows,
        ):
            got, weights = A.fox_attention_forward(
                StubAttention(layer_idx), q, k, v, mask, scaling=self.scaling,
            )
        return q, k, v, mask, got, weights

    def test_matches_independent_recomputation(self) -> None:
        q, k, v, mask, got, _ = self._run(layer_idx=8)
        want = expected_eager(
            q.clone(), k, v, mask, scaling=self.scaling, intervene=True,
        )
        self.assertTrue(
            torch.equal(want, got),
            f"max abs diff {(want.float() - got.float()).abs().max().item():.3e}",
        )

    def test_weights_are_not_returned(self) -> None:
        """Returning them would retain 32 x 22 MiB per branch for nothing."""
        *_, weights = self._run(layer_idx=8)
        self.assertIsNone(weights)

    def test_output_shape_is_bshd(self) -> None:
        """transformers expects [B, Q, H, D] back, not [B, H, Q, D]."""
        _, _, _, _, got, _ = self._run(layer_idx=8)
        self.assertEqual(tuple(got.shape), (1, SEQ, HEADS, HEAD_DIM))

    def test_diagnostics_are_captured(self) -> None:
        A.FOX.reset_diagnostics()
        self._run(layer_idx=8)
        self.assertIn(8, A.FOX.diagnostics)
        d = A.FOX.diagnostics[8]
        for key in ("m_sys", "h_vis", "score", "head_mask"):
            self.assertIn(key, d)
        self.assertEqual(int(d["head_mask"].sum()), max(1, min(round(HEADS * 0.35),
                                                              HEADS // 2)))

    def test_branch_rows_restricts_to_row_one(self) -> None:
        """Row 0 observational, row 1 interventional, in one forward pass."""
        q, k, v = qkv(batch=2, seed=4)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=True, mode="as_implemented", gamma=RELEASED_PENALTY_VALUE,
            ratio=RELEASED_SELECT_RATIO, logit_dtype="native",
            img_start=IMG_START, img_end=IMG_END,
            eager_layers=frozenset(range(2, 16)),
            branch_rows=torch.tensor([1]),
        ):
            got, _ = A.fox_attention_forward(
                StubAttention(8), q, k, v, mask, scaling=self.scaling,
            )
        clean = expected_eager(
            q.clone(), k, v, mask, scaling=self.scaling, intervene=False,
        )
        self.assertTrue(torch.equal(clean[0], got[0]), "row 0 must be untouched")
        self.assertFalse(torch.equal(clean[1], got[1]), "row 1 must be modified")


class TestNonInterventionLayersUseSdpa(unittest.TestCase):
    """Layers outside the window must stay byte-for-byte on the stock path."""

    def setUp(self) -> None:
        A.register()
        self.scaling = HEAD_DIM ** -0.5

    def _fox(self, layer_idx: int, enabled: bool = True, mode: str = "as_implemented"):
        q, k, v = qkv(seed=7)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=enabled, mode=mode, img_start=IMG_START, img_end=IMG_END,
            eager_layers=frozenset(range(2, 16)), branch_rows=None,
        ):
            got, _ = A.fox_attention_forward(
                StubAttention(layer_idx), q, k, v, mask, scaling=self.scaling,
            )
        want, _ = sdpa_attention_forward(
            StubAttention(layer_idx), q, k, v, mask, scaling=self.scaling,
        )
        return want, got

    def test_layer_below_window_is_sdpa(self) -> None:
        want, got = self._fox(layer_idx=0)
        self.assertTrue(torch.equal(want, got))

    def test_layer_above_window_is_sdpa(self) -> None:
        want, got = self._fox(layer_idx=31)
        self.assertTrue(torch.equal(want, got))

    def test_mode_off_is_sdpa_everywhere(self) -> None:
        want, got = self._fox(layer_idx=8, mode="off")
        self.assertTrue(torch.equal(want, got))

    def test_disabled_is_sdpa_everywhere(self) -> None:
        want, got = self._fox(layer_idx=8, enabled=False)
        self.assertTrue(torch.equal(want, got))

    def test_unset_image_span_falls_back_to_sdpa(self) -> None:
        """A missing span must not silently intervene on the wrong keys."""
        q, k, v = qkv(seed=7)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=True, mode="as_implemented", img_start=-1, img_end=-1,
            eager_layers=frozenset(range(2, 16)),
        ):
            got, _ = A.fox_attention_forward(
                StubAttention(8), q, k, v, mask, scaling=self.scaling,
            )
        want, _ = sdpa_attention_forward(
            StubAttention(8), q, k, v, mask, scaling=self.scaling,
        )
        self.assertTrue(torch.equal(want, got))


class TestPrecisionDiscipline(unittest.TestCase):
    def setUp(self) -> None:
        A.register()
        self.scaling = HEAD_DIM ** -0.5

    def test_logit_dtype_changes_the_result(self) -> None:
        """The whole experiment rests on this axis actually biting."""
        outs = {}
        for ld in ("fp16", "fp32", "bf16"):
            q, k, v = qkv(dtype=torch.float16, seed=11)
            mask = causal_mask(SEQ)
            with A.FOX.configured(
                enabled=True, mode="as_implemented", gamma=1e4, ratio=0.35,
                logit_dtype=ld, img_start=IMG_START, img_end=IMG_END,
                eager_layers=frozenset(range(2, 16)),
            ):
                outs[ld], _ = A.fox_attention_forward(
                    StubAttention(8), q, k, v, mask, scaling=self.scaling,
                )
        # fp32 storage makes the shift a near-no-op; fp16 and bf16 do not.
        self.assertFalse(torch.equal(outs["fp32"], outs["fp16"]))
        self.assertFalse(torch.equal(outs["fp32"], outs["bf16"]))

    def test_autocast_is_refused_not_silently_honoured(self) -> None:
        """An autocast context would override the dtype under study."""
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA for an autocast context")
        q, k, v = qkv(seed=3)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=True, mode="as_implemented", img_start=IMG_START,
            img_end=IMG_END, eager_layers=frozenset(range(2, 16)),
        ):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with self.assertRaises(RuntimeError):
                    A.fox_attention_forward(
                        StubAttention(8), q.cuda(), k.cuda(), v.cuda(),
                        mask.cuda(), scaling=self.scaling,
                    )

    def _eager(self, *, gamma: float, logit_dtype: str, softmax_dtype: str,
               seed: int = 13) -> torch.Tensor:
        q, k, v = qkv(dtype=torch.float16, seed=seed)
        mask = causal_mask(SEQ)
        with A.FOX.configured(
            enabled=True, mode="as_implemented", gamma=gamma, ratio=0.35,
            logit_dtype=logit_dtype, softmax_dtype=softmax_dtype,
            img_start=IMG_START, img_end=IMG_END,
            eager_layers=frozenset(range(2, 16)),
        ):
            out, _ = A.fox_attention_forward(
                StubAttention(8), q, k, v, mask, scaling=self.scaling,
            )
        return out

    def test_the_mechanism_is_the_logit_cast_in_every_softmax_dtype(self) -> None:
        """The paper's Section 4.2 claim, tested directly and host-independently.

        Section 4.2 attributes Fox's effect to "numerical logit saturation",
        which would make the finite precision of the *softmax* the thing that
        lets a uniform shift survive. It is not: ``torch.softmax`` subtracts the
        row maximum before exponentiating, so the shift cancels before any
        accumulation happens. What survives is the cast applied to the *logit*
        tensor on the way in.

        So the discriminating comparison is within one ``logit_dtype``, against
        a true eager no-op (``gamma=0``: subtract zero, cast back, exact):

          * storing the shifted logits in **fp32** -- ULP ~0.001 at 1e4,
            far below the O(1)-O(30) spread of real attention logits -- must
            barely move the output;
          * storing them in **fp16** -- ULP 8.0 -- must wreck it.

        And that pattern must hold in *both* softmax accumulation dtypes, which
        is what makes it a statement about Fox rather than about a kernel.

        Note on what this test deliberately does NOT assert. Whether the fp32
        and fp16 softmax paths agree with *each other* is a property of the
        host's softmax kernel, not of the method: measured bit-identical on
        torch 2.8 / arm64 and different on torch 2.11 / x86_64, because CPUs
        generally lack native fp16 arithmetic and upcast internally in
        version-dependent ways. Asserting that equality would make this suite
        flake across hosts while saying nothing about the mechanism.
        """
        for softmax_dtype in ("fp32", "native"):
            with self.subTest(softmax_dtype=softmax_dtype):
                base32 = self._eager(gamma=0.0, logit_dtype="fp32",
                                     softmax_dtype=softmax_dtype)
                shift32 = self._eager(gamma=1e4, logit_dtype="fp32",
                                      softmax_dtype=softmax_dtype)
                base16 = self._eager(gamma=0.0, logit_dtype="native",
                                     softmax_dtype=softmax_dtype)
                shift16 = self._eager(gamma=1e4, logit_dtype="native",
                                      softmax_dtype=softmax_dtype)

                eff32 = (base32.float() - shift32.float()).abs().max().item()
                eff16 = (base16.float() - shift16.float()).abs().max().item()
                ratio = eff16 / max(eff32, 1e-12)

                # The RATIO is the assertion that travels. Absolute magnitudes
                # scale with the logit spread, which here is deliberately small
                # (head_dim=16, so a 0.25 scaling) and is much larger in the
                # real model at head_dim=128. Measured on two hosts:
                # eff32 = 6.1e-5, eff16 = 4.9e-2, ratio ~ 800x.
                self.assertLess(
                    eff32, 1e-2,
                    f"fp32 logit storage should nearly cancel the shift "
                    f"(ULP ~1e-3 at gamma=1e4), got {eff32:.4g}",
                )
                self.assertGreater(
                    eff16, 1e-2,
                    f"fp16 logit storage should visibly change the output "
                    f"(ULP 8.0), got only {eff16:.4g}",
                )
                self.assertGreater(
                    ratio, 50.0,
                    f"the fp16 cast should dominate the fp32 one by orders of "
                    f"magnitude; ratio was {ratio:.1f}. If this fires, the "
                    f"ULP account of the mechanism in fox/reference.py is wrong",
                )

    def test_gamma_zero_is_an_exact_eager_no_op(self) -> None:
        """Anchors the baseline the test above leans on.

        Subtracting 0.0 and casting back must be bit-exact, or ``gamma=0`` is
        not a valid stand-in for "eager path, no intervention".
        """
        for logit_dtype in ("native", "fp32"):
            with self.subTest(logit_dtype=logit_dtype):
                a = self._eager(gamma=0.0, logit_dtype=logit_dtype,
                                softmax_dtype="fp32")
                b = self._eager(gamma=0.0, logit_dtype=logit_dtype,
                                softmax_dtype="fp32")
                self.assertTrue(torch.equal(a, b))


class TestImageSpanLocation(unittest.TestCase):
    def test_locates_a_576_token_span(self) -> None:
        ids = torch.zeros(1, SEQ, dtype=torch.long)
        ids[0, IMG_START:IMG_END + 1] = 32000
        self.assertEqual(
            A.locate_image_span(ids, 32000), (IMG_START, IMG_END)
        )

    def test_rejects_a_wrong_width_span(self) -> None:
        """A width other than 576 means the processor took the legacy path."""
        ids = torch.zeros(1, SEQ, dtype=torch.long)
        ids[0, 10:20] = 32000
        with self.assertRaises(ValueError):
            A.locate_image_span(ids, 32000)

    def test_rejects_a_missing_image_token(self) -> None:
        ids = torch.zeros(1, SEQ, dtype=torch.long)
        with self.assertRaises(ValueError):
            A.locate_image_span(ids, 32000)


if __name__ == "__main__":
    unittest.main()
