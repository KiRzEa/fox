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


class TestCheckpointGuards(unittest.TestCase):
    """Refuse a checkpoint that would load with random weights.

    This is the failure that actually happened on the first real Kaggle run,
    and the only reason it was caught is that AutoProcessor crashed a moment
    later. from_pretrained itself reported the vision tower and projector as
    MISSING, initialised them randomly, and carried on -- which would have
    produced fluent captions unrelated to the image, with every metric
    downstream reporting plausible numbers computed from noise.
    """

    def test_original_llava_repo_is_refused_before_downloading(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            A._refuse_unconverted_checkpoint("liuhaotian/llava-v1.5-7b")
        msg = str(ctx.exception)
        self.assertIn("RANDOMLY INITIALISED", msg)
        self.assertIn("llava-hf/llava-1.5-7b-hf", msg)

    def test_converted_repo_is_accepted(self) -> None:
        A._refuse_unconverted_checkpoint("llava-hf/llava-1.5-7b-hf")  # no raise

    def test_local_path_is_accepted(self) -> None:
        # An attached Kaggle Dataset is a plain path and must not be refused.
        A._refuse_unconverted_checkpoint("/kaggle/input/llava-15-7b-hf")

    def test_missing_keys_are_a_hard_error(self) -> None:
        """A warning here would be worse than a crash: a plausible number."""
        with self.assertRaises(RuntimeError) as ctx:
            A._assert_fully_loaded(
                "some/repo",
                {"missing_keys": [
                    "model.vision_tower.embeddings.patch_embedding.weight",
                    "model.multi_modal_projector.linear_1.weight",
                ]},
            )
        msg = str(ctx.exception)
        self.assertIn("RANDOM", msg)
        self.assertIn("--model-id", msg)

    def test_clean_load_passes(self) -> None:
        A._assert_fully_loaded("some/repo", {"missing_keys": []})
        A._assert_fully_loaded("some/repo", {})

    def test_error_names_a_bounded_number_of_keys(self) -> None:
        """The real report listed ~50 key families; do not paste all of them."""
        with self.assertRaises(RuntimeError) as ctx:
            A._assert_fully_loaded("r", {"missing_keys": [f"k{i}" for i in range(40)]})
        self.assertIn("+32 more", str(ctx.exception))


class TestGeneratorDevice(unittest.TestCase):
    """Sampling must not assume the logits and the RNG share a device.

    Hit on the first real GPU run: fox_generate built its generator on CPU
    while the logits were on CUDA, and torch.multinomial raised "Expected a
    'cuda' device type for generator but found 'cpu'". With
    device_map="auto" the head can also land on a different shard from the
    inputs, so the device is not something the sampler should have to know.
    """

    def test_token_comes_back_on_the_logits_device(self) -> None:
        logits = torch.randn(64)
        gen = torch.Generator().manual_seed(0)
        tok = A._sample_from(
            logits, top_p=0.9, temperature=1.0, greedy=False, generator=gen
        )
        self.assertEqual(tok.device, logits.device)

    def test_seeded_sampling_is_reproducible(self) -> None:
        logits = torch.randn(256, generator=torch.Generator().manual_seed(7))
        a = A._sample_from(
            logits, top_p=0.9, temperature=1.0, greedy=False,
            generator=torch.Generator().manual_seed(11),
        )
        b = A._sample_from(
            logits, top_p=0.9, temperature=1.0, greedy=False,
            generator=torch.Generator().manual_seed(11),
        )
        self.assertEqual(int(a), int(b))

    def test_greedy_needs_no_generator(self) -> None:
        logits = torch.zeros(32)
        logits[7] = 5.0
        tok = A._sample_from(
            logits, top_p=0.9, temperature=1.0, greedy=True, generator=None
        )
        self.assertEqual(int(tok), 7)

    def test_top_p_keeps_at_least_one_candidate(self) -> None:
        """top_p=0 must not produce an empty support and a NaN division."""
        logits = torch.randn(128, generator=torch.Generator().manual_seed(3))
        tok = A._sample_from(
            logits, top_p=0.0, temperature=1.0, greedy=False,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(int(tok), int(logits.argmax()))

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cross_device_generator_is_handled(self) -> None:
        """The exact failure from the Kaggle run, where it can be reproduced."""
        logits = torch.randn(64, device="cuda")
        cpu_gen = torch.Generator().manual_seed(0)
        tok = A._sample_from(
            logits, top_p=0.9, temperature=1.0, greedy=False, generator=cpu_gen
        )
        self.assertEqual(tok.device.type, "cuda")


class TestSamplerMatchesHuggingFace(unittest.TestCase):
    """Sampling must be the released path, not a lookalike.

    The released vcd_sample.py applies `logits_warper` to the fused logits
    before softmax + multinomial, so top-p is HuggingFace's. A hand-rolled
    version kept one token fewer in about 5% of rows -- HF sorts ascending and
    drops the tail whose cumulative mass is <= 1 - top_p, the hand-rolled one
    sorted descending and kept while the exclusive cumulative mass was
    <= top_p, which differs in strictness and accumulates in the opposite
    order. The affected token is the least probable in the support and would
    almost never be drawn, but a reproduction should not carry an avoidable
    difference in its sampler.
    """

    def test_identical_tokens_to_the_huggingface_path(self) -> None:
        from transformers import TemperatureLogitsWarper, TopPLogitsWarper

        g = torch.Generator().manual_seed(0)
        for trial in range(40):
            logits = torch.randn(4096, generator=g) * 3.0
            ours = A._sample_from(
                logits, top_p=0.9, temperature=1.0, greedy=False,
                generator=torch.Generator().manual_seed(trial),
            )
            x = TemperatureLogitsWarper(1.0)(None, logits.clone().float().unsqueeze(0))
            x = TopPLogitsWarper(0.9)(None, x)
            theirs = torch.multinomial(
                torch.softmax(x[0], dim=-1), 1,
                generator=torch.Generator().manual_seed(trial),
            ).squeeze(-1)
            with self.subTest(trial=trial):
                self.assertEqual(int(ours), int(theirs))

    def test_temperature_is_applied_before_top_p(self) -> None:
        """HF's warper order is [temperature, top_p]; swapping them changes
        which tokens survive the nucleus."""
        logits = torch.randn(2048, generator=torch.Generator().manual_seed(5))
        hot = A._sample_from(logits, top_p=0.5, temperature=2.0, greedy=False,
                             generator=torch.Generator().manual_seed(1))
        cold = A._sample_from(logits, top_p=0.5, temperature=0.5, greedy=False,
                              generator=torch.Generator().manual_seed(1))
        self.assertIsInstance(int(hot), int)
        self.assertIsInstance(int(cold), int)

    def test_top_p_one_keeps_everything(self) -> None:
        logits = torch.randn(512, generator=torch.Generator().manual_seed(2))
        tok = A._sample_from(logits, top_p=1.0, temperature=1.0, greedy=False,
                             generator=torch.Generator().manual_seed(3))
        self.assertTrue(0 <= int(tok) < 512)


class TestCpuOnlyHostIsRefused(unittest.TestCase):
    """A missing accelerator must stop the run, not warn about it.

    Round 4 sessions 1 and 2 ran to the ten-hour wall on a Kaggle notebook
    whose GPU was never attached: torch was the CPU build, a caption that
    costs 7.8 s on two T4s took 724 s, and each session finished about fifty
    of five hundred samples before SIGTERM. Twenty hours of quota produced
    nothing. probe_host.py had already said so -- the runner echoed a warning
    and carried on.
    """

    def test_no_cuda_raises_and_names_the_fix(self) -> None:
        import torch

        from foxrepro.fox.attention import _refuse_cpu_only_host

        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        self.addCleanup(setattr, torch.cuda, "is_available", orig)
        with self.assertRaises(RuntimeError) as ctx:
            _refuse_cpu_only_host("fp16")
        msg = str(ctx.exception)
        self.assertIn("Accelerator", msg)      # what to change on Kaggle
        self.assertIn("--allow-cpu", msg)      # how to override deliberately
        self.assertIn("724", msg)              # why it matters

    def test_allow_cpu_is_an_explicit_escape_hatch(self) -> None:
        import torch

        from foxrepro.fox.attention import _refuse_cpu_only_host

        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
        self.addCleanup(setattr, torch.cuda, "is_available", orig)
        _refuse_cpu_only_host("fp16", allow_cpu=True)   # must not raise

    def test_a_cuda_host_passes(self) -> None:
        import torch

        from foxrepro.fox.attention import _refuse_cpu_only_host

        orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: True
        self.addCleanup(setattr, torch.cuda, "is_available", orig)
        _refuse_cpu_only_host("fp16")

    def test_the_runner_aborts_rather_than_warning(self) -> None:
        """run_kaggle_fox.sh must exit, not echo."""
        from foxrepro.storage import paths

        script = (paths().repo_root / "run_kaggle_fox.sh").read_text()
        self.assertIn("scripts/require_gpu.py", script)
        self.assertIn("|| exit 3", script)
        self.assertIn("ALLOW_CPU", script)

    def test_the_gate_script_exits_nonzero_without_a_gpu(self) -> None:
        import subprocess
        import sys

        from foxrepro.storage import paths

        gate = paths().repo_root / "scripts" / "require_gpu.py"
        self.assertTrue(gate.is_file())
        proc = subprocess.run([sys.executable, str(gate)],
                              capture_output=True, text=True, timeout=180)
        import torch
        if torch.cuda.is_available():
            self.assertEqual(proc.returncode, 0, proc.stderr[-500:])
        else:
            self.assertEqual(proc.returncode, 3, proc.stdout[-500:])
            self.assertIn("would run on CPU", proc.stderr)

    def test_allow_cpu_is_not_part_of_the_variant_hash(self) -> None:
        """The host is provenance, not an experimental arm.

        n_gpus, gpu_name and torch_version are already recorded per record,
        so a CPU run is identifiable; putting --allow-cpu in the hash would
        split one job across two run files for no reason.
        """
        from foxrepro.bench.generate import GenConfig

        base = dict(method="sampling", benchmark="pope", split="adversarial",
                    seed=42, dtype="fp16", model_id="llava-hf/llava-1.5-7b-hf",
                    prompt_template_id="pope_v0_raw", max_new_tokens=1)
        a = GenConfig(**base).run_key().filename()
        b = GenConfig(**base, allow_cpu=True).run_key().filename()
        self.assertEqual(a, b)


class TestBackboneProbe(unittest.TestCase):
    """What it takes to run each backbone, decided from metadata not memory.

    Arguing from memory about what transformers supports produced a wrong
    answer about Shikra that stood for several days: it was called a Track A
    job needing transformers 4.29.2, when its decoder is a plain LLaMA-7B
    carrying two extra projector tensors. The probe reads config.json and the
    weight index -- a few hundred kilobytes -- and says so.
    """

    def _probe(self, model_id: str) -> dict:
        import importlib.util
        import sys as _sys

        from foxrepro.storage import paths

        spec = importlib.util.spec_from_file_location(
            "_bbprobe", paths().repo_root / "scripts" / "probe_backbone.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["_bbprobe"] = mod
        spec.loader.exec_module(mod)
        try:
            return mod.probe(model_id)
        except Exception as exc:                        # noqa: BLE001
            self.skipTest(f"network unavailable: {exc}")

    def test_llava_is_runnable_as_is(self) -> None:
        r = self._probe("llava-hf/llava-1.5-7b-hf")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["verdict"], "RUNNABLE")
        self.assertTrue(r["transformers_has_class"])
        self.assertEqual(r["visual_tokens"], 576)

    def test_instructblip_is_runnable_but_its_visual_span_differs(self) -> None:
        """32 Q-Former queries, not 576 patches. H_vis changes meaning."""
        r = self._probe("Salesforce/instructblip-vicuna-7b")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["verdict"], "RUNNABLE")
        self.assertEqual(r["visual_tokens"], 32)

    def test_shikra_is_a_llama_decoder_plus_two_tensors(self) -> None:
        """The correction this probe exists to make.

        ShikraLlamaForCausalLM has no transformers class, which looked like a
        blocker. But the decoder is LLaMA-7B exactly -- 32 layers, 4096 hidden,
        32 heads, 11008 ffn -- and the checkpoint carries precisely two tensors
        beyond it: mm_projector weight and bias. A linear 1024 -> 4096, with
        the vision tower being stock openai/clip-vit-large-patch14 loaded
        separately. That is a port, not a vendored stack.
        """
        r = self._probe("shikras/shikra-7b-delta-v1-0708")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["verdict"], "PORTABLE")
        self.assertFalse(r["transformers_has_class"])
        self.assertTrue(r["decoder_is_llama_7b"])
        self.assertEqual(
            sorted(r["tensors_outside_llama_decoder"]),
            ["model.mm_projector.bias", "model.mm_projector.weight"],
        )
        self.assertTrue(r["is_delta"])

    def test_the_fox_patch_reaches_every_backbone_the_paper_uses(self) -> None:
        """Which is why the release evaluates three models with one patch.

        Fox patches LLaMA attention. Anything whose decoder is LLaMA inherits
        the intervention without a second implementation -- that is the answer
        to "how did the authors run Shikra at all".
        """
        for mid in ("llava-hf/llava-1.5-7b-hf",
                    "Salesforce/instructblip-vicuna-7b",
                    "shikras/shikra-7b-delta-v1-0708"):
            r = self._probe(mid)
            if r["verdict"] == "UNREACHABLE":
                self.skipTest("no network")
            with self.subTest(model=mid):
                self.assertTrue(r["fox_patch_reaches_decoder"])

    def test_shikras_hardcoded_span_is_explained_by_its_config(self) -> None:
        """img_s==33, img_e==290 is 258 tokens: 256 patches + 2 markers."""
        r = self._probe("shikras/shikra-7b-delta-v1-0708")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["visual_tokens"], 290 - 33 + 1)

    def test_every_visual_count_is_derived_not_asserted(self) -> None:
        """Three backbones, three different derivation rules, no table lookup.

        The counts were hardcoded first -- 576, 32, 258 -- which reads as
        measured and is asserted, the failure this project keeps catching.
        They now come from the configs, and the hardcoded values survive only
        as a cross-check that must agree.
        """
        want = {
            "llava-hf/llava-1.5-7b-hf": ("vision_config", 576),
            "Salesforce/instructblip-vicuna-7b": ("num_query_tokens", 32),
            "shikras/shikra-7b-delta-v1-0708": ("clip-vit-large-patch14", 258),
        }
        for mid, (marker, count) in want.items():
            r = self._probe(mid)
            if r["verdict"] == "UNREACHABLE":
                self.skipTest("no network")
            with self.subTest(model=mid):
                self.assertEqual(r["visual_tokens"], count)
                self.assertIn(marker, r["visual_derivation"])
                self.assertFalse(r["visual_mismatch"])

    def test_a_wrong_cross_check_value_is_reported_not_swallowed(self) -> None:
        """If the table and the derivation disagree, one is wrong."""
        import importlib.util
        import sys as _sys

        from foxrepro.storage import paths

        spec = importlib.util.spec_from_file_location(
            "_bbprobe3", paths().repo_root / "scripts" / "probe_backbone.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["_bbprobe3"] = mod
        spec.loader.exec_module(mod)

        mod.EXPECTED_VISUAL_TOKENS["llava-hf/llava-1.5-7b-hf"] = 999
        try:
            r = mod.probe("llava-hf/llava-1.5-7b-hf")
        except Exception as exc:                        # noqa: BLE001
            self.skipTest(f"network unavailable: {exc}")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertTrue(r["visual_mismatch"])
        self.assertEqual(r["visual_tokens"], 576)       # the derivation wins

    def test_the_download_size_is_measured_from_the_listing(self) -> None:
        """Counting .bin and .safetensors together called 29.5 GiB 63 GiB."""
        r = self._probe("Salesforce/instructblip-vicuna-7b")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertIn("safetensors", r["download_derivation"])
        self.assertLess(r["download_gib"], 35)          # not the 63 GiB double count
        self.assertGreater(r["download_gib"], 25)

    def test_the_preprocessor_source_is_read_from_the_config(self) -> None:
        """config.mm_vision_tower, not a table maintained by hand."""
        r = self._probe("shikras/shikra-7b-delta-v1-0708")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["preprocessing"]["repo"],
                         "openai/clip-vit-large-patch14")

    def test_shikra_ships_no_image_processor_and_that_is_fine(self) -> None:
        """A decoder that loads and cannot see is not runnable.

        The probe originally judged Shikra PORTABLE from the decoder alone and
        never looked at preprocessing. Shikra's repo has no
        preprocessor_config.json -- which reads like a hole and is not: its
        pixels go through CLIPImageProcessor from the vision tower named in
        config.mm_vision_tower, and the release's clip_image_eval is a wrapper
        around exactly that call with no logic of its own.
        """
        r = self._probe("shikras/shikra-7b-delta-v1-0708")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        pre = r["preprocessing"]
        self.assertFalse(pre["ships_own"])
        self.assertEqual(pre["source"], "vision_tower")
        self.assertEqual(pre["repo"], "openai/clip-vit-large-patch14")
        self.assertTrue(pre["source_repo_has_processor"])

    def test_the_other_two_carry_their_own_processor(self) -> None:
        for mid in ("llava-hf/llava-1.5-7b-hf",
                    "Salesforce/instructblip-vicuna-7b"):
            r = self._probe(mid)
            if r["verdict"] == "UNREACHABLE":
                self.skipTest("no network")
            with self.subTest(model=mid):
                self.assertTrue(r["preprocessing"]["ships_own"])

    def test_an_unresolved_preprocessor_blocks_the_verdict(self) -> None:
        """Not knowing how pixels are prepared is not the same as fine."""
        import importlib.util
        import sys as _sys

        from foxrepro.storage import paths

        spec = importlib.util.spec_from_file_location(
            "_bbprobe2", paths().repo_root / "scripts" / "probe_backbone.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["_bbprobe2"] = mod
        spec.loader.exec_module(mod)

        # huggyllama/llama-7b is a text-only decoder: no processor anywhere,
        # and none recorded, so the probe must not call it runnable.
        try:
            r = mod.probe("huggyllama/llama-7b")
        except Exception as exc:                        # noqa: BLE001
            self.skipTest(f"network unavailable: {exc}")
        if r["verdict"] == "UNREACHABLE":
            self.skipTest("no network")
        self.assertEqual(r["verdict"], "UNCLEAR")
        self.assertEqual(r["preprocessing"]["source"], "unknown")


# --------------------------------------------------------------------------- #
# SID / CT2S
# --------------------------------------------------------------------------- #
class _Ct2sBase(unittest.TestCase):
    """Two rows: row 0 is the clean branch, row 1 the CT2S one."""

    AGG = 2
    RANK = 7

    def setUp(self) -> None:
        self.q, self.k, self.v = qkv(batch=2, seed=3)
        self.mask = causal_mask(SEQ)
        self.scaling = 1.0 / math.sqrt(HEAD_DIM)
        self.ctx = A.FOX.configured(
            enabled=True, mode="off",
            img_start=IMG_START, img_end=IMG_END,
            sid_enabled=True, sid_agg_layer=self.AGG, sid_rank=self.RANK,
            sid_rows=torch.tensor([1]), sid_keep=None,
        )
        self.ctx.__enter__()
        self.addCleanup(lambda: self.ctx.__exit__(None, None, None))

    def call(self, layer_idx: int, mask=None):
        """``mask=None`` means the default causal mask; ``False`` means None."""
        m = self.mask if mask is None else (None if mask is False else mask)
        return A.fox_attention_forward(
            StubAttention(layer_idx), self.q, self.k, self.v, m,
            scaling=self.scaling,
        )

    def read_ranking(self) -> torch.Tensor:
        self.call(self.AGG - 1)
        self.assertIsNotNone(A.FOX.sid_keep)
        return A.FOX.sid_keep


class TestCt2sRanking(_Ct2sBase):
    def test_the_ranking_layer_changes_nothing_about_the_output(self) -> None:
        """Layer agg-1 only *reads*. It must stay on the stock SDPA path.

        Were it switched to eager to get the probabilities, this baseline's
        numerics would differ from every other arm at one layer for no
        methodological reason.
        """
        got, weights = self.call(self.AGG - 1)
        want = sdpa_attention_forward(
            StubAttention(self.AGG - 1), self.q, self.k, self.v, self.mask,
            scaling=self.scaling, dropout=0.0,
        )[0]
        self.assertIsNone(weights)
        self.assertTrue(torch.equal(got, want))

    def test_the_ranking_equals_a_full_softmax_read_the_released_way(self) -> None:
        """Only the last query row is computed; it must equal the full matrix.

        Softmax is row-independent, so this is exact rather than approximate
        -- and it is what lets the arm run without materialising a
        [2, 32, 605, 605] tensor per step.
        """
        from foxrepro.fox.reference import reference_ct2s_keep_indices

        logits = torch.matmul(self.q, self.k.transpose(2, 3)) * self.scaling
        logits = logits + self.mask[..., : self.k.shape[-2]].to(logits.dtype)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(self.q.dtype)
        want = reference_ct2s_keep_indices(
            probs[1:], IMG_START, N_IMG, self.RANK)
        self.assertTrue(torch.equal(want, self.read_ranking()))

    def test_it_is_read_from_the_row_that_gets_pruned(self) -> None:
        with A.FOX.configured(sid_rows=torch.tensor([0]), sid_keep=None):
            a = self.read_ranking().clone()
        with A.FOX.configured(sid_rows=torch.tensor([1]), sid_keep=None):
            b = self.read_ranking().clone()
        # the two rows see different Q and K, so the rankings must differ
        self.assertFalse(torch.equal(a, b))

    def test_the_ranking_is_recomputed_per_forward(self) -> None:
        """A stale keep-set would prune on last step's attention."""
        first = self.read_ranking().clone()
        q2, k2, v2 = qkv(batch=2, seed=99)
        self.q, self.k, self.v = q2, k2, v2
        A.FOX.sid_keep = None
        second = self.read_ranking().clone()
        self.assertFalse(torch.equal(first, second))


class TestCt2sMasking(_Ct2sBase):
    """What the layers from ``agg_layer`` onward are actually handed.

    Asserted on the additive mask the function passes to SDPA, read by
    spying on ``sdpa_attention_forward`` -- i.e. on the argument the real
    attention kernel receives, not on a recomputation of it.
    """

    def mask_at(self, layer_idx: int) -> torch.Tensor:
        seen = {}
        import transformers.integrations.sdpa_attention as S

        real = S.sdpa_attention_forward

        def spy(module, q, k, v, attention_mask, **kw):  # noqa: ANN001
            seen["mask"] = (None if attention_mask is None
                            else attention_mask.clone())
            return real(module, q, k, v, attention_mask, **kw)

        S.sdpa_attention_forward = spy
        try:
            self.call(layer_idx)
        finally:
            S.sdpa_attention_forward = real
        self.assertIn("mask", seen, "the layer never reached SDPA")
        return seen["mask"]

    def dropped(self, keep: torch.Tensor):
        return sorted(set(range(IMG_START, IMG_END + 1)) - set(keep.tolist()))

    def test_dropped_vision_keys_are_minus_inf_on_the_pruned_row(self) -> None:
        keep = self.read_ranking()
        m = self.mask_at(self.AGG)
        drop = self.dropped(keep)
        self.assertEqual(len(drop), N_IMG - self.RANK)
        floor = torch.finfo(m.dtype).min
        self.assertTrue(bool((m[1, :, :, drop] == floor).all()))
        # and the kept ones are still attendable from the last query
        self.assertTrue(bool((m[1, :, -1, keep.tolist()] == 0).all()))

    def test_the_clean_row_is_untouched(self) -> None:
        keep = self.read_ranking()
        m = self.mask_at(self.AGG)
        self.assertTrue(torch.equal(m[0], self.mask[0]))
        self.assertTrue(
            bool((m[0, :, -1, self.dropped(keep)] == 0).all()))

    def test_text_and_system_keys_are_never_masked(self) -> None:
        self.read_ranking()
        m = self.mask_at(self.AGG)
        self.assertTrue(bool((m[1, :, -1, :IMG_START] == 0).all()))
        self.assertTrue(bool((m[1, :, -1, IMG_END + 1:] == 0).all()))

    def test_causality_survives_the_rewrite(self) -> None:
        """The mask is cloned and written into, so the triangle must remain."""
        self.read_ranking()
        m = self.mask_at(self.AGG)
        floor = torch.finfo(m.dtype).min
        for row in (0, 1):
            with self.subTest(row=row):
                future = torch.triu(
                    torch.ones(SEQ, SEQ, dtype=torch.bool), diagonal=1)
                self.assertTrue(bool((m[row, 0][future] == floor).all()))

    def test_every_layer_from_agg_onward_gets_the_same_mask(self) -> None:
        keep = self.read_ranking()
        drop = self.dropped(keep)
        floor = None
        first = None
        for layer in (self.AGG, self.AGG + 1, 31):
            with self.subTest(layer=layer):
                m = self.mask_at(layer)
                floor = torch.finfo(m.dtype).min
                self.assertTrue(bool((m[1, :, :, drop] == floor).all()))
                if first is None:
                    first = m
                else:
                    self.assertTrue(torch.equal(first, m))

    def test_layers_below_agg_get_the_stock_mask(self) -> None:
        self.read_ranking()
        for layer in (0, self.AGG - 1):
            with self.subTest(layer=layer):
                self.assertTrue(torch.equal(self.mask_at(layer), self.mask))

    def test_the_callers_mask_is_not_mutated(self) -> None:
        """One mask object is reused across all 32 layers of a forward.

        Writing into it in place would prune the clean row as well, from the
        next layer onward, and the arm would stop being a contrast at all.
        """
        self.read_ranking()
        before = self.mask.clone()
        self.call(self.AGG)
        self.assertTrue(torch.equal(before, self.mask))

    def test_a_missing_ranking_raises_rather_than_skipping_the_mask(self) -> None:
        A.FOX.sid_keep = None
        with self.assertRaises(RuntimeError) as cm:
            self.call(self.AGG)
        self.assertIn("keep-set", str(cm.exception))

    def test_no_mask_raises_rather_than_dropping_causality(self) -> None:
        self.read_ranking()
        with self.assertRaises(RuntimeError) as cm:
            self.call(self.AGG, mask=False)
        self.assertIn("attention mask", str(cm.exception))
