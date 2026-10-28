"""The method implementations, exercised against a fake model.

No GPU, no checkpoint, no 14 GB download. A stub model returns deterministic
logits and a stub processor returns plausible tensors, which is enough to pin
the things most likely to break silently:

* ``sampling`` must take the single-branch path and ``fox`` the dual-branch
  one -- if the baseline accidentally ran the fusion, every table would be
  measured against the wrong reference;
* both must go through the SAME sampling code, so that the only difference
  between them is the intervention;
* the LLaVA-1.5 conversation wrapper must be applied, since a prompt mismatch
  is the commonest silent source of a 2-4 point POPE gap in reproductions and
  would be mistaken for a method effect;
* Fox must resolve its hyperparameters from the released code's values, not
  the paper's, and must record which it used;
* the VCD row must be labelled, because what this repo can run is VCD's noised
  branch inside Fox's fusion, not VCD's own contrastive formula.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover - CPU-only checkout
    raise unittest.SkipTest("torch is not installed") from None

try:
    import transformers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("transformers is not installed") from None

from foxrepro.bench.generate import GenConfig, get_method  # noqa: E402
from foxrepro.bench.methods import (  # noqa: E402
    FOX_DEFAULTS, LLAVA_WRAPPERS, parse_layer_window, render_full_prompt,
)
from foxrepro.data.loaders import Sample  # noqa: E402

VOCAB = 64
N_IMG = 576
IMAGE_TOKEN_ID = 32000
EOS = 2


class FakeTextConfig:
    eos_token_id = EOS


class FakeConfig:
    image_token_index = IMAGE_TOKEN_ID
    text_config = FakeTextConfig()


class FakeOutput:
    def __init__(self, logits: torch.Tensor, cache: Any):
        self.logits = logits
        self.past_key_values = cache


class FakeModel:
    """Records how it was called and returns deterministic logits.

    ``rows_seen`` is the point of the whole fixture: it is how the tests tell a
    single-branch baseline from a dual-branch method.
    """

    def __init__(self, n_steps_until_eos: int = 3):
        self.config = FakeConfig()
        self.rows_seen: List[int] = []
        self.pixel_rows_seen: List[int] = []
        self.calls = 0
        self._n = n_steps_until_eos
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def __call__(self, **kw: Any) -> FakeOutput:
        ids = kw["input_ids"]
        b, q = ids.shape
        self.rows_seen.append(b)
        if kw.get("pixel_values") is not None:
            self.pixel_rows_seen.append(int(kw["pixel_values"].shape[0]))
        self.calls += 1
        g = torch.Generator().manual_seed(self.calls)
        logits = torch.randn(b, q, VOCAB, generator=g)
        # Force EOS after a fixed number of steps so the loop terminates.
        if self.calls >= self._n:
            logits[:, -1, :] = -20.0
            logits[:, -1, EOS] = 20.0
        return FakeOutput(logits, cache=object())


class FakeTokenizer:
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"t{i}" for i in ids if i != EOS)


class FakeProcessor:
    """Captures the text it was handed, so the wrapper can be asserted."""

    def __init__(self) -> None:
        self.tokenizer = FakeTokenizer()
        self.seen_text: List[str] = []

    def __call__(self, images: Any, text: str, return_tensors: str = "pt"):
        self.seen_text.append(text)
        n_sys, n_tail = 24, 5
        total = n_sys + N_IMG + n_tail
        ids = torch.zeros(1, total, dtype=torch.long)
        ids[0, n_sys:n_sys + N_IMG] = IMAGE_TOKEN_ID
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(1, total, dtype=torch.long),
            "pixel_values": torch.zeros(1, 3, 336, 336),
        }


def make_cfg(method: str, **over: Any) -> GenConfig:
    base: Dict[str, Any] = dict(
        method=method, benchmark="pope", split="random", seed=42,
        dtype="fp16", model_id="fake/llava", prompt_template_id="pope_v1",
        max_new_tokens=8,
    )
    base.update(over)
    return GenConfig(**base)


def make_sample(tmp: Path) -> Sample:
    img = tmp / "COCO_val2014_000000000042.jpg"
    img.write_bytes(b"not-a-real-jpeg")
    return Sample(
        sample_id="pope/random/q0000001", image_id=42, image_path=img,
        benchmark="pope", split="random",
        question="Is there a dog in the image?", label="no", extra={},
    )


class _Base(unittest.TestCase):
    """Installs the fakes in place of load_llava and PIL."""

    def setUp(self) -> None:
        import foxrepro.bench.methods as M
        from foxrepro.fox import attention as A

        self.M = M
        # enterContext() is 3.11+; addCleanup keeps this runnable on 3.9/3.10 too.
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.sample = make_sample(self.tmp)
        self.model = FakeModel()
        self.processor = FakeProcessor()

        # Patch the loader and the image read; everything else is real code.
        self._orig_load = A.load_llava
        A.load_llava = lambda *a, **k: (self.model, self.processor)  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(A, "load_llava", self._orig_load))

        self._orig_encode = M._LlavaMethod._encode

        def _encode(inner_self, sample, text):  # noqa: ANN001
            # Only the PIL image read is stubbed out. The prompt still goes
            # through the real _prepare -> render_full_prompt, so the
            # system-prompt arms and ICD's disturbed prompt are actually
            # exercised rather than bypassed.
            return inner_self.processor(
                images=None, text=text, return_tensors="pt")

        M._LlavaMethod._encode = _encode  # type: ignore[assignment]
        self.addCleanup(
            lambda: setattr(M._LlavaMethod, "_encode", self._orig_encode)
        )

    def run_method(self, name: str, **over: Any):
        cls = get_method(name)
        m = cls(make_cfg(name, **over))
        m.load()
        out = m.generate(self.sample, "Is there a dog in the image?")
        return m, out


class TestBranchStructure(_Base):
    def test_sampling_is_single_branch(self) -> None:
        """If the baseline ran the fusion, every table would be wrong."""
        _, out = self.run_method("sampling")
        self.assertEqual(set(self.model.rows_seen), {1})
        self.assertGreater(out.n_gen_tokens, 0)

    def test_fox_is_dual_branch(self) -> None:
        _, out = self.run_method("fox", mode="as_implemented", gamma=10000.0)
        self.assertEqual(set(self.model.rows_seen), {2})
        self.assertGreater(out.n_gen_tokens, 0)

    def test_fox_mode_off_is_still_dual_branch(self) -> None:
        """mode=off disables the intervention but keeps the fusion.

        That is the control which isolates the additive fusion, and it is NOT
        the same thing as the single-branch baseline.
        """
        self.run_method("fox", mode="off")
        self.assertEqual(set(self.model.rows_seen), {2})

    def test_vcd_feeds_two_different_images(self) -> None:
        self.run_method("vcd")
        self.assertEqual(set(self.model.rows_seen), {2})
        self.assertEqual(set(self.model.pixel_rows_seen), {2})

    def test_vcd_fox_fusion_also_feeds_two_images(self) -> None:
        """The old arm, kept under its honest name."""
        self.run_method("vcd_fox_fusion")
        self.assertEqual(set(self.model.rows_seen), {2})
        self.assertEqual(set(self.model.pixel_rows_seen), {2})

    def test_icd_runs_two_passes_with_one_image_each(self) -> None:
        """ICD disturbs the instruction, so the sequences differ in length.

        They cannot share a batch; the loop must run two single-row passes.
        """
        self.run_method("icd")
        self.assertEqual(set(self.model.rows_seen), {1})
        self.assertEqual(set(self.model.pixel_rows_seen), {1})


class TestPromptWrapper(_Base):
    def test_llava_wrapper_is_applied(self) -> None:
        """The turn structure, independent of which system prompt precedes it."""
        self.run_method("sampling")
        text = self.processor.seen_text[0]
        self.assertIn("USER: <image>", text)
        self.assertTrue(text.rstrip().endswith("ASSISTANT:"))
        self.assertIn("Is there a dog in the image?", text)
        # the image marker must come after the system prompt, not before it
        self.assertLess(text.index("USER:"), text.index("<image>"))

    def test_wrapper_is_recorded_in_provenance(self) -> None:
        m, _ = self.run_method("sampling")
        self.assertEqual(
            m.describe()["wrapper"], LLAVA_WRAPPERS["released"]
        )


class TestFoxHyperparameters(_Base):
    def test_defaults_come_from_the_code_not_the_paper(self) -> None:
        m, _ = self.run_method("fox", mode="as_implemented")
        r = m.describe()["resolved_hyperparameters"]
        self.assertEqual(r["gamma"], 10000.0)
        self.assertEqual(r["ratio"], 0.35)      # paper says 0.45
        self.assertAlmostEqual(r["tau_js"], 0.15)  # paper says 0.2
        self.assertEqual(r["alpha"], 2.0)
        self.assertAlmostEqual(r["beta"], 0.1)

    def test_explicit_values_override_the_defaults(self) -> None:
        m, _ = self.run_method(
            "fox", mode="as_implemented", gamma=1e3, k=0.45, tau_js=0.2
        )
        r = m.describe()["resolved_hyperparameters"]
        self.assertEqual(r["gamma"], 1e3)
        self.assertEqual(r["ratio"], 0.45)
        self.assertAlmostEqual(r["tau_js"], 0.2)

    def test_the_disagreement_is_recorded(self) -> None:
        m, _ = self.run_method("fox", mode="as_implemented")
        src = m.describe()["hyperparameter_source"]
        self.assertIn("released code", src)
        self.assertIn("2-13", src)  # names what the paper says, for the reader

    def test_default_layer_window_is_the_code_s(self) -> None:
        m, _ = self.run_method("fox", mode="as_implemented")
        self.assertEqual(m.describe()["eager_layers"], list(range(2, 16)))

    def test_layer_window_can_be_overridden(self) -> None:
        m, _ = self.run_method(
            "fox", mode="as_implemented", intervention_layers="2-13"
        )
        self.assertEqual(m.describe()["eager_layers"], list(range(2, 14)))


class TestVcdLabelling(_Base):
    def test_fusion_hybrid_is_labelled_as_not_faithful(self) -> None:
        """No table may present the hybrid as VCD.

        ``vcd`` is now VCD's own contrastive rule; the hybrid that ran in
        rounds 1-5 keeps its own name and its own caveat.
        """
        m, out = self.run_method("vcd_fox_fusion")
        self.assertEqual(out.extra["variant"], "vcd-noise-fox-fusion")
        d = m.describe()
        self.assertEqual(d["variant"], "vcd-noise-fox-fusion")
        self.assertIn("upstream", d["caveat"])

    def test_noise_step_is_recorded(self) -> None:
        for arm in ("vcd", "vcd_fox_fusion"):
            with self.subTest(arm=arm):
                m, out = self.run_method(arm)
                self.assertEqual(out.extra["noise_step"], 500)
                self.assertEqual(
                    m.describe()["resolved_hyperparameters"]["noise_step"]
                    if arm == "vcd" else m.describe()["noise_step"],
                    500,
                )

    def test_real_vcd_is_not_labelled_as_the_hybrid(self) -> None:
        m, out = self.run_method("vcd")
        self.assertEqual(out.extra["variant"], "vcd")
        self.assertEqual(m.describe()["fusion"], "contrastive")
        self.assertNotIn("caveat", m.describe())


class TestDiagnostics(_Base):
    def test_trace_summary_is_attached(self) -> None:
        _, out = self.run_method("fox", mode="as_implemented")
        for key in ("d_t_mean", "d_t_max", "lambda_mean", "n_diverged_tokens"):
            self.assertIn(key, out.extra)

    def test_image_span_is_576_wide(self) -> None:
        _, out = self.run_method("sampling")
        lo, hi = out.extra["image_span"]
        self.assertEqual(hi - lo + 1, N_IMG)

    def test_precision_flags_are_pinned_and_recorded(self) -> None:
        m, _ = self.run_method("sampling")
        flags = m.describe()["pinned_precision_flags"]
        if flags is not None:  # None only when torch has no cuda build at all
            self.assertFalse(flags["matmul_tf32"])
            self.assertFalse(flags["cudnn_tf32"])
            self.assertFalse(flags["fp16_reduced"])


class TestLayerWindowParsing(unittest.TestCase):
    def test_range_form(self) -> None:
        self.assertEqual(parse_layer_window("2-5"), frozenset({2, 3, 4, 5}))

    def test_list_form(self) -> None:
        self.assertEqual(parse_layer_window("2,4,8"), frozenset({2, 4, 8}))

    def test_none_gives_the_released_window(self) -> None:
        lo, hi = FOX_DEFAULTS["layers"]
        self.assertEqual(parse_layer_window(None), frozenset(range(lo, hi + 1)))

    def test_garbage_raises_rather_than_silently_misintervening(self) -> None:
        for bad in ("abc", "5-2", "2--3", "2-"):
            with self.subTest(spec=bad):
                with self.assertRaises(ValueError):
                    parse_layer_window(bad)


if __name__ == "__main__":
    unittest.main()


class TestRegistryUnderDashM(unittest.TestCase):
    """The `python -m` double-import must not hide the real implementations.

    Under ``python -m foxrepro.bench.generate`` that module lands in
    sys.modules as ``"__main__"``, so ``methods.py``'s
    ``from foxrepro.bench.generate import register_method`` imports a SECOND
    copy of the module with its own registry, and the decorator populates that
    one. ``get_method`` therefore merges explicitly, and the merge has to be
    duck-typed: the ``Method`` base in ``__main__`` is a different class object
    from the one ``methods.py`` subclassed, so ``issubclass`` returns False for
    every real implementation.

    This bug is invisible via ordinary imports and via the console scripts, and
    appears only in the invocation path the Kaggle notebook cells actually use.
    """

    def test_dash_m_lists_the_real_methods(self) -> None:
        import subprocess
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.generate", "--list-methods"],
            env=env, capture_output=True, text=True, timeout=180,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        for name in ("sampling", "fox", "vcd", "echo", "sid", "opera"):
            self.assertIn(name, proc.stdout, f"{name} missing from --list-methods")
        # the real ones must show their own docstrings, not a stub's
        self.assertIn("One branch, no fusion", proc.stdout)


class TestSystemPrompt(_Base):
    """The system prompt is part of the method, not presentation.

    Fox scores heads by ``S = m_sys * H_vis``, where ``m_sys`` is the attention
    mass on the keys *before* the image. The paper calls those keys "the
    primary anchor for latent priors" and the origin of the shortcut it claims
    to sever. Omitting the system prompt leaves only ``<s>``, ``USER`` and
    ``:`` there -- an image span starting at index 5 instead of ~40 -- so
    m_sys loses its subject and the causal story loses its premise.

    The released code does send one, but not where a reader looks first:
    ``pope_eval.py`` never touches ``conv_templates``, and the string arrives
    instead from ``eval_configs/llava-1.5_eval.yaml`` via
    ``llava.py:344`` -> ``llava.py:228``. ``none`` is kept as the ablation.
    """

    RELEASED_SYSTEM = (
        "A chat between a curious user and an artificial intelligence "
        "assistant. The assistant gives helpful, detailed, and polite answers "
        "to the user's questions."
    )

    def test_default_reproduces_the_released_concatenation_verbatim(self) -> None:
        """``llava.py:228`` is a bare ``+``, so there is NO separator.

        ``system_message + "USER: <ImageHere> <question> ASSISTANT:"`` runs the
        period straight into ``USER``. That string, not a tidied-up version of
        it, is what produced the paper's numbers, so it is what the default
        has to render. The missing space is asserted explicitly because it is
        the kind of detail a later cleanup would silently "fix".
        """
        self.run_method("sampling")
        text = self.processor.seen_text[0]
        expected = (
            self.RELEASED_SYSTEM
            + "USER: <image> Is there a dog in the image? ASSISTANT:"
        )
        self.assertEqual(text, expected)
        self.assertIn("questions.USER:", text)
        self.assertNotIn("questions. USER:", text)

    def test_the_spaced_arm_differs_by_exactly_one_character(self) -> None:
        self.run_method("sampling", system_prompt_id="llava_v1_spaced")
        text = self.processor.seen_text[0]
        self.assertIn("questions. USER:", text)
        self.assertEqual(len(text), len(self.RELEASED_SYSTEM) + 1 + len(
            "USER: <image> Is there a dog in the image? ASSISTANT:"))

    def test_the_no_system_arm_leaves_the_released_turn_alone(self) -> None:
        """`pope_eval.py:64`: "USER: <ImageHere> <question> ASSISTANT:"."""
        self.run_method("sampling", system_prompt_id="none")
        text = self.processor.seen_text[0]
        self.assertTrue(text.startswith("USER: <image> "), text[:40])
        self.assertNotIn("curious user", text)
        self.assertNotIn("<image>\n", text)
        self.assertTrue(text.rstrip().endswith("ASSISTANT:"))

    def test_the_newline_wrapper_arm_changes_the_separator(self) -> None:
        self.run_method("sampling", wrapper_id="newline")
        self.assertIn("<image>\n", self.processor.seen_text[0])

    def test_both_are_recorded_in_provenance(self) -> None:
        for sid in ("none", "llava_v1", "llava_v1_spaced"):
            with self.subTest(sid=sid):
                m, _ = self.run_method("sampling", system_prompt_id=sid)
                d = m.describe()
                self.assertEqual(d["system_prompt_id"], sid)
                self.assertEqual(bool(d["system_prompt"]), sid != "none")

    def test_unknown_id_is_rejected(self) -> None:
        from foxrepro.bench.methods import render_full_prompt

        with self.assertRaises(ValueError):
            render_full_prompt("q", "helpful_assistant")


# --------------------------------------------------------------------------- #
# the compared baselines: is the contrastive rule actually the one deciding?
# --------------------------------------------------------------------------- #
class TwoBranchModel:
    """Returns one fixed logit vector for the clean branch and another for the
    negative branch, so the decided token identifies the decoding rule.

    The prefill logits are chosen in :class:`TestContrastiveDecoding` so that
    ``argmax`` differs between the observational logits, Fox's additive
    fusion and the contrastive rule. Any single-branch or wrongly-fused
    implementation therefore emits a different first token, rather than a
    plausible-looking one.

    Branch identification: a prefill call with two rows is the batched form
    (row 0 clean, row 1 negative, as ``fox_generate`` builds it); two prefill
    calls with one row each are the split form, in which the clean pass runs
    first. Decode steps (``q == 1``) always return EOS so the loop stops after
    the one token under test.
    """

    def __init__(self, z_obs: torch.Tensor, z_neg: torch.Tensor):
        self.config = FakeConfig()
        self.rows_seen: List[int] = []
        self.pixel_rows_seen: List[int] = []
        self.pixel_sums: List[float] = []
        self.prefills = 0
        self.z_obs = z_obs
        self.z_neg = z_neg
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def __call__(self, **kw: Any) -> FakeOutput:
        ids = kw["input_ids"]
        b, q = ids.shape
        self.rows_seen.append(b)
        px = kw.get("pixel_values")
        if px is not None:
            self.pixel_rows_seen.append(int(px.shape[0]))
            self.pixel_sums.extend(
                float(px[i].abs().sum()) for i in range(px.shape[0]))
        logits = torch.zeros(b, q, VOCAB)
        if q == 1:
            logits[:, -1, :] = -20.0
            logits[:, -1, EOS] = 20.0
            return FakeOutput(logits, cache=object())
        if b == 2:
            logits[0, -1] = self.z_obs
            logits[1, -1] = self.z_neg
        else:
            logits[0, -1] = self.z_obs if self.prefills == 0 else self.z_neg
        self.prefills += 1
        return FakeOutput(logits, cache=object())


class TestContrastiveDecoding(unittest.TestCase):
    """VCD and ICD must decide by ``(1+a) z_obs - a z_neg``, not by Fox's sum.

    With these two vectors the three candidate rules disagree:

    =========================  ===========================  ========
    rule                       value at t10 / t11           argmax
    =========================  ===========================  ========
    observational only         2.0 / 1.9                    10
    Fox additive (lambda > 0)  2.0 + l*5.0 / 1.9 + l*0.0    10
    contrastive (a = 1)        -1.0 / 3.8                   11
    =========================  ===========================  ========

    So a first token of ``t11`` is only reachable through the contrastive
    rule, and ``t10`` means the negative branch was ignored or added.
    """

    T_OBS, T_CD = 10, 11

    def setUp(self) -> None:
        import foxrepro.bench.methods as M
        from foxrepro.fox import attention as A

        z_obs = torch.full((VOCAB,), -5.0)
        z_obs[self.T_OBS] = 2.0
        z_obs[self.T_CD] = 1.9
        z_neg = torch.full((VOCAB,), -5.0)
        z_neg[self.T_OBS] = 5.0
        z_neg[self.T_CD] = 0.0

        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.sample = make_sample(Path(td.name))
        self.model = TwoBranchModel(z_obs, z_neg)
        self.processor = FakeProcessor()

        orig_load = A.load_llava
        A.load_llava = lambda *a, **k: (self.model, self.processor)  # type: ignore[assignment]
        self.addCleanup(lambda: setattr(A, "load_llava", orig_load))

        orig_encode = M._LlavaMethod._encode

        def _encode(inner_self, sample, text):  # noqa: ANN001
            return inner_self.processor(
                images=None, text=text, return_tensors="pt")

        M._LlavaMethod._encode = _encode  # type: ignore[assignment]
        self.addCleanup(
            lambda: setattr(M._LlavaMethod, "_encode", orig_encode))

    def run_method(self, name: str, **over: Any):
        cls = get_method(name)
        m = cls(make_cfg(name, decoding="greedy", **over))
        m.load()
        out = m.generate(self.sample, "Is there a dog in the image?")
        return m, out

    def first_token(self, out: Any) -> str:
        self.assertTrue(out.text, "the method emitted nothing")
        return out.text.split()[0]

    # -- the rule itself ---------------------------------------------------- #
    def test_vcd_decides_by_the_contrastive_rule(self) -> None:
        _, out = self.run_method("vcd")
        self.assertEqual(self.first_token(out), f"t{self.T_CD}")

    def test_icd_decides_by_the_contrastive_rule(self) -> None:
        _, out = self.run_method("icd")
        self.assertEqual(self.first_token(out), f"t{self.T_CD}")

    def test_sid_decides_by_the_contrastive_rule(self) -> None:
        """At SID's own cd_alpha = 0.5, 1.5*z_obs - 0.5*z_neg still flips."""
        _, out = self.run_method("sid")
        self.assertEqual(out.extra["cd_alpha"], 0.5)
        self.assertEqual(self.first_token(out), f"t{self.T_CD}")

    def test_fox_additive_fusion_decides_differently(self) -> None:
        """The contrast that makes the two tests above meaningful.

        Same model, same two branches, Fox's fusion: the observational token
        wins. If this ever returned ``t11`` too, the tests above would be
        passing for the wrong reason.
        """
        _, out = self.run_method("vcd_fox_fusion")
        self.assertEqual(self.first_token(out), f"t{self.T_OBS}")

    def test_sampling_sees_only_the_observational_branch(self) -> None:
        _, out = self.run_method("sampling")
        self.assertEqual(self.first_token(out), f"t{self.T_OBS}")

    # -- what each branch is shown ------------------------------------------ #
    def test_vcd_negative_branch_gets_a_corrupted_image(self) -> None:
        """Same prompt, different pixels -- and the corruption must be real.

        The stub image is all zeros, so the clean row sums to 0 and any
        non-zero sum is noise that actually arrived.
        """
        self.run_method("vcd")
        self.assertEqual(set(self.model.pixel_rows_seen), {2})
        clean, noised = self.model.pixel_sums[0], self.model.pixel_sums[1]
        self.assertEqual(clean, 0.0)
        self.assertGreater(noised, 0.0)
        self.assertEqual(len(set(self.processor.seen_text)), 1)

    def test_icd_negative_branch_gets_the_disturbed_instruction(self) -> None:
        m, _ = self.run_method("icd")
        self.assertEqual(len(self.processor.seen_text), 2)
        clean, disturbed = self.processor.seen_text
        self.assertNotIn("confused object detector", clean)
        self.assertEqual(disturbed, m.disturbed_prompt(
            "Is there a dog in the image?"))
        # verbatim from pope_eval.py:291-295: after the marker, one space
        # either side, and the question still follows it
        self.assertIn("<image> You are a confused object detector.", disturbed)
        self.assertLess(
            disturbed.index("confused object detector"),
            disturbed.index("Is there a dog"),
        )
        # and the two branches differ only by that insertion
        self.assertEqual(
            disturbed.replace(" You are a confused object detector.", "", 1),
            clean,
        )

    def test_icd_runs_two_prefills_one_row_each(self) -> None:
        self.run_method("icd")
        self.assertEqual(self.model.prefills, 2)
        self.assertEqual(set(self.model.rows_seen), {1})

    def test_vcd_runs_one_prefill_of_two_rows(self) -> None:
        self.run_method("vcd")
        self.assertEqual(self.model.prefills, 1)
        self.assertEqual(set(self.model.rows_seen), {2})

    def test_sid_shows_both_rows_the_same_image_and_prompt(self) -> None:
        """CT2S is the whole perturbation; anything else would double up.

        ``tests/test_attention_wiring.py`` proves the masking itself. Here
        the point is the negative side of SID: no second encode, no noise,
        one prompt, one image, two identical rows.
        """
        self.run_method("sid")
        self.assertEqual(self.model.prefills, 1)
        self.assertEqual(set(self.model.rows_seen), {2})
        self.assertEqual(set(self.model.pixel_rows_seen), {2})
        self.assertEqual(len(set(self.processor.seen_text)), 1)
        self.assertEqual(self.model.pixel_sums, [0.0, 0.0])

    def test_sid_records_its_ct2s_configuration(self) -> None:
        m, out = self.run_method("sid")
        self.assertEqual(out.extra["sid_agg_layer"], 2)
        self.assertEqual(out.extra["sid_rank"], 100)
        self.assertEqual(
            m.describe()["resolved_hyperparameters"]["sid_rank"], 100)

    def test_stacking_two_negative_branches_is_refused(self) -> None:
        """One method is one perturbation. Two is nobody's method.

        SID's own harness does stack two: ``--use-vcd`` leaves
        ``--fast-v-attention-rank`` at 100, so that row is VCD+CT2S.
        Reproducing it would file a number under a paper that did not propose
        it, so every pair is refused rather than offered.
        """
        from foxrepro.fox.attention import fox_generate

        px = torch.zeros(1, 3, 4, 4)
        ids = torch.zeros(1, 4, dtype=torch.long)
        pairs = (
            ("sid+vcd", {"sid": True, "pixel_values_do": px}),
            ("sid+icd", {"sid": True, "input_ids_do": ids}),
            ("sid+causalmm", {"sid": True, "cf_vision": True}),
            ("causalmm+vcd", {"cf_vision": True, "pixel_values_do": px}),
            ("causalmm+icd", {"cf_vision": True, "input_ids_do": ids}),
            ("vcd+icd", {"pixel_values_do": px, "input_ids_do": ids}),
        )
        for label, extra in pairs:
            with self.subTest(pair=label):
                with self.assertRaises(ValueError) as cm:
                    fox_generate(
                        self.model,
                        self.processor(images=None, text="x"),
                        max_new_tokens=1, fusion="contrastive",
                        mode="off", **extra,
                    )
                self.assertIn("perturbation", str(cm.exception))

    # -- the hyperparameters that multiply the gap -------------------------- #
    def test_cd_alpha_is_recorded_and_overridable(self) -> None:
        m, out = self.run_method("vcd")
        self.assertEqual(out.extra["cd_alpha"], 1.0)
        self.assertAlmostEqual(out.extra["cd_beta"], 0.1)
        m, out = self.run_method("vcd", alpha=0.5)
        self.assertEqual(out.extra["cd_alpha"], 0.5)
        self.assertEqual(
            m.describe()["resolved_hyperparameters"]["cd_alpha"], 0.5)

    def test_alpha_zero_collapses_to_the_baseline(self) -> None:
        """a = 0 makes ``diffs == z_obs``; the negative branch stops mattering.

        This is the sanity check that the coefficient is wired to the right
        side of the subtraction -- with the sign flipped, a = 0 would still
        change the answer.
        """
        _, out = self.run_method("vcd", alpha=0.0)
        self.assertEqual(self.first_token(out), f"t{self.T_OBS}")

    def test_the_intervention_stays_off(self) -> None:
        """VCD and ICD are not Fox-with-a-different-negative-branch.

        If the head edit were left on, these rows would measure Fox plus a
        perturbation and the comparison table would be meaningless.
        """
        for arm in ("vcd", "icd", "sid"):
            with self.subTest(arm=arm):
                m, _ = self.run_method(arm)
                d = m.describe()
                self.assertEqual(d["fusion"], "contrastive")
                self.assertEqual(d["fox_intervention"], "off")
                self.assertEqual(d["intervention"], "none")
                # Must NOT be called "mode": the record's own `mode` field is
                # cfg.mode, and two keys with one name and two values is how a
                # provenance field stops being evidence.
                self.assertNotIn("mode", d)


class TestUntrackedOverridesAreRefused(_Base):
    """A hyperparameter the run filename cannot see must not be sweepable.

    ``storage.VARIANT_FIELDS`` builds the run file name. ``noise_step``,
    ``sid_rank`` and ``sid_agg_layer`` are not in it, so two runs differing
    only in one of them would land in the same file and resume from each
    other -- half one sweep point, half another, no error, plausible number.
    The published value works; anything else raises.
    """

    CASES = (
        ("vcd", "noise_step", 500, 300),
        ("vcd_fox_fusion", "noise_step", 500, 300),
        ("sid", "sid_rank", 100, 200),
        ("sid", "sid_agg_layer", 2, 5),
    )

    def test_the_published_value_is_accepted_explicitly(self) -> None:
        for name, key, default, _ in self.CASES:
            with self.subTest(method=name, key=key):
                m, _ = self.run_method(name, extra={key: default})
                self.assertIsNotNone(m)

    def test_any_other_value_raises_and_names_variant_fields(self) -> None:
        for name, key, _, other in self.CASES:
            with self.subTest(method=name, key=key):
                with self.assertRaises(ValueError) as cm:
                    self.run_method(name, extra={key: other})
                msg = str(cm.exception)
                self.assertIn("VARIANT_FIELDS", msg)
                self.assertIn(key, msg)


class TestPerSampleSeedIsStable(unittest.TestCase):
    """The same sample must get the same seed in every process.

    ``hash()`` on a str is salted per interpreter unless PYTHONHASHSEED is
    set, and the per-sample seed used it. Three consequences, all silent:
    an identical command did not reproduce its own tokens; a job resumed
    after the session wall continued on a different stream; and two arms at
    "the same seed" never saw the same draw.

    Found by the alpha=0 validation arm, which scored 76.4-78.2 against the
    baseline's 77.0 -- the same distribution -- while 14-19% of individual
    answers differed.
    """

    IDS = ("pope/adversarial/q0000001", "pope/adversarial/q0000002",
           "chair/val2014_500/391895")

    def test_it_does_not_depend_on_the_interpreter(self) -> None:
        import subprocess
        import sys

        from foxrepro.bench.methods import sample_seed

        want = [sample_seed(42, s) for s in self.IDS]
        prog = (
            "import sys; sys.path.insert(0, 'src');"
            "from foxrepro.bench.methods import sample_seed;"
            f"print([sample_seed(42, s) for s in {self.IDS!r}])"
        )
        # A fresh interpreter picks a fresh hash salt, which is exactly the
        # condition that broke this.
        for _ in range(3):
            out = subprocess.run(
                [sys.executable, "-c", prog], capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parents[1]),
            )
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(eval(out.stdout.strip()), want)

    def test_different_samples_get_different_seeds(self) -> None:
        from foxrepro.bench.methods import sample_seed

        seeds = [sample_seed(42, s) for s in self.IDS]
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_different_run_seeds_separate_the_streams(self) -> None:
        from foxrepro.bench.methods import sample_seed

        a = [sample_seed(42, s) for s in self.IDS]
        b = [sample_seed(43, s) for s in self.IDS]
        self.assertEqual(len(set(a) & set(b)), 0)

    def test_no_seed_means_no_seed(self) -> None:
        from foxrepro.bench.methods import sample_seed

        self.assertIsNone(sample_seed(None, "anything"))

    def test_nothing_still_calls_the_salted_hash(self) -> None:
        """The three call sites are easy to add a fourth to."""
        src = (Path(__file__).resolve().parents[1]
               / "src" / "foxrepro" / "bench" / "methods.py").read_text()
        self.assertNotIn("hash(sample.sample_id)", src)
