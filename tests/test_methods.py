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

        self._orig_prepare = M._LlavaMethod._prepare

        def _prepare(inner_self, sample, prompt):  # noqa: ANN001
            # Only the PIL image read is stubbed out; the prompt still goes
            # through the real render_full_prompt, so the system-prompt arms
            # are actually exercised rather than bypassed.
            return inner_self.processor(
                images=None,
                text=render_full_prompt(
                    prompt,
                    inner_self._system_prompt_id(),
                    inner_self._wrapper_id(),
                ),
                return_tensors="pt",
            )

        M._LlavaMethod._prepare = _prepare  # type: ignore[assignment]
        self.addCleanup(
            lambda: setattr(M._LlavaMethod, "_prepare", self._orig_prepare)
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
    def test_vcd_is_labelled_as_not_faithful(self) -> None:
        """No table may silently present this as VCD."""
        m, out = self.run_method("vcd")
        self.assertEqual(out.extra["variant"], "vcd-noise-fox-fusion")
        d = m.describe()
        self.assertEqual(d["variant"], "vcd-noise-fox-fusion")
        self.assertIn("upstream", d["caveat"])

    def test_noise_step_is_recorded(self) -> None:
        m, out = self.run_method("vcd")
        self.assertEqual(out.extra["noise_step"], 500)
        self.assertEqual(m.describe()["noise_step"], 500)


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
