"""The precision axes and the Kaggle host layout.

Two properties are load-bearing for this project and are easy to break by
accident, so they are pinned here.

1. **Every arm of the central experiment must land in its own run file.**
   The resume key is fixed at ``(method, benchmark, split, seed, dtype,
   sample_id)``, which deliberately excludes ``gamma``, ``logit_dtype``,
   ``mode`` and ``decoding`` -- all of which this project sweeps. If two arms
   shared a filename, the second run would read the first's records, consider
   those sample_ids done, and silently emit a file mixing two configurations
   whose metrics would then be meaningless. The variant tag in the filename is
   what prevents that, so it is tested rather than trusted.

2. **The Kaggle layout must keep the 14 GB checkpoint out of /kaggle/working.**
   That directory persists as the notebook's output but is capped at 20 GB.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


from foxrepro import storage  # noqa: E402
from foxrepro.bench.generate import GenConfig, host_fingerprint  # noqa: E402
from foxrepro.storage import (  # noqa: E402
    INTERVENTION_MODES, LOGIT_DTYPES, PROVENANCE_FIELDS, SOFTMAX_DTYPES,
    VARIANT_FIELDS, detect_environment, paths,
)

BASE = dict(
    method="fox",
    benchmark="chair",
    split="val2014_500",
    seed=42,
    dtype="fp16",
    model_id="llava-hf/llava-1.5-7b-hf",
    prompt_template_id="chair_v1",
    max_new_tokens=512,
    gamma=1e4,
    k=0.35,
    tau_js=0.15,
    alpha=2.0,
    beta=0.1,
    intervention_layers="2-15",
)


class TestArmsDoNotCollide(unittest.TestCase):
    """Each experiment arm gets a distinct run file."""

    def _fname(self, **over: object) -> str:
        return GenConfig(**{**BASE, **over}).run_key().filename()  # type: ignore[arg-type]

    def test_gamma_sweep_arms_are_distinct(self) -> None:
        # The magnitude sweep: fp16 ULP at |gamma| is 2^(floor(log2 g) - 10),
        # giving 0.0625 / 0.5 / 8 / 32 for these four. gamma=1e5 overflows
        # fp16 (max 65504) and is deliberately not in the grid.
        names = {self._fname(gamma=g) for g in (1e2, 1e3, 1e4, 6e4)}
        self.assertEqual(len(names), 4)

    def test_every_logit_dtype_is_distinct(self) -> None:
        names = {self._fname(logit_dtype=d) for d in LOGIT_DTYPES}
        self.assertEqual(len(names), len(LOGIT_DTYPES))

    def test_every_softmax_dtype_is_distinct(self) -> None:
        names = {self._fname(softmax_dtype=d) for d in SOFTMAX_DTYPES}
        self.assertEqual(len(names), len(SOFTMAX_DTYPES))

    def test_every_mode_is_distinct(self) -> None:
        names = {self._fname(mode=m) for m in INTERVENTION_MODES}
        self.assertEqual(len(names), len(INTERVENTION_MODES))

    def test_greedy_control_is_distinct_from_sampling(self) -> None:
        # Table 9 is the greedy control of Table 1 at otherwise identical
        # settings; these must not share a file.
        self.assertNotEqual(self._fname(decoding="greedy"), self._fname())

    def test_full_grid_has_no_collisions(self) -> None:
        names = [
            self._fname(gamma=g, logit_dtype=d, mode=m, decoding=dec)
            for g in (1e3, 1e4)
            for d in LOGIT_DTYPES
            for m in INTERVENTION_MODES
            for dec in ("sampling", "greedy")
        ]
        self.assertEqual(len(set(names)), len(names), "variant tag collision")

    def test_system_prompt_arms_are_distinct(self) -> None:
        """m_sys measures attention over the pre-image keys, so the system
        prompt is part of the method. The two arms must not share a file."""
        a = self._fname(system_prompt_id="llava_v1")
        b = self._fname(system_prompt_id="none")
        self.assertNotEqual(a, b)

    def test_axes_are_declared_as_variant_fields(self) -> None:
        for fld in ("logit_dtype", "softmax_dtype", "mode", "gamma",
                    "decoding", "system_prompt_id"):
            self.assertIn(fld, VARIANT_FIELDS, f"{fld} missing from VARIANT_FIELDS")


class TestValidation(unittest.TestCase):
    def test_bad_logit_dtype_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenConfig(**{**BASE, "logit_dtype": "int8"})  # type: ignore[arg-type]

    def test_bad_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenConfig(**{**BASE, "mode": "magic"})  # type: ignore[arg-type]

    def test_bad_softmax_dtype_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GenConfig(**{**BASE, "softmax_dtype": "fp16"})  # type: ignore[arg-type]

    def test_quantization_is_not_an_option(self) -> None:
        # 4-bit / 8-bit would change the very quantity under study, so no
        # dtype spelling may suggest it is available.
        for d in storage.DTYPES + LOGIT_DTYPES:
            self.assertNotIn("int", d)
            self.assertNotIn("4bit", d)
            self.assertNotIn("8bit", d)


class TestProvenanceCompleteness(unittest.TestCase):
    def test_provenance_covers_every_new_axis(self) -> None:
        cfg = GenConfig(**BASE)  # type: ignore[arg-type]
        prov = cfg.provenance()
        for fld in ("logit_dtype", "softmax_dtype", "mode", "device_map",
                    "n_gpus", "gpu_name", "tf32_flags", "system_prompt_id"):
            self.assertIn(fld, prov, f"{fld} missing from provenance()")
            self.assertIn(fld, PROVENANCE_FIELDS)

    def test_host_fingerprint_never_raises_without_torch(self) -> None:
        fp = host_fingerprint()
        self.assertIn("n_gpus", fp)
        self.assertIn("tf32_flags", fp)
        self.assertIsInstance(fp["n_gpus"], int)


class TestKaggleLayout(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            k: os.environ.get(k)
            for k in ("FOX_ENV", "FOX_PERSIST_ROOT", "FOX_LOCAL_ROOT",
                      "FOX_RUNS_ROOT", "FOX_HF_HOME")
        }
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        paths(refresh=True)

    def test_kaggle_is_detected_from_env(self) -> None:
        os.environ["FOX_ENV"] = "kaggle"
        self.assertEqual(detect_environment(), "kaggle")

    def test_checkpoint_cache_stays_out_of_kaggle_working(self) -> None:
        """/kaggle/working is capped at 20 GB and becomes the notebook output.

        A 14 GB HuggingFace cache must not land there.
        """
        os.environ["FOX_ENV"] = "kaggle"
        p = paths(refresh=True)
        self.assertEqual(p.env, "kaggle")
        self.assertNotIn("/kaggle/working", str(p.hf_home))
        self.assertIn("/kaggle/temp", str(p.hf_home))
        # run JSONL is irreplaceable, so it DOES belong on the persistent side
        self.assertIn("/kaggle/working", str(p.runs_root))

    def test_unknown_fox_env_is_rejected(self) -> None:
        os.environ["FOX_ENV"] = "sagemaker"
        with self.assertRaises(ValueError):
            detect_environment()

    def test_kaggle_input_dirs_is_empty_off_kaggle(self) -> None:
        if Path("/kaggle/input").is_dir():  # pragma: no cover - only on Kaggle
            self.skipTest("running on Kaggle")
        self.assertEqual(storage.kaggle_input_dirs(), [])


if __name__ == "__main__":
    unittest.main()
