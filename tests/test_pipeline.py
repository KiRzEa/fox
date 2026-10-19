"""End-to-end tests: generate (CPU echo) -> evaluate -> report.

Also pins the POPE metric arithmetic (including the yes-ratio and every
denominator) against a hand-computed confusion matrix, and checks that the
unparsable-answer policies behave as documented.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List


from foxrepro.bench.evaluate import evaluate_pope, parse_yes_no, read_run  # noqa: E402
from foxrepro.bench.report import load_results, render_report  # noqa: E402
from foxrepro.storage import PROVENANCE_FIELDS  # noqa: E402
from tests._fixtures import make_workspace  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

def _with_src_on_path(env: dict) -> dict:
    """Ensure ``src`` is importable in a subprocess.

    The CLIs are invoked as ``python -m foxrepro.bench.*``. That resolves via
    the installed package when there is one, and via PYTHONPATH from a bare
    checkout; setting it here makes the suite pass either way.
    """
    src = str(REPO / "src")
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src + (os.pathsep + prev if prev else "")
    return env



def _record(sample_id: str, label: str, output: str, **over: Any) -> Dict[str, Any]:
    """A complete, schema-valid JSONL record."""
    rec: Dict[str, Any] = {
        "method": "echo", "benchmark": "pope", "split": "random", "seed": 1,
        "dtype": "fp16", "gamma": None, "k": None, "tau_js": None,
        "alpha": None, "beta": None, "max_new_tokens": 1,
        "prompt_template_id": "pope_v1", "model_id": "test-model",
        "commit_hash": "abc1234", "transformers_path": "/vendored/transformers/__init__.py",
        "sample_id": sample_id, "image_id": 1000, "prompt": "Is there a dog?",
        "output": output, "latency_ms": 12.5, "n_prompt_tokens": 600,
        "n_gen_tokens": 1, "timestamp": "2026-01-01T00:00:00Z",
        "label": label, "decoding": "sampling", "top_p": 0.9, "temperature": 1.0,
        "intervention_layers": None,
        # precision axes of the central experiment
        "logit_dtype": "native", "softmax_dtype": "fp32", "mode": "off",
        "system_prompt_id": "none", "wrapper_id": "released",
        # host / sharding fingerprint
        "device_map": "auto", "n_gpus": 0, "gpu_name": None, "tf32_flags": None,
        "torch_version": None, "transformers_version": None,
        "compute_capability": None,
    }
    rec.update(over)
    assert not [f for f in PROVENANCE_FIELDS if f not in rec]
    return rec


class TestParseYesNo(unittest.TestCase):
    def test_single_token_answers(self) -> None:
        self.assertEqual(parse_yes_no("Yes"), "yes")
        self.assertEqual(parse_yes_no("No"), "no")
        self.assertEqual(parse_yes_no("yes."), "yes")
        self.assertEqual(parse_yes_no(" NO "), "no")

    def test_sentence_answers(self) -> None:
        self.assertEqual(parse_yes_no("Yes, there is a dog."), "yes")
        self.assertEqual(parse_yes_no("No, I don't see one."), "no")
        self.assertEqual(parse_yes_no("There is no dog in the image."), "no")

    def test_unparsable(self) -> None:
        for text in ("", "   ", "maybe", "42", "I cannot tell"):
            self.assertIsNone(parse_yes_no(text), text)


class TestPopeArithmetic(unittest.TestCase):
    """Hand-computed confusion matrix.

    5 "yes" items answered yes,yes,yes,no,no  -> tp=3, fn=2
    5 "no"  items answered yes,yes,no,no,no   -> fp=2, tn=3

    accuracy  = 6/10 = 0.60   (denominator 10)
    precision = 3/5  = 0.60   (denominator = predicted-yes = 5)
    recall    = 3/5  = 0.60   (denominator = ground-truth-yes = 5)
    F1        = 0.60
    yes-ratio = 5/10 = 0.50   (denominator 10)
    """

    def setUp(self) -> None:
        yes_outputs = ["Yes", "Yes", "Yes", "No", "No"]
        no_outputs = ["Yes", "Yes", "No", "No", "No"]
        self.records = (
            [_record(f"q{i}", "yes", o) for i, o in enumerate(yes_outputs)]
            + [_record(f"q{5 + i}", "no", o) for i, o in enumerate(no_outputs)]
        )

    def test_metrics(self) -> None:
        met, outcomes = evaluate_pope(self.records, n_boot=200)
        self.assertEqual(met["confusion"], {"tp": 3, "fp": 2, "fn": 2, "tn": 3})
        self.assertAlmostEqual(met["accuracy"]["point"], 0.60, places=12)
        self.assertAlmostEqual(met["precision"]["point"], 0.60, places=12)
        self.assertAlmostEqual(met["recall"]["point"], 0.60, places=12)
        self.assertAlmostEqual(met["f1"]["point"], 0.60, places=12)
        self.assertAlmostEqual(met["yes_ratio"]["point"], 0.50, places=12)
        self.assertEqual(len(outcomes), 10)
        self.assertEqual(sum(o.correct for o in outcomes), 6)

    def test_denominators_are_explicit(self) -> None:
        met, _ = evaluate_pope(self.records, n_boot=200)
        self.assertEqual(met["denominators"], {
            "accuracy": 10, "precision": 5, "recall": 5, "f1": 10,
            "yes_ratio": 10, "n_ground_truth_yes": 5, "n_ground_truth_no": 5,
        })
        self.assertEqual(met["accuracy"]["n"], 10)
        self.assertEqual(met["precision"]["n"], 5)

    def test_intervals_bracket_the_point_estimate(self) -> None:
        met, _ = evaluate_pope(self.records, n_boot=200)
        for name in ("accuracy", "precision", "recall", "yes_ratio"):
            m = met[name]
            self.assertLessEqual(m["ci_lo"], m["point"])
            self.assertGreaterEqual(m["ci_hi"], m["point"])

    def test_yes_ratio_detects_an_all_no_method(self) -> None:
        """A method that always says "no" scores 50% accuracy with yes-ratio 0."""
        records = [_record(f"q{i}", "yes" if i < 5 else "no", "No") for i in range(10)]
        met, _ = evaluate_pope(records, n_boot=200)
        self.assertAlmostEqual(met["accuracy"]["point"], 0.5, places=12)
        self.assertAlmostEqual(met["yes_ratio"]["point"], 0.0, places=12)
        self.assertEqual(met["f1"]["point"], 0.0)

    def test_unparsed_policies(self) -> None:
        """Only the strict parser has an unparsable class to have a policy for."""
        records = self.records + [_record("q10", "yes", "I cannot tell")]
        kw = dict(pope_parser="strict", n_boot=200)
        negative, _ = evaluate_pope(records, unparsed_policy="negative", **kw)
        self.assertEqual(negative["n_unparsed"], 1)
        self.assertEqual(negative["n_samples"], 11)
        self.assertEqual(negative["confusion"]["fn"], 3)  # counted as "no"

        excluded, _ = evaluate_pope(records, unparsed_policy="exclude", **kw)
        self.assertEqual(excluded["n_unparsed"], 1)
        self.assertEqual(excluded["n_samples"], 10)
        self.assertAlmostEqual(excluded["accuracy"]["point"], 0.60, places=12)

    def test_the_released_parser_scores_a_non_answer_as_yes(self) -> None:
        """``pope_eval.py`` has no unparsable class, and that moves accuracy.

        The released rule only looks for a negation token, so "I cannot tell"
        on a "yes" sample is a true positive there and a false negative under
        the strict parser. With the released prompt (no "single word or
        phrase" suffix) and ``max_new_tokens = 1``, answers like this are the
        norm, not the exception, so the two rules do not give the same table.
        """
        records = self.records + [_record("q10", "yes", "I cannot tell")]
        rel, _ = evaluate_pope(records, pope_parser="released", n_boot=200)
        self.assertEqual(rel["n_unparsed"], 0)
        self.assertEqual(rel["n_samples"], 11)
        self.assertEqual(rel["confusion"]["tp"], 4)      # q10 counted yes

        strict, _ = evaluate_pope(records, pope_parser="strict", n_boot=200)
        self.assertEqual(strict["confusion"]["tp"], 3)   # q10 counted no
        self.assertEqual(strict["confusion"]["fn"], 3)
        self.assertGreater(rel["accuracy"]["point"], strict["accuracy"]["point"])

    def test_the_parser_choice_is_recorded(self) -> None:
        for name in ("released", "strict"):
            with self.subTest(name=name):
                met, _ = evaluate_pope(self.records, pope_parser=name, n_boot=200)
                self.assertEqual(met["pope_parser"], name)

    def test_an_unknown_parser_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_pope(self.records, pope_parser="lenient", n_boot=200)

    def test_the_default_parser_is_the_released_one(self) -> None:
        """The default has to be the rule that produced the paper's numbers."""
        met, _ = evaluate_pope(self.records, n_boot=200)
        self.assertEqual(met["pope_parser"], "released")

    def test_label_disagreement_raises(self) -> None:
        labels = {r["sample_id"]: "no" for r in self.records}
        with self.assertRaises(ValueError):
            evaluate_pope(self.records, labels=labels, n_boot=200)


class TestReadRun(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "run.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, records: List[Dict[str, Any]], tail: str = "") -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
            fh.write(tail)

    def test_shared_provenance_is_extracted(self) -> None:
        self._write([_record("a", "yes", "Yes"), _record("b", "no", "No")])
        records, meta = read_run(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(meta["method"], "echo")
        self.assertEqual(meta["dtype"], "fp16")
        self.assertEqual(meta["n_duplicate_sample_ids"], 0)
        self.assertAlmostEqual(meta["latency_ms_mean"], 12.5, places=9)

    def test_duplicate_sample_id_keeps_first(self) -> None:
        self._write([
            _record("a", "yes", "Yes"),
            _record("a", "yes", "No"),
            _record("b", "no", "No"),
        ])
        records, meta = read_run(self.path)
        self.assertEqual(len(records), 2)
        self.assertEqual(meta["n_duplicate_sample_ids"], 1)
        self.assertEqual(records[0]["output"], "Yes")

    def test_mixed_provenance_raises_in_strict_mode(self) -> None:
        self._write([_record("a", "yes", "Yes"), _record("b", "no", "No", dtype="bf16")])
        with self.assertRaises(ValueError):
            read_run(self.path, strict=True)

    def test_trailing_partial_line_is_ignored(self) -> None:
        self._write([_record("a", "yes", "Yes")], tail='{"sample_id": "b')
        records, meta = read_run(self.path)
        self.assertEqual(len(records), 1)
        self.assertGreater(meta["trailing_partial_bytes"], 0)


class TestEndToEnd(unittest.TestCase):
    """generate --dry-run -> evaluate -> report, as real subprocesses."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.ws = make_workspace(self.root, n_images=12, n_per_image=6)
        self.reports = self.root / "reports"
        self.env = _with_src_on_path(dict(os.environ))
        self.env.update({
            "FOX_ENV": "local",
            "FOX_PERSIST_ROOT": str(self.root / "persist"),
            "FOX_LOCAL_ROOT": str(self.root / "local"),
        })
        self.env.pop("FOX_TEST_CRASH_AFTER", None)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _generate(self, benchmark: str, split: str) -> None:
        cmd = [
            sys.executable, "-m", "foxrepro.bench.generate", "--dry-run",
            "--benchmark", benchmark, "--split", split, "--seed", "42",
            "--dtype", "fp16",
            "--pope-dir", str(self.ws["pope_dir"]),
            "--images-dir", str(self.ws["images_dir"]),
            "--chair-image-list", str(self.ws["chair_list"]),
            "--runs-dir", str(self.ws["runs_dir"]), "--progress-every", "0",
        ]
        proc = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])

    def test_pope_and_chair_pipeline(self) -> None:
        self._generate("pope", "random")
        self._generate("pope", "adversarial")
        self._generate("chair", "val2014_500")

        proc = subprocess.run(
            [
                sys.executable, "-m", "foxrepro.bench.evaluate",
                "--runs-dir", str(self.ws["runs_dir"]),
                "--out-dir", str(self.reports),
                "--pope-dir", str(self.ws["pope_dir"]),
                "--instances", str(self.ws["ann_dir"] / "instances_val2014_subset.json"),
                "--captions", str(self.ws["ann_dir"] / "captions_val2014_subset.json"),
                "--n-boot", "200", "--format", "both",
            ],
            env=self.env, capture_output=True, text=True, timeout=600,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        metrics_files = sorted(self.reports.glob("*.metrics.json"))
        self.assertEqual(len(metrics_files), 3)
        outcome_files = sorted(self.reports.glob("*.outcomes.jsonl"))
        self.assertEqual(len(outcome_files), 3)
        self.assertIn("CHAIR_I", proc.stdout)
        self.assertIn("yes_ratio", proc.stdout)

        # the CHAIR_I denominator must be a real count, not a placeholder
        chair_doc = next(
            json.loads(f.read_text()) for f in metrics_files
            if json.loads(f.read_text())["provenance"]["benchmark"] == "chair"
        )
        denom = chair_doc["metrics"]["denominators"]["chair_i"]
        self.assertGreater(denom, 0)
        self.assertEqual(denom, chair_doc["metrics"]["totals"]["chair_i_denominator"])

        results = load_results(self.reports)
        self.assertEqual(len(results), 3)
        report = render_report(results, baseline="echo")
        for needle in ("Table 1", "Table 2", "Resolution budget",
                       "Provenance audit", "CHAIR_I n (mentions)"):
            self.assertIn(needle, report)

    def test_report_cli_writes_markdown(self) -> None:
        self._generate("pope", "random")
        subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.evaluate",
             "--runs-dir", str(self.ws["runs_dir"]), "--out-dir", str(self.reports),
             "--n-boot", "200", "--format", "json"],
            env=self.env, capture_output=True, text=True, timeout=300, check=True,
        )
        out = self.root / "report.md"
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.report",
             "--reports-dir", str(self.reports), "--out", str(out),
             "--baseline", "echo"],
            env=self.env, capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        text = out.read_text(encoding="utf-8")
        self.assertIn("# Fox reproduction", text)
        self.assertIn("95% CI", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestReportPromptArms(unittest.TestCase):
    """Two archived rounds of one job must not silently become one table.

    Session 1 ran twice: once with the prompt this project invented, once
    byte-faithful to the release. Both are kept, and they differ by four
    points of POPE-adversarial accuracy. A report that picked whichever
    loaded first would be worse than one that refused to build.
    """

    FAITHFUL = {"system_prompt_id": "llava_v1", "wrapper_id": "released",
                "prompt_template_id": "pope_v0_raw"}
    OLD = {"system_prompt_id": "llava_v1", "wrapper_id": "newline",
           "prompt_template_id": "pope_v1"}

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.reports = Path(self._tmp.name) / "reports"
        self.reports.mkdir()

    def _write(self, name: str, method: str, arm: Dict[str, str],
               accuracy: float) -> None:
        n = 3000
        k = int(round(accuracy * n))
        prov = {
            "method": method, "benchmark": "pope", "split": "adversarial",
            "seed": 42, "dtype": "fp16", "decoding": "sampling",
            "pope_parser": "released", **arm,
        }
        met = {
            "benchmark": "pope", "n_samples": n, "n_unparsed": 0,
            "accuracy": {"name": "pope_accuracy", "point": accuracy,
                         "ci_lo": accuracy - 0.014, "ci_hi": accuracy + 0.014,
                         "n": n, "successes": k, "confidence": 0.95,
                         "interval_method": "wilson"},
            "f1": {"name": "pope_f1", "point": accuracy, "ci_lo": accuracy - 0.02,
                   "ci_hi": accuracy + 0.02, "n": n, "confidence": 0.95,
                   "interval_method": "bootstrap"},
            "yes_ratio": {"name": "pope_yes_ratio", "point": 0.5,
                          "ci_lo": 0.48, "ci_hi": 0.52, "n": n,
                          "confidence": 0.95, "interval_method": "wilson"},
            "confusion": {"tp": k // 2, "fp": (n - k) // 2,
                          "fn": (n - k) - (n - k) // 2, "tn": k - k // 2},
            "denominators": {"accuracy": n, "precision": n // 2, "recall": n // 2,
                             "f1": n, "yes_ratio": n,
                             "n_ground_truth_yes": n // 2,
                             "n_ground_truth_no": n // 2},
        }
        (self.reports / name).write_text(
            json.dumps({"provenance": prov, "metrics": met}), encoding="utf-8")

    def _both_rounds(self) -> None:
        self._write("r2_sampling.metrics.json", "sampling", self.FAITHFUL, 0.7717)
        self._write("r2_fox.metrics.json", "fox", self.FAITHFUL, 0.8140)
        self._write("r1_sampling.metrics.json", "sampling", self.OLD, 0.8190)
        self._write("r1_fox.metrics.json", "fox", self.OLD, 0.8333)

    def test_the_faithful_filter_keeps_exactly_one_round(self) -> None:
        from foxrepro.bench.report import load_results

        self._both_rounds()
        self.assertEqual(len(load_results(self.reports)), 4)
        kept = load_results(self.reports, system_prompt_id="llava_v1",
                            wrapper_id="released",
                            prompt_template_ids=["pope_v0_raw", "chair_v1"])
        self.assertEqual(len(kept), 2)
        self.assertEqual({r.method for r in kept}, {"sampling", "fox"})
        self.assertEqual(
            {round(r.met["accuracy"]["point"], 4) for r in kept},
            {0.7717, 0.8140},
        )

    def test_an_unfiltered_archive_refuses_to_build_a_table(self) -> None:
        from foxrepro.bench.report import load_results, render_report

        self._both_rounds()
        with self.assertRaises(ValueError) as ctx:
            render_report(load_results(self.reports), baseline="sampling")
        msg = str(ctx.exception)
        # It must name the fields that actually differ, not send the reader
        # after --dtype, which cannot separate these two.
        self.assertIn("wrapper_id", msg)
        self.assertIn("prompt_template_id", msg)
        self.assertIn("--wrapper", msg)

    def test_the_cli_defaults_to_the_faithful_arm(self) -> None:
        self._both_rounds()
        out = Path(self._tmp.name) / "report.md"
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.report",
             "--reports-dir", str(self.reports), "--out", str(out)],
            env=_with_src_on_path(dict(os.environ)),
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        text = out.read_text(encoding="utf-8")
        self.assertIn("Prompt arm:", text)
        self.assertIn("pope_v0_raw", text)
        self.assertIn("77.17", text)      # the faithful baseline
        self.assertNotIn("81.90", text)   # the old one must not appear

    def test_filtering_everything_out_says_which_arms_exist(self) -> None:
        """Not "run evaluate.py first": the files are there."""
        self._write("r1_sampling.metrics.json", "sampling", self.OLD, 0.8190)
        proc = subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.report",
             "--reports-dir", str(self.reports)],
            env=_with_src_on_path(dict(os.environ)),
            capture_output=True, text=True, timeout=300,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("run evaluate.py first", proc.stderr)
        self.assertIn("Arms present", proc.stderr)
        self.assertIn("newline", proc.stderr)
        self.assertIn("--prompt-arm any", proc.stderr)


class TestResolvedHyperparametersAreRecorded(unittest.TestCase):
    """A fox record must say which gamma produced it.

    ``--gamma`` defaults to None meaning "use the released value", and None is
    what belongs in the provenance and in the variant hash, since "default" is
    itself an arm. But the archive is the deliverable, and session 1's records
    could not say that gamma was 10000.0 -- the report could only warn
    "gamma = None: the intervention constant was not recorded". The resolved
    values now ride in ``method_extra``, which is outside the variant hash so
    adding them does not rename any run file.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.ws = make_workspace(self.root, n_images=4, n_per_image=2)
        self.env = _with_src_on_path(dict(os.environ))
        self.env.update({
            "FOX_ENV": "local",
            "FOX_PERSIST_ROOT": str(self.root / "persist"),
            "FOX_LOCAL_ROOT": str(self.root / "local"),
        })

    def test_the_echo_run_carries_no_fox_hyperparameters(self) -> None:
        """The dry-run method has none, and must not invent any."""
        subprocess.run(
            [sys.executable, "-m", "foxrepro.bench.generate", "--dry-run",
             "--benchmark", "pope", "--split", "random", "--seed", "42",
             "--dtype", "fp16",
             "--pope-dir", str(self.ws["pope_dir"]),
             "--images-dir", str(self.ws["images_dir"]),
             "--chair-image-list", str(self.ws["chair_list"]),
             "--runs-dir", str(self.ws["runs_dir"]), "--progress-every", "0"],
            env=self.env, capture_output=True, text=True, timeout=300, check=True,
        )
        runs = sorted(Path(self.ws["runs_dir"]).glob("*.jsonl"))
        self.assertTrue(runs)
        rec = json.loads(runs[0].read_text().splitlines()[0])
        self.assertNotIn(
            "resolved_hyperparameters", rec.get("method_extra", {})
        )

    def test_a_method_with_resolved_hyperparameters_gets_them_recorded(self) -> None:
        """Drives ``run()`` with a fake method, so no GPU and no checkpoint."""
        from foxrepro.bench.generate import (
            GenConfig, Method, MethodOutput, register_method, run,
        )
        from foxrepro.data.loaders import Sample

        resolved = {"gamma": 10000.0, "ratio": 0.35, "tau_js": 0.15}

        @register_method
        class _Probe(Method):
            name = "probe_resolved_hp"

            def load(self) -> None:
                return None

            def describe(self):
                d = dict(super().describe())
                d["resolved_hyperparameters"] = dict(resolved)
                return d

            def generate(self, sample, prompt):
                return MethodOutput(text="No", n_prompt_tokens=5,
                                    n_gen_tokens=1, extra={"n_steps": 1})

        img = self.root / "COCO_val2014_000000000042.jpg"
        img.write_bytes(b"x")
        samples = [
            Sample(sample_id=f"pope/random/q{i:07d}", image_id=42,
                   image_path=img, benchmark="pope", split="random",
                   question="Is there a dog in the image?", label="no",
                   extra={})
            for i in range(3)
        ]
        cfg = GenConfig(
            method="probe_resolved_hp", benchmark="pope", split="random",
            seed=42, dtype="fp16", model_id="fake/llava",
            prompt_template_id="pope_v0_raw", max_new_tokens=1,
        )
        out = self.root / "probe.jsonl"
        run(cfg, samples, out, progress_every=0, fsync=False)

        recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        self.assertEqual(len(recs), 3)
        for rec in recs:
            # gamma stays None in the provenance and therefore in the variant
            # hash; the resolved value lives alongside it.
            self.assertIsNone(rec["gamma"])
            self.assertEqual(
                rec["method_extra"]["resolved_hyperparameters"], resolved
            )
            # the method's own per-sample extras must survive the merge
            self.assertEqual(rec["method_extra"]["n_steps"], 1)
