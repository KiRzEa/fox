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
        records = self.records + [_record("q10", "yes", "I cannot tell")]
        negative, _ = evaluate_pope(records, unparsed_policy="negative", n_boot=200)
        self.assertEqual(negative["n_unparsed"], 1)
        self.assertEqual(negative["n_samples"], 11)
        self.assertEqual(negative["confusion"]["fn"], 3)  # counted as "no"

        excluded, _ = evaluate_pope(records, unparsed_policy="exclude", n_boot=200)
        self.assertEqual(excluded["n_unparsed"], 1)
        self.assertEqual(excluded["n_samples"], 10)
        self.assertAlmostEqual(excluded["accuracy"]["point"], 0.60, places=12)

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
