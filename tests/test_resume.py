"""Tests for the resume / crash-safety logic in :mod:`bench.generate`.

The scenarios cover the failure this project actually hits: a Colab session
dies at sample 317 and the run must continue at 318 with no duplicate and no
lost sample.

1. **Hard kill** (``os._exit`` right after a flushed record, simulating a
   dropped VM): resume must produce exactly the full sample set, once each.
2. **Hard kill mid-write** (a trailing partial JSON line): the partial line
   must be truncated away so the file stays valid JSONL, and the sample must
   be redone.
3. **SIGTERM**: the signal handler must finish the in-flight sample, close the
   file, and exit with ``EXIT_STOPPED_EARLY``.
4. **Repeated interruptions**: killing the run many times must still converge
   to a complete, duplicate-free file.
5. A corrupt line in the *middle* of the file is not guessed at: it raises.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, List, Sequence


from foxrepro.bench.generate import (  # noqa: E402
    EXIT_STOPPED_EARLY, GenConfig, Method, MethodOutput, get_method, run,
    scan_run_file,
)
from foxrepro.data.loaders import Sample, load_pope  # noqa: E402
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



def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    """Parse a JSONL file, failing loudly on any unparsable line."""
    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{i} is not valid JSON: {line!r}") from exc
    return rows


class _SubprocessRunCase(unittest.TestCase):
    """Drives ``foxrepro.bench.generate`` as a real subprocess in a temp workspace."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=10, n_per_image=6)
        self.expected_ids = [
            s.sample_id for s in load_pope(
                self.ws["pope_dir"], self.ws["images_dir"], "random"
            )
        ]
        self.assertEqual(len(self.expected_ids), 60)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def generate(self, crash_after: int = 0, crash_mode: str = "hard",
                 run_path: Path = None) -> subprocess.CompletedProcess:
        """Invoke generate.py once; optionally crash after N records."""
        env = _with_src_on_path(dict(os.environ))
        env["FOX_ENV"] = "local"
        env["FOX_PERSIST_ROOT"] = str(self.ws["root"] / "persist")
        env["FOX_LOCAL_ROOT"] = str(self.ws["root"] / "local")
        env.pop("FOX_TEST_CRASH_AFTER", None)
        if crash_after:
            env["FOX_TEST_CRASH_AFTER"] = str(crash_after)
            env["FOX_TEST_CRASH_MODE"] = crash_mode
            if run_path is not None:
                env["FOX_TEST_CRASH_PATH"] = str(run_path)
        return subprocess.run(
            [
                sys.executable, "-m", "foxrepro.bench.generate",
                "--dry-run", "--benchmark", "pope", "--split", "random",
                "--seed", "42", "--dtype", "fp16",
                "--pope-dir", str(self.ws["pope_dir"]),
                "--images-dir", str(self.ws["images_dir"]),
                "--runs-dir", str(self.ws["runs_dir"]),
                "--progress-every", "0",
            ],
            env=env, capture_output=True, text=True, timeout=300,
        )

    def the_run_file(self) -> Path:
        files = sorted(Path(self.ws["runs_dir"]).glob("*.jsonl"))
        self.assertEqual(len(files), 1, f"expected one run file, got {files}")
        return files[0]


class TestHardKillResume(_SubprocessRunCase):
    def test_resume_after_hard_kill_is_exact(self) -> None:
        first = self.generate(crash_after=17)
        self.assertEqual(first.returncode, 99, first.stderr[-2000:])
        path = self.the_run_file()
        partial = _read_jsonl(path)
        self.assertEqual(len(partial), 17)

        second = self.generate()
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        rows = _read_jsonl(path)
        ids = [r["sample_id"] for r in rows]
        self.assertEqual(len(ids), 60, "lost or duplicated samples")
        self.assertEqual(len(set(ids)), 60, "duplicate sample_ids after resume")
        self.assertEqual(sorted(ids), sorted(self.expected_ids))
        # the resumed process must not have redone finished samples
        self.assertIn("17 done", second.stdout)

    def test_every_record_carries_full_provenance(self) -> None:
        self.assertEqual(self.generate().returncode, 0)
        for row in _read_jsonl(self.the_run_file()):
            missing = [f for f in PROVENANCE_FIELDS if f not in row]
            self.assertEqual(missing, [], f"record missing {missing}")
            self.assertEqual(row["method"], "echo")
            self.assertEqual(row["dtype"], "fp16")
            self.assertEqual(row["max_new_tokens"], 1)  # POPE protocol
            self.assertEqual(row["n_gen_tokens"], 1)
            self.assertGreater(row["latency_ms"], 0.0)
            self.assertIn("transformers_path", row)

    def test_second_run_is_a_no_op(self) -> None:
        self.assertEqual(self.generate().returncode, 0)
        before = self.the_run_file().read_bytes()
        again = self.generate()
        self.assertEqual(again.returncode, 0)
        self.assertIn("already complete", again.stdout)
        self.assertEqual(self.the_run_file().read_bytes(), before)

    def test_repeated_interruptions_converge(self) -> None:
        # ``crash_after`` counts records written by *that* process, so a late
        # pass may finish the remaining samples before the hook fires; both
        # outcomes are legal, the invariants below are not negotiable.
        for n in (5, 11, 23, 41):
            result = self.generate(crash_after=n)
            self.assertIn(result.returncode, (0, 99), result.stderr[-2000:])
            rows = _read_jsonl(self.the_run_file())
            self.assertEqual(len(set(r["sample_id"] for r in rows)), len(rows),
                             "a crash introduced duplicate sample_ids")
            self.assertLessEqual(len(rows), 60)
        self.assertEqual(self.generate().returncode, 0)
        ids = [r["sample_id"] for r in _read_jsonl(self.the_run_file())]
        self.assertEqual(sorted(ids), sorted(self.expected_ids))
        self.assertEqual(len(set(ids)), 60)


class TestTruncatedLineRecovery(_SubprocessRunCase):
    def test_partial_trailing_line_is_repaired(self) -> None:
        # first pass crashes *after* writing a deliberate partial line
        probe = self.generate(crash_after=9, crash_mode="hard")
        self.assertEqual(probe.returncode, 99)
        path = self.the_run_file()

        # emulate a kill in the middle of a write
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"method": "echo", "sample_id": "pope/random/q000')
        with open(path, "rb") as fh:
            self.assertFalse(fh.read().endswith(b"\n"))

        second = self.generate()
        self.assertEqual(second.returncode, 0, second.stderr[-2000:])
        self.assertIn("repaired", second.stdout)
        rows = _read_jsonl(path)
        ids = [r["sample_id"] for r in rows]
        self.assertEqual(sorted(ids), sorted(self.expected_ids))
        self.assertEqual(len(set(ids)), 60)

    def test_crash_mode_truncate_leaves_recoverable_file(self) -> None:
        first = self.generate(crash_after=4, crash_mode="hard")
        self.assertEqual(first.returncode, 99)
        path = self.the_run_file()
        second = self.generate(crash_after=8, crash_mode="truncate", run_path=path)
        self.assertEqual(second.returncode, 99)
        with open(path, "rb") as fh:
            self.assertFalse(fh.read().endswith(b"\n"))
        third = self.generate()
        self.assertEqual(third.returncode, 0)
        ids = [r["sample_id"] for r in _read_jsonl(path)]
        self.assertEqual(sorted(ids), sorted(self.expected_ids))


class TestScanRunFile(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "run.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, lines: Sequence[str]) -> None:
        self.path.write_text("".join(lines), encoding="utf-8")

    def test_missing_file_is_empty_state(self) -> None:
        result = scan_run_file(self.path)
        self.assertEqual(result.done_ids, set())
        self.assertEqual(result.n_records, 0)

    def test_counts_duplicates(self) -> None:
        self._write([
            '{"sample_id": "a"}\n', '{"sample_id": "b"}\n', '{"sample_id": "a"}\n',
        ])
        result = scan_run_file(self.path)
        self.assertEqual(result.done_ids, {"a", "b"})
        self.assertEqual((result.n_records, result.n_duplicates), (3, 1))

    def test_truncates_trailing_partial_line(self) -> None:
        self._write(['{"sample_id": "a"}\n', '{"sample_id": "b'])
        result = scan_run_file(self.path)
        self.assertEqual(result.done_ids, {"a"})
        self.assertGreater(result.truncated_bytes, 0)
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))
        self.assertEqual(len(self.path.read_text().splitlines()), 1)

    def test_keeps_partial_line_when_repair_is_off(self) -> None:
        self._write(['{"sample_id": "a"}\n', '{"sample_id": "b'])
        scan_run_file(self.path, repair=False)
        self.assertFalse(self.path.read_bytes().endswith(b"\n"))

    def test_corrupt_middle_line_raises(self) -> None:
        self._write(['{"sample_id": "a"}\n', 'not json\n', '{"sample_id": "b"}\n'])
        with self.assertRaises(ValueError):
            scan_run_file(self.path)

    def test_record_without_sample_id_raises(self) -> None:
        self._write(['{"method": "echo"}\n'])
        with self.assertRaises(ValueError):
            scan_run_file(self.path)

    def test_run_key_mismatch_raises(self) -> None:
        from foxrepro.storage import RunKey

        key = RunKey("echo", "pope", "random", 42, "fp16")
        self._write([json.dumps({
            "sample_id": "a", "method": "echo", "benchmark": "pope",
            "split": "random", "seed": 42, "dtype": "bf16",
        }) + "\n"])
        with self.assertRaises(ValueError):
            scan_run_file(self.path, key)


class _SlowEcho(Method):
    """Echo with a delay, so a signal can land mid-run."""

    name = "slow_echo_test_only"
    requires_gpu = False

    def load(self) -> None:
        pass

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        time.sleep(0.02)
        return MethodOutput("Yes", 10, 1, {})


class TestSigtermHandling(unittest.TestCase):
    """SIGTERM must leave a valid file and an EXIT_STOPPED_EARLY-style result."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=10, n_per_image=6)
        self.samples = load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")
        self.cfg = GenConfig(
            method="slow_echo_test_only", benchmark="pope", split="random",
            seed=1, dtype="fp16", model_id="test", prompt_template_id="pope_v1",
            max_new_tokens=1,
        )
        from foxrepro.bench.generate import _REGISTRY

        _REGISTRY.setdefault(_SlowEcho.name, _SlowEcho)
        self.path = Path(self.ws["runs_dir"]) / self.cfg.run_key().filename()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sigterm_stops_cleanly_and_run_resumes(self) -> None:
        def killer() -> None:
            time.sleep(0.25)
            os.kill(os.getpid(), signal.SIGTERM)

        threading.Thread(target=killer, daemon=True).start()
        summary = run(self.cfg, self.samples, self.path, progress_every=0, fsync=False)
        self.assertTrue(summary["stopped_early"])
        self.assertLess(summary["n_written"], 60)
        self.assertGreater(summary["n_written"], 0)

        rows = _read_jsonl(self.path)
        self.assertEqual(len(rows), summary["n_written"])

        resumed = run(self.cfg, self.samples, self.path, progress_every=0, fsync=False)
        self.assertFalse(resumed["stopped_early"])
        self.assertEqual(resumed["n_done_at_start"], summary["n_written"])
        ids = [r["sample_id"] for r in _read_jsonl(self.path)]
        self.assertEqual(len(ids), 60)
        self.assertEqual(len(set(ids)), 60)
        self.assertEqual(sorted(ids), sorted(s.sample_id for s in self.samples))

    def test_exit_code_constant_is_distinct(self) -> None:
        self.assertNotIn(EXIT_STOPPED_EARLY, (0, 1, 2))


class TestLimitAndRunKey(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=5, n_per_image=6)
        self.samples = load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _cfg(self, **over: object) -> GenConfig:
        base = dict(
            method="echo", benchmark="pope", split="random", seed=7, dtype="fp16",
            model_id="test", prompt_template_id="pope_v1", max_new_tokens=1,
        )
        base.update(over)
        return GenConfig(**base)  # type: ignore[arg-type]

    def test_limit_is_counted_against_completed_samples(self) -> None:
        cfg = self._cfg(limit=10)
        path = Path(self.ws["runs_dir"]) / cfg.run_key().filename()
        first = run(cfg, self.samples, path, progress_every=0, fsync=False)
        self.assertEqual(first["n_written"], 10)
        second = run(cfg, self.samples, path, progress_every=0, fsync=False)
        self.assertEqual(second["n_written"], 0)
        self.assertEqual(len(_read_jsonl(path)), 10)

    def test_gamma_sweep_does_not_share_a_run_file(self) -> None:
        """Two gammas must not alias into one file even though gamma is not
        part of the resume key."""
        a = self._cfg(method="echo", gamma=1000.0)
        b = self._cfg(method="echo", gamma=50.0)
        self.assertNotEqual(a.run_key().filename(), b.run_key().filename())

    def test_greedy_control_does_not_share_a_run_file(self) -> None:
        a = self._cfg(decoding="sampling")
        b = self._cfg(decoding="greedy")
        self.assertNotEqual(a.run_key().filename(), b.run_key().filename())

    def test_dtype_is_in_the_file_name(self) -> None:
        self.assertIn("dt-bf16", self._cfg(dtype="bf16").run_key().filename())

    def test_remaining_stub_methods_fail_loudly(self) -> None:
        """Only sid and opera are still stubs.

        ``sampling``, ``fox`` and ``vcd`` are implemented in
        ``foxrepro.bench.methods`` and tested in ``tests/test_methods.py``
        against a fake model, so they are deliberately absent here: calling
        ``load()`` on them would try to fetch a checkpoint.

        SID needs its upstream repo (the Fox release ships only a FastV
        skeleton, no reweighting), and OPERA is a beam-search method with
        rollback whose reference implementation is its own transformers fork.
        """
        for name in ("sid", "opera"):
            with self.subTest(method=name):
                method = get_method(name)(self._cfg(method=name))
                with self.assertRaises(NotImplementedError):
                    method.load()

    def test_implemented_methods_are_not_stubs(self) -> None:
        from foxrepro.bench.generate import _NotImplementedMethod

        for name in ("sampling", "fox", "vcd"):
            with self.subTest(method=name):
                cls = get_method(name)
                self.assertFalse(
                    issubclass(cls, _NotImplementedMethod),
                    f"{name} should be a real implementation by now",
                )
                self.assertEqual(cls.__module__, "foxrepro.bench.methods")


if __name__ == "__main__":
    unittest.main(verbosity=2)
