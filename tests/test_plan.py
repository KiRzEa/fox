"""The reproduction plan, and the one invariant it must not break.

``plan.py`` tracks progress by predicting the run file each job will write and
counting the records in it. If that prediction ever drifts from what
``generate.py`` actually writes, the plan silently reports zero progress
forever, re-emits completed jobs, and a multi-session reproduction loses track
of itself. So the round-trip -- job -> command line -> GenConfig -> filename --
is asserted for every job in the plan, not a sample of them.
"""

from __future__ import annotations

import shlex
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

from foxrepro.bench.generate import GenConfig
from foxrepro.bench.plan import (
    COST_S, DUAL_BRANCH, PLAN, Job, measured_cost, next_jobs, render_plan,
    scan_progress,
)
from foxrepro.storage import MAX_NEW_TOKENS_DEFAULT


def cfg_from_command(cmd: str) -> GenConfig:
    """Rebuild a GenConfig from a plan command, the way the CLI would."""
    toks = shlex.split(cmd)
    args: Dict[str, Optional[str]] = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:].replace("-", "_")
            nxt = (
                toks[i + 1]
                if i + 1 < len(toks) and not toks[i + 1].startswith("--")
                else None
            )
            args[key] = nxt
            i += 2 if nxt else 1
        else:
            i += 1
    bench = args["benchmark"]
    is_fox = args["method"] == "fox"
    return GenConfig(
        method=str(args["method"]), benchmark=str(bench), split=str(args["split"]),
        seed=int(str(args["seed"])), dtype=str(args["dtype"]),
        model_id=str(args["model_id"]),
        prompt_template_id="pope_v1" if bench == "pope" else "chair_v1",
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT[str(bench)],
        decoding=str(args["decoding"]),
        system_prompt_id=str(args["system_prompt"]),
        mode=str(args.get("mode") or "as_implemented") if is_fox else "as_implemented",
        logit_dtype=str(args.get("logit_dtype") or "native") if is_fox else "native",
        gamma=float(str(args["gamma"])) if "gamma" in args else None,
        k=float(str(args["k"])) if "k" in args else None,
        tau_js=float(str(args["tau_js"])) if "tau_js" in args else None,
        alpha=float(str(args["alpha"])) if "alpha" in args else None,
    )


class TestPlanMatchesTheCli(unittest.TestCase):
    def test_every_job_round_trips_to_the_same_run_file(self) -> None:
        """The invariant the whole multi-session workflow rests on."""
        for job in PLAN:
            with self.subTest(target=job.target, note=job.note):
                self.assertEqual(
                    cfg_from_command(job.command()).run_key().filename(),
                    job.run_key().filename(),
                )

    def test_every_command_is_a_valid_generate_invocation(self) -> None:
        for job in PLAN:
            with self.subTest(note=job.note):
                cfg_from_command(job.command())  # raises on anything invalid

    def test_commands_always_name_the_converted_checkpoint(self) -> None:
        """A plan that emits the original LLaVA repo would load random weights."""
        for job in PLAN:
            self.assertIn("llava-hf/llava-1.5-7b-hf", job.command())
            self.assertNotIn("liuhaotian", job.command())


class TestPlanShape(unittest.TestCase):
    def test_covers_the_main_tables_and_the_mechanism_audit(self) -> None:
        targets = {j.target for j in PLAN}
        for t in ("table1", "table2", "table9", "gamma_sweep", "controls",
                  "dtype_arms", "system_prompt"):
            self.assertIn(t, targets)

    def test_gamma_sweep_spans_the_fp16_ulp_grid(self) -> None:
        """0.0625 / 0.5 / 8 / 32 -- a 512x span, entirely inside fp16."""
        gammas = sorted({j.gamma for j in PLAN if j.target == "gamma_sweep"})
        self.assertEqual(gammas, [100.0, 1000.0, 10000.0, 60000.0])

    def test_gamma_sweep_excludes_the_overflowing_value(self) -> None:
        # 1e5 exceeds fp16's 65504 and becomes inf, so it is not a data point.
        self.assertNotIn(1e5, {j.gamma for j in PLAN})

    def test_every_control_mode_is_planned(self) -> None:
        modes = {j.mode for j in PLAN if j.target == "controls"}
        self.assertEqual(
            modes, {"off", "as_described", "ulp_control", "noise_control"}
        )

    def test_dual_branch_jobs_cost_more(self) -> None:
        single = Job("t", "sampling", "pope", "random")
        dual = Job("t", "fox", "pope", "random")
        self.assertGreater(dual.est_seconds(), single.est_seconds())
        self.assertIn("fox", DUAL_BRANCH)

    def test_limited_jobs_cost_proportionally_less(self) -> None:
        full = Job("t", "fox", "pope", "adversarial")
        small = Job("t", "fox", "pope", "adversarial", limit=500)
        self.assertEqual(small.n_samples, 500)
        self.assertLess(small.est_seconds(), full.est_seconds())

    def test_cost_model_covers_every_planned_combination(self) -> None:
        for job in PLAN:
            with self.subTest(note=job.note):
                self.assertIn((job.benchmark, job.branches), COST_S)


class TestProgressTracking(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.runs = Path(td.name)

    def test_empty_runs_dir_means_nothing_done(self) -> None:
        prog = scan_progress(runs_root=self.runs)
        self.assertEqual(prog["n_complete"], 0)
        self.assertGreater(float(prog["remaining_hours"]), 0)

    def test_next_jobs_respects_the_budget(self) -> None:
        small = next_jobs(2.0, runs_root=self.runs)
        large = next_jobs(20.0, runs_root=self.runs)
        self.assertLess(len(small), len(large))
        self.assertLessEqual(
            sum(float(j["est_hours"]) for j in small[1:]), 2.0 + 1e-9
        )

    def test_an_oversized_job_is_still_emitted(self) -> None:
        """It resumes, so a partial session is progress rather than waste."""
        jobs = next_jobs(0.01, runs_root=self.runs)
        self.assertTrue(jobs)
        self.assertTrue(jobs[0]["exceeds_budget"])

    def test_measured_cost_is_empty_without_runs(self) -> None:
        self.assertEqual(measured_cost(self.runs), {})

    def test_render_plan_produces_runnable_commands(self) -> None:
        out = render_plan(3.0, self.runs)
        self.assertIn("python -m foxrepro.bench.generate", out)
        self.assertIn("GPU-hours remaining", out)
        self.assertIn("| target |", out)


if __name__ == "__main__":
    unittest.main()


class TestSessionPartition(unittest.TestCase):
    """Disjointness first: two sessions running the same job burns a T4 twice."""

    def test_partition_is_disjoint_and_complete(self) -> None:
        from foxrepro.bench.plan import partition_jobs

        for n in (1, 2, 3, 5, 8):
            with self.subTest(n_sessions=n):
                groups = partition_jobs(n)
                ids = [j.job_id() for g in groups for j in g]
                self.assertEqual(len(ids), len(PLAN), "jobs lost or duplicated")
                self.assertEqual(len(set(ids)), len(PLAN), "a job is in two sessions")

    def test_partition_is_cost_balanced(self) -> None:
        """Round-robin over costs ranging 0.2-2.3 h leaves one session double."""
        from foxrepro.bench.plan import partition_jobs

        loads = [
            sum(j.est_seconds() for j in g) / 3600 for g in partition_jobs(5)
        ]
        self.assertLess(max(loads) - min(loads), 0.1 * max(loads))

    def test_cheap_jobs_run_first_within_a_session(self) -> None:
        """A session cut early then leaves more completed jobs behind."""
        from foxrepro.bench.plan import partition_jobs

        for group in partition_jobs(5):
            costs = [j.est_seconds() for j in group]
            self.assertEqual(costs, sorted(costs))

    def test_rejects_a_bad_session_count(self) -> None:
        from foxrepro.bench.plan import partition_jobs

        with self.assertRaises(ValueError):
            partition_jobs(0)

    def test_job_ids_are_unique(self) -> None:
        """The frozen plan and the calibration TSV are both keyed on them."""
        ids = [j.job_id() for j in PLAN]
        self.assertEqual(len(set(ids)), len(ids))


class TestFrozenPlan(unittest.TestCase):
    """The split is written to a file so sessions agree after a plan edit."""

    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)

    def test_freeze_writes_every_job_once(self) -> None:
        from foxrepro.bench.plan import freeze_plan

        out = self.tmp / "run_plan.json"
        payload = freeze_plan(4, 10.5, out)
        self.assertTrue(out.is_file())
        assigned = [j for s in payload["sessions"] for j in s["jobs"]]
        self.assertEqual(len(assigned), len(PLAN))
        self.assertEqual(len(set(assigned)), len(PLAN))
        self.assertEqual(set(assigned), set(payload["jobs"]))

    def test_frozen_commands_match_the_live_plan(self) -> None:
        from foxrepro.bench.plan import freeze_plan

        payload = freeze_plan(3, 10.5, self.tmp / "p.json")
        live = {j.job_id(): j.command() for j in PLAN}
        for jid, rec in payload["jobs"].items():
            self.assertEqual(rec["command"], live[jid])

    def test_calibration_replaces_estimates(self) -> None:
        from foxrepro.bench.plan import freeze_plan

        jid = PLAN[0].job_id()
        tsv = self.tmp / "_job_times.tsv"
        tsv.write_text(f"{jid}\t42.0\t0\nnot-a-job\tbroken\n")
        payload = freeze_plan(2, 10.5, self.tmp / "p.json", calibrate=tsv)
        self.assertEqual(payload["n_calibrated"], 1)
        self.assertEqual(payload["jobs"][jid]["est_s"], 42.0 * 60)
        self.assertTrue(payload["jobs"][jid]["measured"])

    def test_run_file_names_in_the_plan_match_the_cli(self) -> None:
        """If these drift, a session re-runs work another already finished."""
        from foxrepro.bench.plan import freeze_plan

        payload = freeze_plan(2, 10.5, self.tmp / "p.json")
        for jid, rec in payload["jobs"].items():
            cmd = rec["command"]
            self.assertEqual(
                cfg_from_command(cmd).run_key().filename(), rec["run_file"]
            )

    def test_dry_run_prints_without_executing(self) -> None:
        import io
        from contextlib import redirect_stdout

        from foxrepro.bench.plan import freeze_plan, run_session

        out = self.tmp / "p.json"
        freeze_plan(2, 10.5, out)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_session(1, out, dry_run=True)
        self.assertEqual(rc, 0)
        self.assertIn("foxrepro.bench.generate", buf.getvalue())

    def test_unknown_session_is_an_error_not_a_silent_no_op(self) -> None:
        from foxrepro.bench.plan import freeze_plan, run_session

        out = self.tmp / "p.json"
        freeze_plan(2, 10.5, out)
        self.assertEqual(run_session(9, out, dry_run=True), 2)


class TestIngest(unittest.TestCase):
    """Each session writes to its own /kaggle/working; the files must merge."""

    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)

    def _write(self, d: Path, name: str, ids) -> None:
        import json as _json

        d.mkdir(parents=True, exist_ok=True)
        with (d / name).open("w") as fh:
            for sid in ids:
                fh.write(_json.dumps({"sample_id": sid, "output": "Yes"}) + "\n")

    def test_distinct_files_are_copied(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        s1, s2, dest = self.tmp / "s1", self.tmp / "s2", self.tmp / "dest"
        self._write(s1, "a.jsonl", ["x1", "x2"])
        self._write(s2, "b.jsonl", ["y1"])
        rep = ingest_runs([s1, s2], dest)
        self.assertEqual(rep["copied"], 2)
        self.assertTrue((dest / "a.jsonl").is_file())
        self.assertTrue((dest / "b.jsonl").is_file())

    def test_partial_files_for_the_same_job_merge_by_union(self) -> None:
        """A plan edit can move a job between sessions; both halves are real."""
        from foxrepro.bench.plan import ingest_runs

        s1, s2, dest = self.tmp / "s1", self.tmp / "s2", self.tmp / "dest"
        self._write(s1, "same.jsonl", ["x1", "x2"])
        self._write(s2, "same.jsonl", ["x2", "x3"])
        ingest_runs([s1], dest)
        rep = ingest_runs([s2], dest)
        self.assertEqual(rep["merged"], 1)
        text = (dest / "same.jsonl").read_text()
        for sid in ("x1", "x2", "x3"):
            self.assertIn(sid, text)
        self.assertEqual(text.count('"x2"'), 1, "duplicate record")

    def test_nothing_new_is_skipped_not_duplicated(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        s1, dest = self.tmp / "s1", self.tmp / "dest"
        self._write(s1, "same.jsonl", ["x1"])
        ingest_runs([s1], dest)
        rep = ingest_runs([s1], dest)
        self.assertEqual(rep["skipped"], 1)
        self.assertEqual(rep["merged"], 0)

    def test_dry_run_changes_nothing(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        s1, dest = self.tmp / "s1", self.tmp / "dest"
        self._write(s1, "a.jsonl", ["x1"])
        ingest_runs([s1], dest, dry_run=True)
        self.assertFalse((dest / "a.jsonl").exists())


class TestProgressActuallyCounts(unittest.TestCase):
    """Progress must be read from the file, not silently reported as zero.

    scan_progress once wrapped the scan in a broad `except Exception: done = 0`
    while calling a field that did not exist. Every job therefore reported no
    progress, the plan re-emitted finished work forever, and the round-trip
    test above still passed because it only checked filenames. So the count
    itself is now asserted against a file written with real provenance.
    """

    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.runs = Path(td.name)

    def _fill(self, job, n: int) -> None:
        import json as _json

        prov = cfg_from_command(job.command()).provenance()
        with (self.runs / job.run_key().filename()).open("w") as fh:
            for i in range(n):
                rec = dict(prov)
                rec.update(
                    sample_id=f"{job.benchmark}/{job.split}/q{i:07d}",
                    image_id=i, prompt="p", output="Yes", latency_ms=1.0,
                    n_prompt_tokens=600, n_gen_tokens=1,
                    timestamp="2026-01-01T00:00:00Z", commit_hash="x",
                    transformers_path="y", label="yes",
                )
                fh.write(_json.dumps(rec) + "\n")

    def test_partial_progress_is_counted(self) -> None:
        job = PLAN[0]
        self._fill(job, 7)
        row = scan_progress([job], self.runs)["rows"][0]
        self.assertEqual(row["done"], 7)
        self.assertFalse(row["complete"])

    def test_a_full_file_marks_the_job_complete(self) -> None:
        job = next(j for j in PLAN if j.n_samples == 500)
        self._fill(job, job.n_samples)
        row = scan_progress([job], self.runs)["rows"][0]
        self.assertTrue(row["complete"])
        self.assertEqual(row["remaining_s"], 0)

    def test_a_completed_job_is_not_re_emitted(self) -> None:
        from foxrepro.bench.plan import next_jobs

        job = next(j for j in PLAN if j.n_samples == 500)
        self._fill(job, job.n_samples)
        emitted = [j["command"] for j in next_jobs(50.0, [job], self.runs)]
        self.assertEqual(emitted, [])

    def test_measured_cost_reads_real_latencies(self) -> None:
        from foxrepro.bench.plan import measured_cost

        self._fill(PLAN[0], 5)
        cost = measured_cost(self.runs)
        self.assertIn("pope/single", cost)
        self.assertAlmostEqual(cost["pope/single"], 0.001, places=3)
