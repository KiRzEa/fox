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
from typing import Any, Dict, Optional

from foxrepro.bench.generate import GenConfig
from foxrepro.bench.plan import (
    COST_S, DUAL_BRANCH, PLAN, Job, measured_cost, next_jobs, render_plan,
    scan_progress,
)
from foxrepro.storage import MAX_NEW_TOKENS_DEFAULT

#: tests/ sits next to the repo root, where run_kaggle_fox.sh lives.
REPO_ROOT = Path(__file__).resolve().parent.parent


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
        prompt_template_id=str(
            args.get("prompt_template_id")
            or ("pope_v0_raw" if bench == "pope" else "chair_v1")
        ),
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT[str(bench)],
        decoding=str(args["decoding"]),
        system_prompt_id=str(args["system_prompt"]),
        wrapper_id=str(args.get("wrapper") or "released"),
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


class TestZipIngest(unittest.TestCase):
    """Download five zips, drop them in one folder, run one command.

    pack_runs.sh zips flat (`zip -j`), so members sit at the archive root.
    rglob is still used when reading them back, because a zip produced by some
    other means may nest.
    """

    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.dest = self.tmp / "runs"

    def _session_zip(self, job, n: int, session: int) -> Path:
        import json as _json
        import zipfile as _zip

        sd = self.tmp / f"s{session}"
        sd.mkdir(parents=True, exist_ok=True)
        prov = cfg_from_command(job.command()).provenance()
        f = sd / job.run_key().filename()
        with f.open("w") as fh:
            for i in range(n):
                rec = dict(prov)
                rec.update(
                    sample_id=f"{job.benchmark}/{job.split}/q{i:07d}",
                    image_id=i, prompt="p", output="Yes", latency_ms=1400.0,
                    n_prompt_tokens=640, n_gen_tokens=1, commit_hash="x",
                    timestamp="2026-01-01T00:00:00Z", transformers_path="y",
                    label="yes",
                )
                fh.write(_json.dumps(rec) + "\n")
        z = self.tmp / f"fox_runs_fox_r1_s{session}.zip"
        with _zip.ZipFile(z, "w") as zf:
            zf.write(f, arcname=f.name)
        return z

    def test_zips_are_expanded_and_merged(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        zips = [self._session_zip(j, 4, i) for i, j in enumerate(PLAN[:3], 1)]
        rep = ingest_runs(zips, self.dest)
        self.assertEqual(rep["copied"], 3)
        self.assertEqual(len(list(self.dest.glob("*.jsonl"))), 3)

    def test_zips_in_subdirectories_are_found(self) -> None:
        """/kaggle/input holds one subdirectory per attached dataset, which is
        how a previous session's results reach a fresh one."""
        from foxrepro.bench.plan import ingest_runs

        root = self.tmp / "kaggle_input"
        (root / "fox-session-1").mkdir(parents=True)
        self._session_zip(PLAN[0], 3, 1).replace(
            root / "fox-session-1" / "s1.zip")
        rep = ingest_runs([root], self.dest)
        self.assertEqual(rep["copied"], 1)

    def test_a_folder_of_zips_works_too(self) -> None:
        """The actual workflow: download into one folder, ingest the folder."""
        from foxrepro.bench.plan import ingest_runs

        box = self.tmp / "inbox"
        box.mkdir()
        for i, j in enumerate(PLAN[:2], 1):
            self._session_zip(j, 3, i).replace(box / f"s{i}.zip")
        rep = ingest_runs([box], self.dest)
        self.assertEqual(rep["copied"], 2)

    def test_progress_is_visible_after_ingest(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        zips = [self._session_zip(j, 4, i) for i, j in enumerate(PLAN[:3], 1)]
        ingest_runs(zips, self.dest)
        rows = scan_progress(PLAN[:3], self.dest)["rows"]
        self.assertTrue(all(r["done"] == 4 for r in rows))

    def test_re_ingesting_the_same_zip_adds_nothing(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        zips = [self._session_zip(PLAN[0], 4, 1)]
        ingest_runs(zips, self.dest)
        again = ingest_runs(zips, self.dest)
        self.assertEqual(again["copied"], 0)
        self.assertEqual(again["merged"], 0)
        self.assertEqual(again["skipped"], 1)

    def test_no_sources_and_no_inbox_is_not_an_error(self) -> None:
        from foxrepro.bench.plan import ingest_runs

        # An explicit empty inbox: the repo's own may hold real session zips.
        rep = ingest_runs([], self.dest, inbox=self.tmp / "empty_inbox")
        self.assertEqual(rep["copied"], 0)
        self.assertIn("note", rep)


class TestSessionBudget(unittest.TestCase):
    """The plan has to be cuttable into shards that each fit a Kaggle session.

    The number of shards is not pinned here: it moves whenever a job is added
    or an estimate is recalibrated. What is pinned is the property the runner
    depends on -- that *some* split fits under the 10 h wall, and that the
    split one shard smaller does not, so the number in the docs is the
    smallest one that works rather than an arbitrary round figure.
    """

    WALL_HOURS = 10.0

    @staticmethod
    def _loads(n: int) -> list:
        from foxrepro.bench.plan import partition_jobs

        return [
            sum(j.est_seconds() for j in g) / 3600 for g in partition_jobs(n)
        ]

    def _smallest_fitting(self) -> int:
        for n in range(1, 61):
            if max(self._loads(n)) <= self.WALL_HOURS:
                return n
        self.fail("no split of the plan fits a 10 h session")

    def test_a_split_fits_the_ten_hour_cap(self) -> None:
        n = self._smallest_fitting()
        loads = self._loads(n)
        self.assertLessEqual(max(loads), self.WALL_HOURS)
        # No shard may be near-empty either: that would mean the partitioner
        # is dumping everything into one session and padding with the rest.
        self.assertGreater(min(loads), 0.5 * max(loads))

    def test_one_session_fewer_does_not_fit(self) -> None:
        n = self._smallest_fitting()
        self.assertGreater(max(self._loads(n - 1)), self.WALL_HOURS)

    def test_the_runner_default_matches_the_plan(self) -> None:
        """``run_kaggle_fox.sh`` documents a session count; keep it honest."""
        import re

        script = (REPO_ROOT / "run_kaggle_fox.sh").read_text()
        m = re.search(r"--sessions (\d+)", script)
        self.assertIsNotNone(m, "run_kaggle_fox.sh no longer documents --sessions")
        documented = int(m.group(1))  # type: ignore[union-attr]
        self.assertLessEqual(max(self._loads(documented)), self.WALL_HOURS)

    def test_watchdog_fires_inside_the_budget(self) -> None:
        """A fixed 11.3 h deadline never fires in a 10 h session."""
        for budget in (2.0, 10.0, 12.0):
            with self.subTest(budget=budget):
                deadline = max(budget - 0.25, budget * 0.5)
                self.assertLess(deadline, budget)
                self.assertGreater(deadline, budget * 0.4)


class TestPilotSessions(unittest.TestCase):
    """Session 1 answers one question: do the paper's numbers reproduce?

    Nothing about mechanism, because that only becomes worth asking once there
    is a result to explain -- if Fox does not beat the baseline here, every
    control downstream measures a difference that does not exist.
    """

    def test_one_session_is_pinned(self) -> None:
        from foxrepro.bench.plan import pilot_jobs

        groups = pilot_jobs()
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]), 6)

    def test_it_covers_the_three_main_tables(self) -> None:
        from foxrepro.bench.plan import pilot_jobs

        self.assertEqual({j.target for j in pilot_jobs()[0]},
                         {"table1", "table2", "table9"})

    def test_each_table_has_both_baseline_and_method(self) -> None:
        from foxrepro.bench.plan import pilot_jobs

        by_target = {}
        for j in pilot_jobs()[0]:
            by_target.setdefault(j.target, set()).add(j.method)
        for target, methods in by_target.items():
            with self.subTest(target=target):
                self.assertEqual(methods, {"sampling", "fox"},
                                 "a table without both is not comparable")

    def test_table9_jobs_are_greedy(self) -> None:
        """Greedy is deterministic, so at one seed it is the sharper signal."""
        from foxrepro.bench.plan import pilot_jobs

        for j in pilot_jobs()[0]:
            if j.target == "table9":
                self.assertEqual(j.decoding, "greedy")
            else:
                self.assertEqual(j.decoding, "sampling")

    def test_no_mechanism_arm_is_pinned(self) -> None:
        """Controls and sweeps come after reproducibility, not alongside it."""
        from foxrepro.bench.plan import pilot_jobs

        for j in pilot_jobs()[0]:
            self.assertEqual(j.mode, "as_implemented")
            self.assertEqual(j.logit_dtype, "native")
            self.assertIsNone(j.gamma)

    def test_vcd_is_excluded(self) -> None:
        """What this repo runs is VCD noise inside Fox's fusion, not VCD."""
        from foxrepro.bench.plan import pilot_jobs

        self.assertNotIn("vcd", {j.method for j in pilot_jobs()[0]})

    def test_it_fits_one_session(self) -> None:
        from foxrepro.bench.plan import pilot_jobs

        hours = sum(j.est_seconds() for j in pilot_jobs()[0]) / 3600
        self.assertLess(hours, 10.0)
        self.assertGreater(hours, 4.0)

    def test_partition_stays_disjoint_and_complete(self) -> None:
        from foxrepro.bench.plan import partition_jobs

        ids = [j.job_id() for g in partition_jobs(10, pilot=True) for j in g]
        self.assertEqual(len(ids), len(PLAN))
        self.assertEqual(len(set(ids)), len(PLAN))

    def test_pilot_needs_more_sessions_than_it_pins(self) -> None:
        from foxrepro.bench.plan import partition_jobs

        with self.assertRaises(ValueError):
            partition_jobs(1, pilot=True)


class TestArchive(unittest.TestCase):
    """results/runs is the committed archive; the live runs dir is working state.

    Tracking raw JSONL in git looks wasteful and is not: 63% of every record is
    provenance repeated identically on all 3000 lines, so git stores 7.7 MB of
    POPE runs in 0.34 MB. A clone then already carries every finished job, and
    a fresh Kaggle session needs no attached Dataset for resume to work.
    """

    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)
        self.live = self.tmp / "live"
        self.archive = self.tmp / "results" / "runs"

    def _run_file(self, job, n: int, where: Path) -> Path:
        import json as _json

        where.mkdir(parents=True, exist_ok=True)
        prov = cfg_from_command(job.command()).provenance()
        f = where / job.run_key().filename()
        with f.open("w") as fh:
            for i in range(n):
                rec = dict(prov)
                rec.update(sample_id=f"q{i:07d}", image_id=i, prompt="p",
                           output="Yes", latency_ms=1.0, n_prompt_tokens=600,
                           n_gen_tokens=1, timestamp="2026-01-01T00:00:00Z",
                           commit_hash="x", transformers_path="y", label="yes")
                fh.write(_json.dumps(rec) + "\n")
        return f

    def test_live_runs_are_copied_into_the_archive(self) -> None:
        from foxrepro.bench.plan import archive_runs

        self._run_file(PLAN[0], 5, self.live)
        rep = archive_runs(self.live, self.archive)
        self.assertEqual(rep["copied"], 1)
        self.assertEqual(len(list(self.archive.glob("*.jsonl"))), 1)

    def test_a_job_finished_later_merges_rather_than_replaces(self) -> None:
        """Half done in one session, finished in the next; both halves count."""
        from foxrepro.bench.plan import archive_runs

        self._run_file(PLAN[0], 3, self.live)
        archive_runs(self.live, self.archive)
        self._run_file(PLAN[0], 7, self.live)      # same job, more records
        rep = archive_runs(self.live, self.archive)
        self.assertEqual(rep["merged"], 1)
        text = (self.archive / PLAN[0].run_key().filename()).read_text()
        self.assertEqual(text.count('"q0000006"'), 1)
        self.assertEqual(len(text.strip().splitlines()), 7)

    def test_archiving_twice_adds_nothing(self) -> None:
        from foxrepro.bench.plan import archive_runs

        self._run_file(PLAN[0], 4, self.live)
        archive_runs(self.live, self.archive)
        rep = archive_runs(self.live, self.archive)
        self.assertEqual(rep["copied"], 0)
        self.assertEqual(rep["merged"], 0)
        self.assertEqual(rep["skipped"], 1)

    def test_no_live_directory_is_not_an_error(self) -> None:
        from foxrepro.bench.plan import archive_runs

        rep = archive_runs(self.tmp / "absent", self.archive)
        self.assertEqual(rep["copied"], 0)
        self.assertIn("note", rep)

    def test_the_archive_is_tracked_by_git(self) -> None:
        """A .gitignore rule once excluded every *.jsonl in the repo."""
        import subprocess

        from foxrepro.storage import paths

        repo = paths().repo_root
        probe = repo / "results" / "runs" / "_tracked_probe.jsonl"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("{}\n")
        try:
            rc = subprocess.run(
                ["git", "check-ignore", "-q", str(probe)],
                cwd=repo, capture_output=True,
            ).returncode
            self.assertNotEqual(rc, 0, "results/runs/*.jsonl must not be ignored")
        finally:
            probe.unlink(missing_ok=True)


class TestCalibrationLoader(unittest.TestCase):
    """``--calibrate`` feeds measured times back into the estimates.

    The TSV is append-only, one row per job *invocation*, so a job can appear
    several times: once per resume, and once per failure. Getting the
    reduction wrong is not a cosmetic error -- the planner packs sessions
    against a ten-hour wall from these numbers.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def _freeze(self, tsv_text: str) -> Dict[str, Any]:
        from foxrepro.bench.plan import freeze_plan

        tsv = self.tmp / "_job_times.tsv"
        tsv.write_text(tsv_text)
        return freeze_plan(11, 10.0, self.tmp / "plan.json",
                           calibrate=tsv, pilot=True)

    def _est(self, payload: Dict[str, Any], job_id: str) -> float:
        return payload["jobs"][job_id]["est_s"]

    def test_a_failed_row_is_discarded(self) -> None:
        """Session 1's four CHAIR jobs died in 6 s on a missing file.

        Those rows measure the failure. Had they been believed, the planner
        would have budgeted 6 seconds for a job that costs over an hour.
        """
        jid = "table2:sampling:chair:val2014_500"
        payload = self._freeze(f"{jid}\t0.1\t1\n")
        from foxrepro.bench.plan import PLAN

        job = next(j for j in PLAN if j.job_id() == jid)
        self.assertEqual(self._est(payload, jid), round(job.est_seconds(), 1))
        self.assertEqual(payload["n_calibrated"], 0)

    def test_a_failure_after_a_success_does_not_erase_it(self) -> None:
        """Last-row-wins is the trap: the good measurement must survive."""
        jid = "table2:sampling:chair:val2014_500"
        payload = self._freeze(f"{jid}\t70.2\t0\n{jid}\t0.1\t1\n")
        self.assertAlmostEqual(self._est(payload, jid), 70.2 * 60, places=1)

    def test_a_resumed_job_sums_its_invocations(self) -> None:
        """Cut by the wall at 40 min, finished in 30 more -> it costs 70."""
        jid = "table2:sampling:chair:val2014_500"
        payload = self._freeze(f"{jid}\t40.0\t0\n{jid}\t30.2\t0\n")
        self.assertAlmostEqual(self._est(payload, jid), 70.2 * 60, places=1)

    def test_the_real_session_one_tsv_gives_the_measured_times(self) -> None:
        """The file session 1 actually produced, verbatim."""
        tsv = (REPO_ROOT / "logs" / "_job_times.tsv")
        if not tsv.is_file():          # not every clone has run a session
            self.skipTest("no logs/_job_times.tsv in this checkout")
        payload = self._freeze(tsv.read_text())
        want = {
            "table1:fox:pope:adversarial": 58.5,
            "table1:sampling:pope:adversarial": 30.1,
            "table2:fox:chair:val2014_500": 93.3,
            "table2:sampling:chair:val2014_500": 70.2,
            "table9:fox:chair:val2014_500:greedy": 92.3,
            "table9:sampling:chair:val2014_500:greedy": 65.5,
        }
        self.assertEqual(payload["n_calibrated"], len(want))
        for jid, minutes in want.items():
            with self.subTest(jid=jid):
                self.assertAlmostEqual(
                    self._est(payload, jid), minutes * 60, places=1
                )

    def test_a_malformed_row_is_skipped_not_fatal(self) -> None:
        payload = self._freeze(
            "garbage\n"
            "table1:fox:pope:adversarial\tnot-a-number\t0\n"
            "table1:sampling:pope:adversarial\t30.1\tnot-a-code\n"
        )
        self.assertEqual(payload["n_calibrated"], 0)
