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
