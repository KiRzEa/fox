"""The direct COCO image fetcher. No network in these tests.

The one property worth protecting hardest is truncation detection. A connection
dropped mid-transfer leaves a file with a valid JPEG header and a missing tail,
which PIL will often open and silently return a partial image for -- so a
header-only check would let a half-downloaded image into a benchmark run and
corrupt the numbers in a way nothing downstream could detect.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from foxrepro.data.fetch_images import (
    COCO_IMAGE_URL, _is_complete_jpeg, fetch_images, needed_image_ids,
)
from foxrepro.data.loaders import image_filename_for_id

_SOI = b"\xff\xd8\xff"
_EOI = b"\xff\xd9"


def write_jpeg(path: Path, *, size: int = 4096, truncated: bool = False) -> None:
    body = _SOI + b"\x00" * max(size - 5, 0)
    path.write_bytes(body if truncated else body + _EOI)


class TestJpegValidation(unittest.TestCase):
    def setUp(self) -> None:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.tmp = Path(td.name)

    def test_complete_jpeg_accepted(self) -> None:
        f = self.tmp / "ok.jpg"
        write_jpeg(f)
        self.assertTrue(_is_complete_jpeg(f))

    def test_truncated_jpeg_rejected(self) -> None:
        """The failure mode that would silently corrupt a benchmark run."""
        f = self.tmp / "trunc.jpg"
        write_jpeg(f, truncated=True)
        self.assertFalse(_is_complete_jpeg(f))

    def test_wrong_magic_rejected(self) -> None:
        f = self.tmp / "nope.jpg"
        f.write_bytes(b"<html>404 not found</html>" + b"\x00" * 4096 + _EOI)
        self.assertFalse(_is_complete_jpeg(f))

    def test_tiny_file_rejected(self) -> None:
        f = self.tmp / "tiny.jpg"
        f.write_bytes(_SOI + _EOI)
        self.assertFalse(_is_complete_jpeg(f))

    def test_missing_file_rejected(self) -> None:
        self.assertFalse(_is_complete_jpeg(self.tmp / "absent.jpg"))


class TestNeededImageIds(unittest.TestCase):
    def test_pope_ids_come_from_the_shipped_annotations(self) -> None:
        """No network and no COCO download needed to answer this."""
        sets = needed_image_ids()
        self.assertEqual(len(sets["pope"]), 500)
        # POPE uses the same 500 images in all three splits.
        self.assertEqual(sets["union"] & sets["pope"], sets["pope"])
        self.assertTrue(all(isinstance(i, int) for i in sets["pope"]))

    def test_missing_pope_dir_yields_empty_rather_than_raising(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sets = needed_image_ids(pope_dir=Path(td), chair_list=Path(td) / "x.json")
            self.assertEqual(sets["pope"], set())
            self.assertEqual(sets["union"], set())

    def test_chair_list_is_read_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lst = Path(td) / "chair.json"
            lst.write_text(json.dumps([42, 1171, 42]))
            sets = needed_image_ids(pope_dir=Path(td), chair_list=lst)
            self.assertEqual(sets["chair"], {42, 1171})


class TestFetchSkipsExistingFiles(unittest.TestCase):
    """Resumability: a dead Kaggle session must cost only the in-flight image."""

    def test_present_images_are_skipped_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            ids = [42, 1171, 3845]
            for i in ids:
                write_jpeg(dest / image_filename_for_id(i))
            # If any download were attempted this would need the network and
            # the test would be slow or flaky; it is neither.
            report = fetch_images(ids, dest, workers=2, progress_every=0)
            self.assertEqual(report["skipped"], 3)
            self.assertEqual(report["downloaded"], 0)
            self.assertEqual(report["failed"], 0)

    def test_truncated_present_image_is_refetched_not_skipped(self) -> None:
        """A half-downloaded file must not be mistaken for a finished one."""
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td)
            write_jpeg(dest / image_filename_for_id(42), truncated=True)
            self.assertFalse(_is_complete_jpeg(dest / image_filename_for_id(42)))


class TestUrlShape(unittest.TestCase):
    def test_url_matches_coco_layout(self) -> None:
        name = image_filename_for_id(42)
        self.assertEqual(name, "COCO_val2014_000000000042.jpg")
        self.assertEqual(
            COCO_IMAGE_URL.format(name=name),
            "http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg",
        )


if __name__ == "__main__":
    unittest.main()


class TestShippedChairList(unittest.TestCase):
    """The CHAIR image list must travel with the package.

    Deriving the 500 ids needs instances_val2014.json, which is distributed
    only inside a 241 MB zip. Without a shipped list, every CHAIR job fails at
    startup in a fresh session -- which is what happened on the first real
    run, four jobs dead in forty seconds.
    """

    def test_the_list_is_shipped_and_resolvable(self) -> None:
        from foxrepro.storage import paths

        path = paths().chair_image_list
        self.assertTrue(
            path.is_file(),
            f"{path} is missing; it must ship in assets/ so a fresh session "
            f"needs no COCO download to run CHAIR",
        )

    def test_it_holds_exactly_500_distinct_ids(self) -> None:
        import json as _json

        from foxrepro.storage import paths

        payload = _json.loads(paths().chair_image_list.read_text())
        ids = payload["image_ids"] if isinstance(payload, dict) else payload
        self.assertEqual(len(ids), 500)
        self.assertEqual(len(set(ids)), 500)
        self.assertTrue(all(isinstance(i, int) and i > 0 for i in ids))

    def test_the_loader_accepts_it(self) -> None:
        from foxrepro.data.loaders import load_chair
        from foxrepro.storage import paths

        samples = load_chair(
            paths().chair_image_list, Path("/nonexistent"),
            require_images=False,
        )
        self.assertEqual(len(samples), 500)
        self.assertEqual(samples[0].benchmark, "chair")

    def test_a_missing_list_explains_itself(self) -> None:
        """A bare pathlib FileNotFoundError said nothing about the cause."""
        from foxrepro.data.loaders import load_chair

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError) as ctx:
                load_chair(Path(td) / "absent.json", Path(td),
                           require_images=False)
            msg = str(ctx.exception)
            self.assertIn("assets/chair_images.json", msg)
            self.assertIn("pip install -e .", msg)


class TestSessionAwareFetch(unittest.TestCase):
    """The data step and the run step must not be able to disagree.

    Session 1 mixes POPE and CHAIR jobs; the bootstrap fetched only POPE, and
    all four CHAIR jobs died within seconds. `--what auto` reads the frozen
    plan instead of trusting a hardcoded flag.
    """

    def test_auto_reads_the_plan(self) -> None:
        from foxrepro.data.fetch_images import _session_needs
        from foxrepro.storage import paths

        plan = paths().repo_root / "scripts" / "run_plan.json"
        if not plan.is_file():
            self.skipTest("no frozen plan in this checkout")
        what, _ = _session_needs(1, plan)
        self.assertIn(what, ("pope", "chair", "union"))

    def test_an_unknown_session_fetches_everything(self) -> None:
        """Guessing wrong costs a whole session; over-fetching costs minutes."""
        from foxrepro.data.fetch_images import _session_needs

        self.assertEqual(_session_needs(99, Path("/nonexistent")), ("union", None))
        self.assertEqual(_session_needs(None, None), ("union", None))

    def test_a_session_fetches_only_the_chair_draws_it_runs(self) -> None:
        """Every Kaggle session starts from an empty disk.

        There is no carry-over between sessions, so over-fetching is not paid
        once -- it is paid again in every session. With five CHAIR image lists
        in the package, unioning all of them would make a session that scores
        500 images download 2452.
        """
        import json

        from foxrepro.data.fetch_images import _session_needs, needed_image_ids
        from foxrepro.storage import paths

        p = paths()
        plan = p.repo_root / "scripts" / "run_plan.json"
        if not plan.is_file():
            self.skipTest("no frozen plan in this checkout")
        doc = json.loads(plan.read_text())

        everything = needed_image_ids(chair_splits=None)["chair"]
        for sess in doc["sessions"]:
            n = int(sess["session"])
            want = {doc["jobs"][j]["id"].split(":")[3]
                    for j in sess["jobs"]
                    if doc["jobs"][j]["id"].split(":")[2] == "chair"}
            what, splits = _session_needs(n, plan)
            with self.subTest(session=n):
                self.assertEqual(set(splits or ()), want)
                got = needed_image_ids(chair_splits=splits)["chair"]
                # The union of the lists this session actually scores. Draws
                # are independent samples of ~40k images, so they overlap by
                # a handful and the union is slightly under 500 per draw.
                expect = set()
                for sp in want:
                    expect |= set(json.loads(
                        p.chair_image_list_for(sp).read_text())["image_ids"])
                self.assertEqual(got, expect)
                self.assertLessEqual(len(got), len(everything))
                if want:
                    self.assertGreater(len(got), 450 * len(want))

    def test_the_benchmarks_and_the_draws_are_decided_separately(self) -> None:
        """A POPE-only session must not be handed a CHAIR list."""
        import json

        from foxrepro.data.fetch_images import _session_needs
        from foxrepro.storage import paths

        plan = paths().repo_root / "scripts" / "run_plan.json"
        if not plan.is_file():
            self.skipTest("no frozen plan in this checkout")
        doc = json.loads(plan.read_text())
        for sess in doc["sessions"]:
            n = int(sess["session"])
            benches = {doc["jobs"][j]["id"].split(":")[2] for j in sess["jobs"]}
            what, splits = _session_needs(n, plan)
            with self.subTest(session=n):
                if benches == {"pope"}:
                    self.assertEqual(what, "pope")
                    self.assertEqual(splits, [])
                elif benches == {"chair"}:
                    self.assertEqual(what, "chair")
                    self.assertTrue(splits)
                else:
                    self.assertEqual(what, "union")
                    self.assertTrue(splits)


class TestDiskProbe(unittest.TestCase):
    """The checkpoint and the run files land on different filesystems.

    On Kaggle that distinction has a limit attached: /kaggle/working is capped
    at 20 GB because it is the notebook's output, while the HuggingFace cache
    goes to /kaggle/temp. A 30 GB checkpoint is only a problem if the cache
    root cannot hold it, and finding that out ten hours in costs ten hours --
    the same lesson as the missing GPU.
    """

    def _probe(self, model_id: str) -> dict:
        import json as _json
        import subprocess
        import sys as _sys

        from foxrepro.storage import paths

        proc = subprocess.run(
            [_sys.executable, str(paths().repo_root / "scripts" / "probe_host.py"),
             "--json", "--model-id", model_id],
            capture_output=True, text=True, timeout=600,
        )
        doc = _json.loads(proc.stdout)
        return next(c for c in doc["checks"] if c["check"] == "disk")

    def test_it_reports_the_three_roots_separately(self) -> None:
        d = self._probe("llava-hf/llava-1.5-7b-hf")
        self.assertEqual(
            set(d["roots"]),
            {"hf_cache (checkpoint)", "runs + reports", "images"},
        )
        for name, info in d["roots"].items():
            with self.subTest(root=name):
                self.assertIn("free_gib", info)
                self.assertGreater(info["total_gib"], 0)

    def test_instructblip_is_reported_as_the_larger_download(self) -> None:
        """It is stored fp32, so it transfers at ~2x LLaVA for ~1x the VRAM."""
        small = self._probe("llava-hf/llava-1.5-7b-hf")
        large = self._probe("Salesforce/instructblip-vicuna-7b")
        self.assertGreater(large["checkpoint_download_gib"],
                           1.8 * small["checkpoint_download_gib"])

    def test_an_unknown_model_falls_back_rather_than_crashing(self) -> None:
        d = self._probe("some/unlisted-model")
        self.assertGreater(d["checkpoint_download_gib"], 0)
        self.assertIn(d["status"], ("PASS", "WARN", "FAIL"))

    def test_the_gpu_check_follows_the_model_id_too(self) -> None:
        """A probe that passes for the wrong model is worse than no probe.

        --model-id was added for the disk check first, and check_gpus kept
        using LLaVA's VRAM figures whatever was named -- so probing for
        InstructBLIP would have reported a margin computed for a different
        checkpoint.
        """
        import json as _json
        import subprocess
        import sys as _sys

        import torch

        from foxrepro.storage import paths

        if not torch.cuda.is_available():
            self.skipTest("the gpus check only runs with CUDA present")

        def gpus(model_id):
            proc = subprocess.run(
                [_sys.executable,
                 str(paths().repo_root / "scripts" / "probe_host.py"),
                 "--json", "--model-id", model_id],
                capture_output=True, text=True, timeout=600,
            )
            doc = _json.loads(proc.stdout)
            return next(c for c in doc["checks"] if c["check"] == "gpus")

        small = gpus("llava-hf/llava-1.5-7b-hf")
        large = gpus("Salesforce/instructblip-vicuna-7b")
        self.assertGreater(large["weights_gib_fp16"], small["weights_gib_fp16"])
        self.assertGreater(large["est_per_gpu_need_gib"],
                           small["est_per_gpu_need_gib"])
        self.assertFalse(large["memory_model_estimated"])

    def test_an_unlisted_model_says_its_vram_figures_are_borrowed(self) -> None:
        from foxrepro.bench import report  # noqa: F401  (keeps import style)

        import importlib.util
        import sys as _sys

        from foxrepro.storage import paths

        spec = importlib.util.spec_from_file_location(
            "_probe", paths().repo_root / "scripts" / "probe_host.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules["_probe"] = mod
        spec.loader.exec_module(mod)

        known = mod._memory_model("llava-hf/llava-1.5-7b-hf")
        guess = mod._memory_model("some/unlisted-model")
        self.assertFalse(known["estimated"])
        self.assertTrue(guess["estimated"])
        self.assertEqual(guess["weights_fp16"], known["weights_fp16"])
