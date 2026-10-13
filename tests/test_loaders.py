"""Tests for :mod:`data.loaders`: stable ids, deterministic order, hard errors."""

from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


from foxrepro.data.loaders import (  # noqa: E402
    PROMPT_TEMPLATES, describe_split, image_filename_for_id,
    image_id_from_filename, iter_samples, load_benchmark, load_chair, load_pope,
    pope_file_for_split, render_prompt, template_benchmark,
)
from tests._fixtures import make_workspace  # noqa: E402


class TestIdParsing(unittest.TestCase):
    def test_round_trip(self) -> None:
        for image_id in (1, 42, 131089, 581929):
            self.assertEqual(image_id_from_filename(image_filename_for_id(image_id)), image_id)

    def test_matches_reference_last_six_digit_parse(self) -> None:
        """The reference code uses int(name.split('.jpg')[0][-6:])."""
        for image_id in (1, 42, 131089, 581929):
            name = image_filename_for_id(image_id)
            self.assertEqual(
                image_id_from_filename(name), int(name.split(".jpg")[0][-6:])
            )

    def test_rejects_a_name_without_digits(self) -> None:
        with self.assertRaises(ValueError):
            image_id_from_filename("not_a_coco_image.jpg")


class TestPrompts(unittest.TestCase):
    def test_pope_template_needs_a_question(self) -> None:
        with self.assertRaises(ValueError):
            render_prompt("pope_v1")

    def test_chair_template_ignores_the_question(self) -> None:
        self.assertEqual(render_prompt("chair_v1", "ignored"),
                         PROMPT_TEMPLATES["chair_v1"])

    def test_unknown_template(self) -> None:
        with self.assertRaises(KeyError):
            render_prompt("nope")

    def test_template_benchmark_mapping(self) -> None:
        self.assertEqual(template_benchmark("pope_v1"), "pope")
        self.assertEqual(template_benchmark("chair_v2_short"), "chair")


class TestPopeLoader(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=8, n_per_image=6)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_counts_and_balance(self) -> None:
        samples = load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")
        info = describe_split(samples)
        self.assertEqual(info["n_samples"], 48)
        self.assertEqual(info["n_images"], 8)
        self.assertEqual(info["n_label_yes"], 24)
        self.assertEqual(info["n_label_no"], 24)

    def test_sample_ids_are_stable_under_file_reordering(self) -> None:
        path = pope_file_for_split(self.ws["pope_dir"], "random")
        original = [s.sample_id for s in
                    load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")]
        lines = path.read_text(encoding="utf-8").splitlines()
        random.Random(0).shuffle(lines)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        reordered = [s.sample_id for s in
                     load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")]
        self.assertEqual(original, reordered)
        self.assertEqual(original, sorted(original))

    def test_accepts_json_array_format(self) -> None:
        path = pope_file_for_split(self.ws["pope_dir"], "random")
        records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        array_path = self.ws["pope_dir"] / "coco_pope_popular.json"
        array_path.write_text(json.dumps(records), encoding="utf-8")
        samples = load_pope(self.ws["pope_dir"], self.ws["images_dir"], "popular")
        self.assertEqual(len(samples), 48)

    def test_missing_image_raises(self) -> None:
        next(Path(self.ws["images_dir"]).glob("*.jpg")).unlink()
        with self.assertRaises(FileNotFoundError):
            load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")

    def test_require_images_false_skips_the_check(self) -> None:
        samples = load_pope(self.ws["pope_dir"], Path("/nonexistent"), "random",
                            require_images=False)
        self.assertEqual(len(samples), 48)

    def test_duplicate_question_id_raises(self) -> None:
        path = pope_file_for_split(self.ws["pope_dir"], "random")
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines + [lines[0]]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")

    def test_non_binary_label_raises(self) -> None:
        path = pope_file_for_split(self.ws["pope_dir"], "random")
        bad = json.dumps({"question_id": 9999, "image": "COCO_val2014_000000001000.jpg",
                          "text": "Is there a dog?", "label": "maybe"})
        path.write_text(path.read_text(encoding="utf-8") + bad + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            load_pope(self.ws["pope_dir"], self.ws["images_dir"], "random")

    def test_missing_split_file_raises_with_the_names_it_looked_for(self) -> None:
        with self.assertRaises(FileNotFoundError) as ctx:
            load_pope(self.ws["pope_dir"], self.ws["images_dir"], "popular")
        self.assertIn("coco_popular.jsonl", str(ctx.exception))


class TestChairLoader(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=6)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sorted_by_image_id_with_stable_ids(self) -> None:
        samples = load_chair(self.ws["chair_list"], self.ws["images_dir"])
        self.assertEqual([s.image_id for s in samples], sorted(s.image_id for s in samples))
        self.assertEqual(samples[0].sample_id, f"chair/val2014_500/{samples[0].image_id:012d}")
        self.assertTrue(all(s.question is None for s in samples))
        self.assertTrue(all(s.label is None for s in samples))

    def test_accepts_filenames_and_bare_lists(self) -> None:
        path = Path(self._tmp.name) / "list.json"
        ids = [1000, 1007]
        path.write_text(json.dumps([image_filename_for_id(i) for i in ids]), encoding="utf-8")
        samples = load_chair(path, self.ws["images_dir"])
        self.assertEqual([s.image_id for s in samples], ids)

    def test_duplicate_ids_raise(self) -> None:
        path = Path(self._tmp.name) / "dupes.json"
        path.write_text(json.dumps([1000, 1000]), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_chair(path, self.ws["images_dir"])

    def test_prompt_is_attached(self) -> None:
        samples = load_chair(self.ws["chair_list"], self.ws["images_dir"],
                             prompt_template_id="chair_v2_short")
        self.assertEqual(samples[0].extra["caption_prompt"],
                         PROMPT_TEMPLATES["chair_v2_short"])


class TestDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = make_workspace(Path(self._tmp.name), n_images=4)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _load(self, benchmark: str, split: str, template: str):
        return load_benchmark(
            benchmark=benchmark, split=split, pope_dir=self.ws["pope_dir"],
            images_dir=self.ws["images_dir"], chair_image_list=self.ws["chair_list"],
            prompt_template_id=template,
        )

    def test_dispatch_pope_and_chair(self) -> None:
        self.assertEqual(len(self._load("pope", "random", "pope_v1")), 24)
        self.assertEqual(len(self._load("chair", "val2014_500", "chair_v1")), 4)

    def test_template_benchmark_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load("pope", "random", "chair_v1")

    def test_unknown_benchmark_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._load("mme", "x", "pope_v1")

    def test_iter_samples_skips_done_ids(self) -> None:
        samples = self._load("pope", "random", "pope_v1")
        skip = {samples[0].sample_id, samples[3].sample_id}
        kept = list(iter_samples(samples, skip))
        self.assertEqual(len(kept), len(samples) - 2)
        self.assertNotIn(samples[0].sample_id, {s.sample_id for s in kept})


if __name__ == "__main__":
    unittest.main(verbosity=2)
