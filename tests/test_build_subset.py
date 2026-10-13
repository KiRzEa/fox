"""Tests for :mod:`data.build_coco_subset` (the one-time archive builder).

Uses a synthetic "val2014" of a few dozen tiny JPEGs, so the build/extract/
verify cycle is exercised without the 6 GB download.
"""

from __future__ import annotations

import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List


from foxrepro.data.build_coco_subset import (  # noqa: E402
    build_subset, extract_subset, pope_image_ids, sample_chair_image_ids,
    verify_subset,
)
from foxrepro.data.loaders import image_filename_for_id, load_chair, load_pope  # noqa: E402
from tests._fixtures import CATEGORIES, write_pope_split  # noqa: E402


def _write_full_val2014(root: Path, image_ids: List[int]) -> Dict[str, Path]:
    """Create a fake full val2014 download: images + annotation files."""
    images = root / "val2014"
    anns = root / "annotations"
    images.mkdir(parents=True, exist_ok=True)
    anns.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        # distinct content per image so checksums differ
        (images / image_filename_for_id(image_id)).write_bytes(
            b"\xff\xd8\xff" + str(image_id).encode() * 32 + b"\xff\xd9"
        )
    instances = {
        "info": {"year": 2014},
        "licenses": [],
        "categories": [{"id": c, "name": n, "supercategory": "x"} for c, n in CATEGORIES],
        "images": [
            {"id": i, "file_name": image_filename_for_id(i), "height": 1, "width": 1}
            for i in image_ids
        ],
        "annotations": [
            {"id": j, "image_id": i, "category_id": 18, "iscrowd": 0, "area": 1.0}
            for j, i in enumerate(image_ids, start=1)
        ],
    }
    captions = {
        "info": {"year": 2014},
        "licenses": [],
        "images": instances["images"],
        "annotations": [
            {"id": j, "image_id": i, "caption": "A dog on a couch."}
            for j, i in enumerate(image_ids, start=1)
        ],
    }
    (anns / "instances_val2014.json").write_text(json.dumps(instances), encoding="utf-8")
    (anns / "captions_val2014.json").write_text(json.dumps(captions), encoding="utf-8")
    return {"images": images, "annotations": anns}


class TestBuildSubset(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.all_ids = [1000 + 3 * i for i in range(40)]
        dirs = _write_full_val2014(self.root, self.all_ids)
        self.images_dir = dirs["images"]
        self.ann_dir = dirs["annotations"]
        self.pope_dir = self.root / "POPE"
        self.pope_ids = self.all_ids[:10]
        for split in ("random", "popular", "adversarial"):
            write_pope_split(self.pope_dir, split, self.pope_ids, n_per_image=6)
        self.out_tar = self.root / "out" / "subset.tar"
        self.out_manifest = self.root / "out" / "subset.manifest.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _build(self, **over: object):
        kwargs: Dict[str, object] = dict(
            val2014_dir=self.images_dir, annotations_dir=self.ann_dir,
            pope_dir=self.pope_dir, out_tar=self.out_tar,
            out_manifest=self.out_manifest, chair_sample_n=5, chair_seed=0,
        )
        kwargs.update(over)
        return build_subset(**kwargs)  # type: ignore[arg-type]

    def test_pope_image_ids_per_split(self) -> None:
        ids = pope_image_ids(self.pope_dir)
        self.assertEqual(set(ids), {"random", "popular", "adversarial"})
        self.assertEqual(ids["random"], set(self.pope_ids))

    def test_chair_sampling_is_deterministic(self) -> None:
        inst = self.ann_dir / "instances_val2014.json"
        a = sample_chair_image_ids(inst, 5, seed=0)
        b = sample_chair_image_ids(inst, 5, seed=0)
        c = sample_chair_image_ids(inst, 5, seed=1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(a, sorted(a))
        self.assertTrue(set(a) <= set(self.all_ids))

    def test_chair_sampling_rejects_too_large_n(self) -> None:
        with self.assertRaises(ValueError):
            sample_chair_image_ids(self.ann_dir / "instances_val2014.json", 10**6, 0)

    def test_build_contains_only_needed_images(self) -> None:
        result = self._build()
        with tarfile.open(result.tar_path) as tar:
            names = tar.getnames()
        jpgs = [n for n in names if n.startswith("images/")]
        self.assertEqual(len(jpgs), result.n_images)
        self.assertLessEqual(result.n_images, 15)  # 10 POPE + <=5 CHAIR
        self.assertGreaterEqual(result.n_images, 10)
        for required in (
            "pope/coco_pope_random.json", "pope/coco_pope_popular.json",
            "pope/coco_pope_adversarial.json", "chair_images.json",
            "BUILD_INFO.json", "annotations/instances_val2014_subset.json",
            "annotations/captions_val2014_subset.json",
        ):
            self.assertIn(required, names)

    def test_build_is_byte_reproducible(self) -> None:
        first = self._build()
        sha_first = first.tar_sha256
        second = self._build()
        self.assertEqual(sha_first, second.tar_sha256)

    def test_manifest_records_checksums_and_counts(self) -> None:
        result = self._build()
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["tar_sha256"], result.tar_sha256)
        self.assertEqual(manifest["n_images"], result.n_images)
        self.assertEqual(manifest["tar_bytes"], result.tar_path.stat().st_size)
        self.assertEqual(len(manifest["members"]), manifest["n_members"])
        for meta in manifest["members"].values():
            self.assertEqual(len(meta["sha256"]), 64)
        self.assertIn("chair_source", manifest["build_info"])

    def test_annotation_subset_is_filtered(self) -> None:
        result = self._build()
        with tarfile.open(result.tar_path) as tar:
            payload = json.loads(
                tar.extractfile("annotations/instances_val2014_subset.json").read()
            )
        self.assertEqual(len(payload["images"]), result.n_images)
        self.assertEqual(len(payload["categories"]), len(CATEGORIES))
        self.assertTrue(all(
            a["image_id"] in {i["id"] for i in payload["images"]}
            for a in payload["annotations"]
        ))

    def test_missing_image_is_reported_not_skipped(self) -> None:
        (self.images_dir / image_filename_for_id(self.pope_ids[0])).unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            self._build()
        self.assertIn("val2014", str(ctx.exception))

    def test_extract_verify_and_load(self) -> None:
        result = self._build()
        dest = self.root / "local" / "data"
        first = extract_subset(result.tar_path, dest, result.manifest_path)
        self.assertTrue(first["extracted"])
        again = extract_subset(result.tar_path, dest, result.manifest_path)
        self.assertFalse(again["extracted"], "extraction is not idempotent")

        for level in ("fast", "full"):
            report = verify_subset(dest, result.manifest_path, level=level)
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["n_missing"], 0)

        # the loaders must work off the extracted tree
        samples = load_pope(dest / "pope", dest / "images", "random")
        self.assertEqual(len(samples), 60)
        chair = load_chair(dest / "chair_images.json", dest / "images")
        self.assertEqual(len(chair), 5)
        self.assertTrue(all(s.image_path.is_file() for s in chair))

    def test_verify_detects_a_truncated_file(self) -> None:
        result = self._build()
        dest = self.root / "local" / "data"
        extract_subset(result.tar_path, dest, result.manifest_path)
        victim = next((dest / "images").glob("*.jpg"))
        victim.write_bytes(b"\xff\xd8")
        self.assertFalse(verify_subset(dest, result.manifest_path, "fast")["ok"])

    def test_verify_detects_a_same_size_corruption(self) -> None:
        result = self._build()
        dest = self.root / "local" / "data"
        extract_subset(result.tar_path, dest, result.manifest_path)
        victim = next((dest / "images").glob("*.jpg"))
        data = bytearray(victim.read_bytes())
        data[5] ^= 0xFF
        victim.write_bytes(bytes(data))
        self.assertTrue(verify_subset(dest, result.manifest_path, "fast")["ok"])
        full = verify_subset(dest, result.manifest_path, "full")
        self.assertFalse(full["ok"])
        self.assertEqual(len(full["bad_hash"]), 1)

    def test_extract_rejects_a_tar_that_disagrees_with_the_manifest(self) -> None:
        result = self._build()
        with open(result.tar_path, "r+b") as fh:
            fh.truncate(result.tar_path.stat().st_size - 1024)
        with self.assertRaises(ValueError):
            extract_subset(result.tar_path, self.root / "x", result.manifest_path)

    def test_chair_from_pope_shrinks_the_archive(self) -> None:
        shared = self._build(chair_from_pope=True)
        self.assertEqual(shared.n_images, len(self.pope_ids))

    def test_explicit_chair_list_is_frozen_into_the_tar(self) -> None:
        listing = self.root / "my_chair.json"
        wanted = self.all_ids[20:24]
        listing.write_text(json.dumps(wanted), encoding="utf-8")
        result = self._build(chair_image_list=listing)
        with tarfile.open(result.tar_path) as tar:
            payload = json.loads(tar.extractfile("chair_images.json").read())
        self.assertEqual(payload["image_ids"], sorted(wanted))
        self.assertIn("explicit:", payload["source"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
