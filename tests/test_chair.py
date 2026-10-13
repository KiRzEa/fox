"""Tests for :mod:`bench.chair`, including hand-calculated CHAIR_S / CHAIR_I.

Hand calculation (see ``TestHandCalculatedChair``):

================  =================================  ==========  ===========
image / GT        caption                            mentions    hallucinated
================  =================================  ==========  ===========
1 {dog, person,   "A man and his dog sit on a couch   person,     cat
  couch}           next to a cat."                    dog,
                                                      couch, cat
2 {cat}           "A cat and a dog."                  cat, dog    dog
3 {zebra}         "Two zebras standing in a field."   zebra       --
================  =================================  ==========  ===========

CHAIR_S = 2 captions with a hallucination / 3 captions   = 0.6667
CHAIR_I = 2 hallucinated mentions / 7 total mentions     = 0.2857
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


from foxrepro.bench.chair import CHAIRScorer, load_synonyms, singularize  # noqa: E402
from foxrepro.storage import paths  # noqa: E402
from tests._fixtures import write_annotations  # noqa: E402

SYNONYMS = paths().chair_synonyms


class TestSynonymFile(unittest.TestCase):
    def test_ships_eighty_categories(self) -> None:
        groups, digest, provenance = load_synonyms(SYNONYMS)
        self.assertEqual(len(groups), 80)
        self.assertEqual(len(digest), 64)
        self.assertIn("rohrbach", provenance.lower())

    def test_canonical_names_are_the_coco_category_names(self) -> None:
        groups, _, _ = load_synonyms(SYNONYMS)
        names = {g[0] for g in groups}
        for expected in (
            "person", "traffic light", "fire hydrant", "sports ball",
            "baseball glove", "wine glass", "hot dog", "potted plant",
            "dining table", "cell phone", "teddy bear", "hair drier", "tv",
        ):
            self.assertIn(expected, names)

    def test_no_surface_form_claimed_twice(self) -> None:
        # load_synonyms raises on a cross-category duplicate; this pins it.
        groups, _, _ = load_synonyms(SYNONYMS)
        surfaces = [s for g in groups for s in g]
        self.assertEqual(len(surfaces), len(set(surfaces)))


class TestSingularize(unittest.TestCase):
    CASES = {
        "dogs": "dog", "zebras": "zebra", "horses": "horse", "gloves": "glove",
        "microwaves": "microwave", "buses": "bus", "glasses": "glass",
        "benches": "bench", "sandwiches": "sandwich", "knives": "knife",
        "men": "man", "women": "woman", "people": "person", "children": "child",
        "mice": "mouse", "babies": "baby", "ties": "tie", "vases": "vase",
        "toothbrushes": "toothbrush", "sheep": "sheep", "scissors": "scissors",
        "skis": "skis", "broccoli": "broccoli", "grass": "grass",
        "tennis": "tennis", "bus": "bus", "is": "is", "sizes": "size",
        "couches": "couch", "boxes": "box", "potatoes": "potato",
    }

    def test_vocabulary(self) -> None:
        for plural, singular in self.CASES.items():
            with self.subTest(word=plural):
                self.assertEqual(singularize(plural), singular)


class _ScorerCase(unittest.TestCase):
    """Base class that builds a scorer over synthetic annotations."""

    instances: dict = {}
    captions: dict = {}

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        inst_path, cap_path = write_annotations(root, self.instances, self.captions)
        self.scorer = CHAIRScorer(
            synonyms_path=SYNONYMS, instances_json=inst_path,
            captions_json=cap_path, gt_source="both",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestHandCalculatedChair(_ScorerCase):
    instances = {1: ["dog", "person"], 2: ["cat"], 3: ["zebra"]}
    captions = {1: ["A dog and a person on a couch."], 2: ["A cat."], 3: ["A zebra."]}

    def test_ground_truth_unions_instances_and_captions(self) -> None:
        self.assertEqual(self.scorer.imid_to_objects[1], {"dog", "person", "couch"})
        self.assertEqual(self.scorer.imid_to_objects[2], {"cat"})

    def test_per_caption_scores(self) -> None:
        s1 = self.scorer.score_caption("a", 1, "A man and his dog sit on a couch next to a cat.")
        self.assertEqual(s1.mentions, ["person", "dog", "couch", "cat"])
        self.assertEqual(s1.hallucinated, ["cat"])
        self.assertEqual((s1.n_mentions, s1.n_hallucinated), (4, 1))

        s2 = self.scorer.score_caption("b", 2, "A cat and a dog.")
        self.assertEqual(s2.mentions, ["cat", "dog"])
        self.assertEqual(s2.hallucinated, ["dog"])

        s3 = self.scorer.score_caption("c", 3, "Two zebras standing in a field.")
        self.assertEqual(s3.mentions, ["zebra"])
        self.assertEqual(s3.hallucinated, [])
        self.assertFalse(s3.has_hallucination)

    def test_totals_match_hand_calculation(self) -> None:
        scores = self.scorer.score_many([
            ("a", 1, "A man and his dog sit on a couch next to a cat."),
            ("b", 2, "A cat and a dog."),
            ("c", 3, "Two zebras standing in a field."),
        ])
        totals = self.scorer.totals(scores)
        self.assertEqual(totals.n_captions, 3)
        self.assertEqual(totals.n_captions_with_hallucination, 2)
        self.assertEqual(totals.n_mentions, 7)
        self.assertEqual(totals.n_hallucinated_mentions, 2)
        self.assertAlmostEqual(totals.chair_s, 2 / 3, places=12)
        self.assertAlmostEqual(totals.chair_i, 2 / 7, places=12)

    def test_repeated_mentions_count_once_each(self) -> None:
        """CHAIR_I counts mentions, not distinct categories."""
        score = self.scorer.score_caption("d", 2, "A dog and another dog and a dog.")
        self.assertEqual(score.n_mentions, 3)
        self.assertEqual(score.n_hallucinated, 3)

    def test_empty_caption(self) -> None:
        score = self.scorer.score_caption("e", 1, "")
        self.assertEqual((score.n_mentions, score.n_words), (0, 0))
        totals = self.scorer.totals([score])
        self.assertEqual(totals.n_empty_captions, 1)
        self.assertEqual(totals.chair_s, 0.0)
        self.assertTrue(totals.chair_i != totals.chair_i)  # nan: no mentions

    def test_unknown_image_is_an_error_not_a_zero(self) -> None:
        with self.assertRaises(KeyError):
            self.scorer.score_caption("x", 999999, "A dog.")

    def test_config_records_synonym_checksum(self) -> None:
        cfg = self.scorer.config()
        self.assertEqual(len(cfg["synonyms_sha256"]), 64)
        self.assertEqual(cfg["n_categories"], 80)
        self.assertEqual(cfg["gt_source"], "both")


class TestReferenceHardRules(_ScorerCase):
    instances = {
        10: ["hot dog", "dining table"], 11: ["toilet"], 12: ["elephant"],
        13: ["person"], 14: ["car"],
    }
    captions = {}

    def test_double_words_are_single_objects(self) -> None:
        score = self.scorer.score_caption(
            "a", 10, "A hot dog on a dining table next to a wine glass."
        )
        self.assertEqual(score.mentions, ["hot dog", "dining table", "wine glass"])
        self.assertEqual(score.hallucinated, ["wine glass"])

    def test_toilet_seat_is_not_a_chair(self) -> None:
        score = self.scorer.score_caption("b", 11, "The seat of the toilet is white.")
        self.assertEqual(score.mentions, ["toilet"])
        self.assertEqual(score.hallucinated, [])

    def test_seat_alone_is_a_chair(self) -> None:
        score = self.scorer.score_caption("c", 13, "A person sits on a seat.")
        self.assertEqual(score.mentions, ["person", "chair"])

    def test_baby_animal_maps_to_the_animal(self) -> None:
        score = self.scorer.score_caption("d", 12, "A baby elephant walks.")
        self.assertEqual(score.mentions, ["elephant"])

    def test_bow_tie_is_a_tie(self) -> None:
        score = self.scorer.score_caption("e", 13, "A man wearing a bow tie.")
        self.assertEqual(score.mentions, ["person", "tie"])

    def test_synonyms_map_to_canonical_category(self) -> None:
        score = self.scorer.score_caption("f", 14, "A taxi and a suv and a puppy.")
        self.assertEqual(score.mentions, ["car", "car", "dog"])
        self.assertEqual(score.hallucinated, ["dog"])


class TestGtSourceMatters(unittest.TestCase):
    """Dropping reference captions from the ground truth inflates CHAIR."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.inst, self.caps = write_annotations(
            root, instances={1: ["dog"]}, captions={1: ["A dog on a couch."]}
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_instances_only_counts_the_couch_as_hallucinated(self) -> None:
        caption = "A dog on a couch."
        both = CHAIRScorer(SYNONYMS, self.inst, self.caps, gt_source="both")
        inst_only = CHAIRScorer(SYNONYMS, self.inst, None, gt_source="instances")
        self.assertEqual(both.score_caption("a", 1, caption).hallucinated, [])
        self.assertEqual(inst_only.score_caption("a", 1, caption).hallucinated, ["couch"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
