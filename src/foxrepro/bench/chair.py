"""CHAIR (Caption Hallucination Assessment with Image Relevance) scorer.

This is a from-scratch implementation of the procedure in Rohrbach et al.,
"Object Hallucination in Image Captioning" (EMNLP 2018), reproducing the
behaviour of the reference ``chair.py``:

1. Lower-case and tokenize the caption; singularize every token.
2. Merge known two-word objects (``hot dog``, ``teddy bear``, ``traffic
   light``, ...) and apply the reference script's hard-coded rules
   (``baby <animal>`` -> ``<animal>``, ``passenger train`` -> ``train``,
   ``bow tie`` -> ``tie``, ``toilet seat`` -> ``toilet``, and the rule that
   drops ``seat`` when ``toilet`` is also present so that "the seat of the
   toilet" is not counted as a *chair*).
3. Keep the tokens that are MS-COCO objects or synonyms thereof and map each
   to its canonical category ("node word"). **Repeated mentions are kept**:
   CHAIR_I counts mentions, not distinct categories.
4. Ground-truth objects for an image are the union of (a) the categories of
   its ``instances_val2014`` segmentation annotations and (b) the objects
   mentioned in its five reference captions, both mapped through the same
   synonym table.
5. ``CHAIR_S`` = captions with at least one hallucinated mention / captions.
   ``CHAIR_I`` = hallucinated mentions / **all** mentions.

Denominators
------------
Every count that goes into a metric is returned explicitly:
``n_captions`` for CHAIR_S and ``n_mentions`` for CHAIR_I. The Fox paper does
not state ``n_mentions``, and it cannot be recovered from a published CHAIR_I
value, so it is reported everywhere in this repository.

Known deviations from the reference script (recorded in every result):
* The reference uses ``nltk.word_tokenize`` and ``pattern.en.singularize``.
  Both are heavyweight dependencies and the CPU side of this project must run
  on a bare laptop, so a regex tokenizer and a rule-based singularizer are
  used instead. They agree with the reference on the COCO object vocabulary
  (that agreement is what ``tests/test_chair.py`` pins down). Pass
  ``tokenizer="nltk"`` to use nltk when it happens to be installed.
* ``CHAIR_S`` is computed per generated *caption* (= per response), which is
  what the reference script does, even though the paper's prose says
  "sentences".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "CHAIRScorer",
    "CaptionScore",
    "ChairTotals",
    "singularize",
    "load_synonyms",
]

# --------------------------------------------------------------------------- #
# singularization
# --------------------------------------------------------------------------- #
#: Irregular plurals that matter for the COCO object vocabulary.
_IRREGULAR: Dict[str, str] = {
    "men": "man", "women": "woman", "children": "child", "people": "person",
    "mice": "mouse", "geese": "goose", "oxen": "ox", "teeth": "tooth",
    "feet": "foot", "knives": "knife", "leaves": "leaf", "loaves": "loaf",
    "wolves": "wolf", "shelves": "shelf", "halves": "half", "lives": "life",
    "scarves": "scarf", "calves": "calf", "elves": "elf", "thieves": "thief",
    "hooves": "hoof", "cacti": "cactus", "fungi": "fungus",
    "benches": "bench", "sandwiches": "sandwich", "buses": "bus",
    "busses": "bus", "glasses": "glass", "dishes": "dish", "boxes": "box",
    "couches": "couch", "watches": "watch", "sheep": "sheep", "fish": "fish",
    "deer": "deer", "moose": "moose", "aircraft": "aircraft",
    "broccoli": "broccoli", "scissors": "scissors", "skis": "skis",
    "species": "species", "series": "series", "jeans": "jeans",
    "shoes": "shoe", "gases": "gas", "noses": "nose",
}

#: Words that already look singular despite the trailing "s"
#: ("glass", "bus", "tennis"). Deliberately excludes "as"/"os", which would
#: otherwise freeze "zebras" and "pianos".
_S_INVARIANT_SUFFIXES = ("ss", "us", "is")

#: Sibilant stems that take "-es". "ses" and "zes" are NOT listed: they would
#: mangle "horses" -> "hors" and "sizes" -> "siz". The handful of genuine
#: "-ses" plurals in this vocabulary ("buses", "glasses", "dishes") are in
#: :data:`_IRREGULAR` instead.
_ES_SUFFIXES = ("sses", "shes", "ches", "xes")


def singularize(word: str) -> str:
    """Rule-based singularizer sufficient for the COCO object vocabulary.

    Deterministic and dependency-free. Not a general-purpose English
    singularizer; it is tuned to leave non-nouns alone and to map every plural
    form of a COCO object or synonym onto its singular.

    Args:
        word: a lower-case token.

    Returns:
        The singular form, or ``word`` unchanged.
    """
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if len(word) <= 3 or not word.endswith("s"):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    for suffix in _ES_SUFFIXES:
        if word.endswith(suffix):
            return word[: -len("es")]
    # No general "-ves -> -f" rule: it would turn "gloves" into "glof" and
    # "microwaves" into "microwaf". The real "-ves" plurals are irregulars.
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]
    if word.endswith(_S_INVARIANT_SUFFIXES):
        return word
    return word[:-1]


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def _tokenize(text: str, tokenizer: str = "regex") -> List[str]:
    """Tokenize a caption. ``tokenizer`` is ``"regex"`` (default) or ``"nltk"``."""
    lowered = text.lower()
    if tokenizer == "nltk":
        import nltk  # imported lazily; never required

        return nltk.word_tokenize(lowered)
    if tokenizer != "regex":
        raise ValueError(f"unknown tokenizer {tokenizer!r}")
    return _TOKEN_RE.findall(lowered)


# --------------------------------------------------------------------------- #
# synonym table
# --------------------------------------------------------------------------- #
def load_synonyms(path: Path) -> Tuple[List[List[str]], str, str]:
    """Load the synonym list.

    Args:
        path: a file in the format documented in ``data/chair_synonyms.txt``.

    Returns:
        ``(groups, sha256, provenance)`` where ``groups[i][0]`` is a canonical
        COCO category name and ``groups[i][1:]`` its synonyms; ``sha256`` is
        the digest of the file as loaded; ``provenance`` is the value of the
        ``# provenance:`` header line, or ``"unknown"``.

    Raises:
        ValueError: duplicate surface form across two categories, or a file
            that does not contain exactly 80 categories.
    """
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    provenance = "unknown"
    groups: List[List[str]] = []
    for line in raw.decode("utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if "provenance:" in stripped:
                provenance = stripped.split("provenance:", 1)[1].strip()
            continue
        fields = [f.strip().lower() for f in stripped.split(",") if f.strip()]
        if fields:
            # drop within-line duplicates, preserving order
            seen: Set[str] = set()
            groups.append([f for f in fields if not (f in seen or seen.add(f))])
    if len(groups) != 80:
        raise ValueError(f"{path} defines {len(groups)} categories, expected 80")
    owner: Dict[str, str] = {}
    for group in groups:
        for surface in group:
            if surface in owner and owner[surface] != group[0]:
                raise ValueError(
                    f"{path}: surface form {surface!r} claimed by both "
                    f"{owner[surface]!r} and {group[0]!r}"
                )
            owner[surface] = group[0]
    return groups, digest, provenance


# ---- hard-coded rules transcribed from the reference chair.py -------------- #
_COCO_DOUBLE_WORDS = (
    "motor bike", "motor cycle", "air plane", "traffic light", "street light",
    "traffic signal", "stop light", "fire hydrant", "stop sign",
    "parking meter", "suit case", "sports ball", "baseball bat",
    "baseball glove", "tennis racket", "wine glass", "hot dog", "cell phone",
    "mobile phone", "teddy bear", "hair drier", "potted plant", "bow tie",
    "laptop computer", "stove top oven", "home plate", "train track",
    "dining table",
)
_ANIMAL_WORDS = (
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "animal", "cub",
)
_VEHICLE_WORDS = ("jet", "train")


def _build_double_word_dict() -> Dict[str, str]:
    """Two-word phrases that collapse to a single object word."""
    table: Dict[str, str] = {w: w for w in _COCO_DOUBLE_WORDS}
    for animal in _ANIMAL_WORDS:
        table[f"baby {animal}"] = animal
        table[f"adult {animal}"] = animal
    for vehicle in _VEHICLE_WORDS:
        table[f"passenger {vehicle}"] = vehicle
    table["bow tie"] = "tie"
    table["toilet seat"] = "toilet"
    table["wine glas"] = "wine glass"
    return table


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #
@dataclass
class CaptionScore:
    """Per-caption CHAIR result; the unit the paired bootstrap resamples.

    Attributes:
        sample_id: run-level sample id.
        image_id: COCO image id.
        mentions: canonical category per mention, in order, duplicates kept.
        hallucinated: the subset of ``mentions`` not in the ground truth.
        gt_objects: ground-truth categories for the image.
        n_mentions: ``len(mentions)`` -- CHAIR_I's per-caption denominator.
        n_hallucinated: ``len(hallucinated)``.
        has_hallucination: whether this caption counts towards CHAIR_S.
        n_words: caption length in tokens (used to detect degenerate outputs).
    """

    sample_id: str
    image_id: int
    mentions: List[str] = field(default_factory=list)
    hallucinated: List[str] = field(default_factory=list)
    gt_objects: List[str] = field(default_factory=list)
    n_words: int = 0

    @property
    def n_mentions(self) -> int:
        return len(self.mentions)

    @property
    def n_hallucinated(self) -> int:
        return len(self.hallucinated)

    @property
    def has_hallucination(self) -> bool:
        return bool(self.hallucinated)

    @property
    def n_correct_distinct(self) -> int:
        """Distinct ground-truth categories correctly mentioned (for recall)."""
        return len(set(self.mentions) & set(self.gt_objects))

    def as_dict(self) -> Dict[str, object]:
        """JSON-ready per-caption record."""
        return {
            "sample_id": self.sample_id, "image_id": self.image_id,
            "mentions": self.mentions, "hallucinated": self.hallucinated,
            "gt_objects": sorted(self.gt_objects),
            "n_mentions": self.n_mentions, "n_hallucinated": self.n_hallucinated,
            "has_hallucination": self.has_hallucination, "n_words": self.n_words,
        }


@dataclass
class ChairTotals:
    """Aggregated CHAIR counts with every denominator spelled out."""

    n_captions: int
    n_captions_with_hallucination: int
    n_mentions: int
    n_hallucinated_mentions: int
    n_distinct_correct: int
    n_gt_objects: int
    n_distinct_mentions: int
    n_empty_captions: int
    mean_caption_words: float

    @property
    def chair_s(self) -> float:
        """Sentence-level CHAIR; denominator ``n_captions``."""
        return (
            self.n_captions_with_hallucination / self.n_captions
            if self.n_captions else float("nan")
        )

    @property
    def chair_i(self) -> float:
        """Instance-level CHAIR; denominator ``n_mentions``."""
        return (
            self.n_hallucinated_mentions / self.n_mentions
            if self.n_mentions else float("nan")
        )

    @property
    def recall(self) -> float:
        """Coverage of ground-truth objects (paper eq. 27), guards evasion."""
        return (
            self.n_distinct_correct / self.n_gt_objects
            if self.n_gt_objects else float("nan")
        )

    @property
    def precision(self) -> float:
        """Distinct mentioned objects that are real (paper eq. 28)."""
        return (
            self.n_distinct_correct / self.n_distinct_mentions
            if self.n_distinct_mentions else float("nan")
        )

    @property
    def f1(self) -> float:
        """Harmonic mean of :attr:`precision` and :attr:`recall` (eq. 29)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r > 0 else 0.0

    def as_dict(self) -> Dict[str, object]:
        """JSON-ready totals, metrics first and denominators alongside."""
        return {
            "chair_s": self.chair_s,
            "chair_s_numerator": self.n_captions_with_hallucination,
            "chair_s_denominator": self.n_captions,
            "chair_i": self.chair_i,
            "chair_i_numerator": self.n_hallucinated_mentions,
            "chair_i_denominator": self.n_mentions,
            "recall": self.recall,
            "recall_denominator": self.n_gt_objects,
            "precision": self.precision,
            "precision_denominator": self.n_distinct_mentions,
            "f1": self.f1,
            "n_distinct_correct": self.n_distinct_correct,
            "n_empty_captions": self.n_empty_captions,
            "mean_caption_words": self.mean_caption_words,
        }


# --------------------------------------------------------------------------- #
# scorer
# --------------------------------------------------------------------------- #
class CHAIRScorer:
    """Scores generated captions against COCO ground truth.

    Args:
        synonyms_path: the synonym list (see ``data/chair_synonyms.txt``).
        instances_json: ``instances_val2014[_subset].json``; supplies
            segmentation-derived ground truth.
        captions_json: ``captions_val2014[_subset].json``; supplies
            reference-caption-derived ground truth. Optional but strongly
            recommended -- the reference implementation uses both, and
            dropping it inflates CHAIR by counting real-but-unsegmented
            objects as hallucinations.
        tokenizer: ``"regex"`` (default, dependency-free) or ``"nltk"``.
        gt_source: ``"both"`` (reference behaviour), ``"instances"`` or
            ``"captions"``.
    """

    def __init__(
        self,
        synonyms_path: Path,
        instances_json: Path,
        captions_json: Optional[Path] = None,
        tokenizer: str = "regex",
        gt_source: str = "both",
    ) -> None:
        if gt_source not in ("both", "instances", "captions"):
            raise ValueError("gt_source must be both/instances/captions")
        if gt_source == "both" and captions_json is None:
            raise ValueError("gt_source='both' needs captions_json")
        self.tokenizer = tokenizer
        self.gt_source = gt_source
        groups, self.synonyms_sha256, self.synonyms_provenance = load_synonyms(synonyms_path)
        self.synonyms_path = str(synonyms_path)
        self.coco_objects: Set[str] = set()
        self.inverse_synonym: Dict[str, str] = {}
        for group in groups:
            self.coco_objects.update(group)
            for surface in group:
                self.inverse_synonym[surface] = group[0]
        self.double_word_dict = _build_double_word_dict()
        self.imid_to_objects: Dict[int, Set[str]] = {}
        self._gt_stats: Dict[str, int] = {}
        if gt_source in ("both", "instances"):
            self._load_instances(Path(instances_json))
        if gt_source in ("both", "captions") and captions_json is not None:
            self._load_captions(Path(captions_json))

    # ---- ground truth ---- #
    def _load_instances(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        id_to_name = {int(c["id"]): str(c["name"]).lower() for c in data["categories"]}
        unknown: Set[str] = set()
        n = 0
        for ann in data["annotations"]:
            name = id_to_name[int(ann["category_id"])]
            node = self.inverse_synonym.get(name)
            if node is None:
                unknown.add(name)
                continue
            self.imid_to_objects.setdefault(int(ann["image_id"]), set()).add(node)
            n += 1
        for img in data["images"]:
            self.imid_to_objects.setdefault(int(img["id"]), set())
        if unknown:
            raise ValueError(
                f"COCO categories absent from the synonym list: {sorted(unknown)}; "
                "the synonym file must cover all 80 category names."
            )
        self._gt_stats["n_instance_annotations"] = n
        self._gt_stats["n_images_with_instances"] = len(
            [1 for v in self.imid_to_objects.values() if v]
        )

    def _load_captions(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        n = 0
        for ann in data["annotations"]:
            _, nodes = self.caption_to_words(str(ann["caption"]))
            self.imid_to_objects.setdefault(int(ann["image_id"]), set()).update(nodes)
            n += 1
        self._gt_stats["n_reference_captions"] = n

    # ---- caption parsing ---- #
    def caption_to_words(self, caption: str) -> Tuple[List[str], List[str]]:
        """Extract COCO object mentions from a caption.

        Returns:
            ``(surface_forms, node_words)``, aligned and in order of
            appearance, duplicates preserved.
        """
        words = [singularize(w) for w in _tokenize(caption, self.tokenizer)]

        merged: List[str] = []
        i = 0
        while i < len(words):
            double = " ".join(words[i : i + 2])
            if i + 1 < len(words) and double in self.double_word_dict:
                merged.append(self.double_word_dict[double])
                i += 2
            else:
                merged.append(words[i])
                i += 1

        # "the seat of the toilet" must not count as a chair
        if "toilet" in merged and "seat" in merged:
            merged = [w for w in merged if w != "seat"]

        surfaces = [w for w in merged if w in self.coco_objects]
        nodes = [self.inverse_synonym[w] for w in surfaces]
        return surfaces, nodes

    # ---- scoring ---- #
    def score_caption(
        self, sample_id: str, image_id: int, caption: str
    ) -> CaptionScore:
        """Score one caption against the ground truth of ``image_id``.

        Raises:
            KeyError: the image has no ground-truth entry, which means the
                annotation subset does not cover the generated run.
        """
        if image_id not in self.imid_to_objects:
            raise KeyError(
                f"image_id {image_id} has no ground-truth annotations; the "
                "annotation subset does not match this run's images."
            )
        gt = self.imid_to_objects[image_id]
        _, nodes = self.caption_to_words(caption)
        return CaptionScore(
            sample_id=sample_id,
            image_id=image_id,
            mentions=nodes,
            hallucinated=[n for n in nodes if n not in gt],
            gt_objects=sorted(gt),
            n_words=len(_tokenize(caption, self.tokenizer)),
        )

    def score_many(
        self, items: Iterable[Tuple[str, int, str]]
    ) -> List[CaptionScore]:
        """Score an iterable of ``(sample_id, image_id, caption)`` triples."""
        return [self.score_caption(sid, iid, cap) for sid, iid, cap in items]

    @staticmethod
    def totals(scores: Sequence[CaptionScore]) -> ChairTotals:
        """Aggregate per-caption scores into :class:`ChairTotals`."""
        n_captions = len(scores)
        distinct_mentions = 0
        distinct_correct = 0
        n_gt = 0
        for s in scores:
            distinct_mentions += len(set(s.mentions))
            distinct_correct += s.n_correct_distinct
            n_gt += len(s.gt_objects)
        return ChairTotals(
            n_captions=n_captions,
            n_captions_with_hallucination=sum(1 for s in scores if s.has_hallucination),
            n_mentions=sum(s.n_mentions for s in scores),
            n_hallucinated_mentions=sum(s.n_hallucinated for s in scores),
            n_distinct_correct=distinct_correct,
            n_gt_objects=n_gt,
            n_distinct_mentions=distinct_mentions,
            n_empty_captions=sum(1 for s in scores if s.n_words == 0),
            mean_caption_words=(
                sum(s.n_words for s in scores) / n_captions if n_captions else 0.0
            ),
        )

    def config(self) -> Dict[str, object]:
        """Everything about this scorer that could change a CHAIR number."""
        return {
            "synonyms_path": self.synonyms_path,
            "synonyms_sha256": self.synonyms_sha256,
            "synonyms_provenance": self.synonyms_provenance,
            "n_categories": len(set(self.inverse_synonym.values())),
            "n_surface_forms": len(self.coco_objects),
            "tokenizer": self.tokenizer,
            "singularizer": "fox-repro rule-based (not pattern.en)",
            "gt_source": self.gt_source,
            "gt_stats": dict(self._gt_stats),
            "chair_s_unit": "per generated caption (reference chair.py behaviour)",
        }
