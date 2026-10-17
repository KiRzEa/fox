"""Deterministic iterators over the POPE and CHAIR evaluation sets.

Two hard requirements shape this module:

* **Stable ``sample_id``.** Resume (Decision 4) keys on
  ``(method, benchmark, split, seed, dtype, sample_id)``, so a ``sample_id``
  must be a pure function of the benchmark item, never of its position in a
  list or of a wall-clock value. POPE ids come from the official
  ``question_id``; CHAIR ids come from the COCO ``image_id``.
* **Deterministic ordering.** Items are always yielded in ascending
  ``sample_id`` order, so a partially finished run and a fresh run agree on
  what "the next sample" is.

Generation length is a per-benchmark protocol constant, not a free knob:
POPE is generated with ``max_new_tokens = 1`` (the answer is one yes/no token,
so POPE is prefill-dominated and cheap per sample), CHAIR with
``max_new_tokens = 512``. See ``storage.MAX_NEW_TOKENS_DEFAULT``.

Prompt templates live here rather than in the model code, and every record
stores the ``prompt_template_id`` that produced it: the exact wording of the
captioning prompt is not stated in the Fox paper and is known to move CHAIR by
several points, so it must be part of the provenance.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

__all__ = [
    "PROMPT_TEMPLATES",
    "render_prompt",
    "Sample",
    "load_pope",
    "load_chair",
    "load_benchmark",
    "pope_file_for_split",
    "image_id_from_filename",
    "image_filename_for_id",
]

# --------------------------------------------------------------------------- #
# prompt templates
# --------------------------------------------------------------------------- #
#: ``prompt_template_id -> template``. ``{question}`` is substituted for POPE.
#:
#: ``pope_v1`` is the standard LLaVA-1.5 POPE instruction used by the VCD /
#: SID / OPERA code releases; the Fox paper does not print its POPE prompt, so
#: this is a reproduction decision and is recorded per sample.
#: ``chair_v1`` likewise follows the prompt those releases use for the CHAIR
#: captioning task. The paper only states max 512 new tokens.
PROMPT_TEMPLATES: Dict[str, str] = {
    "pope_v1": "{question}\nAnswer the question using a single word or phrase.",
    "pope_v0_raw": "{question}",
    "chair_v1": "Please describe this image in detail.",
    "chair_v2_short": "Describe this image in detail.",
}

#: Which templates are legal for which benchmark (guards CLI typos).
_TEMPLATE_BENCHMARK = {
    "pope_v1": "pope",
    "pope_v0_raw": "pope",
    "chair_v1": "chair",
    "chair_v2_short": "chair",
}

_IMAGE_ID_RE = re.compile(r"(\d{6,})")


def render_prompt(template_id: str, question: Optional[str] = None) -> str:
    """Render a prompt from the registry.

    Args:
        template_id: key of :data:`PROMPT_TEMPLATES`.
        question: required for POPE templates, ignored for CHAIR templates.

    Raises:
        KeyError: unknown template id.
        ValueError: a POPE template was used without a question.
    """
    template = PROMPT_TEMPLATES[template_id]
    if "{question}" in template:
        if question is None:
            raise ValueError(f"template {template_id!r} needs a question")
        return template.format(question=question)
    return template


def template_benchmark(template_id: str) -> str:
    """Benchmark a template belongs to (``"pope"`` or ``"chair"``)."""
    return _TEMPLATE_BENCHMARK[template_id]


# --------------------------------------------------------------------------- #
# sample record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    """One evaluation item, independent of any model.

    Attributes:
        sample_id: stable identifier used by the resume key.
        image_id: COCO numeric image id (val2014).
        image_path: absolute path to the decoded JPEG on fast local disk.
        question: the POPE question, or ``None`` for CHAIR.
        label: POPE ground truth, ``"yes"`` / ``"no"``; ``None`` for CHAIR.
        benchmark: ``"pope"`` or ``"chair"``.
        split: benchmark split name.
        extra: anything else worth keeping (e.g. the POPE ``question_id``).
    """

    sample_id: str
    image_id: int
    image_path: Path
    benchmark: str
    split: str
    question: Optional[str] = None
    label: Optional[str] = None
    extra: Dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            object.__setattr__(self, "extra", {})


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def image_id_from_filename(filename: str) -> int:
    """``"COCO_val2014_000000131089.jpg"`` -> ``131089``.

    The Fox reference code parses CHAIR image ids as
    ``int(name.split(".jpg")[0][-6:])``, i.e. the last six digits. That agrees
    with this function for every val2014 id (the largest is 581929, six
    digits), but it would silently truncate a seven-digit id, so the full
    numeric field is parsed here instead. ``sample_id`` never depends on this
    parse: it is built from the zero-padded canonical id.

    Raises:
        ValueError: when no numeric id can be found.
    """
    match = _IMAGE_ID_RE.search(Path(filename).stem)
    if not match:
        raise ValueError(f"cannot parse a COCO image id out of {filename!r}")
    return int(match.group(1))


def image_filename_for_id(image_id: int) -> str:
    """``131089`` -> ``"COCO_val2014_000000131089.jpg"`` (val2014 convention)."""
    return f"COCO_val2014_{image_id:012d}.jpg"


def pope_file_for_split(pope_dir: Path, split: str) -> Path:
    """Locate the POPE annotation file for a split.

    Accepts the Fox reference-repo names (``coco_random.jsonl`` /
    ``coco_popular.jsonl`` / ``coco_adversarial.jsonl``, which is what this
    repository ships in ``data/pope/``), the official POPE release names
    (``coco_pope_random.json`` ...), and the bare ``random.json`` form.
    """
    candidates = [
        # layout used by the Fox reference repo (pope/coco/coco_random.jsonl)
        pope_dir / f"coco_{split}.jsonl",
        pope_dir / f"coco_{split}.json",
        # layout of the official POPE release
        pope_dir / f"coco_pope_{split}.json",
        pope_dir / f"coco_pope_{split}.jsonl",
        # bare names, for a hand-assembled directory
        pope_dir / f"{split}.jsonl",
        pope_dir / f"{split}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"no POPE file for split {split!r} in {pope_dir}; "
        f"looked for {[c.name for c in candidates]}"
    )


def _read_json_records(path: Path) -> List[Dict[str, object]]:
    """Read either a JSON array or JSON-lines file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} is not a list of records")
        return data
    records: List[Dict[str, object]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} is not valid JSON") from exc
    return records


# --------------------------------------------------------------------------- #
# POPE
# --------------------------------------------------------------------------- #
def load_pope(
    pope_dir: Path,
    images_dir: Path,
    split: str,
    require_images: bool = True,
) -> List[Sample]:
    """Load one POPE split.

    Each POPE record is ``{question_id, image, text, label}`` where ``label``
    is ``"yes"``/``"no"``. The official COCO-POPE release has 500 val2014
    images x 6 questions = 3000 items per split; this loader does not enforce
    that count but :func:`describe_split` reports it.

    Args:
        pope_dir: directory containing ``coco_pope_<split>.json``.
        images_dir: directory of extracted val2014 JPEGs.
        split: ``"random"`` / ``"popular"`` / ``"adversarial"``.
        require_images: raise if a referenced image is missing locally. Set
            ``False`` for CPU-side inspection without the image tar.

    Returns:
        Samples sorted by ``question_id``.
    """
    path = pope_file_for_split(Path(pope_dir), split)
    raw = _read_json_records(path)
    if not raw:
        raise ValueError(f"{path} contained no records")

    samples: List[Sample] = []
    seen: set[str] = set()
    missing: List[str] = []
    for idx, rec in enumerate(raw):
        question = rec.get("text") or rec.get("question")
        label = rec.get("label") or rec.get("answer")
        image_field = rec.get("image") or rec.get("image_file") or rec.get("image_id")
        if question is None or label is None or image_field is None:
            raise ValueError(f"{path} record {idx} missing text/label/image: {rec!r}")
        label_norm = str(label).strip().lower()
        if label_norm not in ("yes", "no"):
            raise ValueError(f"{path} record {idx} has non-binary label {label!r}")
        if isinstance(image_field, int):
            image_id = image_field
            image_name = image_filename_for_id(image_id)
        else:
            image_name = str(image_field)
            image_id = image_id_from_filename(image_name)
        qid = rec.get("question_id", idx)
        sample_id = f"pope/{split}/q{int(qid):07d}"
        if sample_id in seen:
            raise ValueError(f"duplicate question_id {qid} in {path}")
        seen.add(sample_id)
        image_path = Path(images_dir) / image_name
        if require_images and not image_path.is_file():
            missing.append(image_name)
        samples.append(
            Sample(
                sample_id=sample_id,
                image_id=image_id,
                image_path=image_path,
                benchmark="pope",
                split=split,
                question=str(question),
                label=label_norm,
                extra={"question_id": int(qid), "source_file": path.name},
            )
        )
    if missing:
        uniq = sorted(set(missing))
        raise FileNotFoundError(
            f"{len(uniq)} POPE images missing from {images_dir} "
            f"(first few: {uniq[:5]}). Extract the COCO subset tar first."
        )
    samples.sort(key=lambda s: s.sample_id)
    return samples


# --------------------------------------------------------------------------- #
# CHAIR
# --------------------------------------------------------------------------- #
def load_chair(
    image_list_path: Path,
    images_dir: Path,
    split: str = "val2014_500",
    prompt_template_id: str = "chair_v1",
    require_images: bool = True,
) -> List[Sample]:
    """Load the CHAIR captioning set (500 COCO val2014 images).

    Args:
        image_list_path: ``chair_images.json`` as frozen into the subset tar.
            Accepts a bare list of ids, a list of filenames, or
            ``{"image_ids": [...]}``.
        images_dir: directory of extracted val2014 JPEGs.
        split: split label recorded in provenance (default ``"val2014_500"``).
        prompt_template_id: which captioning prompt to attach.
        require_images: raise if an image is missing locally.

    Returns:
        Samples sorted by ``image_id``.
    """
    path = Path(image_list_path)
    if not path.is_file():
        # A bare FileNotFoundError out of pathlib is what the first real
        # session produced, four times in forty seconds, and it said nothing
        # about which of several causes applied.
        raise FileNotFoundError(
            f"CHAIR image list not found at {path}.\n"
            f"It ships with the package at assets/chair_images.json, so the "
            f"usual cause is a stale install -- re-run `pip install -e .`.\n"
            f"To regenerate it instead, build the COCO subset:\n"
            f"    python -m foxrepro.data.build_coco_subset build "
            f"--val2014-dir <dir> --annotations-dir <dir> --pope-dir <dir>"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("image_ids") or payload.get("images")
        if entries is None:
            raise ValueError(f"{path} has neither 'image_ids' nor 'images'")
    else:
        entries = payload
    if not entries:
        raise ValueError(f"{path} contained no images")

    image_ids: List[int] = []
    for e in entries:
        if isinstance(e, int):
            image_ids.append(e)
        elif isinstance(e, str):
            image_ids.append(image_id_from_filename(e))
        elif isinstance(e, dict) and "id" in e:
            image_ids.append(int(e["id"]))
        elif isinstance(e, dict) and "image_id" in e:
            image_ids.append(int(e["image_id"]))
        else:
            raise ValueError(f"cannot interpret CHAIR image entry {e!r}")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path} contains duplicate image ids")

    prompt = render_prompt(prompt_template_id)
    samples: List[Sample] = []
    missing: List[str] = []
    for image_id in sorted(image_ids):
        name = image_filename_for_id(image_id)
        image_path = Path(images_dir) / name
        if require_images and not image_path.is_file():
            missing.append(name)
        samples.append(
            Sample(
                sample_id=f"chair/{split}/{image_id:012d}",
                image_id=image_id,
                image_path=image_path,
                benchmark="chair",
                split=split,
                question=None,
                label=None,
                extra={"caption_prompt": prompt, "source_file": path.name},
            )
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} CHAIR images missing from {images_dir} "
            f"(first few: {missing[:5]}). Extract the COCO subset tar first."
        )
    return samples


def load_benchmark(
    benchmark: str,
    split: str,
    pope_dir: Path,
    images_dir: Path,
    chair_image_list: Path,
    prompt_template_id: str,
    require_images: bool = True,
) -> List[Sample]:
    """Dispatch to :func:`load_pope` or :func:`load_chair`.

    Args:
        benchmark: ``"pope"`` or ``"chair"``.
        split: split name; for CHAIR this is just a provenance label.
        pope_dir: POPE annotation directory.
        images_dir: extracted val2014 images.
        chair_image_list: frozen CHAIR image-id list.
        prompt_template_id: must belong to ``benchmark``.
        require_images: raise on missing images.
    """
    if template_benchmark(prompt_template_id) != benchmark:
        raise ValueError(
            f"template {prompt_template_id!r} is for "
            f"{template_benchmark(prompt_template_id)!r}, not {benchmark!r}"
        )
    if benchmark == "pope":
        return load_pope(pope_dir, images_dir, split, require_images=require_images)
    if benchmark == "chair":
        return load_chair(
            chair_image_list, images_dir, split=split,
            prompt_template_id=prompt_template_id, require_images=require_images,
        )
    raise ValueError(f"unknown benchmark {benchmark!r}")


def describe_split(samples: Sequence[Sample]) -> Dict[str, object]:
    """Summary counts for a loaded split (printed by the setup readiness check)."""
    n_yes = sum(1 for s in samples if s.label == "yes")
    return {
        "n_samples": len(samples),
        "n_images": len({s.image_id for s in samples}),
        "n_label_yes": n_yes,
        "n_label_no": sum(1 for s in samples if s.label == "no"),
        "first_sample_id": samples[0].sample_id if samples else None,
        "last_sample_id": samples[-1].sample_id if samples else None,
    }


def iter_samples(samples: Iterable[Sample], skip: Optional[set] = None) -> Iterator[Sample]:
    """Yield samples whose ``sample_id`` is not in ``skip`` (resume helper)."""
    skip = skip or set()
    for s in samples:
        if s.sample_id not in skip:
            yield s
