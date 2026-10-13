"""Shared test fixtures: a throwaway workspace that mimics the real layout.

No GPU, no model, no COCO download. The POPE files, the CHAIR image list, the
annotation subsets and (empty) JPEG files are synthesised, so the tests
exercise the real loaders, the real resume logic and the real metric code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: (category_id, name) pairs used by the synthetic annotation files.
CATEGORIES: Sequence[Tuple[int, str]] = (
    (1, "person"), (17, "cat"), (18, "dog"), (24, "zebra"), (22, "elephant"),
    (63, "couch"), (67, "dining table"), (58, "hot dog"), (46, "wine glass"),
    (70, "toilet"), (62, "chair"), (3, "car"),
)
_NAME_TO_ID = {name: cid for cid, name in CATEGORIES}


def write_pope_split(
    pope_dir: Path, split: str, image_ids: Sequence[int], n_per_image: int = 6
) -> int:
    """Write a synthetic ``coco_<split>.jsonl`` POPE file.

    Labels alternate yes/no so the split is balanced, matching real POPE.

    Returns:
        Number of questions written.
    """
    pope_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    qid = 1
    for image_id in image_ids:
        for j in range(n_per_image):
            lines.append(json.dumps({
                "question_id": qid,
                "image": f"COCO_val2014_{image_id:012d}.jpg",
                "text": f"Is there a {'dog' if j % 2 == 0 else 'car'} in the image?",
                "label": "yes" if j % 2 == 0 else "no",
            }))
            qid += 1
    (pope_dir / f"coco_{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_images(images_dir: Path, image_ids: Sequence[int]) -> None:
    """Create empty placeholder JPEGs so loaders' existence checks pass."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for image_id in image_ids:
        (images_dir / f"COCO_val2014_{image_id:012d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def write_chair_list(path: Path, image_ids: Sequence[int]) -> None:
    """Write a frozen CHAIR image-id list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"split": "val2014_500", "source": "test", "image_ids": list(image_ids)}),
        encoding="utf-8",
    )


def write_annotations(
    ann_dir: Path,
    instances: Dict[int, Sequence[str]],
    captions: Optional[Dict[int, Sequence[str]]] = None,
) -> Tuple[Path, Path]:
    """Write synthetic ``instances``/``captions`` subset files.

    Args:
        ann_dir: destination directory.
        instances: ``image_id -> [category name, ...]`` segmentation truth.
        captions: ``image_id -> [reference caption, ...]``.

    Returns:
        ``(instances_path, captions_path)``.
    """
    ann_dir.mkdir(parents=True, exist_ok=True)
    inst = {
        "categories": [{"id": cid, "name": name, "supercategory": "x"}
                       for cid, name in CATEGORIES],
        "images": [{"id": int(i), "file_name": f"COCO_val2014_{int(i):012d}.jpg"}
                   for i in sorted(instances)],
        "annotations": [],
    }
    ann_id = 1
    for image_id, names in sorted(instances.items()):
        for name in names:
            inst["annotations"].append({
                "id": ann_id, "image_id": int(image_id),
                "category_id": _NAME_TO_ID[name], "iscrowd": 0, "area": 1.0,
            })
            ann_id += 1
    inst_path = ann_dir / "instances_val2014_subset.json"
    inst_path.write_text(json.dumps(inst), encoding="utf-8")

    caps = {
        "images": inst["images"],
        "annotations": [],
    }
    cap_id = 1
    for image_id, texts in sorted((captions or {}).items()):
        for text in texts:
            caps["annotations"].append(
                {"id": cap_id, "image_id": int(image_id), "caption": text}
            )
            cap_id += 1
    cap_path = ann_dir / "captions_val2014_subset.json"
    cap_path.write_text(json.dumps(caps), encoding="utf-8")
    return inst_path, cap_path


def make_workspace(
    root: Path, n_images: int = 10, n_per_image: int = 6
) -> Dict[str, object]:
    """Build a complete synthetic workspace under ``root``.

    Returns:
        A dict of paths and counts used by the tests.
    """
    image_ids = [1000 + 7 * i for i in range(n_images)]
    pope_dir = root / "pope"
    images_dir = root / "data" / "images"
    ann_dir = root / "data" / "annotations"
    chair_list = root / "data" / "chair_images.json"
    runs_dir = root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    n_questions = write_pope_split(pope_dir, "random", image_ids, n_per_image)
    write_pope_split(pope_dir, "adversarial", image_ids, n_per_image)
    write_images(images_dir, image_ids)
    write_chair_list(chair_list, image_ids)
    write_annotations(
        ann_dir,
        instances={i: ["person", "dog"] for i in image_ids},
        captions={i: ["A person with a dog on a couch."] for i in image_ids},
    )
    return {
        "root": root, "pope_dir": pope_dir, "images_dir": images_dir,
        "ann_dir": ann_dir, "chair_list": chair_list, "runs_dir": runs_dir,
        "image_ids": image_ids, "n_questions": n_questions,
    }
