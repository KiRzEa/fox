"""One-time builder for the COCO val2014 evaluation subset archive.

Why an archive
--------------
Full COCO ``val2014`` is ~40k JPEGs / ~6.2 GB. Google Drive is unusably slow
with tens of thousands of small files, and a Colab session re-copies them every
time. This script extracts only the images the benchmarks actually touch
(500 POPE images + 500 CHAIR images, overlapping or not: at most 1000 files,
typically 150-200 MB) and packs them, together with the annotation subsets, in
a **single deterministic ``.tar``** that lives on Drive and is extracted to fast
local disk once per session.

The tar is uncompressed on purpose: JPEG is already compressed, so gzip buys
~1% at the cost of a slow single-threaded decompress on every session start.

Where to get the inputs
-----------------------
``val2014`` images (6.2 GB)::

    wget http://images.cocodataset.org/zips/val2014.zip
    unzip -q val2014.zip            # -> val2014/COCO_val2014_*.jpg

COCO 2014 train/val annotations (241 MB), which contain
``instances_val2014.json`` and ``captions_val2014.json``::

    wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    unzip -q annotations_trainval2014.zip   # -> annotations/

POPE annotation files (the COCO ones; small)::

    git clone https://github.com/RUCAIBox/POPE
    # POPE/output/coco/coco_pope_random.json
    # POPE/output/coco/coco_pope_popular.json
    # POPE/output/coco/coco_pope_adversarial.json

Those three POPE files reference 500 val2014 images and 3000 questions each.

Layout inside the tar
---------------------
::

    images/COCO_val2014_000000XXXXXX.jpg      only the needed images
    pope/coco_pope_random.json                verbatim copies
    pope/coco_pope_popular.json
    pope/coco_pope_adversarial.json
    annotations/instances_val2014_subset.json  filtered to the subset images
    annotations/captions_val2014_subset.json   filtered to the subset images
    chair_images.json                          frozen CHAIR image-id list
    BUILD_INFO.json                            build parameters

The manifest (written next to the tar, never inside it, since a file cannot
contain its own hash) records the SHA-256 of the tar and of every member.

Usage
-----
::

    # one time, on a machine that has the full val2014 download
    python -m data.build_coco_subset build \
        --val2014-dir /data/val2014 \
        --annotations-dir /data/annotations \
        --pope-dir /data/POPE/output/coco \
        --chair-sample-n 500 --chair-seed 0

    # every session, on Colab / RunPod
    python -m data.build_coco_subset extract
    python -m data.build_coco_subset verify --level full
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


from foxrepro.data.loaders import (  # noqa: E402
    image_filename_for_id,
    image_id_from_filename,
    pope_file_for_split,
)
from foxrepro.storage import paths, sha256_file  # noqa: E402

__all__ = [
    "build_subset",
    "extract_subset",
    "verify_subset",
    "pope_image_ids",
    "sample_chair_image_ids",
]

POPE_SPLITS = ("random", "popular", "adversarial")

#: Fixed metadata so the tar bytes (and therefore its SHA-256) are reproducible.
_FIXED_MTIME = 0
_FIXED_MODE = 0o644


# --------------------------------------------------------------------------- #
# image id selection
# --------------------------------------------------------------------------- #
def pope_image_ids(pope_dir: Path, splits: Sequence[str] = POPE_SPLITS) -> Dict[str, Set[int]]:
    """Image ids referenced by each POPE split.

    Returns:
        ``{split: {image_id, ...}}``.
    """
    out: Dict[str, Set[int]] = {}
    for split in splits:
        path = pope_file_for_split(Path(pope_dir), split)
        ids: Set[int] = set()
        text = path.read_text(encoding="utf-8").strip()
        records = json.loads(text) if text.startswith("[") else [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        for rec in records:
            image_field = rec.get("image") or rec.get("image_file") or rec.get("image_id")
            if image_field is None:
                raise ValueError(f"{path}: record without an image field: {rec!r}")
            ids.add(
                int(image_field) if isinstance(image_field, int)
                else image_id_from_filename(str(image_field))
            )
        out[split] = ids
    return out


def sample_chair_image_ids(
    instances_json: Path, n: int, seed: int, exclude: Optional[Set[int]] = None
) -> List[int]:
    """Deterministically sample ``n`` val2014 image ids for CHAIR.

    The population is every image listed in ``instances_val2014.json``, sorted
    by id before sampling so the result depends only on ``(n, seed)`` and not
    on JSON ordering or filesystem order.

    Args:
        instances_json: path to ``instances_val2014.json``.
        n: how many images (the paper uses 500).
        seed: sampling seed, recorded in ``BUILD_INFO.json``.
        exclude: ids to keep out of the sample (unused by default; pass the
            POPE ids if you want POPE and CHAIR to be disjoint).

    Returns:
        Sorted list of ``n`` image ids.
    """
    data = json.loads(Path(instances_json).read_text(encoding="utf-8"))
    population = sorted({int(img["id"]) for img in data["images"]})
    if exclude:
        population = [i for i in population if i not in exclude]
    if len(population) < n:
        raise ValueError(f"need {n} images, population has {len(population)}")
    rng = random.Random(seed)
    return sorted(rng.sample(population, n))


# --------------------------------------------------------------------------- #
# annotation filtering
# --------------------------------------------------------------------------- #
def _filter_coco_annotation_file(path: Path, keep: Set[int]) -> Dict[str, object]:
    """Filter a COCO annotation file down to ``keep`` image ids.

    Keeps ``info``, ``licenses`` and ``categories`` untouched and subsets
    ``images`` and ``annotations``. Works for both ``instances_*`` and
    ``captions_*`` files.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, object] = {}
    for meta_key in ("info", "licenses", "categories"):
        if meta_key in data:
            out[meta_key] = data[meta_key]
    images = [img for img in data.get("images", []) if int(img["id"]) in keep]
    anns = [a for a in data.get("annotations", []) if int(a["image_id"]) in keep]
    images.sort(key=lambda img: int(img["id"]))
    anns.sort(key=lambda a: (int(a["image_id"]), int(a.get("id", 0))))
    out["images"] = images
    out["annotations"] = anns
    found = {int(img["id"]) for img in images}
    missing = keep - found
    if missing:
        raise ValueError(
            f"{path.name} is missing {len(missing)} of the required images "
            f"(e.g. {sorted(missing)[:5]}); is this really val2014?"
        )
    return out


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
@dataclass
class BuildResult:
    """Outcome of :func:`build_subset`."""

    tar_path: Path
    manifest_path: Path
    n_images: int
    total_bytes: int
    tar_sha256: str


def _add_bytes(tar: tarfile.TarFile, arcname: str, payload: bytes) -> Tuple[str, int]:
    """Add in-memory bytes as a deterministic tar member; return (sha256, size)."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    info.mtime = _FIXED_MTIME
    info.mode = _FIXED_MODE
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    tar.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _add_file(tar: tarfile.TarFile, arcname: str, src: Path) -> Tuple[str, int]:
    """Add a file as a deterministic tar member; return (sha256, size)."""
    return _add_bytes(tar, arcname, src.read_bytes())


def build_subset(
    val2014_dir: Path,
    annotations_dir: Path,
    pope_dir: Path,
    out_tar: Path,
    out_manifest: Path,
    chair_image_list: Optional[Path] = None,
    chair_sample_n: int = 500,
    chair_seed: int = 0,
    chair_from_pope: bool = False,
    pope_splits: Sequence[str] = POPE_SPLITS,
) -> BuildResult:
    """Build the single-tar COCO subset plus its checksum manifest.

    Args:
        val2014_dir: directory holding the full ``COCO_val2014_*.jpg`` set.
        annotations_dir: directory holding ``instances_val2014.json`` and
            ``captions_val2014.json``.
        pope_dir: directory holding the three ``coco_pope_*.json`` files.
        out_tar: destination tar (typically on Drive / the persistent volume).
        out_manifest: destination manifest JSON.
        chair_image_list: an explicit, pre-existing CHAIR image list to freeze
            instead of sampling. Mutually exclusive with sampling options.
        chair_sample_n: how many CHAIR images to sample (500 per the paper).
        chair_seed: seed for that sample.
        chair_from_pope: reuse the POPE image set for CHAIR instead of
            sampling. Halves the tar, at the cost of deviating from the usual
            CHAIR protocol of an independent random val2014 sample.
        pope_splits: which POPE splits to include.

    Returns:
        A :class:`BuildResult`.
    """
    val2014_dir = Path(val2014_dir)
    annotations_dir = Path(annotations_dir)
    pope_dir = Path(pope_dir)
    instances = annotations_dir / "instances_val2014.json"
    captions = annotations_dir / "captions_val2014.json"
    for required in (instances, captions):
        if not required.is_file():
            raise FileNotFoundError(f"{required} not found (need COCO *2014* annotations)")
    if not val2014_dir.is_dir():
        raise FileNotFoundError(f"{val2014_dir} is not a directory")

    per_split = pope_image_ids(pope_dir, pope_splits)
    pope_ids: Set[int] = set().union(*per_split.values()) if per_split else set()

    if chair_image_list is not None:
        payload = json.loads(Path(chair_image_list).read_text(encoding="utf-8"))
        entries = payload["image_ids"] if isinstance(payload, dict) else payload
        chair_ids = sorted(
            int(e) if not isinstance(e, str) else image_id_from_filename(e) for e in entries
        )
        chair_source = f"explicit:{Path(chair_image_list).name}"
    elif chair_from_pope:
        chair_ids = sorted(pope_ids)[:chair_sample_n]
        chair_source = "pope-image-set"
    else:
        chair_ids = sample_chair_image_ids(instances, chair_sample_n, chair_seed)
        chair_source = f"random(instances_val2014, n={chair_sample_n}, seed={chair_seed})"

    needed = sorted(pope_ids | set(chair_ids))
    print(
        f"[build] POPE images: {len(pope_ids)}  CHAIR images: {len(chair_ids)}  "
        f"union: {len(needed)}",
        flush=True,
    )

    missing_files = [
        image_filename_for_id(i) for i in needed
        if not (val2014_dir / image_filename_for_id(i)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} required images are not in {val2014_dir} "
            f"(e.g. {missing_files[:5]}). Note the benchmarks use val2014, not val2017."
        )

    out_tar = Path(out_tar)
    out_tar.parent.mkdir(parents=True, exist_ok=True)
    members: Dict[str, Dict[str, object]] = {}
    tmp_tar = out_tar.with_suffix(out_tar.suffix + ".partial")

    build_info = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "val2014_dir": str(val2014_dir),
        "pope_dir": str(pope_dir),
        "pope_splits": list(pope_splits),
        "pope_image_counts": {k: len(v) for k, v in per_split.items()},
        "chair_source": chair_source,
        "chair_sample_n": len(chair_ids),
        "chair_seed": chair_seed,
        "n_images": len(needed),
        "builder_version": 1,
    }

    with tarfile.open(tmp_tar, "w", format=tarfile.PAX_FORMAT) as tar:
        for image_id in needed:
            name = image_filename_for_id(image_id)
            digest, size = _add_file(tar, f"images/{name}", val2014_dir / name)
            members[f"images/{name}"] = {"sha256": digest, "size": size}

        for split in pope_splits:
            src = pope_file_for_split(pope_dir, split)
            arc = f"pope/coco_pope_{split}.json"
            digest, size = _add_file(tar, arc, src)
            members[arc] = {"sha256": digest, "size": size, "source": src.name}

        keep = set(needed)
        for src, arc in (
            (instances, "annotations/instances_val2014_subset.json"),
            (captions, "annotations/captions_val2014_subset.json"),
        ):
            print(f"[build] filtering {src.name} ...", flush=True)
            filtered = _filter_coco_annotation_file(src, keep)
            payload_bytes = json.dumps(filtered, sort_keys=True, separators=(",", ":")).encode()
            digest, size = _add_bytes(tar, arc, payload_bytes)
            members[arc] = {
                "sha256": digest, "size": size, "source": src.name,
                "n_images": len(filtered["images"]),          # type: ignore[index]
                "n_annotations": len(filtered["annotations"]),  # type: ignore[index]
            }

        chair_payload = json.dumps(
            {"split": "val2014_500", "source": chair_source, "image_ids": chair_ids},
            sort_keys=True, indent=1,
        ).encode()
        digest, size = _add_bytes(tar, "chair_images.json", chair_payload)
        members["chair_images.json"] = {"sha256": digest, "size": size}

        info_bytes = json.dumps(build_info, sort_keys=True, indent=1).encode()
        digest, size = _add_bytes(tar, "BUILD_INFO.json", info_bytes)
        members["BUILD_INFO.json"] = {"sha256": digest, "size": size}

    os.replace(tmp_tar, out_tar)
    tar_sha = sha256_file(out_tar)
    total_bytes = sum(int(m["size"]) for m in members.values())

    manifest = {
        "schema": "fox-coco-subset/1",
        "tar_name": out_tar.name,
        "tar_sha256": tar_sha,
        "tar_bytes": out_tar.stat().st_size,
        "n_members": len(members),
        "n_images": len(needed),
        "member_bytes_total": total_bytes,
        "build_info": build_info,
        "members": members,
    }
    out_manifest = Path(out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=1), encoding="utf-8")

    print(
        f"[build] wrote {out_tar} "
        f"({out_tar.stat().st_size / 1e6:.1f} MB, {len(needed)} images)\n"
        f"[build] manifest {out_manifest}\n"
        f"[build] tar sha256 {tar_sha}",
        flush=True,
    )
    return BuildResult(out_tar, out_manifest, len(needed), total_bytes, tar_sha)


# --------------------------------------------------------------------------- #
# extract / verify
# --------------------------------------------------------------------------- #
def _safe_members(tar: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    """Yield only plain-file members with safe relative names.

    The tar is one we built ourselves, but extraction code that trusts member
    names is a standing hazard, so the check stays.
    """
    for member in tar.getmembers():
        name = member.name
        if member.islnk() or member.issym():
            raise ValueError(f"refusing link member {name!r}")
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"refusing unsafe member name {name!r}")
        if member.isdir():
            continue
        if not member.isfile():
            raise ValueError(f"refusing non-regular member {name!r}")
        yield member


def extract_subset(
    tar_path: Path,
    dest_dir: Path,
    manifest_path: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, object]:
    """Extract the subset tar to fast local disk, idempotently.

    If ``dest_dir`` already holds the expected number of members and
    ``force`` is false, extraction is skipped -- this is what makes
    ``setup_colab.sh`` cheap to re-run after a disconnect.

    Args:
        tar_path: the archive, usually on Drive.
        dest_dir: local destination (``paths().data_root``).
        manifest_path: manifest to verify the tar's SHA-256 against before
            extracting. Highly recommended: a truncated Drive copy is the
            single most common failure in this environment.
        force: extract even if the destination looks complete.

    Returns:
        ``{"extracted": bool, "n_members": int, "dest": str}``.
    """
    tar_path, dest_dir = Path(tar_path), Path(dest_dir)
    if not tar_path.is_file():
        raise FileNotFoundError(f"{tar_path} not found")

    manifest: Optional[Dict[str, object]] = None
    if manifest_path and Path(manifest_path).is_file():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        expected_bytes = int(manifest["tar_bytes"])  # type: ignore[index]
        actual_bytes = tar_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"{tar_path} is {actual_bytes} bytes, manifest says {expected_bytes}: "
                "the Drive copy is incomplete or stale."
            )
        actual_sha = sha256_file(tar_path)
        if actual_sha != manifest["tar_sha256"]:
            raise ValueError(
                f"{tar_path} sha256 {actual_sha} != manifest {manifest['tar_sha256']}"
            )

    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r") as tar:
        member_list = list(_safe_members(tar))
        if not force:
            present = sum(1 for m in member_list if (dest_dir / m.name).is_file())
            if present == len(member_list):
                return {"extracted": False, "n_members": len(member_list), "dest": str(dest_dir)}
        for member in member_list:
            member.mode = _FIXED_MODE
            # filter="data" is the strict extraction policy: it rejects
            # absolute paths, parent traversal, links, devices and special
            # modes. `_safe_members` already screens for those, so this is
            # belt-and-braces -- but passing it explicitly also silences the
            # DeprecationWarning on 3.12/3.13 and keeps working on 3.14, where
            # filtering becomes the default and an unfiltered extract is an
            # error. Added as a keyword with a fallback because the parameter
            # does not exist before 3.11.4.
            try:
                tar.extract(member, path=dest_dir, set_attrs=False, filter="data")
            except TypeError:  # pragma: no cover - Python < 3.11.4
                tar.extract(member, path=dest_dir, set_attrs=False)
    return {"extracted": True, "n_members": len(member_list), "dest": str(dest_dir)}


def verify_subset(
    dest_dir: Path, manifest_path: Path, level: str = "fast"
) -> Dict[str, object]:
    """Verify an extracted subset against the manifest.

    Args:
        dest_dir: the extracted data root.
        manifest_path: manifest written by :func:`build_subset`.
        level: ``"fast"`` checks presence and byte sizes (seconds);
            ``"full"`` recomputes every SHA-256 (a minute or so for 1000 JPEGs).

    Returns:
        ``{"ok": bool, "checked": int, "missing": [...], "bad_size": [...],
        "bad_hash": [...]}``.
    """
    if level not in ("fast", "full"):
        raise ValueError("level must be 'fast' or 'full'")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    members: Dict[str, Dict[str, object]] = manifest["members"]
    missing: List[str] = []
    bad_size: List[str] = []
    bad_hash: List[str] = []
    for name, meta in sorted(members.items()):
        path = Path(dest_dir) / name
        if not path.is_file():
            missing.append(name)
            continue
        if path.stat().st_size != int(meta["size"]):
            bad_size.append(name)
            continue
        if level == "full" and sha256_file(path) != meta["sha256"]:
            bad_hash.append(name)
    return {
        "ok": not (missing or bad_size or bad_hash),
        "level": level,
        "checked": len(members),
        "missing": missing[:20],
        "n_missing": len(missing),
        "bad_size": bad_size[:20],
        "bad_hash": bad_hash[:20],
        "n_images": manifest.get("n_images"),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the subset tar (one time, needs full val2014)")
    b.add_argument("--val2014-dir", type=Path, required=True)
    b.add_argument("--annotations-dir", type=Path, required=True)
    b.add_argument("--pope-dir", type=Path, required=True)
    b.add_argument("--out-tar", type=Path, default=p.coco_subset_tar)
    b.add_argument("--out-manifest", type=Path, default=p.coco_subset_manifest)
    g = b.add_mutually_exclusive_group()
    g.add_argument("--chair-image-list", type=Path, help="freeze an existing CHAIR id list")
    g.add_argument("--chair-from-pope", action="store_true",
                   help="reuse POPE's images for CHAIR (smaller tar, protocol deviation)")
    b.add_argument("--chair-sample-n", type=int, default=500)
    b.add_argument("--chair-seed", type=int, default=0)

    e = sub.add_parser("extract", help="extract the tar to fast local disk (every session)")
    e.add_argument("--tar", type=Path, default=p.coco_subset_tar)
    e.add_argument("--dest", type=Path, default=p.data_root)
    e.add_argument("--manifest", type=Path, default=p.coco_subset_manifest)
    e.add_argument("--force", action="store_true")

    v = sub.add_parser("verify", help="verify an extracted subset against the manifest")
    v.add_argument("--dest", type=Path, default=p.data_root)
    v.add_argument("--manifest", type=Path, default=p.coco_subset_manifest)
    v.add_argument("--level", choices=("fast", "full"), default="fast")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        build_subset(
            val2014_dir=args.val2014_dir,
            annotations_dir=args.annotations_dir,
            pope_dir=args.pope_dir,
            out_tar=args.out_tar,
            out_manifest=args.out_manifest,
            chair_image_list=args.chair_image_list,
            chair_sample_n=args.chair_sample_n,
            chair_seed=args.chair_seed,
            chair_from_pope=args.chair_from_pope,
        )
        return 0
    if args.cmd == "extract":
        result = extract_subset(args.tar, args.dest, args.manifest, force=args.force)
        print(json.dumps(result, indent=1))
        return 0
    report = verify_subset(args.dest, args.manifest, args.level)
    print(json.dumps(report, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
