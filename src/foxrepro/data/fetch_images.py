"""Download exactly the COCO val2014 images this project needs, and no more.

Why this exists
---------------
The documented way to get COCO val2014 is a 6.2 GB zip. This project needs 500
images for POPE and 500 for CHAIR -- at most 1000 of the 40,504, about 200 MB.
COCO serves images individually at a predictable URL, so the whole 6 GB
download, the subset-tar build and the upload-as-a-Kaggle-Dataset dance are all
avoidable for the common case:

    http://images.cocodataset.org/val2014/COCO_val2014_000000000042.jpg

``data/build_coco_subset.py`` remains the right tool when you already have the
full val2014 locally, or when you want one reproducible, checksummed tar to
attach as a read-only Kaggle Dataset so later sessions cost nothing. This
module is the shortcut that gets a first real run going today.

Scope
-----
POPE needs **images only** -- its labels ship with the benchmark and are in
``foxrepro/assets/pope``. So a POPE run is unblocked by this module alone.

CHAIR additionally needs ``instances_val2014.json`` (ground-truth objects) and,
for reference-caption matching, ``captions_val2014.json``. Those only come in
``annotations_trainval2014.zip`` (~241 MB), which ``--with-annotations``
fetches. There is no per-file endpoint for them.

Safety and manners
------------------
* Writes are atomic: each image lands in a temp file and is renamed only after
  the JPEG end-of-image marker is verified, so an interrupted run never leaves
  a truncated image that a later run would treat as complete.
* Already-present, valid files are skipped, so this is resumable and cheap to
  re-run after a Kaggle session dies.
* Concurrency defaults to 8 with retry and backoff. COCO is a free public
  mirror; do not raise this much.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from foxrepro.data.loaders import image_filename_for_id, pope_file_for_split
from foxrepro.storage import BENCHMARK_SPLITS, paths

__all__ = [
    "COCO_IMAGE_URL",
    "needed_image_ids",
    "fetch_images",
    "fetch_annotations",
]

COCO_IMAGE_URL = "http://images.cocodataset.org/val2014/{name}"
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
POPE_SPLITS = ("random", "popular", "adversarial")

_JPEG_SOI = b"\xff\xd8\xff"
_JPEG_EOI = b"\xff\xd9"


def _is_complete_jpeg(path: Path) -> bool:
    """A file is usable only if it both starts and ends like a JPEG.

    Checking the end matters more than the start: a connection dropped
    mid-transfer leaves a file with a correct header and missing tail, which
    PIL may happily open and silently return a partial image for.
    """
    try:
        if path.stat().st_size < 1024:
            return False
        with path.open("rb") as fh:
            if fh.read(3) != _JPEG_SOI:
                return False
            fh.seek(-2, 2)
            return fh.read(2) == _JPEG_EOI
    except OSError:
        return False


def needed_image_ids(
    pope_dir: Optional[Path] = None,
    chair_list: Optional[Path] = None,
    splits: Sequence[str] = POPE_SPLITS,
    chair_splits: Optional[Sequence[str]] = None,
) -> Dict[str, Set[int]]:
    """Which COCO ids POPE and CHAIR actually reference.

    Returns ``{"pope": {...}, "chair": {...}, "union": {...}}``. The POPE ids
    come from the shipped annotations, so this needs no network and no COCO
    download to answer.

    Args:
        pope_dir: POPE annotation directory.
        chair_list: an explicit CHAIR id list, overriding ``chair_splits``.
        splits: POPE splits to union.
        chair_splits: which CHAIR draws to union. **Pass the session's own
            splits.** Every Kaggle session starts from an empty disk and
            re-downloads whatever it needs, so defaulting to all five draws
            would make every session fetch 2452 CHAIR images when it uses 500
            -- roughly 400 MB of waste per session, paid again each time.
            ``None`` means all of them, which is the right answer only when
            the caller genuinely does not know.
    """
    p = paths()
    pope_dir = pope_dir or p.pope_ann_dir
    pope_ids: Set[int] = set()
    for split in splits:
        try:
            f = pope_file_for_split(pope_dir, split)
        except FileNotFoundError:
            continue
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                name = rec.get("image") or rec.get("image_id")
                if isinstance(name, str):
                    pope_ids.add(int(Path(name).stem[-12:]))
                elif isinstance(name, int):
                    pope_ids.add(name)

    chair_ids: Set[int] = set()

    def _ids_from(path: Path) -> Set[int]:
        if not path.is_file():
            return set()
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data = data.get("image_ids", [])
        return {int(x) for x in data}

    if chair_list:
        chair_ids = _ids_from(Path(chair_list))
    else:
        # Exactly the draws asked for. Fetching the d0 list for a d2 job would
        # fail at startup on images that were never downloaded -- the same
        # mismatch between the data step and the run step that killed four
        # jobs in session 1 -- and fetching all five would re-download 2000
        # unused images in every session. The lists ship with the package, so
        # resolving this needs no COCO download.
        for split in (chair_splits
                      if chair_splits is not None
                      else BENCHMARK_SPLITS["chair"]):
            chair_ids |= _ids_from(p.chair_image_list_for(split))

    return {"pope": pope_ids, "chair": chair_ids,
            "union": pope_ids | chair_ids}


def _download_one(
    image_id: int, dest_dir: Path, *, retries: int = 4, timeout: int = 60
) -> Tuple[int, str]:
    """Fetch one image atomically. Returns ``(image_id, status)``."""
    name = image_filename_for_id(image_id)
    final = dest_dir / name
    if _is_complete_jpeg(final):
        return image_id, "skipped"

    url = COCO_IMAGE_URL.format(name=name)
    tmp = dest_dir / f".{name}.part"
    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "foxrepro/0.1 (research reproduction)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
                while True:
                    chunk = r.read(1 << 16)
                    if not chunk:
                        break
                    fh.write(chunk)
            if not _is_complete_jpeg(tmp):
                last = "incomplete JPEG"
                tmp.unlink(missing_ok=True)
                raise OSError(last)
            # Rename only after validation, so an interrupted run never leaves
            # a truncated file that a later run would accept as done.
            tmp.replace(final)
            return image_id, "downloaded"
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            tmp.unlink(missing_ok=True)
            if attempt < retries - 1:
                # Jittered backoff: a public mirror deserves it, and a thundering
                # herd of 8 workers retrying in lockstep is what gets you rate
                # limited.
                time.sleep(min(2 ** attempt, 8) * (0.5 + random.random()))
    return image_id, f"failed ({last})"


def fetch_images(
    image_ids: Iterable[int],
    dest_dir: Path,
    *,
    workers: int = 8,
    progress_every: int = 50,
) -> Dict[str, object]:
    """Download the given ids into ``dest_dir``. Resumable and idempotent."""
    ids = sorted(set(int(i) for i in image_ids))
    dest_dir.mkdir(parents=True, exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []
    t0 = time.time()

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, i, dest_dir): i for i in ids}
        for n, fut in enumerate(cf.as_completed(futures), 1):
            image_id, status = fut.result()
            if status.startswith("failed"):
                counts["failed"] += 1
                failures.append(f"{image_id}: {status}")
            else:
                counts[status] += 1
            if progress_every and n % progress_every == 0:
                rate = n / max(time.time() - t0, 1e-6)
                print(
                    f"[fetch] {n}/{len(ids)}  {rate:.1f}/s  "
                    f"ok={counts['downloaded']} skip={counts['skipped']} "
                    f"fail={counts['failed']}",
                    flush=True,
                )

    total_mb = sum(
        (dest_dir / image_filename_for_id(i)).stat().st_size
        for i in ids
        if (dest_dir / image_filename_for_id(i)).is_file()
    ) / 1024 / 1024
    out = {
        "requested": len(ids), "dest": str(dest_dir),
        "elapsed_s": round(time.time() - t0, 1),
        "total_mb": round(total_mb, 1), **counts,
    }
    if failures:
        out["failures"] = failures[:20]
    return out


def fetch_annotations(dest_dir: Path, *, timeout: int = 600) -> Dict[str, object]:
    """Fetch and unpack the two val2014 annotation files CHAIR needs.

    Only ``instances_val2014.json`` and ``captions_val2014.json`` are kept; the
    train2014 members of the archive are several times larger and unused here.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = ("annotations/instances_val2014.json",
              "annotations/captions_val2014.json")
    if all((dest_dir / Path(w).name).is_file() for w in wanted):
        return {"status": "already present", "dest": str(dest_dir)}

    tmp = dest_dir / ".annotations_trainval2014.zip.part"
    print(f"[fetch] annotations zip (~241 MB) -> {tmp}", flush=True)
    req = urllib.request.Request(
        COCO_ANNOTATIONS_URL,
        headers={"User-Agent": "foxrepro/0.1 (research reproduction)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)

    extracted = []
    with zipfile.ZipFile(tmp) as z:
        for member in wanted:
            if member not in z.namelist():
                continue
            target = dest_dir / Path(member).name
            with z.open(member) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            extracted.append(target.name)
    tmp.unlink(missing_ok=True)
    return {"status": "ok", "dest": str(dest_dir), "extracted": extracted}


def _session_needs(
    session: Optional[int], plan_path: Optional[Path]
) -> Tuple[str, Optional[List[str]]]:
    """``(what, chair_splits)`` for one session, read from the frozen plan.

    Two separate questions, and getting either wrong costs a session:

    * **which benchmarks** -- fetching only POPE made all four CHAIR jobs of
      the first real session fail within seconds of starting;
    * **which CHAIR draws** -- there are five image lists now, and a Kaggle
      session starts from an empty disk every time. Fetching all of them
      would re-download about 400 MB of unused images per session, and
      fetching the wrong one would fail at startup exactly like the first
      case.

    Returns ``(what, None)`` when there is no usable plan, which fetches
    everything rather than guessing.
    """
    if session is None:
        return "union", None
    plan_path = plan_path or (paths().repo_root / "scripts" / "run_plan.json")
    try:
        plan = json.loads(Path(plan_path).read_text())
        ids = plan["sessions"][session - 1]["jobs"]
    except (OSError, KeyError, IndexError, json.JSONDecodeError):
        return "union", None

    benches: Set[str] = set()
    chair_splits: Set[str] = set()
    for j in ids:
        parts = plan["jobs"][j]["id"].split(":")
        bench, split = parts[2], parts[3]
        benches.add(bench)
        if bench == "chair":
            chair_splits.add(split)
    want = sorted(chair_splits) if chair_splits else []
    if benches == {"pope"}:
        return "pope", want
    if benches == {"chair"}:
        return "chair", want
    return "union", want


def _main(argv: Optional[Sequence[str]] = None) -> int:
    p = paths()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--what", choices=("pope", "chair", "union", "auto"), default="auto",
        help="which image set to fetch. 'pope' (500 images, ~78 MB) is enough "
             "for a full POPE run and needs no annotations. 'chair' fetches "
             "the 500 captioning images. 'auto' reads the frozen plan and "
             "fetches exactly what the given --session needs, which is the "
             "right default: fetching only POPE and then running a CHAIR job "
             "fails at startup.",
    )
    ap.add_argument(
        "--session", type=int, default=None,
        help="with --what auto, the session whose jobs decide what to fetch",
    )
    ap.add_argument(
        "--plan", type=Path, default=None,
        help="frozen plan for --what auto (default scripts/run_plan.json)",
    )
    ap.add_argument("--dest", type=Path, default=p.coco_images_dir)
    ap.add_argument("--pope-dir", type=Path, default=p.pope_ann_dir)
    ap.add_argument("--chair-list", type=Path, default=None,
                    help="default: the union of every CHAIR draw")
    ap.add_argument("--workers", type=int, default=8,
                    help="keep this modest; COCO is a free public mirror")
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch only the first N ids (for a smoke test)")
    ap.add_argument("--with-annotations", action="store_true",
                    help="also fetch instances/captions_val2014.json (~241 MB "
                         "zip), required for CHAIR scoring")
    ap.add_argument("--progress-every", type=int, default=50,
                    help="print a progress line every N images (0 = quiet)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be fetched and stop")
    args = ap.parse_args(argv)

    what = args.what
    chair_splits: Optional[List[str]] = None
    if what == "auto":
        what, chair_splits = _session_needs(args.session, args.plan)
        note = f" chair_splits={','.join(chair_splits)}" if chair_splits else ""
        print(f"[fetch] auto -> {what}{note}")
    sets = needed_image_ids(args.pope_dir, args.chair_list,
                            chair_splits=chair_splits)
    ids = sorted(sets[what])
    if not ids:
        print(
            f"[fetch] nothing to do: no {what} image ids found.\n"
            f"        POPE ids come from {args.pope_dir} (shipped in the "
            f"package).\n"
            f"        CHAIR ids come from {args.chair_list}, which "
            f"build_coco_subset.py writes.",
            file=sys.stderr,
        )
        return 1
    if args.limit:
        ids = ids[: args.limit]

    print(f"[fetch] set={what}  ids={len(ids)}  dest={args.dest}")
    print(f"[fetch] pope={len(sets['pope'])} chair={len(sets['chair'])} "
          f"union={len(sets['union'])}")
    if args.dry_run:
        print(f"[fetch] dry run; ~{len(ids) * 208 / 1024:.0f} MB at ~208 KB/image")
        return 0

    report = fetch_images(
        ids, args.dest, workers=args.workers,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2))

    if args.with_annotations:
        print(json.dumps(fetch_annotations(p.coco_ann_dir), indent=2))

    return 1 if report.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(_main())
