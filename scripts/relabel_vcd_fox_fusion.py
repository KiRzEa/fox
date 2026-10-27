#!/usr/bin/env python3
"""Relabel the archived `vcd` runs as `vcd_fox_fusion`. CPU only, no GPU.

Rounds 1-5 ran one compared-method arm under the name ``vcd``. It was never
VCD: it put VCD's noised image into **Fox's** additive gated fusion, and
every record says so in ``method_extra.variant``.

Now that VCD's own contrastive rule exists, ``vcd`` means VCD -- and the two
collide in the run key, because ``method`` is part of the file name but the
variant hash is identical. Concretely, before this script:

    the new real-VCD job's file name == an archived hybrid file name

so ``scan_progress`` marked the job finished and six VCD jobs dropped out of
the plan without a word; and a run launched anyway would have resumed from
the hybrid's 3000 records, skipped every sample, and reported a finished
"VCD" row carrying the hybrid's numbers.

This rewrites the ``method`` field and the file name. It is a correction to a
LABEL, not to a measurement: not one output, score or provenance field other
than ``method`` changes, and the script refuses to touch a file whose records
do not all carry the hybrid's variant tag.

    python scripts/relabel_vcd_fox_fusion.py --dry-run
    python scripts/relabel_vcd_fox_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OLD = "vcd"
NEW = "vcd_fox_fusion"
VARIANT = "vcd-noise-fox-fusion"


def expected_name(records: list, old_path: Path) -> str:
    """The file name the same run would get under the corrected method.

    Derived from ``RunKey`` rather than by string surgery on the old name, so
    a change to the naming scheme cannot silently produce an orphan.
    """
    from foxrepro.storage import VARIANT_FIELDS, RunKey

    rec = records[0]
    key = RunKey(
        method=NEW, benchmark=rec["benchmark"], split=rec["split"],
        seed=rec["seed"], dtype=rec["dtype"],
        variant_fields={f: rec.get(f) for f in VARIANT_FIELDS},
    )
    name = key.filename()
    # The variant hash must not have moved: `method` is a separate RunKey
    # field, so only the m- prefix may differ.
    if name.split("__", 1)[1] != old_path.name.split("__", 1)[1]:
        raise SystemExit(
            f"refusing to rename {old_path.name}: the variant hash changed to "
            f"{name}. Something other than the method label is different."
        )
    return name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    files = sorted(args.runs_dir.glob(f"m-{OLD}__*.jsonl"))
    if not files:
        print(f"nothing named m-{OLD}__* in {args.runs_dir}")
        return 0

    for path in files:
        records = [json.loads(line) for line in
                   path.read_text().splitlines() if line.strip()]
        if not records:
            raise SystemExit(f"{path} is empty")

        variants = {
            (r.get("method_extra") or {}).get("variant") for r in records
        }
        methods = {r.get("method") for r in records}
        if methods != {OLD} or variants != {VARIANT}:
            raise SystemExit(
                f"refusing to touch {path.name}: methods={sorted(methods)} "
                f"variants={sorted(v for v in variants if v)}. Only records "
                f"that all say variant={VARIANT!r} are the hybrid."
            )

        target = path.with_name(expected_name(records, path))
        if target.exists():
            raise SystemExit(f"{target.name} already exists; refusing")

        print(f"{path.name}\n  -> {target.name}  ({len(records)} records)")
        if args.dry_run:
            continue

        for r in records:
            r["method"] = NEW
        body = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
        target.write_text(body)

        # Verify before deleting: read it back and check nothing but the
        # label moved.
        back = [json.loads(line) for line in
                target.read_text().splitlines() if line.strip()]
        if len(back) != len(records):
            raise SystemExit(f"{target.name}: wrote {len(back)} of {len(records)}")
        for a, b in zip(records, back):
            if a != b:
                raise SystemExit(f"{target.name}: record mismatch after write")
        path.unlink()

    if args.dry_run:
        print("\n(dry run -- nothing written)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
