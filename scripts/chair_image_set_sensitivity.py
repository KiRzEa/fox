#!/usr/bin/env python3
"""How much does CHAIR move when only the image set changes?

The question this answers is whether the 3.88-point gap between our CHAIR_S
baseline and the paper's needs an explanation beyond "a different 500 images".

It cannot be settled by looking harder at one draw. ``char_eval.py:136`` takes
``os.listdir`` of ``/data-disk/only_test`` -- a directory never published --
shuffles it and keeps 500, so the paper's set is not recoverable and no amount
of care reproduces it. What *is* available is the spread: run the same
baseline over five independent 500-image draws and see how far the number
travels on its own.

Two questions, not one, and they can disagree:

1. **Absolute.** Does the observed range of CHAIR_S cover the paper's 57.60?
   If it does, the discrepancy is image-set noise and needs no further
   account. If 57.60 sits well outside, something else is going on.

2. **The gap.** Does ``baseline - fox`` move too? This matters more. Every
   CHAIR interval reported so far is computed within one draw, so it sees
   item difficulty and generation noise but is blind to which 500 images were
   drawn -- the same shape of blindness that one seed had before the ten-seed
   replication fixed it. If the gap swings as widely as the absolute number,
   then "CHAIR_S -6.98 [-9.25, -4.71] excludes the paper's -11.20" is a
   weaker claim than it reads, and should be restated.

Usage::

    python3 scripts/chair_image_set_sensitivity.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: What the paper prints for LLaVA-1.5 CHAIR under greedy decoding (Table 9),
#: which is the decoding these draws use.
PAPER_GREEDY = {"baseline_s": 50.90, "fox_s": 42.60,
                "baseline_i": 15.40, "fox_i": 12.16}
#: ...and under sampling (Table 2), where our ten-seed comparison lives.
PAPER_SAMPLING = {"baseline_s": 57.60, "fox_s": 46.40,
                  "baseline_i": 17.18, "fox_i": 12.90}


def main() -> int:
    from foxrepro.storage import BENCHMARK_SPLITS, paths

    rows = {}
    for f in paths().reports_dir.glob("*.metrics.json"):
        d = json.loads(f.read_text())
        m, p = d["metrics"], d["provenance"]
        if m["benchmark"] != "chair" or (p.get("decoding") or "sampling") != "greedy":
            continue
        if p.get("mode") not in (None, "as_implemented"):
            continue
        if p.get("logit_dtype") not in (None, "native"):
            continue
        if int(m["totals"]["chair_s_denominator"]) != 500:
            continue
        rows.setdefault(p["split"], {})[p["method"]] = m["totals"]

    draws = [s for s in BENCHMARK_SPLITS["chair"]
             if s in rows and {"sampling", "fox"} <= set(rows[s])]
    if len(draws) < 2:
        print(f"need at least two scored draws, have {len(draws)}: {draws}\n"
              f"run bench.evaluate over the chair_draws runs first",
              file=sys.stderr)
        return 1

    def col(split, method, key):
        return 100 * rows[split][method][key]

    print(f"CHAIR under greedy, {len(draws)} independent 500-image draws\n")
    hdr = f"{'draw':20s} {'base_S':>8s} {'fox_S':>8s} {'gap_S':>8s}   " \
          f"{'base_I':>7s} {'fox_I':>7s} {'gap_I':>7s}"
    print(hdr)
    print("-" * len(hdr))
    bs, fs, gs, bi, fi, gi = [], [], [], [], [], []
    for d in draws:
        b_s, f_s = col(d, "sampling", "chair_s"), col(d, "fox", "chair_s")
        b_i, f_i = col(d, "sampling", "chair_i"), col(d, "fox", "chair_i")
        bs.append(b_s); fs.append(f_s); gs.append(f_s - b_s)
        bi.append(b_i); fi.append(f_i); gi.append(f_i - b_i)
        print(f"{d:20s} {b_s:8.2f} {f_s:8.2f} {f_s-b_s:+8.2f}   "
              f"{b_i:7.2f} {f_i:7.2f} {f_i-b_i:+7.2f}")

    def summarise(name, v, paper=None):
        lo, hi = min(v), max(v)
        sd = st.stdev(v) if len(v) > 1 else 0.0
        line = (f"{name:20s} mean {st.mean(v):7.2f}  sd {sd:5.2f}  "
                f"range [{lo:.2f}, {hi:.2f}]  width {hi-lo:5.2f}")
        if paper is not None:
            inside = lo <= paper <= hi
            line += f"   paper {paper:6.2f} -> {'INSIDE' if inside else 'OUTSIDE'}"
        print(line)

    print()
    summarise("baseline CHAIR_S", bs, PAPER_GREEDY["baseline_s"])
    summarise("fox CHAIR_S", fs, PAPER_GREEDY["fox_s"])
    summarise("gap CHAIR_S", gs,
              PAPER_GREEDY["fox_s"] - PAPER_GREEDY["baseline_s"])
    print()
    summarise("baseline CHAIR_I", bi, PAPER_GREEDY["baseline_i"])
    summarise("fox CHAIR_I", fi, PAPER_GREEDY["fox_i"])
    summarise("gap CHAIR_I", gi,
              PAPER_GREEDY["fox_i"] - PAPER_GREEDY["baseline_i"])

    print("\nWhat to read from this:")
    width_abs, width_gap = max(bs) - min(bs), max(gs) - min(gs)
    print(f"  * the absolute CHAIR_S of an unchanged baseline moves "
          f"{width_abs:.2f} points across draws")
    print(f"  * the gap moves {width_gap:.2f} points")
    if width_gap < width_abs / 2:
        print("    -> pairing absorbs most of it, as intended: the gap is the "
              "more stable quantity and the comparison survives the image set")
    else:
        print("    -> pairing does NOT absorb it. Every CHAIR interval in this "
              "project is computed within one draw and is therefore blind to "
              "this. Restate the CHAIR conclusions accordingly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
