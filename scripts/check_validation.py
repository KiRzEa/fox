#!/usr/bin/env python3
"""Read the validation arms and decide each pre-registered prediction.

**CPU only. No GPU, no model, no checkpoint.** Run it locally after session 1
comes back, on the ingested run files.

    python scripts/check_validation.py                 # all available arms
    python scripts/check_validation.py --strict        # exit 1 on any FAIL

The predictions and their readings are fixed in docs/VALIDATION.md, written
before any of these arms ran. This script only applies them; it does not
choose them. Where an arm is missing it says so and moves on, so it is
useful after a partial session.

What it checks, by section of that document:

V1  negative_only   each negative branch's POPE accuracy against the
                    baseline's. PREDICTION: every one is lower.
V2  alpha = 0       token-identity against the baseline, per sample.
                    PREDICTION: identical. One differing token fails.
V4  cf_*            CausalMM's shuffle against constant and blind.
                    PREDICTION: all three agree.
V5  opera_*         OPERA's three-way decomposition.
V6  diverged_frac   read off every paired run. PREDICTION: strictly between
                    0 and 1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PASS, FAIL, SKIP, INFO = "PASS", "FAIL", "n/a ", "    "


def load(path: Path) -> Tuple[List[dict], dict]:
    from foxrepro.bench.evaluate import read_run

    return read_run(path, strict=False)


def find(runs: Path, method: str, *, benchmark: str = "pope",
         split: str = "adversarial", seed: int = 42,
         control: Optional[str] = None, decoding: str = "sampling",
         **variant) -> Optional[Path]:
    """Locate a run file by what it is, not by guessing its hash."""
    from foxrepro.bench.generate import GenConfig
    from foxrepro.storage import MAX_NEW_TOKENS_DEFAULT

    cfg = GenConfig(
        method=method, benchmark=benchmark, split=split, seed=seed,
        dtype="fp16", model_id="llava-hf/llava-1.5-7b-hf",
        prompt_template_id=("pope_v0_raw" if benchmark == "pope"
                            else "chair_v1"),
        max_new_tokens=MAX_NEW_TOKENS_DEFAULT[benchmark],
        decoding=decoding, control=control, **variant,
    )
    path = runs / cfg.run_key().filename()
    return path if path.is_file() else None


def pope_accuracy(records: List[dict], split: str) -> float:
    """Accuracy as a percentage, the way every table in this project prints it."""
    from foxrepro.bench.evaluate import evaluate_pope

    metrics, _ = evaluate_pope(records, labels=None)
    return 100.0 * float(metrics["accuracy"]["point"])


def outputs_by_sample(records: List[dict]) -> Dict[str, str]:
    return {r["sample_id"]: r["output"] for r in records}


def say(status: str, text: str) -> None:
    colour = {PASS: "32", FAIL: "31", SKIP: "33"}.get(status)
    tag = f"\033[{colour}m[{status}]\033[0m" if colour else f" {status} "
    print(f"  {tag} {text}")


# --------------------------------------------------------------------------- #
def v1_negative_branch_is_worse(runs: Path) -> List[str]:
    """Every contrastive method claims it subtracts a WORSE model."""
    print("\nV1  Is the negative branch actually worse?")
    base = find(runs, "sampling")
    if base is None:
        say(SKIP, "no sampling baseline at seed 42; cannot compare")
        return []
    base_acc = pope_accuracy(load(base)[0], "adversarial")
    say(INFO, f"baseline accuracy {base_acc:.2f}")
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        path = find(runs, method, control="negative_only")
        if path is None:
            say(SKIP, f"{method}: negative_only not run yet")
            continue
        acc = pope_accuracy(load(path)[0], "adversarial")
        delta = acc - base_acc
        if delta < 0:
            say(PASS, f"{method}: {acc:.2f} ({delta:+.2f}) -- premise holds")
            verdicts.append(PASS)
        else:
            say(FAIL, f"{method}: {acc:.2f} ({delta:+.2f}) -- the branch this "
                      f"method SUBTRACTS is not worse than the baseline, so "
                      f"whatever it gains is not the described mechanism")
            verdicts.append(FAIL)
    return verdicts


def v2_alpha_zero_collapses(runs: Path) -> List[str]:
    """At alpha = 0 the rule is the identity, so the tokens must match."""
    print("\nV2  Does alpha = 0 reproduce the baseline exactly?")
    base = find(runs, "sampling")
    if base is None:
        say(SKIP, "no sampling baseline at seed 42; cannot compare")
        return []
    base_out = outputs_by_sample(load(base)[0])
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        path = find(runs, method, alpha=0.0)
        if path is None:
            say(SKIP, f"{method}: alpha=0 arm not run yet")
            continue
        got = outputs_by_sample(load(path)[0])
        shared = sorted(set(got) & set(base_out))
        if not shared:
            say(SKIP, f"{method}: no overlapping sample ids")
            continue
        diff = [s for s in shared if got[s] != base_out[s]]
        if not diff:
            say(PASS, f"{method}: {len(shared)} samples, all identical")
            verdicts.append(PASS)
        else:
            say(FAIL, f"{method}: {len(diff)}/{len(shared)} samples differ "
                      f"(e.g. {diff[0]}: {base_out[diff[0]]!r} vs "
                      f"{got[diff[0]]!r}). At alpha=0 the rule is the "
                      f"identity, so this is a wiring fault, not a tuning "
                      f"difference")
            verdicts.append(FAIL)
    return verdicts


def v4_is_the_shuffle_load_bearing(runs: Path) -> List[str]:
    """CausalMM's gain is 1/577 whatever the attention is."""
    print("\nV4  Is CausalMM's shuffle doing the work, or just its magnitude?")
    base = find(runs, "causalmm")
    if base is None:
        say(SKIP, "plain causalmm not run yet")
        return []
    ref = outputs_by_sample(load(base)[0])
    verdicts = []
    for control in ("cf_constant", "cf_blind"):
        path = find(runs, "causalmm", control=control)
        if path is None:
            say(SKIP, f"{control} not run yet")
            continue
        got = outputs_by_sample(load(path)[0])
        shared = sorted(set(got) & set(ref))
        if not shared:
            say(SKIP, f"{control}: no overlapping sample ids")
            continue
        same = sum(1 for s in shared if got[s] == ref[s])
        frac = same / len(shared)
        verdict = PASS if frac > 0.99 else INFO
        say(verdict, f"{control}: agrees with shuffle on {same}/{len(shared)} "
                     f"({frac:.1%})"
                     + ("  -- the vision counterfactual is not what works; "
                        "CausalMM is a language-prior contrast"
                        if frac > 0.99 else
                        "  -- the shuffle carries signal after all"))
        verdicts.append(verdict)
    return verdicts


def v5_opera_decomposition(runs: Path) -> List[str]:
    """Which third of OPERA is working?"""
    print("\nV5  OPERA: beam search, candidate restriction, or penalty?")
    plain = find(runs, "opera", decoding="greedy")
    if plain is None:
        say(SKIP, "plain opera not run yet")
        return []
    plain_acc = pope_accuracy(load(plain)[0], "adversarial")
    say(INFO, f"opera accuracy {plain_acc:.2f}")
    for control in ("opera_no_penalty", "opera_unrestricted"):
        path = find(runs, "opera", decoding="greedy", control=control)
        if path is None:
            say(SKIP, f"{control} not run yet")
            continue
        acc = pope_accuracy(load(path)[0], "adversarial")
        delta = acc - plain_acc
        note = ""
        if control == "opera_no_penalty" and abs(delta) < 0.2:
            note = ("  -- the over-trust penalty is INERT; OPERA's POPE "
                    "number is beam search with a truncated candidate set")
        say(INFO, f"{control}: {acc:.2f} ({delta:+.2f}){note}")
    return []


def v6_divergence(runs: Path) -> List[str]:
    """Did each rule ever change a decision?"""
    print("\nV6  Does each rule change any decision?")
    verdicts = []
    for method in ("vcd", "icd", "sid", "causalmm"):
        for benchmark, split in (("pope", "adversarial"),
                                 ("chair", "val2014_500")):
            path = find(runs, method, benchmark=benchmark, split=split)
            if path is None:
                continue
            records = load(path)[0]
            fracs = [
                (r.get("method_extra") or {}).get("diverged_frac")
                for r in records
            ]
            fracs = [f for f in fracs if f is not None]
            if not fracs:
                say(SKIP, f"{method}/{benchmark}: no diverged_frac recorded")
                continue
            mean = sum(fracs) / len(fracs)
            if mean <= 0.0:
                say(FAIL, f"{method}/{benchmark}: diverged_frac 0.000 -- the "
                          f"rule never changed a decision. The method is "
                          f"inert and no accuracy figure would show it")
                verdicts.append(FAIL)
            elif mean >= 1.0:
                say(FAIL, f"{method}/{benchmark}: diverged_frac 1.000 -- the "
                          f"rule overrode the model on every token")
                verdicts.append(FAIL)
            else:
                say(PASS, f"{method}/{benchmark}: diverged_frac {mean:.3f}")
                verdicts.append(PASS)
    return verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=None,
                    help="default: the live runs directory")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any prediction failed")
    args = ap.parse_args(argv)

    from foxrepro.storage import paths

    runs = args.runs_dir or paths().runs_root
    if not runs.is_dir():
        print(f"no runs directory at {runs}", file=sys.stderr)
        return 2
    print(f"# Validation arms, read from {runs}")
    print("# Predictions are pre-registered in docs/VALIDATION.md.")

    verdicts: List[str] = []
    for check in (v1_negative_branch_is_worse, v2_alpha_zero_collapses,
                  v4_is_the_shuffle_load_bearing, v5_opera_decomposition,
                  v6_divergence):
        verdicts += check(runs)

    n_fail = verdicts.count(FAIL)
    n_pass = verdicts.count(PASS)
    print(f"\n{n_pass} predictions held, {n_fail} failed.")
    if n_fail:
        print("A FAIL means the method is not doing what its paper describes "
              "on this stack. Read docs/VALIDATION.md for what each one "
              "implies before running the table rows.")
    return 1 if (args.strict and n_fail) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    raise SystemExit(main())
