#!/usr/bin/env python3
"""Probe a rented/ephemeral GPU host for Fox-reproduction feasibility.

Run this FIRST on any new host (Kaggle T4x2, Colab, RunPod, Vast) before
downloading 14 GB of weights. It answers the four things that cannot be
determined remotely:

  1. Is the Python version compatible with Track A?
     (transformers 4.29.2 -> tokenizers 0.13.3, wheels cp37-cp311 only)
  2. Does this GPU support the dtypes the experiment needs?
     In particular bfloat16 CAST + ADD on pre-Ampere cards (T4 = sm75).
  3. Can the model be sharded across multiple GPUs, and is there headroom?
  4. What is the actual fp16 ULP at each candidate gamma?
     This is the mechanism under study, so measure it, do not assume it.

Usage:
    python probe_host.py            # full probe
    python probe_host.py --json     # machine-readable

Needs only torch. Reports PASS / WARN / FAIL per check and an overall verdict.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys

# Candidate gammas for the magnitude sweep. The claim under test is that the
# intervention's effect is governed by the fp16 ULP at |gamma|, not by gamma
# itself, so these are chosen to land in four different binades.
GAMMAS = [1e2, 1e3, 1e4, 6e4, 1e5]

# Memory model for LLaVA-1.5-7B (derived from config.json, not guessed):
#   7.063e9 params; 32 layers x 32 heads x 128 head_dim, no GQA
#   KV = 2 * 32 * 32 * 128 * bytes = 0.5 MiB/token at fp16
#   seq = 576 image + ~24 prompt + 512 generated = 1112 tokens, x2 branches
WEIGHTS_GIB_FP16 = 13.16
KV_GIB_FP16_2BRANCH = 1.09
OVERHEAD_GIB = 1.2  # CUDA context + allocator slack + eager transient


def _status(ok: bool, warn: bool = False) -> str:
    return "WARN" if warn else ("PASS" if ok else "FAIL")


def check_python() -> dict:
    major, minor = sys.version_info[:2]
    track_a = (major, minor) <= (3, 11) and (major, minor) >= (3, 8)
    return {
        "check": "python_version",
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "track_a_ok": track_a,
        "status": _status(track_a),
        "note": (
            "Track A (vendored transformers 4.29.2) is installable."
            if track_a
            else "Track A needs Python <=3.11: tokenizers 0.13.3 has no cp312+ "
            "wheel and would build from Rust source. Use `uv venv --python 3.10`, "
            "or run Track B only on this host."
        ),
    }


def check_torch() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "check": "torch",
            "status": "FAIL",
            "note": "torch not importable; install it before probing further.",
        }
    cuda = torch.cuda.is_available()
    return {
        "check": "torch",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda,
        "arch_list": torch.cuda.get_arch_list() if cuda else [],
        "status": _status(cuda),
        "note": (
            "Check your GPU's sm_XX appears in arch_list. If not, the wheel has "
            "no native cubin and relies on CUDA minor-version binary "
            "compatibility (usually fine for sm_89; NOT available for Blackwell "
            "sm_100/sm_120)."
        ),
    }


def check_gpus() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"check": "gpus", "status": "FAIL", "note": "no CUDA device"}

    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append(
            {
                "index": i,
                "name": p.name,
                "sm": f"{p.major}.{p.minor}",
                "total_gib": round(p.total_memory / 1024**3, 2),
                "bf16_native_compute": p.major >= 8,  # bf16 matmul needs sm80+
            }
        )

    n = len(devices)
    total = sum(d["total_gib"] for d in devices)
    smallest = min(d["total_gib"] for d in devices)

    # Sharding by layer splits weights and KV cache evenly; overhead is per-GPU.
    per_gpu_need = (WEIGHTS_GIB_FP16 + KV_GIB_FP16_2BRANCH) / n + OVERHEAD_GIB
    fits = per_gpu_need < smallest * 0.95

    return {
        "check": "gpus",
        "count": n,
        "devices": devices,
        "total_gib": round(total, 2),
        "est_per_gpu_need_gib": round(per_gpu_need, 2),
        "smallest_gpu_gib": smallest,
        "fits_sharded": fits,
        "status": _status(fits),
        "note": (
            f"{n} GPU(s). Estimated {per_gpu_need:.1f} GiB needed per GPU when "
            f"sharded with device_map='auto', against {smallest:.1f} GiB "
            f"available on the smallest. "
            + (
                "Fits with headroom."
                if fits
                else "Does NOT fit. Quantization is ruled out for this project "
                "(it changes the quantity under study), so this host is unusable "
                "for the reproduction."
            )
        ),
    }


def check_dtypes() -> dict:
    """The decisive check on pre-Ampere cards.

    The experiment needs, inside the attention function:
        logits.to(TARGET_DTYPE) -> add a constant -> cast back -> softmax(fp32)
    That is a CAST and an elementwise ADD, not a matmul. bf16 matmul needs
    sm80+, but cast+add may still work on sm75. Measure rather than assume.
    """
    import torch

    results = {}
    for name, dt in [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ]:
        entry: dict = {"cast_add": False, "matmul": False, "softmax_fp32": False}
        try:
            x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
            y = (x.to(dt) - 10000.0).to(torch.float16)
            torch.cuda.synchronize()
            entry["cast_add"] = bool(torch.isfinite(y).any().item())
        except Exception as exc:  # noqa: BLE001 - we want the reason verbatim
            entry["cast_add_error"] = f"{type(exc).__name__}: {exc}"
        try:
            a = torch.randn(64, 64, device="cuda", dtype=dt)
            (a @ a).sum().item()
            torch.cuda.synchronize()
            entry["matmul"] = True
        except Exception as exc:  # noqa: BLE001
            entry["matmul_error"] = f"{type(exc).__name__}: {exc}"
        try:
            a = torch.randn(8, 64, device="cuda", dtype=dt)
            torch.nn.functional.softmax(a, dim=-1, dtype=torch.float32).sum().item()
            torch.cuda.synchronize()
            entry["softmax_fp32"] = True
        except Exception as exc:  # noqa: BLE001
            entry["softmax_fp32_error"] = f"{type(exc).__name__}: {exc}"
        results[name] = entry

    fp16_ok = results["float16"]["cast_add"]
    fp32_ok = results["float32"]["cast_add"]
    bf16_ok = results["bfloat16"]["cast_add"]

    core_ok = fp16_ok and fp32_ok
    return {
        "check": "dtypes",
        "results": results,
        "fp16_arm": fp16_ok,
        "fp32_arm": fp32_ok,
        "bf16_arm": bf16_ok,
        "status": _status(core_ok, warn=core_ok and not bf16_ok),
        "note": (
            "All three dtype arms available."
            if core_ok and bf16_ok
            else (
                "fp16 and fp32 arms work; bf16 cast+add is NOT available on this "
                "card. This is survivable: the gamma magnitude sweep spans ULP "
                "0.0625 -> 32 entirely within fp16, which gives the full "
                "dose-response curve. bf16 (ULP 64) becomes an optional "
                "confirmatory row, not a required arm."
                if core_ok
                else "Core fp16/fp32 arms unavailable -- investigate before proceeding."
            )
        ),
    }


def check_ulp() -> dict:
    """Measure the fp16 ULP at each candidate gamma, empirically.

    This is the mechanism the paper's effect actually rides on, so it is worth
    measuring on the real host rather than trusting the binade arithmetic.
    """
    import torch

    rows = []
    for g in GAMMAS:
        t = torch.tensor([g], dtype=torch.float16)
        overflow = not bool(torch.isfinite(t).all().item())
        if overflow:
            rows.append(
                {
                    "gamma": g,
                    "overflows_fp16": True,
                    "ulp": None,
                    "note": "exceeds fp16 max (65504) -> becomes inf; unusable",
                }
            )
            continue
        nxt = torch.nextafter(t, torch.tensor([float("inf")], dtype=torch.float16))
        ulp = float((nxt - t).item())
        binade = int(math.floor(math.log2(abs(g))))
        rows.append(
            {
                "gamma": g,
                "overflows_fp16": False,
                "ulp": ulp,
                "binade": f"[2^{binade}, 2^{binade + 1})",
                "predicted_ulp": 2.0 ** (binade - 10),  # fp16 has 10 mantissa bits
            }
        )

    usable = [r for r in rows if not r["overflows_fp16"]]
    spread = (
        max(r["ulp"] for r in usable) / min(r["ulp"] for r in usable)
        if len(usable) > 1
        else 1.0
    )
    ok = spread >= 100
    return {
        "check": "fp16_ulp_sweep",
        "rows": rows,
        "ulp_spread_factor": round(spread, 1),
        "status": _status(ok),
        "note": (
            f"Usable gammas span a {spread:.0f}x range of ULP, which is enough "
            "for a dose-response curve in fp16 alone. A monotone response in ULP "
            "rather than in gamma magnitude is the signature of the "
            "requantization account of the paper's mechanism."
        ),
    }


def check_tf32() -> dict:
    """TF32 silently reduces an 'fp32' matmul to ~10 mantissa bits on Ampere+.

    That would contaminate the fp32 arm into looking like fp16. Report the
    current flags so they can be pinned and logged.
    """
    import torch

    flags = {
        "cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn.allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda.matmul.allow_fp16_reduced_precision_reduction": (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
    }
    try:
        flags["cuda.matmul.allow_bf16_reduced_precision_reduction"] = (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        )
    except AttributeError:
        pass

    risky = [k for k, v in flags.items() if v]
    return {
        "check": "tf32_precision_flags",
        "flags": flags,
        "need_pinning": risky,
        "status": _status(True, warn=bool(risky)),
        "note": (
            "Set all of these to False before the dtype experiment and LOG them "
            "into every results file. cudnn.allow_tf32 defaults to True, and on "
            "Ampere+ a TF32 matmul has ~10 mantissa bits -- roughly fp16 "
            "precision in the QK product -- which would make the fp32 arm "
            "indistinguishable from fp16 for the wrong reason."
            if risky
            else "All precision flags already off."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    checks = [check_python(), check_torch()]
    if checks[-1].get("cuda_available"):
        checks += [check_gpus(), check_dtypes(), check_ulp(), check_tf32()]

    fails = [c for c in checks if c["status"] == "FAIL"]
    warns = [c for c in checks if c["status"] == "WARN"]
    verdict = "UNUSABLE" if fails else ("USABLE WITH CAVEATS" if warns else "READY")

    if args.json:
        print(json.dumps({"verdict": verdict, "checks": checks}, indent=2))
        return 1 if fails else 0

    print("=" * 72)
    print("Fox reproduction -- host probe")
    print("=" * 72)
    for c in checks:
        print(f"\n[{c['status']:4s}] {c['check']}")
        for k, v in c.items():
            if k in ("check", "status", "note"):
                continue
            print(f"       {k}: {v}")
        if c.get("note"):
            print(f"       -> {c['note']}")
    print("\n" + "=" * 72)
    print(f"VERDICT: {verdict}")
    if fails:
        print("  blocking: " + ", ".join(c["check"] for c in fails))
    if warns:
        print("  caveats : " + ", ".join(c["check"] for c in warns))
    print("=" * 72)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
