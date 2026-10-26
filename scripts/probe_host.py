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
from pathlib import Path

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

#: Per-checkpoint VRAM model, so ``--model-id`` means the same thing to the
#: GPU check as it does to the disk check. Without this the GPU estimate
#: silently stayed LLaVA's whichever checkpoint was named -- a probe that
#: passes for the wrong model is worse than no probe.
#:
#: InstructBLIP's weights are the awkward entry: 29.5 GiB of fp32 safetensors
#: is ~7.9e9 parameters, so ~14.8 GiB at fp16, plus a ViT-g vision tower and a
#: Q-Former that LLaVA has no counterpart for. Its visual stream is 32
#: Q-Former query tokens rather than 576 patch tokens, so the KV cache is
#: smaller even though the weights are larger.
MODEL_MEMORY_GIB = {
    "llava-hf/llava-1.5-7b-hf": {
        "weights_fp16": 13.16, "kv_fp16_2branch": 1.09,
    },
    "Salesforce/instructblip-vicuna-7b": {
        "weights_fp16": 15.80, "kv_fp16_2branch": 0.55,
    },
}


def _memory_model(model_id: str) -> dict:
    """VRAM figures for ``model_id``, falling back to LLaVA's with a note."""
    known = MODEL_MEMORY_GIB.get(model_id)
    if known:
        return {**known, "estimated": False}
    return {"weights_fp16": WEIGHTS_GIB_FP16,
            "kv_fp16_2branch": KV_GIB_FP16_2BRANCH, "estimated": True}


def _status(ok: bool, warn: bool = False) -> str:
    return "WARN" if warn else ("PASS" if ok else "FAIL")


def check_python() -> dict:
    """Python version, and which tracks it admits.

    Track B (the primary environment) needs only >=3.10. Track A -- the
    fidelity oracle that runs the paper's own vendored transformers 4.29.2 --
    additionally needs <=3.11, because ``tokenizers==0.13.3`` publishes wheels
    for cp37-cp311 only and would otherwise build from Rust source.

    Track A being unavailable is a CAVEAT, not a blocker: thanks to the
    bit-equivalence proof in ``tests/test_kernel_equivalence.py``, the
    mechanism is validated without running any model, and Track A is reduced
    to one end-to-end token-level cross-check that can run on any other host.
    So this returns WARN, never FAIL.
    """
    major, minor = sys.version_info[:2]
    track_b = (major, minor) >= (3, 10)
    track_a = (3, 8) <= (major, minor) <= (3, 11)
    return {
        "check": "python_version",
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "track_b_ok": track_b,
        "track_a_ok": track_a,
        "status": _status(track_b, warn=track_b and not track_a),
        "note": (
            "Both tracks installable."
            if track_a and track_b
            else (
                "Track B (primary) is fine. Track A is NOT installable here: it "
                "needs Python <=3.11 for tokenizers 0.13.3. This is a caveat, "
                "not a blocker -- the bit-equivalence proof validates the "
                "mechanism with no model at all, so Track A is only an "
                "end-to-end cross-check and can run elsewhere (a local "
                "`uv venv --python 3.10`, or a RunPod pytorch:2.1.2 image)."
                if track_b
                else f"Python {major}.{minor} is below the 3.10 floor; "
                "neither track will install."
            )
        ),
    }


def check_preinstalled_stack() -> dict:
    """What the host already provides, and whether it can do what Track B needs.

    The decisive question is not a version number but a capability: Track B
    implements the intervention as a custom attention function registered into
    transformers' pluggable attention interface, which is the only mechanism
    that grants write access to the pre-softmax attention logits at a chosen
    (layer, head, query-position) slice. If ``AttentionInterface`` and
    ``AttentionMaskInterface`` are both importable, the host can run Track B
    with whatever it already has, and forcing a torch downgrade is pure cost.
    """
    out: dict = {"check": "preinstalled_stack", "versions": {}}
    for mod in ("torch", "transformers", "tokenizers", "huggingface_hub",
                "accelerate", "numpy", "gradio"):
        try:
            m = __import__(mod)
            out["versions"][mod] = getattr(m, "__version__", "unknown")
        except BaseException:  # noqa: BLE001
            out["versions"][mod] = None

    caps: dict = {}
    try:
        import transformers  # noqa: PLC0415

        caps["AttentionInterface"] = hasattr(transformers, "AttentionInterface")
        caps["AttentionMaskInterface"] = hasattr(transformers, "AttentionMaskInterface")
        try:
            from transformers.modeling_utils import (  # noqa: PLC0415
                ALL_ATTENTION_FUNCTIONS,
            )

            caps["ALL_ATTENTION_FUNCTIONS"] = True
            caps["registered_backends"] = sorted(
                getattr(ALL_ATTENTION_FUNCTIONS, "valid_keys", lambda: [])()
            ) or "unlistable"
        except BaseException:  # noqa: BLE001
            caps["ALL_ATTENTION_FUNCTIONS"] = False
        try:
            from transformers.integrations.sdpa_attention import (  # noqa: PLC0415, F401
                sdpa_attention_forward,
            )

            caps["sdpa_attention_forward"] = True
        except BaseException:  # noqa: BLE001
            caps["sdpa_attention_forward"] = False
        try:
            from transformers.masking_utils import eager_mask  # noqa: PLC0415, F401

            caps["eager_mask"] = True
        except BaseException:  # noqa: BLE001
            caps["eager_mask"] = False
        try:
            from transformers import (  # noqa: PLC0415, F401
                LlavaForConditionalGeneration,
            )

            caps["LlavaForConditionalGeneration"] = True
        except BaseException:  # noqa: BLE001
            caps["LlavaForConditionalGeneration"] = False
    except BaseException:  # noqa: BLE001
        caps["transformers_importable"] = False

    out["track_b_capabilities"] = caps
    needed = (
        "AttentionInterface", "AttentionMaskInterface", "sdpa_attention_forward",
        "eager_mask", "LlavaForConditionalGeneration",
    )
    missing = [k for k in needed if not caps.get(k)]
    out["missing_for_track_b"] = missing
    out["status"] = _status(not missing, warn=bool(missing))
    out["note"] = (
        "The preinstalled transformers already exposes everything Track B "
        "needs. Do NOT downgrade torch -- use what is here and pin only what "
        "is missing."
        if not missing
        else "Missing: " + ", ".join(missing) + ". Install "
        "requirements-trackB.lock, then re-run this probe."
    )
    return out


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


def check_gpus(model_id: str = "llava-hf/llava-1.5-7b-hf") -> dict:
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
    mem = _memory_model(model_id)
    per_gpu_need = (mem["weights_fp16"] + mem["kv_fp16_2branch"]) / n + OVERHEAD_GIB
    fits = per_gpu_need < smallest * 0.95

    return {
        "check": "gpus",
        "model_id": model_id,
        "weights_gib_fp16": mem["weights_fp16"],
        "memory_model_estimated": mem["estimated"],
        "count": n,
        "devices": devices,
        "total_gib": round(total, 2),
        "est_per_gpu_need_gib": round(per_gpu_need, 2),
        "smallest_gpu_gib": smallest,
        "fits_sharded": fits,
        "status": _status(fits),
        "note": (
            f"{n} GPU(s). {model_id} needs an estimated "
            f"{per_gpu_need:.1f} GiB per GPU when sharded with "
            f"device_map='auto', against {smallest:.1f} GiB on the smallest. "
            + ("No VRAM figures are recorded for this checkpoint, so LLaVA's "
               "are used -- treat the margin with suspicion. " if mem["estimated"]
               else "")
            + (
                "Fits with headroom."
                if fits
                else "Does NOT fit. Quantization is ruled out for this project "
                "(it changes the quantity under study), so this host is unusable "
                "for the reproduction."
            )
        ),
    }


#: Fallback download sizes, in GiB, used only when the HuggingFace API cannot
#: be reached. The live figure is derived from the repo's own file listing --
#: a hardcoded number here would read as measured and be asserted, which is
#: the mistake probe_backbone.py was corrected for.
#:
#: These are what HuggingFace actually transfers -- `from_pretrained` prefers
#: safetensors and skips the duplicate `.bin` shards, so the figure is the
#: safetensors total, not the repository total.
#:
#: InstructBLIP is the awkward one: it is stored in **fp32**, so it downloads
#: at roughly twice LLaVA-1.5 even though it occupies about the same VRAM once
#: cast to fp16. The fix is not a format conversion -- it is already a
#: HuggingFace repo -- but a one-time re-save at fp16. See docs/DEVIATIONS.md.
CHECKPOINT_DOWNLOAD_GIB = {
    "llava-hf/llava-1.5-7b-hf": 13.2,
    "Salesforce/instructblip-vicuna-7b": 29.5,
}


def _download_gib(model_id: str):
    """Transfer size, measured from the repo listing where the network allows.

    Falls back to the recorded figure offline, and says which it used, so a
    number in the output is never ambiguous about whether it was observed.
    """
    import json as _json
    import urllib.error
    import urllib.request

    try:
        url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
        with urllib.request.urlopen(url, timeout=30) as fh:
            sib = _json.loads(fh.read()).get("siblings", [])
        st = [f.get("size") or 0 for f in sib
              if f["rfilename"].endswith(".safetensors")]
        bins = [f.get("size") or 0 for f in sib
                if f["rfilename"].endswith(".bin")]
        use = st or bins
        if use:
            return (round(sum(use) / 1024**3, 1),
                    f"measured from the repo listing, "
                    f"{len(use)} {'safetensors' if st else '.bin'} shards")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError,
            _json.JSONDecodeError, KeyError):
        pass

    known = CHECKPOINT_DOWNLOAD_GIB.get(model_id)
    if known:
        return known, "recorded fallback, the HF API was unreachable"
    return 13.2, ("NOT measured and not recorded for this checkpoint -- "
                  "LLaVA's figure is standing in")


def check_disk(model_id: str = "llava-hf/llava-1.5-7b-hf") -> dict:
    """Free space where the checkpoint lands, which is not where runs land.

    On Kaggle these are two different filesystems with two different limits,
    and confusing them is easy: `/kaggle/working` is capped at 20 GB because
    it is the notebook's *output*, while the HuggingFace cache goes to
    `/kaggle/temp`, which is larger and ephemeral. A 30 GB checkpoint is a
    problem only if the cache root cannot hold it.

    Checked before a session commits, for the same reason the GPU is: finding
    out ten hours in costs ten hours.
    """
    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from foxrepro.storage import paths

    p = paths()
    roots = {
        "hf_cache (checkpoint)": p.hf_home,
        "runs + reports": p.persist_root,
        "images": p.coco_images_dir,
    }
    report = {}
    for label, path in roots.items():
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        report[label] = {
            "path": str(path),
            "free_gib": round(usage.free / 1024**3, 1),
            "total_gib": round(usage.total / 1024**3, 1),
        }

    need, how = _download_gib(model_id)
    free = report["hf_cache (checkpoint)"]["free_gib"]
    fits = free > need * 1.15          # leave room for the partial-download file
    return {
        "check": "disk",
        "model_id": model_id,
        "checkpoint_download_gib": need,
        "download_source": how,
        "roots": report,
        "status": _status(fits, warn=not fits and free > need),
        "note": (
            f"{model_id} downloads {need:.1f} GiB ({how}). The HF cache "
            f"root has {free:.1f} GiB free. "
            + ("Fits." if fits else
               "Does NOT fit with headroom. On Kaggle check that FOX_HF_HOME "
               "points at /kaggle/temp and not /kaggle/working, which is "
               "capped at 20 GB because it is the notebook output.")
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

    # On pre-Ampere cards (T4 = sm_75) bf16 has no native tensor-core path,
    # but PyTorch still accepts bf16 tensors and emulates the ops by upcasting.
    # That is slow for a whole model and irrelevant here: the experiment needs
    # only a CAST and an elementwise ADD on the attention logits, and a cast is
    # exact. So an "available" bf16 arm on sm_75 is numerically valid.
    try:
        import torch as _t

        _pre_ampere = (
            _t.cuda.is_available()
            and _t.cuda.get_device_properties(0).major < 8
        )
    except BaseException:  # noqa: BLE001
        _pre_ampere = False

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
            (
                "All three dtype arms available. NOTE: this GPU is "
                "pre-Ampere, so bf16 is emulated rather than native -- "
                "numerically exact for a cast+add on the logits, which is "
                "all this experiment needs, but do not set the WEIGHT "
                "dtype to bf16 here or everything will crawl."
                if _pre_ampere else "All three dtype arms available."
            )
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

    # Severity depends on the architecture, and conflating the two is how a
    # real hazard gets lost among harmless ones.
    #
    # TF32 exists only on Ampere+ (sm_80+). On a pre-Ampere card such as the
    # T4 (sm_75) the allow_tf32 flags are inert: there is no TF32 path to take.
    #
    # allow_fp16_reduced_precision_reduction is a different matter and DOES
    # bite on sm_75. It governs the accumulation precision inside an fp16
    # matmul -- which is exactly the Q.K^T product whose output tensor this
    # whole experiment studies. Leaving it True adds a second, uncontrolled
    # precision effect on top of the one under test.
    try:
        import torch as _t

        major = _t.cuda.get_device_properties(0).major if _t.cuda.is_available() else 0
    except BaseException:  # noqa: BLE001
        major = 0
    pre_ampere = 0 < major < 8

    tf32_keys = {"cuda.matmul.allow_tf32", "cudnn.allow_tf32"}
    inert = sorted(k for k in risky if pre_ampere and k in tf32_keys)
    material = sorted(k for k in risky if k not in inert)

    if pre_ampere:
        note = (
            f"sm_{major}x is pre-Ampere, so the TF32 flags "
            f"({', '.join(inert) or 'none set'}) are INERT here -- there is no "
            "TF32 path on this GPU. What does matter on this card: "
            f"{', '.join(material) or 'nothing'}. "
            "allow_fp16_reduced_precision_reduction controls accumulation "
            "inside the fp16 Q.K^T matmul, i.e. the very tensor under study, "
            "so leaving it True stacks a second uncontrolled precision effect "
            "on top of the one being measured. pin_precision_flags() turns all "
            "of them off and records the result in every JSONL line."
        )
    else:
        note = (
            "Set all of these to False before the dtype experiment and LOG them "
            "into every results file. cudnn.allow_tf32 defaults to True, and on "
            "Ampere+ a TF32 matmul has ~10 mantissa bits -- roughly fp16 "
            "precision in the QK product -- which would make the fp32 arm "
            "indistinguishable from fp16 for the wrong reason."
        ) if risky else "All precision flags already off."

    return {
        "check": "tf32_precision_flags",
        "flags": flags,
        "compute_capability_major": major,
        "pre_ampere": pre_ampere,
        "inert_on_this_gpu": inert,
        "material_on_this_gpu": material,
        "need_pinning": risky,
        "status": _status(True, warn=bool(material)),
        "note": note,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf",
                    help="checkpoint whose download size the disk check uses")
    args = ap.parse_args()

    checks = [check_python(), check_torch()]
    cuda = checks[-1].get("cuda_available")
    checks.append(check_preinstalled_stack())
    checks.append(check_disk(args.model_id))
    if cuda:
        checks += [check_gpus(args.model_id), check_dtypes(),
                   check_ulp(), check_tf32()]

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
