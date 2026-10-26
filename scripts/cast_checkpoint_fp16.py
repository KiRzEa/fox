#!/usr/bin/env python3
"""Re-save a checkpoint at fp16, once, so every session downloads half as much.

The problem this solves is not a format problem. ``Salesforce/instructblip-
vicuna-7b`` is already a HuggingFace repository -- there is nothing to convert
it *to*. It is simply **stored in fp32**, so it transfers 29.5 GiB to occupy
about 15.8 GiB of VRAM once ``from_pretrained(dtype=torch.float16)`` casts it
at load. Every Kaggle session pays that transfer again, because every session
starts from an empty disk.

Casting once and keeping the result is the fix: 29.5 GiB becomes ~15.5 GiB,
and the session downloads that instead.

**This does not change any number.** ``from_pretrained(dtype=fp16)`` performs
exactly this cast anyway, at load, every time. Doing it once at rest produces
bit-identical weights -- the same round-to-nearest on the same fp32 inputs --
and ``--verify`` proves it on a sample of tensors rather than asserting it.

Quantization would be a different matter and is ruled out for this project:
the quantity under study *is* a precision effect, so an int8 or int4
checkpoint would change the thing being measured. fp16 is not quantization
here; it is the dtype the runs already use.

Usage::

    # once, on a host with ~50 GB free
    python3 scripts/cast_checkpoint_fp16.py \\
        --model-id Salesforce/instructblip-vicuna-7b \\
        --out ~/instructblip-vicuna-7b-fp16 --verify

    # then push it somewhere a Kaggle session can reach
    huggingface-cli upload <you>/instructblip-vicuna-7b-fp16 \\
        ~/instructblip-vicuna-7b-fp16

    # and point a run at it
    python3 -m foxrepro.bench.generate --model-id <you>/instructblip-vicuna-7b-fp16 ...

The run's provenance records ``model_id``, so a cast checkpoint is traceable
in the archive and cannot be confused with the original.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", action="store_true",
                    help="reload the result and assert it is bit-identical to "
                         "casting the original at load time")
    ap.add_argument("--max-shard-size", default="4GB")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor

    cfg = AutoConfig.from_pretrained(args.model_id)
    arch = (cfg.architectures or ["?"])[0]
    print(f"[cast] {args.model_id}  ({arch})")

    model = AutoModel.from_pretrained(args.model_id, dtype=torch.float16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[cast] {n/1e9:.2f}B parameters -> {n*2/1024**3:.1f} GiB at fp16")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, max_shard_size=args.max_shard_size,
                          safe_serialization=True)
    try:
        AutoProcessor.from_pretrained(args.model_id).save_pretrained(args.out)
    except Exception as exc:                      # noqa: BLE001
        print(f"[cast] WARNING: no processor saved ({exc}). A run needs one; "
              f"copy it from the original repo before using this checkpoint.",
              file=sys.stderr)

    on_disk = sum(f.stat().st_size for f in args.out.rglob("*")
                  if f.suffix in (".safetensors", ".bin"))
    print(f"[cast] wrote {args.out}  ({on_disk/1024**3:.1f} GiB)")

    if args.verify:
        print("[cast] verifying bit-identity against a load-time cast...")
        ref = AutoModel.from_pretrained(args.model_id, dtype=torch.float16)
        got = AutoModel.from_pretrained(args.out, dtype=torch.float16)
        a = dict(ref.named_parameters())
        b = dict(got.named_parameters())
        if set(a) != set(b):
            print(f"[cast] FAIL: parameter sets differ "
                  f"({sorted(set(a) ^ set(b))[:5]})", file=sys.stderr)
            return 1
        bad = [k for k, v in a.items() if not torch.equal(v, b[k])]
        if bad:
            print(f"[cast] FAIL: {len(bad)} tensors differ, e.g. {bad[:5]}",
                  file=sys.stderr)
            return 1
        print(f"[cast] OK: all {len(a)} tensors bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
