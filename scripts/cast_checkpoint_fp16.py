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


def _safetensor_files(model_id_or_path) -> list:
    """Local paths to a repo's safetensors shards, downloading only those.

    Both checkpoints here also ship duplicate ``.bin`` shards that
    ``from_pretrained`` ignores; fetching them would double the transfer for
    nothing, so this asks for the safetensors by name.
    """
    from pathlib import Path as _P

    local = _P(str(model_id_or_path))
    if local.is_dir():
        return sorted(local.glob("*.safetensors"))

    from huggingface_hub import hf_hub_download, list_repo_files

    names = [f for f in list_repo_files(str(model_id_or_path))
             if f.endswith(".safetensors")]
    return [_P(hf_hub_download(str(model_id_or_path), n)) for n in sorted(names)]


def _verify_streaming(model_id: str, out_dir) -> int:
    """Compare every tensor without holding two models in memory.

    The obvious check -- load the original at fp16, load the saved copy, and
    diff the parameter dicts -- needs about 32 GB of RAM for a 7B model, which
    is more than the host doing the cast is likely to have. Reading both sides
    shard by shard with ``safe_open`` peaks at one tensor instead, and
    compares the same values.

    Returns a process exit code: 0 if every shared tensor is bit-identical.
    """
    import torch
    from safetensors import safe_open

    def index(files):
        out = {}
        for f in files:
            with safe_open(str(f), framework="pt") as fh:
                for k in fh.keys():
                    out[k] = f
        return out

    src = index(_safetensor_files(model_id))
    dst = index(_safetensor_files(out_dir))
    if not src or not dst:
        print("[cast] FAIL: no safetensors found on one side "
              f"(src={len(src)}, dst={len(dst)})", file=sys.stderr)
        return 1

    # save_pretrained can strip or add a wrapper prefix; match on the suffix
    # when the key sets differ rather than declaring a failure for a rename.
    shared = set(src) & set(dst)
    if not shared:
        by_tail = {k.split(".", 1)[-1]: k for k in dst}
        shared = {k for k in src if k.split(".", 1)[-1] in by_tail}
        dst = {k.split(".", 1)[-1]: v for k, v in dst.items()}
        lookup = lambda k: (dst[k.split(".", 1)[-1]], k.split(".", 1)[-1])
    else:
        lookup = lambda k: (dst[k], k)

    only_src = sorted(set(src) - shared)
    if only_src:
        print(f"[cast] WARNING: {len(only_src)} tensors absent from the copy, "
              f"e.g. {only_src[:3]}", file=sys.stderr)

    bad, checked = [], 0
    for k in sorted(shared):
        with safe_open(str(src[k]), framework="pt") as fh:
            a = fh.get_tensor(k).to(torch.float16)
        f_dst, k_dst = lookup(k)
        with safe_open(str(f_dst), framework="pt") as fh:
            b = fh.get_tensor(k_dst)
        checked += 1
        if a.shape != b.shape or not torch.equal(a, b):
            bad.append(k)
        del a, b
        if checked % 100 == 0:
            print(f"[cast]   {checked}/{len(shared)} tensors checked")

    if bad:
        print(f"[cast] FAIL: {len(bad)} tensors differ, e.g. {bad[:5]}",
              file=sys.stderr)
        return 1
    print(f"[cast] OK: all {checked} tensors bit-identical to a load-time cast")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify", action="store_true",
                    help="reload the result and assert it is bit-identical to "
                         "casting the original at load time")
    ap.add_argument("--max-shard-size", default="4GB")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the transfer and disk plan, download nothing")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoConfig, AutoProcessor

    if args.dry_run:
        from huggingface_hub import list_repo_files, model_info
        info = model_info(args.model_id, files_metadata=True)
        st = [f for f in info.siblings if f.rfilename.endswith(".safetensors")]
        dl = sum(f.size or 0 for f in st) / 1024**3
        print(f"[cast] would download {dl:.1f} GiB ({len(st)} safetensors "
              f"shards; duplicate .bin shards are skipped)")
        print(f"[cast] would write    ~{dl/2:.1f} GiB of fp16 to {args.out}")
        print(f"[cast] peak disk      ~{dl * 1.5:.1f} GiB (cache + output)")
        print(f"[cast] peak RAM with --verify: one tensor at a time, "
              f"not two models")
        return 0

    cfg = AutoConfig.from_pretrained(args.model_id)
    arch = (cfg.architectures or [None])[0]
    print(f"[cast] {args.model_id}  ({arch})")

    # The class named in config.json, not AutoModel. AutoModel resolves to the
    # bare encoder for several architectures and would silently drop the
    # language-model head -- a checkpoint that loads and generates nothing.
    import transformers as _tf

    cls = getattr(_tf, arch, None) if arch else None
    if cls is None:
        print(f"[cast] FAIL: transformers has no class {arch!r}; this "
              f"checkpoint needs its own loader", file=sys.stderr)
        return 1
    model = cls.from_pretrained(args.model_id, dtype=torch.float16)
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
        rc = _verify_streaming(args.model_id, args.out)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
