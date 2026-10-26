#!/usr/bin/env python3
"""Can this repo run a given backbone, and what would it take?

Answers from a checkpoint's ``config.json`` and weight index alone --
a few hundred kilobytes, not the 13-30 GiB of weights. Written because the
alternative was arguing from memory about what transformers supports, and
that produced a wrong answer about Shikra that stood for several days.

What it reports:

* whether transformers has the class the checkpoint names, and if not,
  whether the architecture reduces to one it does have;
* the tensors that sit outside that base architecture -- for Shikra, exactly
  two, which is the difference between "needs a vendored stack" and "needs
  fifty lines";
* the visual token layout, since Fox's head scoring is an entropy over the
  image indices and every backbone lays them out differently;
* VRAM at fp16 against the GPUs present, if any;
* whether the Fox attention patch reaches the decoder at all.

Usage::

    python3 scripts/probe_backbone.py                      # all known
    python3 scripts/probe_backbone.py --model-id some/repo # one
    python3 scripts/probe_backbone.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

#: Backbones the paper evaluates, plus what this repo already runs.
KNOWN = [
    "llava-hf/llava-1.5-7b-hf",
    "Salesforce/instructblip-vicuna-7b",
    "shikras/shikra-7b-delta-v1-0708",
]

#: Base architectures whose attention this repo can already intervene on.
#: Fox patches LLaMA attention, so anything whose decoder IS LLaMA inherits
#: the intervention -- which is how the released code evaluates Shikra
#: without writing a second implementation for it.
LLAMA_SHAPE = ("hidden_size", "intermediate_size", "num_attention_heads",
               "num_hidden_layers")

#: Cross-check only. These are what the derivation below is EXPECTED to
#: produce; they are not the source of truth and are never returned on their
#: own. If a derivation disagrees with one of them the probe says so loudly,
#: because one of the two is then wrong and silence would pick the wrong one.
#:
#: They were hardcoded as the answer at first, which is the failure this
#: project keeps catching: a number that reads as measured and is asserted.
EXPECTED_VISUAL_TOKENS = {
    "llava-hf/llava-1.5-7b-hf": 576,
    "Salesforce/instructblip-vicuna-7b": 32,
    "shikras/shikra-7b-delta-v1-0708": 258,
    "shikras/shikra-7b-delta-v1": 258,
}


def _fetch(repo: str, name: str):
    url = f"https://huggingface.co/{repo}/raw/main/{name}"
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            return json.loads(fh.read())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None


def _repo_files(repo: str):
    """Filenames in a repo, for checking what it ships besides weights."""
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            return [f["rfilename"] for f in json.loads(fh.read()).get("siblings", [])]
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return []


def _grid_tokens(vision_cfg: dict):
    """``(image_size / patch_size) ** 2`` -- the patch count of a ViT grid."""
    try:
        side = int(vision_cfg["image_size"]) // int(vision_cfg["patch_size"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return side * side


def derive_visual_tokens(model_id: str, cfg: dict):
    """How many positions Fox's ``H_vis`` is an entropy over, and why.

    Three different mechanisms across three backbones, so the derivation has
    to branch rather than read one field:

    * **Q-Former** models state ``num_query_tokens`` outright -- the visual
      stream is learned queries, not patches.
    * **Self-contained ViT** models carry a ``vision_config``; the count is
      its grid.
    * **External tower** models (Shikra) name the tower in
      ``mm_vision_tower`` and ship none of it, so the grid comes from that
      repo's config, plus two if ``mm_use_im_start_end`` wraps it in
      ``<im_start>`` / ``<im_end>``.

    Returns ``(count, derivation)``; ``count`` is ``None`` when no rule fits,
    which is a result and not a crash.
    """
    n = cfg.get("num_query_tokens")
    if n:
        return int(n), "config.num_query_tokens (Q-Former queries, not patches)"

    vc = cfg.get("vision_config")
    if isinstance(vc, dict):
        g = _grid_tokens(vc)
        if g:
            return g, (f"({vc['image_size']}/{vc['patch_size']})^2 from "
                       f"config.vision_config")

    tower = cfg.get("mm_vision_tower")
    if tower:
        tcfg = _fetch(str(tower), "config.json") or {}
        g = _grid_tokens(tcfg.get("vision_config", tcfg))
        if g:
            markers = 2 if cfg.get("mm_use_im_start_end") else 0
            how = f"({g}) patches from {tower}"
            if markers:
                how += " + <im_start> + <im_end>"
            return g + markers, how
    return None, "no rule fits this config"


def _repo_sizes(repo: str) -> dict:
    """``filename -> bytes`` from the HF API, for the files it actually ships."""
    url = f"https://huggingface.co/api/models/{repo}?blobs=true"
    try:
        with urllib.request.urlopen(url, timeout=60) as fh:
            doc = json.loads(fh.read())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return {}
    return {f["rfilename"]: (f.get("size") or 0)
            for f in doc.get("siblings", [])}


def derive_download_gib(repo: str):
    """What ``from_pretrained`` actually transfers.

    Safetensors when present, because it prefers them and skips the duplicate
    ``.bin`` shards -- counting both is how a 29.5 GiB checkpoint got called
    63 GiB earlier in this project.
    """
    sizes = _repo_sizes(repo)
    st = {k: v for k, v in sizes.items() if k.endswith(".safetensors")}
    use = st or {k: v for k, v in sizes.items() if k.endswith(".bin")}
    if not use:
        return None, "no weight files listed"
    fmt = "safetensors" if st else ".bin (no safetensors published)"
    return round(sum(use.values()) / 1024**3, 1), f"{len(use)} {fmt} shards"


def _weight_keys(repo: str):
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        idx = _fetch(repo, name)
        if idx and "weight_map" in idx:
            return sorted(idx["weight_map"]), name
    return [], None


def probe(model_id: str) -> dict:
    import transformers

    cfg = _fetch(model_id, "config.json")
    if cfg is None:
        return {"model_id": model_id, "verdict": "UNREACHABLE",
                "note": "no config.json (private, gated, or does not exist)"}

    arch = (cfg.get("architectures") or [None])[0]
    has_class = bool(arch) and hasattr(transformers, arch)

    keys, index_file = _weight_keys(model_id)
    llama_like = all(cfg.get(k) == v for k, v in
                     zip(LLAMA_SHAPE, (4096, 11008, 32, 32))) if cfg else False

    # Tensors outside a plain LLaMA decoder. For a model that is LLaMA plus a
    # projector this is a two-item list, and the port is that list.
    extra = [k for k in keys
             if not k.startswith("model.layers.")
             and k not in ("model.embed_tokens.weight", "model.norm.weight",
                           "lm_head.weight")]

    params = None
    if cfg.get("num_hidden_layers") and cfg.get("hidden_size"):
        h, n = cfg["hidden_size"], cfg["num_hidden_layers"]
        i = cfg.get("intermediate_size", 4 * h)
        v = cfg.get("vocab_size", 32000)
        params = n * (4 * h * h + 3 * h * i) + 2 * v * h

    n_visual, how_visual = derive_visual_tokens(model_id, cfg)
    expected = EXPECTED_VISUAL_TOKENS.get(model_id)
    visual_mismatch = (expected is not None and n_visual is not None
                       and expected != n_visual)

    dl_gib, dl_how = derive_download_gib(model_id)

    # Does this repo carry its own image processor, and if not, is the one it
    # points at reachable? A decoder that loads and cannot see is not runnable.
    own_proc = [f for f in _repo_files(model_id)
                if "preprocessor_config" in f or "processor_config" in f]
    if own_proc:
        pre = {"source": "self", "repo": None,
               "note": f"ships {', '.join(sorted(own_proc))}"}
    elif cfg.get("mm_vision_tower"):
        # Derived, not recorded: the checkpoint names its own tower, and the
        # processor belongs to the tower rather than to the decoder.
        pre = {"source": "vision_tower", "repo": str(cfg["mm_vision_tower"]),
               "note": "no preprocessor in this repo by design -- pixels go "
                       "through the processor of the tower named in "
                       "config.mm_vision_tower"}
    else:
        pre = {"source": "unknown", "repo": None,
               "note": "no preprocessor config in this repo, and the config "
                       "names no vision tower to borrow one from -- find out "
                       "how pixels are prepared before calling this runnable"}
    pre["ships_own"] = bool(own_proc)
    if pre.get("repo"):
        pre["source_repo_has_processor"] = bool(
            [f for f in _repo_files(pre["repo"]) if "preprocessor_config" in f])

    if pre.get("source") == "unknown":
        verdict = "UNCLEAR"
        path = ("the decoder is reachable, but nothing here says how its "
                "pixels are preprocessed. Resolve that before planning a run.")
    elif has_class:
        verdict, path = "RUNNABLE", f"transformers ships {arch}"
    elif llama_like:
        verdict = "PORTABLE"
        path = (f"transformers has no {arch}, but the decoder IS LLaMA "
                f"(32x4096, 32 heads, 11008 ffn). Load the decoder as "
                f"LlamaForCausalLM and carry the {len(extra)} extra tensor(s) "
                f"yourself: {extra or 'none found'}")
    else:
        verdict = "BLOCKED"
        path = (f"transformers has no {arch}, and the architecture does not "
                f"reduce to LLaMA. Needs the release's own modeling code.")

    return {
        "model_id": model_id,
        "architecture": arch,
        "model_type": cfg.get("model_type"),
        "transformers_has_class": has_class,
        "decoder_is_llama_7b": llama_like,
        "fox_patch_reaches_decoder": has_class or llama_like,
        "weight_index": index_file,
        "n_tensors": len(keys),
        "tensors_outside_llama_decoder": extra,
        "params_est_b": round(params / 1e9, 2) if params else None,
        "weights_gib_fp16": round(params * 2 / 1024**3, 1) if params else None,
        "is_delta": "delta" in model_id,
        "visual_tokens": n_visual,
        "visual_derivation": how_visual,
        "visual_expected": expected,
        "visual_mismatch": visual_mismatch,
        "download_gib": dl_gib,
        "download_derivation": dl_how,
        "preprocessing": pre,
        "verdict": verdict,
        "path": path,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", action="append", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    targets = args.model_id or KNOWN
    results = [probe(m) for m in targets]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    try:
        import torch
        gpus = [(torch.cuda.get_device_name(i),
                 torch.cuda.get_device_properties(i).total_memory / 1024**3)
                for i in range(torch.cuda.device_count())]
    except Exception:                                    # noqa: BLE001
        gpus = []

    print("=" * 76)
    print("Backbone deployability -- from config.json and the weight index only")
    print("=" * 76)
    for r in results:
        print(f"\n[{r['verdict']:9s}] {r['model_id']}")
        if r["verdict"] == "UNREACHABLE":
            print(f"            {r['note']}")
            continue
        print(f"            architecture : {r['architecture']} "
              f"(model_type={r['model_type']})")
        print(f"            transformers : "
              f"{'has it' if r['transformers_has_class'] else 'does NOT have it'}"
              f"   decoder is LLaMA-7B: {r['decoder_is_llama_7b']}")
        print(f"            Fox patch reaches the decoder: "
              f"{r['fox_patch_reaches_decoder']}")
        if r["weights_gib_fp16"]:
            line = f"            weights      : ~{r['params_est_b']}B -> " \
                   f"{r['weights_gib_fp16']} GiB at fp16"
            if gpus:
                need = r["weights_gib_fp16"] / max(len(gpus), 1) + 1.2
                line += (f"; ~{need:.1f} GiB/GPU over {len(gpus)}, "
                         f"smallest has {min(g[1] for g in gpus):.1f}")
            print(line)
        if r.get("download_gib"):
            print(f"            download     : {r['download_gib']} GiB "
                  f"({r['download_derivation']})")
        if r["visual_tokens"]:
            print(f"            visual span  : {r['visual_tokens']} tokens  "
                  f"<- {r['visual_derivation']}")
            if r["visual_mismatch"]:
                print(f"            *** MISMATCH : expected "
                      f"{r['visual_expected']}, derived {r['visual_tokens']}. "
                      f"One of the two is wrong -- do not proceed on either.")
        else:
            print(f"            visual span  : UNKNOWN "
                  f"-- {r['visual_derivation']}")
        pre = r.get("preprocessing") or {}
        if pre:
            where = ("its own repo" if pre.get("ships_own")
                     else f"{pre.get('repo') or 'UNKNOWN'}")
            print(f"            image proc   : {where}"
                  + ("  [reachable]" if pre.get("source_repo_has_processor")
                     else "")
                  + f"\n                           {pre.get('note','')}")
        if r["is_delta"]:
            print(f"            NOTE         : this is a DELTA. It must be "
                  f"merged onto its base before use.")
        print(f"            -> {r['path']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
