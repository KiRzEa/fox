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

#: Visual token counts, stated rather than inferred, because each comes from a
#: different place: LLaVA expands <image> into patch tokens in input_ids,
#: InstructBLIP feeds Q-Former queries, Shikra splices a projected CLIP grid
#: between two marker tokens.
VISUAL_LAYOUT = {
    "llava-hf/llava-1.5-7b-hf": {
        "visual_tokens": 576, "note": "24x24 patches from CLIP ViT-L/14-336, "
        "expanded into input_ids by the processor",
    },
    "Salesforce/instructblip-vicuna-7b": {
        "visual_tokens": 32, "note": "Q-Former query tokens, NOT patches -- "
        "H_vis becomes an entropy over 32 positions instead of 576",
    },
    "shikras/shikra-7b-delta-v1-0708": {
        "visual_tokens": 258, "note": "256 CLIP ViT-L/14 patches plus "
        "<im_start> and <im_end>; the released code hardcodes this as "
        "img_s==33 and img_e==290 (290-33+1 = 258)",
    },
}


def _fetch(repo: str, name: str):
    url = f"https://huggingface.co/{repo}/raw/main/{name}"
    try:
        with urllib.request.urlopen(url, timeout=30) as fh:
            return json.loads(fh.read())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None


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

    layout = VISUAL_LAYOUT.get(model_id, {})

    if has_class:
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
        "visual_tokens": layout.get("visual_tokens"),
        "visual_note": layout.get("note"),
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
        if r["visual_tokens"]:
            print(f"            visual span  : {r['visual_tokens']} tokens "
                  f"-- {r['visual_note']}")
        if r["is_delta"]:
            print(f"            NOTE         : this is a DELTA. It must be "
                  f"merged onto its base before use.")
        print(f"            -> {r['path']}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
