"""Wire the Fox kernel into a live LLaVA-1.5, and drive both branches at once.

This is the only file that touches transformers. Everything numerically
load-bearing lives in :mod:`foxrepro.fox.kernel`, which is asserted
bit-identical to the released code by ``tests/test_kernel_equivalence.py``
without needing a GPU. So a bug here is a *plumbing* bug -- it cannot silently
change the mechanism, only fail to deliver it.

How the intervention reaches the pre-softmax logits
---------------------------------------------------
A forward hook cannot do it: ``LlamaAttention.forward`` returns
``(attn_output, attn_weights)`` where the weights are already post-softmax, and
the pre-softmax tensor is a local inside the attention function, never a module
boundary. Monkeypatching ``forward`` would work but means re-owning RoPE, the
KV-cache update and mask handling on every transformers bump.

The supported route is transformers' pluggable attention interface. We register
one function and select it with ``attn_implementation="fox"``. That function
receives ``(module, query, key, value, attention_mask, ...)`` *after* RoPE and
*after* the cache update, and owns everything from ``Q @ K.T`` onward -- so the
pre-softmax logit tensor is an ordinary local variable we can index and write
to at any ``(batch, head, query, key)`` slice.

``attn_implementation`` is per-sub-config, not per-layer, and all 32 attention
modules share one config object. So per-layer behaviour comes from the function
itself: it dispatches on ``module.layer_idx`` and hands every untouched layer
straight to ``sdpa_attention_forward``. Eager is confined to the intervention
window, which halves the peak attention transient and costs nothing in speed
(attention is ~2.4% of prefill FLOPs).

Three traps this file is built around
-------------------------------------
1. **Never use ``output_attentions=True``.** From transformers ~4.52 onward
   there is no eager fallback: ``sdpa_attention_forward`` returns
   ``(output, None)``, the recorder appends only non-None entries, and you get
   ``outputs.attentions == ()`` -- an empty tuple, with a single
   ``warning_once``. Code that indexes it raises far from the cause; code that
   iterates it computes statistics over nothing. We compute the per-head
   statistics inside the attention function and return ``None`` for the
   weights, which also avoids retaining 32 x 22 MiB per branch.
2. **Registering the mask function is not optional.** If the name is absent
   from ``AttentionMaskInterface``, transformers skips mask creation and passes
   ``attention_mask=None``. Causality is then silently dropped: the model
   attends to the future and still produces fluent text, while POPE accuracy
   looks suspiciously good.
3. **Pin the reduced-precision flags before loading.** On the T4 the TF32 flags
   are inert (TF32 is Ampere+), but
   ``allow_fp16_reduced_precision_reduction`` defaults to True and governs
   accumulation *inside* the fp16 ``Q @ K.T`` matmul -- the exact tensor under
   study. See ``bench.generate.pin_precision_flags``.

Both branches in one forward pass
---------------------------------
The released code runs the observational and interventional passes
sequentially, reading all 13.16 GiB of weights twice per token. Decode is
weight-bandwidth bound, so running them as batch rows 0 and 1 of a single pass
roughly halves wall-clock for the cost of one extra KV cache (~0.54 GiB).
``FoxState.branch_rows`` restricts the write to row 1, and the fusion in
``kernel.fuse`` is per-row, unlike the released version which computes the JSD
from row 0 and applies that scalar everywhere.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F

from foxrepro.fox.kernel import intervene, resolve_logit_dtype

__all__ = [
    "FoxState",
    "FOX",
    "register",
    "describe_environment",
    "load_llava",
    "locate_image_span",
    "fox_generate",
    "ATTN_NAME",
]

#: The name we register under and pass as ``attn_implementation``.
ATTN_NAME = "fox"

#: Released intervention window for LLaVA-1.5, inclusive. The paper says 2-13;
#: the code says 2-15 and so does a leftover author log path. See
#: ``foxrepro.fox.reference`` for the full disagreement table.
RELEASED_LAYERS = tuple(range(2, 16))


# --------------------------------------------------------------------------- #
# shared control block
# --------------------------------------------------------------------------- #
@dataclass
class FoxState:
    """One module-level instance, read by all 32 attention modules.

    State goes here rather than through ``forward`` kwargs on purpose: extra
    kwargs do reach the attention function, but ``generate()``'s
    ``_validate_model_kwargs`` is a moving target across transformers versions,
    and this project spans 4.57.x through 5.x.
    """

    # ---- what is active ----
    enabled: bool = False
    eager_layers: frozenset = field(default_factory=lambda: frozenset(RELEASED_LAYERS))
    mode: str = "as_implemented"

    # ---- intervention parameters (forwarded to kernel.intervene) ----
    gamma: float = 10000.0
    ratio: float = 0.35
    logit_dtype: str = "native"
    softmax_dtype: str = "fp32"
    use_m_sys: bool = True
    grid_step: Optional[float] = None
    noise_halfwidth: Optional[float] = None
    generator: Optional[torch.Generator] = None

    # ---- where the image tokens are (absolute key positions) ----
    img_start: int = -1
    img_end: int = -1

    # ---- which batch rows get the intervention ----
    branch_rows: Optional[torch.Tensor] = None

    # ---- readouts, refilled every forward ----
    collect: bool = True
    diagnostics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def reset_diagnostics(self) -> None:
        self.diagnostics.clear()

    def span_is_set(self) -> bool:
        return self.img_start >= 0 and self.img_end >= self.img_start

    @contextlib.contextmanager
    def configured(self, **kw: Any) -> Iterator["FoxState"]:
        """Temporarily override fields, restoring them afterwards."""
        saved = {k: getattr(self, k) for k in kw}
        for k, v in kw.items():
            setattr(self, k, v)
        try:
            yield self
        finally:
            for k, v in saved.items():
                setattr(self, k, v)


FOX = FoxState()


# --------------------------------------------------------------------------- #
# the attention function
# --------------------------------------------------------------------------- #
def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA key/value expansion. A no-op for LLaVA-1.5, which is MHA.

    Inlined rather than imported: its module path has moved between
    transformers versions, and this project must run on both 4.57.x and 5.x.
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv * n_rep, s, d)


def fox_attention_forward(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """Eager attention with a Fox intervention, on the intervention layers only.

    Every other layer is forwarded to SDPA unchanged, so the model's numerics
    outside the window are byte-for-byte the stock path.

    Returns ``(attn_output, None)``. The ``None`` is deliberate: returning the
    weights would make ``output_attentions=True`` retain 32 x 22 MiB per branch
    for the whole prefill, and nothing needs them -- the per-head statistics are
    computed here and stashed in :data:`FOX`.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    layer_idx = getattr(module, "layer_idx", None)

    if (
        not FOX.enabled
        or FOX.mode == "off"
        or layer_idx is None
        or layer_idx not in FOX.eager_layers
        or not FOX.span_is_set()
    ):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs,
        )

    # An autocast context would silently re-cast the logit tensor and invalidate
    # the dtype experiment, so refuse rather than produce a quiet wrong number.
    if torch.is_autocast_enabled():
        raise RuntimeError(
            "autocast is enabled inside the Fox attention path. The experiment "
            "varies the pre-softmax logit dtype explicitly; autocast would "
            "override it. Wrap the decode loop in "
            "torch.autocast(device_type='cuda', enabled=False)."
        )

    n_rep = getattr(module, "num_key_value_groups", 1)
    k_states = _repeat_kv(key, n_rep)
    v_states = _repeat_kv(value, n_rep)

    if scaling is None:  # pragma: no cover - transformers always passes it
        scaling = float(module.head_dim) ** -0.5

    # [B, H, Q, K] pre-softmax, in the model dtype. This is the tensor under
    # study: the released intervention writes to it and casts it back, and the
    # cast's rounding is the entire surviving mechanism.
    logits = torch.matmul(query, k_states.transpose(2, 3)) * scaling

    store = resolve_logit_dtype(FOX.logit_dtype, logits)
    if logits.dtype != store:
        logits = logits.to(store)

    # The kernel computes its head statistics from a separate masked fp32 copy,
    # then writes to `logits` raw -- matching the released order, where the
    # causal mask is added only after the intervention.
    logits, diag = intervene(
        logits,
        FOX.img_start,
        FOX.img_end,
        mode=FOX.mode,
        gamma=FOX.gamma,
        ratio=FOX.ratio,
        logit_dtype=FOX.logit_dtype,
        attention_mask=attention_mask,
        rows=FOX.branch_rows,
        use_m_sys=FOX.use_m_sys,
        grid_step=FOX.grid_step,
        noise_halfwidth=FOX.noise_halfwidth,
        generator=FOX.generator,
    )
    if FOX.collect and diag is not None:
        FOX.diagnostics[layer_idx] = {
            k: v.detach() for k, v in diag.items()
        }

    if attention_mask is not None:
        logits = logits + attention_mask[..., : k_states.shape[-2]].to(logits.dtype)
        logits = torch.max(
            logits,
            torch.tensor(torch.finfo(logits.dtype).min, device=logits.device,
                         dtype=logits.dtype),
        )

    acc = torch.float32 if FOX.softmax_dtype == "fp32" else logits.dtype
    probs = F.softmax(logits, dim=-1, dtype=acc).to(query.dtype)

    out = torch.matmul(probs, v_states).transpose(1, 2).contiguous()
    return out, None


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def register() -> Dict[str, Any]:
    """Register ``"fox"`` as an attention implementation and a mask builder.

    Must be called before ``from_pretrained``. Idempotent.

    Returns a small dict describing what was registered, for the record.
    """
    try:
        from transformers import AttentionInterface, AttentionMaskInterface
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "This transformers release has no AttentionInterface, so there is "
            "no supported way to write to the pre-softmax attention logits. "
            "Run scripts/probe_host.py: it checks the capability rather than "
            "the version number."
        ) from exc

    from transformers.masking_utils import eager_mask

    AttentionInterface.register(ATTN_NAME, fox_attention_forward)
    # Not optional. Without a registered mask builder, transformers passes
    # attention_mask=None and causality is silently dropped. `eager_mask`
    # always materialises an explicit additive float mask, which the eager body
    # needs and which sdpa_attention_forward also accepts. The cost is that the
    # SDPA layers cannot take the flash kernel; attention is ~2.4% of prefill
    # FLOPs, so that is a sub-2% price for correctness.
    AttentionMaskInterface.register(ATTN_NAME, eager_mask)
    return describe_environment()


def describe_environment() -> Dict[str, Any]:
    """Versions and capabilities, to be recorded alongside every run."""
    info: Dict[str, Any] = {"attn_name": ATTN_NAME}
    try:
        import transformers

        info["transformers_version"] = getattr(transformers, "__version__", "unknown")
        info["has_attention_interface"] = hasattr(transformers, "AttentionInterface")
        info["has_mask_interface"] = hasattr(transformers, "AttentionMaskInterface")
    except ImportError:  # pragma: no cover
        info["transformers_version"] = None
    info["torch_version"] = torch.__version__
    if torch.cuda.is_available():
        info["n_gpus"] = torch.cuda.device_count()
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["compute_capability"] = f"{p.major}.{p.minor}"
    return info


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_llava(
    model_id: str = "llava-hf/llava-1.5-7b-hf",
    *,
    dtype: str = "fp16",
    device_map: str = "auto",
    eager_layers: Optional[frozenset] = None,
) -> Tuple[Any, Any]:
    """Load LLaVA-1.5 with the Fox attention active on the chosen layers.

    ``device_map="auto"`` shards the layers across every visible GPU, which is
    what makes a 2x T4 16 GB host viable: 13.16 GiB of fp16 weights do not fit
    one card with headroom, but ~6.6 GiB per card plus ~0.55 GiB of KV does.
    Quantization is deliberately unavailable -- it would change the numerical
    precision of the very tensor under study.

    Returns ``(model, processor)``.
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[dtype]
    if dtype == "bf16" and torch.cuda.is_available():
        if torch.cuda.get_device_properties(0).major < 8:
            raise ValueError(
                "bf16 weights on a pre-Ampere GPU are emulated and will be "
                "extremely slow. bf16 is for --logit-dtype, not --dtype."
            )

    register()
    if eager_layers is not None:
        FOX.eager_layers = eager_layers

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch_dtype,            # `torch_dtype` is deprecated from 4.56
        attn_implementation={"text_config": ATTN_NAME, "vision_config": "sdpa"},
        device_map=device_map,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def locate_image_span(input_ids: torch.Tensor, image_token_id: int) -> Tuple[int, int]:
    """Half-open-exclusive ``(img_start, img_end)`` with ``img_end`` inclusive.

    LLaVA-1.5 with CLIP ViT-L/336 expands ``<image>`` into exactly 576 token
    ids, so this should return a 576-wide span; a different width means the
    processor took the pre-4.44 "legacy" path that spliced embeddings inside
    the model instead, and the key indices would be wrong.

    System keys are everything before ``img_start``, which is what the paper's
    ``m_sys`` measures.
    """
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise ValueError(
            f"no image token (id={image_token_id}) in input_ids. This "
            "checkpoint may use the legacy embedding-splice path, in which "
            "case the image keys are not addressable from input_ids."
        )
    start, end = int(pos[0].item()), int(pos[-1].item())
    width = end - start + 1
    if width != 576:
        raise ValueError(
            f"expected 576 image tokens for LLaVA-1.5/CLIP-336, found {width}. "
            "Check the processor config (patch_size=14, "
            "num_additional_image_tokens=1) before trusting any key index."
        )
    return start, end


# --------------------------------------------------------------------------- #
# dual-branch decode
# --------------------------------------------------------------------------- #
def _sample_from(
    logits: torch.Tensor, *, top_p: float, temperature: float, greedy: bool,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Nucleus sampling or argmax over a single ``[V]`` logit row."""
    if greedy:
        return logits.argmax(dim=-1)
    x = logits / max(temperature, 1e-6)
    probs = F.softmax(x.float(), dim=-1)
    order = probs.argsort(descending=True)
    sorted_probs = probs[order]
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p
    keep[0] = True
    filtered = torch.zeros_like(probs)
    filtered[order[keep]] = sorted_probs[keep]
    filtered = filtered / filtered.sum()
    return torch.multinomial(filtered, 1, generator=generator).squeeze(-1)


@torch.inference_mode()
def fox_generate(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 512,
    mode: str = "as_implemented",
    gamma: float = 10000.0,
    ratio: float = 0.35,
    tau_js: float = 0.15,
    alpha: float = 2.0,
    beta: float = 0.1,
    logit_dtype: str = "native",
    softmax_dtype: str = "fp32",
    greedy: bool = False,
    mask_greedy: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
    eager_layers: Optional[frozenset] = None,
    seed: Optional[int] = None,
    collect_trace: bool = False,
) -> Dict[str, Any]:
    """Generate with Fox: both branches batched, fused per token.

    ``generate()`` cannot express this -- it would sample the two rows
    independently rather than fusing their logits -- so the loop is explicit.

    Args:
        mask_greedy: the released greedy path argmaxes the **unmasked** fused
            logits while its sampling path uses the beta-masked ones, so its
            Table 9 changes the decision rule as well as removing randomness.
            ``False`` reproduces that; ``True`` applies the mask under greedy
            too, which is the clean de-randomisation the paper should have run.
            Report both.
        collect_trace: record per-token ``d_t`` and ``lambda_t``, for the
            demo's conflict-trace panel.

    Returns ``{"token_ids", "text"?, "n_prompt_tokens", "n_gen_tokens",
    "trace"?}``.
    """
    from foxrepro.fox.kernel import fuse, jsd

    device = next(model.parameters()).device
    image_token_id = model.config.image_token_index
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    img_start, img_end = locate_image_span(input_ids, image_token_id)
    n_prompt = int(input_ids.shape[1])

    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(seed)

    # Row 0 observational, row 1 interventional: identical token sequences, so
    # batching them is exact, and decode reads the weights once instead of
    # twice. The released code runs two sequential passes.
    ids2 = input_ids.repeat(2, 1)
    mask2 = attn_mask.repeat(2, 1)
    px2 = pixel_values.repeat(2, *([1] * (pixel_values.dim() - 1)))

    out_ids: List[int] = []
    trace: List[Dict[str, float]] = []
    cache = None

    with FOX.configured(
        enabled=True, mode=mode, gamma=gamma, ratio=ratio,
        logit_dtype=logit_dtype, softmax_dtype=softmax_dtype,
        img_start=img_start, img_end=img_end,
        branch_rows=torch.tensor([1], device=device),
        eager_layers=eager_layers if eager_layers is not None else FOX.eager_layers,
        generator=gen,
    ):
        for step in range(max_new_tokens):
            FOX.reset_diagnostics()
            if step == 0:
                out = model(
                    input_ids=ids2, attention_mask=mask2, pixel_values=px2,
                    use_cache=True,
                )
            else:
                out = model(
                    input_ids=ids2[:, -1:], attention_mask=mask2,
                    past_key_values=cache, use_cache=True,
                )
            cache = out.past_key_values
            z_obs = out.logits[0, -1].float().unsqueeze(0)
            z_do = out.logits[1, -1].float().unsqueeze(0)

            if mode == "off":
                # Fusion still runs: this arm isolates the additive logit
                # fusion from the intervention, which the paper never did.
                d_t = torch.zeros(1, device=z_obs.device, dtype=torch.float64)
            else:
                d_t = jsd(z_obs, z_do)

            fused, masked, lam = fuse(
                z_obs, z_do, d_t, alpha=alpha, tau_js=tau_js, beta=beta
            )
            decision = fused if (greedy and not mask_greedy) else masked
            nxt = _sample_from(
                decision[0], top_p=top_p, temperature=temperature,
                greedy=greedy, generator=gen,
            )
            tok = int(nxt.item())
            out_ids.append(tok)
            if collect_trace:
                trace.append({
                    "step": step,
                    "d_t": float(d_t[0].item()),
                    "lambda_t": float(lam[0].item()),
                    "diverged": bool(int(fused[0].argmax()) != int(z_obs[0].argmax())),
                })
            if tok == model.config.text_config.eos_token_id:
                break

            nxt2 = nxt.view(1, 1).repeat(2, 1).to(device)
            ids2 = torch.cat([ids2, nxt2], dim=1)
            mask2 = torch.cat([mask2, mask2.new_ones(2, 1)], dim=1)

    result: Dict[str, Any] = {
        "token_ids": out_ids,
        "n_prompt_tokens": n_prompt,
        "n_gen_tokens": len(out_ids),
        "image_span": (img_start, img_end),
    }
    if collect_trace:
        result["trace"] = trace
    return result
