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
    "causalmm_vision",
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

    # ---- SID / CT2S: vision-key masking on the negative branch ----
    # A different intervention on a different tensor: Fox edits pre-softmax
    # logits per head, CT2S makes whole vision keys unattendable for every
    # head. They are never on together -- SID is a compared baseline, and it
    # runs with Fox's mode="off".
    sid_enabled: bool = False
    sid_agg_layer: int = 2
    sid_rank: int = 100
    #: Rows the mask applies to. None would mean "all rows", which for the
    #: batched dual pass would prune the clean branch too.
    sid_rows: Optional[torch.Tensor] = None
    #: Refilled at layer ``sid_agg_layer - 1`` of every forward, and cleared
    #: by the decode loop before each one. ``None`` at a layer that needs it
    #: is an error, not a reason to skip the mask -- see _sid_dropped.
    sid_keep: Optional[torch.Tensor] = None

    # ---- OPERA: the last layer's attention, for the over-trust penalty ----
    # Captured here rather than via output_attentions=True, which on this
    # stack returns an empty tuple under SDPA (see the module docstring) and
    # would otherwise retain all 32 layers -- 3.9 GB at 5 beams on a 620-token
    # prefill. Only the last query row is needed, and only at one layer.
    opera_capture: bool = False
    opera_layer: int = -1
    #: ``[B, n_keys]``, max-pooled over heads, in the model dtype. Refilled by
    #: every forward that reaches ``opera_layer``; cleared by the loop first,
    #: so a miss raises instead of reusing the previous candidate's row.
    opera_attn: Optional[torch.Tensor] = None

    # ---- readouts, refilled every forward ----
    collect: bool = True
    diagnostics: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)

    def reset_diagnostics(self) -> None:
        self.diagnostics.clear()

    def sid_read_row(self) -> int:
        """Which row's layer-(agg-1) attention the ranking is read from.

        The released code reads batch element 0 of its own negative forward,
        which is the negative branch. Here the two branches are rows of one
        pass, so it is the first pruned row. Up to ``sid_agg_layer`` the two
        rows are identical anyway -- nothing has happened to either yet -- so
        this only matters if the rows ever stop sharing a prompt.
        """
        if self.sid_rows is None or self.sid_rows.numel() == 0:
            return 0
        return int(self.sid_rows[0])

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


def _opera_capture_attn(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> None:
    """Stash the last query's attention row, max-pooled over heads.

    OPERA reads ``outputs.attentions[-1].max(1, keepdim=True).values`` and
    keeps only the last query row, so that is all this computes: ``q[-1] @
    K.T`` rather than the full matrix. Softmax is row-independent, so the
    values are identical, and it avoids materialising
    ``[5 beams, 32, q, kv]`` per layer.

    Max over heads, not mean -- SID's CT2S averages, OPERA max-pools, and
    the two are different statistics of the same tensor. The pooling runs in
    the model dtype because eager LLaMA casts its fp32 softmax back to the
    query dtype before returning the weights, and OPERA pools that.

    Normalisation is deliberately NOT done here: OPERA divides ``attn_last``
    by its own row sum *after* assembling the square, and a max over heads
    does not sum to 1, so where the division happens is observable.
    """
    n_rep = getattr(module, "num_key_value_groups", 1)
    k_states = _repeat_kv(key, n_rep)
    q_last = query[:, :, -1:, :]
    logits = torch.matmul(q_last, k_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[..., -1:, : k_states.shape[-2]].to(
            logits.dtype)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    FOX.opera_attn = probs.max(dim=1).values[:, 0, :]


def _sid_dropped(n_keys: int, device: Any) -> torch.Tensor:
    """Which keys CT2S makes unattendable, for a forward whose ranking is in.

    Raises rather than returning an empty mask when the ranking is missing:
    a silently unpruned negative branch makes SID identical to the baseline
    with twice the compute, and the run would look finished and be wrong.
    """
    from foxrepro.fox.kernel import ct2s_dropped_keys

    if FOX.sid_keep is None:
        raise RuntimeError(
            f"CT2S reached layer >= {FOX.sid_agg_layer} with no keep-set. "
            f"Layer {FOX.sid_agg_layer - 1} must run through this attention "
            "function to read the ranking. Every layer goes through it when "
            "the model is loaded with attn_implementation='fox', so seeing "
            "this means the image span was not set at that layer."
        )
    return ct2s_dropped_keys(
        FOX.sid_keep, FOX.img_start, FOX.img_end, n_keys, device)


def _sid_read_ranking(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
) -> None:
    """Record CT2S's keep-set from this layer's attention, changing nothing.

    Only the last query row is needed, and softmax is row-independent, so
    this computes ``q[-1] @ K.T`` rather than the full ``[B, H, Q, K]``
    matrix: identical values, and it does not have to materialise 93 MB per
    branch at prefill. The caller then hands the layer to SDPA untouched, so
    every one of the 32 layers stays bit-for-bit the stock path and the only
    thing this arm changes is which keys are visible.
    """
    from foxrepro.fox.kernel import ct2s_keep

    n_rep = getattr(module, "num_key_value_groups", 1)
    k_states = _repeat_kv(key, n_rep)
    q_last = query[:, :, -1:, :]
    logits = torch.matmul(q_last, k_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        logits = logits + attention_mask[..., -1:, : k_states.shape[-2]].to(
            logits.dtype)
    # fp32 accumulation then back to the model dtype: what eager LlamaAttention
    # does, and the released mean-over-heads runs on the fp16 result.
    probs = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    FOX.sid_keep = ct2s_keep(
        probs, FOX.img_start, FOX.img_end,
        rank=FOX.sid_rank, row=FOX.sid_read_row(),
    )


@contextlib.contextmanager
def causalmm_vision(
    model: Any,
    *,
    rows: Tuple[int, ...] = (1,),
    edit: str = "shuffle",
    generator: Optional[torch.Generator] = None,
    stats: Optional[Dict[str, float]] = None,
) -> Iterator[None]:
    """Make ``rows`` see CausalMM's vision counterfactual, for one generate call.

    ``LlavaModel.get_image_features`` is the right seam: its list is
    ``torch.cat``-ed and ``masked_scatter``-ed into ``inputs_embeds``
    positionally, so replacing element *i* makes batch row *i* -- and only
    that row -- read a different image, with every other tensor and the whole
    language model untouched.

    The body re-does what the stock method does because the stock one returns
    only the projected features, and the counterfactual needs the
    *pre-projector* ones plus the attention. The duplication is covered by a
    test that asserts this reconstruction is bit-identical to
    ``get_image_features`` with attentions off, so a transformers change
    cannot make the clean branch drift silently.

    One deviation that cannot be avoided and must be read with the result:
    ``output_attentions=True`` puts CLIP on its eager path, and that changes
    the hidden states -- measured at max|diff| 3.9e-3 in fp16 on a small
    stand-in. So this arm's CLEAN branch is not bit-identical to the clean
    branch of every other arm in this repo. That is faithful to CausalMM,
    which always passes the flag and derives both branches from the one
    forward; it is recorded because this project's own finding is that a
    last-bits fp16 difference in attention moved POPE by 2.8 points, so
    "the last bits do not matter" is exactly the assumption not to make here.
    """
    from foxrepro.fox.kernel import cf_shuffle_attention, cf_weight_features

    inner = model.model
    orig = inner.get_image_features
    cfg = model.config

    def wrapped(pixel_values, vision_feature_layer=None,
                vision_feature_select_strategy=None, **kw):  # noqa: ANN001
        layer = (vision_feature_layer if vision_feature_layer is not None
                 else cfg.vision_feature_layer)
        strategy = (vision_feature_select_strategy
                    if vision_feature_select_strategy is not None
                    else cfg.vision_feature_select_strategy)
        if not isinstance(layer, int):
            raise ValueError(
                f"CausalMM reads one vision layer ({layer!r} is a pool); the "
                "released code indexes hidden_states and attentions with the "
                "same scalar mm_vision_select_layer"
            )
        if strategy != "default":
            raise ValueError(
                f"CausalMM drops the CLS token (features[:, 1:]), which is "
                f"vision_feature_select_strategy='default', not {strategy!r}"
            )
        if kw.get("image_sizes") is not None:
            raise ValueError(
                "CausalMM is a LLaVA-1.5 method: fixed 336x336, one tile. "
                "image_sizes means an anyres backbone, where the 24x24 "
                "feature grid this assumes does not hold"
            )
        n = int(pixel_values.shape[0])
        if n > 1 and not all(
            torch.equal(pixel_values[0], pixel_values[i]) for i in range(1, n)
        ):
            raise ValueError(
                "the batch rows do not share one image, so running the vision "
                "tower once and reusing it would be wrong. CausalMM's "
                "counterfactual is built from the SAME image as the clean "
                "branch; a differing row means some other perturbation is "
                "also active"
            )
        # One row, not n: the rows are identical, and the attentions are
        # [1, 24, 577, 577] per layer x 24 layers ~ 384 MB in fp16 -- doubling
        # that on a T4 that already holds 13 GB of weights is not free.
        with _eager_vision_tower(inner.vision_tower):
            outs = inner.vision_tower(
                pixel_values[:1], output_hidden_states=True,
                output_attentions=True)
        selected = outs.hidden_states[layer][:, 1:]
        attn = outs.attentions[layer] if outs.attentions is not None else None
        if attn is None:
            raise RuntimeError(
                "the vision tower returned no attentions, so the "
                "counterfactual cannot be built. On transformers >= ~4.52 "
                "output_attentions=True no longer falls back to eager -- it "
                "warns and returns None -- so the eager switch above is "
                "mandatory and did not take effect."
            )
        edited = (cf_shuffle_attention(attn, generator=generator)
                  if edit == "shuffle" else _cf_edit_other(attn, edit))
        weighted = cf_weight_features(selected, edited).to(selected.dtype)
        if stats is not None:
            gain = cf_weight_features(
                torch.ones_like(selected[..., :1]), edited).float()
            stats["cf_gain_mean"] = float(gain.mean())
            stats["cf_gain_std"] = float(gain.std())
        clean = inner.multi_modal_projector(selected)[0]
        cf = inner.multi_modal_projector(weighted)[0]
        out = [clean] * n
        for r in rows:
            if r >= n:
                raise ValueError(
                    f"counterfactual row {r} but the batch has {n} rows")
            out[r] = cf
        return out

    # Patch the INSTANCE, and on the way out remove the instance attribute
    # rather than assigning the old bound method back: leaving an attribute
    # that shadows the class method would survive this context for the rest
    # of the process, so a later arm would still be running through it.
    had_own = "get_image_features" in vars(inner)
    inner.get_image_features = wrapped
    try:
        yield
    finally:
        if had_own:  # pragma: no cover - nothing else patches this today
            inner.get_image_features = orig
        else:
            del inner.get_image_features


@contextlib.contextmanager
def _eager_vision_tower(tower: Any) -> Iterator[None]:
    """Put the vision tower on eager attention, then put it back.

    Required, not an optimisation. CausalMM was written against transformers
    4.31.0, where ``output_attentions=True`` silently fell back to eager. From
    ~4.52 onward it does not: ``sdpa_attention_forward`` warns once and
    returns ``None`` for the weights, so a direct port gets ``attentions=None``
    and builds a counterfactual out of nothing.

    This is also why CausalMM's clean branch cannot be bit-identical to the
    other arms': eager is what the method requires, and eager CLIP does not
    agree with SDPA CLIP in the last bits.
    """
    setter = getattr(tower, "set_attn_implementation", None)
    cfg = getattr(tower, "config", None)
    previous = getattr(cfg, "_attn_implementation", None) if cfg else None
    if setter is not None:
        setter("eager")
    elif cfg is not None:  # pragma: no cover - older transformers
        cfg._attn_implementation = "eager"
    else:  # pragma: no cover - not a transformers module
        raise RuntimeError(
            "cannot switch the vision tower to eager attention, so "
            "output_attentions would come back None"
        )
    try:
        yield
    finally:
        if previous is not None:
            if setter is not None:
                setter(previous)
            elif cfg is not None:  # pragma: no cover
                cfg._attn_implementation = previous


def _cf_edit_other(attn: torch.Tensor, edit: str) -> torch.Tensor:
    """The three non-default edits, which are the paper's ablation.

    Unreachable from the released code -- ``edit_attention``'s default is
    ``shuffle`` and the call site passes nothing -- so they are implemented
    straight from the reference rather than given their own kernel.
    """
    from foxrepro.fox.reference import reference_edit_attention

    return reference_edit_attention(attn, edit)


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

    if FOX.opera_capture and layer_idx is not None and layer_idx == FOX.opera_layer:
        _opera_capture_attn(
            module, query, key, attention_mask,
            scaling if scaling is not None
            else float(module.head_dim) ** -0.5,
        )

    if FOX.sid_enabled and layer_idx is not None and FOX.span_is_set():
        agg = FOX.sid_agg_layer
        if layer_idx == agg - 1:
            _sid_read_ranking(module, query, key, attention_mask, scaling
                              if scaling is not None
                              else float(module.head_dim) ** -0.5)
        elif layer_idx >= agg:
            n_keys = key.shape[-2]
            drop = _sid_dropped(n_keys, key.device)
            rows = FOX.sid_rows
            if attention_mask is None:
                raise RuntimeError(
                    "CT2S needs an explicit additive attention mask to write "
                    "into; register() installs eager_mask so that this is "
                    "never None. Seeing None here means the mask builder was "
                    "not registered, which also means causality is gone."
                )
            m = attention_mask[..., :n_keys]
            m = m.expand(query.shape[0], -1, -1, -1).clone()
            fill = torch.finfo(m.dtype).min
            if rows is None:
                m = m.masked_fill(drop.view(1, 1, 1, -1), fill)
            else:
                m[rows] = m[rows].masked_fill(drop.view(1, 1, 1, -1), fill)
            attention_mask = m
        # The layer itself is never altered: fall through to the untouched
        # path below with (possibly) a narrower mask.

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
    allow_cpu: bool = False,
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

    _refuse_cpu_only_host(dtype, allow_cpu=allow_cpu)
    _refuse_unconverted_checkpoint(model_id)

    register()
    if eager_layers is not None:
        FOX.eager_layers = eager_layers

    model, info = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch_dtype,            # `torch_dtype` is deprecated from 4.56
        attn_implementation={"text_config": ATTN_NAME, "vision_config": "sdpa"},
        device_map=device_map,
        output_loading_info=True,
    )
    _assert_fully_loaded(model_id, info)
    model = model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


#: Checkpoints known to use the original LLaVA key layout rather than the
#: HuggingFace-converted one. Weight-identical, but not loadable as a
#: ``LlavaForConditionalGeneration``.
_UNCONVERTED_PREFIXES = ("liuhaotian/",)


def _refuse_cpu_only_host(dtype: str, allow_cpu: bool = False) -> None:
    """Stop before loading 14 GB when there is no GPU to load it onto.

    Round 4 sessions 1 and 2 ran to the ten-hour wall on a Kaggle notebook
    whose accelerator was never attached. ``torch`` was the CPU build, fp16
    matmuls fell back to software, and a caption that costs 7.8 s on two T4s
    took 724 s -- 93x slower. Each session produced about fifty usable
    captions out of five hundred and then died on SIGTERM, and because the
    plan's other jobs never started, twenty GPU-hours of quota bought nothing.

    Nothing stopped it. ``probe_host.py`` runs first and reports the problem,
    but the runner only echoes a warning, and ``device_map="auto"`` places the
    model on CPU without complaint -- which is the right default for a laptop
    and exactly wrong for an unattended ten-hour session.

    Args:
        dtype: the weight dtype requested.
        allow_cpu: set by ``--allow-cpu`` for a deliberate CPU run.

    Raises:
        RuntimeError: when no CUDA device is visible and ``allow_cpu`` is not
            set.
    """
    if allow_cpu or torch.cuda.is_available():
        return
    raise RuntimeError(
        "no CUDA device is visible, so this run would execute on CPU.\n"
        f"  torch {torch.__version__} "
        f"({'CPU build' if '+cpu' in torch.__version__ else 'CUDA build'}), "
        f"torch.cuda.is_available() = False\n"
        "On Kaggle this means the notebook's accelerator is not attached: "
        "set Sidebar -> Accelerator = GPU T4 x2 and restart the session.\n"
        f"A {dtype} LLaVA-1.5-7B caption costs about 7.8 s on two T4s and "
        "about 724 s on CPU, so a ten-hour session would finish roughly one "
        "job in ten.\n"
        "Pass --allow-cpu if you really intend a CPU run."
    )


def _refuse_unconverted_checkpoint(model_id: str) -> None:
    """Fail before downloading 13.5 GB that cannot be used.

    ``liuhaotian/llava-v1.5-7b`` stores the language model under
    ``model.layers.*`` while transformers expects
    ``model.language_model.layers.*``, and it ships no
    ``preprocessor_config.json``. ``from_pretrained`` does not treat that as an
    error: it reports the vision tower and the projector as MISSING and
    **initialises them randomly**, then carries on. A model in that state
    produces fluent captions that have nothing to do with the image, and no
    metric downstream can tell.
    """
    if model_id.startswith(_UNCONVERTED_PREFIXES):
        raise ValueError(
            f"{model_id!r} is the original LLaVA checkpoint, not the "
            f"HuggingFace-converted one. transformers would load it with a "
            f"RANDOMLY INITIALISED vision tower and projector -- producing "
            f"fluent output unrelated to the image, which no metric would "
            f"catch.\n"
            f"Use 'llava-hf/llava-1.5-7b-hf' (weight-identical, correct key "
            f"layout, ships an image processor).\n"
            f"The original repo belongs to Track A, which runs the paper's own "
            f"vendored transformers 4.29.2 and expects that layout."
        )


def _assert_fully_loaded(model_id: str, info: Dict[str, Any]) -> None:
    """Refuse a checkpoint that did not populate every parameter.

    The general form of the guard above: any key transformers reports as
    missing was newly initialised, i.e. random. Silence here is how a whole
    benchmark run ends up measuring noise, so this is a hard error rather than
    a warning -- the one thing worse than a crash is a plausible number.
    """
    missing = list(info.get("missing_keys") or ())
    if not missing:
        return
    shown = ", ".join(missing[:8])
    more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
    raise RuntimeError(
        f"{model_id!r} left {len(missing)} parameters uninitialised, so they "
        f"are RANDOM: {shown}{more}.\n"
        f"This is not recoverable by retrying -- the checkpoint's key layout "
        f"does not match LlavaForConditionalGeneration. Check --model-id; it "
        f"must be the HuggingFace-converted repo ('llava-hf/llava-1.5-7b-hf')."
    )


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
    """Nucleus sampling or argmax over a single ``[V]`` logit row.

    Top-p goes through HuggingFace's own ``TemperatureLogitsWarper`` and
    ``TopPLogitsWarper``, in that order, because that is exactly what the
    released code does: ``vcd_sample.py`` applies ``logits_warper`` to the
    fused logits before ``softmax`` + ``multinomial``.

    This was originally hand-rolled, and the hand-rolled version kept one
    token fewer than HuggingFace in about 5% of rows. HF sorts ascending and
    drops the tail whose cumulative mass is ``<= 1 - top_p``; the hand-rolled
    version sorted descending and kept while the exclusive cumulative mass was
    ``<= top_p``. Those boundaries differ in strictness, and the two cumulative
    sums accumulate in opposite order so they also differ in the last bits. The
    affected token is always the least probable one in the support and would
    almost never be drawn -- but "almost never" is not "never", and a
    reproduction should not carry an avoidable difference in its sampler.

    ``torch.multinomial`` requires its generator on the same device as the
    probabilities. With ``device_map="auto"`` the logits come back on whichever
    device accelerate put the head on, which is not something this function
    should have to know -- so if the devices disagree, the single 32k-wide
    vector is moved to the generator's device. That costs nothing measurable
    and makes a seeded run reproduce the same tokens regardless of how the
    model happened to be sharded.
    """
    if greedy:
        return logits.argmax(dim=-1)

    from transformers import TemperatureLogitsWarper, TopPLogitsWarper

    out_device = logits.device
    if generator is not None and generator.device != logits.device:
        logits = logits.to(generator.device)

    x = logits.float().unsqueeze(0)
    if temperature != 1.0:
        x = TemperatureLogitsWarper(max(temperature, 1e-6))(None, x)
    if top_p < 1.0:
        x = TopPLogitsWarper(top_p)(None, x)
    probs = F.softmax(x[0], dim=-1)
    tok = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    return tok.to(out_device)


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
    single_branch: bool = False,
    pixel_values_do: Optional[torch.Tensor] = None,
    fusion: str = "fox_additive",
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
    input_ids_do: Optional[torch.Tensor] = None,
    attn_mask_do: Optional[torch.Tensor] = None,
    sid: bool = False,
    sid_agg_layer: int = 2,
    sid_rank: int = 100,
    cf_vision: bool = False,
    cf_edit: str = "shuffle",
) -> Dict[str, Any]:
    """Generate with Fox: both branches batched, fused per token.

    ``single_branch=True`` runs one branch with no fusion at all, sampling
    straight from ``z_obs``. That is the plain-nucleus-sampling baseline, and
    routing it through *this* function rather than ``model.generate()`` is
    deliberate: it guarantees the baseline and every Fox arm share one sampling
    implementation, so the only difference between them is the intervention.
    Unfair baselines are the most common silent error in this literature, and
    two different top-p implementations would be exactly that.

    Note that ``single_branch=True`` is NOT the same as ``mode="off"``. The
    latter keeps the dual pass and the fusion while disabling the intervention,
    which is the control that isolates the additive fusion ``z_obs + alpha *
    z_do`` -- and which the paper never ran. Both are needed and they answer
    different questions.

    ``fusion`` selects how the two branches combine, and the two options are
    not variants of one another:

    * ``"fox_additive"`` -- ``z_obs + lambda * z_do``, conflict-gated. Fox.
      Amplifies where the branches AGREE.
    * ``"contrastive"`` -- ``(1 + cd_alpha) * z_obs - cd_alpha * z_do``, the
      rule VCD, ICD and SID share. Amplifies where they DISAGREE.

    How the negative branch is built is then a separate axis, and the three
    compared baselines differ only along it:

    * ``pixel_values_do`` -- a different image. VCD noises it.
    * ``input_ids_do`` / ``attn_mask_do`` -- a different token sequence. ICD
      inserts a negative role description after the image marker.
    * ``sid`` -- the same image and the same prompt, read through CT2S: the
      100 vision keys the model attends to least are the only ones left
      visible from layer ``sid_agg_layer`` onward. SID perturbs nothing
      outside the model, which is why its negative branch costs one forward
      and no preprocessing.
    * ``cf_vision`` -- the same image, re-weighted per visual token by a
      shuffled CLIP attention map before the projector. CausalMM. The
      language model is untouched; only what row 1 reads as the image
      changes. See :func:`causalmm_vision`.

    When ``input_ids_do`` is given the two branches cannot be batched: their
    sequences differ in length and content, so position ids and the cache
    would have to be reconciled by padding. They are run as two passes with
    two caches instead, which is also what the released code does. Fox and
    VCD keep the batched path, where both rows share one token sequence.

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
    from foxrepro.fox.kernel import contrastive_fuse, fuse, jsd

    if fusion not in ("fox_additive", "contrastive"):
        raise ValueError(
            f"fusion must be 'fox_additive' or 'contrastive', got {fusion!r}"
        )
    n_perturbations = sum((
        input_ids_do is not None, pixel_values_do is not None, bool(sid),
        bool(cf_vision),
    ))
    if n_perturbations > 1:
        # SID's own harness DOES stack two of these: `--use-vcd` leaves
        # --fast-v-attention-rank at 100, so its VCD row is VCD+CT2S
        # (vcd_sample.py:141, reached for any images_cd). That is a defect in
        # the harness, not a method, and a number from it would be
        # misattributed to whichever paper it was filed under.
        raise ValueError(
            f"{n_perturbations} negative-branch perturbations are active at "
            "once (of input_ids_do, pixel_values_do, sid/CT2S, cf_vision). "
            "Each compared method is exactly one of them; stacking any two "
            "is not a published method and the result belongs to neither"
        )
    if cf_vision and single_branch:
        raise ValueError(
            "cf_vision=True needs the negative branch; not single_branch")
    if sid and single_branch:
        raise ValueError("sid=True needs the negative branch; not single_branch")
    if sid and sid_agg_layer < 1:
        raise ValueError(
            f"sid_agg_layer must be >= 1 (the ranking is read from the layer "
            f"below it), got {sid_agg_layer}"
        )
    #: ICD's branch has its own token sequence, so it needs its own pass.
    split_pass = input_ids_do is not None and not single_branch

    device = next(model.parameters()).device
    image_token_id = model.config.image_token_index
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    img_start, img_end = locate_image_span(input_ids, image_token_id)
    n_prompt = int(input_ids.shape[1])

    gen = None
    if seed is not None:
        # On the model's device, not CPU. The generator feeds both
        # torch.multinomial here and the uniform_() in the noise_control arm,
        # and both operate on device tensors -- a CPU generator raises
        # "Expected a 'cuda' device type for generator but found 'cpu'".
        gen = torch.Generator(device=device).manual_seed(seed)

    # Row 0 observational, row 1 interventional: identical token sequences, so
    # batching them is exact, and decode reads the weights once instead of
    # twice. The released code runs two sequential passes.
    n_rows = 1 if (single_branch or split_pass) else 2
    ids2 = input_ids.repeat(n_rows, 1)
    mask2 = attn_mask.repeat(n_rows, 1)
    if single_branch or split_pass:
        # split_pass runs one row per forward, so the image must not be
        # doubled here: ids2 has one row and the shapes have to agree.
        px2 = pixel_values
    elif pixel_values_do is not None:
        # VCD: row 0 sees the clean image, row 1 the corrupted one.
        px2 = torch.cat([pixel_values, pixel_values_do.to(device)], dim=0)
    else:
        px2 = pixel_values.repeat(2, *([1] * (pixel_values.dim() - 1)))

    if split_pass:
        ids_do = input_ids_do.to(device)
        mask_do = (attn_mask_do if attn_mask_do is not None
                   else torch.ones_like(ids_do)).to(device)
        px_do = pixel_values

    out_ids: List[int] = []
    trace: List[Dict[str, float]] = []
    sid_kept: List[int] = []
    cache = None
    cache_do = None

    # Populated by causalmm_vision; empty otherwise. Recorded so a run whose
    # counterfactual gain collapsed to ~0 (i.e. the negative branch is
    # reading a blanked image rather than a reshuffled one) is visible in the
    # record instead of only in the score.
    cf_stats: Dict[str, float] = {}
    cf_ctx: Any = (
        causalmm_vision(model, rows=(1,), edit=cf_edit, generator=gen,
                        stats=cf_stats)
        if cf_vision else contextlib.nullcontext()
    )

    with FOX.configured(
        enabled=not single_branch, mode=mode, gamma=gamma, ratio=ratio,
        logit_dtype=logit_dtype, softmax_dtype=softmax_dtype,
        img_start=img_start, img_end=img_end,
        branch_rows=torch.tensor([1], device=device),
        eager_layers=eager_layers if eager_layers is not None else FOX.eager_layers,
        generator=gen,
        sid_enabled=sid, sid_agg_layer=sid_agg_layer, sid_rank=sid_rank,
        sid_rows=torch.tensor([1], device=device) if sid else None,
        sid_keep=None,
    ), cf_ctx:
        for step in range(max_new_tokens):
            FOX.reset_diagnostics()
            # The released code recomputes the keep-set from the current last
            # query at every forward, so it is per-step state, not per-prompt.
            # Clearing it here is what makes a missed read raise instead of
            # silently reusing the previous token's ranking.
            FOX.sid_keep = None
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
            if sid:
                # Recorded per step because the ranking is recomputed per
                # step: a run whose keep count drifts from sid_rank has hit
                # the n_keys clamp and is not the configured method.
                sid_kept.append(
                    0 if FOX.sid_keep is None else int(FOX.sid_keep.numel()))
            z_obs = out.logits[0, -1].float().unsqueeze(0)

            if split_pass:
                # ICD: the negative branch has its own prompt, so its own
                # pass and its own cache. The intervention is OFF for it --
                # the perturbation IS the instruction, not an attention edit.
                if step == 0:
                    out_do = model(
                        input_ids=ids_do, attention_mask=mask_do,
                        pixel_values=px_do, use_cache=True,
                    )
                else:
                    out_do = model(
                        input_ids=ids_do[:, -1:], attention_mask=mask_do,
                        past_key_values=cache_do, use_cache=True,
                    )
                cache_do = out_do.past_key_values
                z_do = out_do.logits[0, -1].float().unsqueeze(0)

            if single_branch:
                # No second branch, no fusion: this IS the baseline. Sampling
                # goes through the same _sample_from as every Fox arm.
                d_t = torch.zeros(1, device=z_obs.device, dtype=torch.float64)
                lam = torch.zeros(1, device=z_obs.device)
                fused = masked = z_obs
                decision = z_obs
            else:
                if not split_pass:
                    z_do = out.logits[1, -1].float().unsqueeze(0)
                if fusion == "contrastive":
                    # VCD / ICD / SID. A fixed subtraction, no JSD gate, so
                    # d_t stays 0 and lambda records -cd_alpha -- the trace
                    # keeps one schema across every arm rather than growing a
                    # second shape that every consumer must branch on.
                    d_t = torch.zeros(1, device=z_obs.device,
                                      dtype=torch.float64)
                    lam = torch.full((1,), -float(cd_alpha),
                                     device=z_obs.device)
                    fused, masked = contrastive_fuse(
                        z_obs, z_do, cd_alpha=cd_alpha, cd_beta=cd_beta
                    )
                    decision = fused if (greedy and not mask_greedy) else masked
                else:
                    if mode == "off":
                        # The fusion still runs. This arm isolates the
                        # additive logit fusion from the intervention -- the
                        # control the paper never ran.
                        d_t = torch.zeros(1, device=z_obs.device,
                                          dtype=torch.float64)
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

            nxt2 = nxt.view(1, 1).repeat(n_rows, 1).to(device)
            ids2 = torch.cat([ids2, nxt2], dim=1)
            mask2 = torch.cat([mask2, mask2.new_ones(n_rows, 1)], dim=1)
            if split_pass:
                # The negative branch generates the SAME token: both branches
                # must condition on the text actually emitted, or the contrast
                # after step 1 is between two different continuations rather
                # than between two readings of one.
                ids_do = torch.cat([ids_do, nxt.view(1, 1).to(device)], dim=1)
                mask_do = torch.cat([mask_do, mask_do.new_ones(1, 1)], dim=1)

    result: Dict[str, Any] = {
        "token_ids": out_ids,
        "n_prompt_tokens": n_prompt,
        "n_gen_tokens": len(out_ids),
        "image_span": (img_start, img_end),
    }
    if collect_trace:
        result["trace"] = trace
    if sid:
        result["sid_n_kept"] = sid_kept
    if cf_stats:
        result.update(cf_stats)
    return result
