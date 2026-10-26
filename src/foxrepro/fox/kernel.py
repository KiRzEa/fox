"""The Fox mechanism, clean and parameterised. Proven bit-equal to the original.

This is the port. ``fox/reference.py`` is the literal transcription of the
released code and acts as the numerical oracle;
``tests/test_kernel_equivalence.py`` asserts that ``intervene`` in
``mode="as_implemented"`` is **bit-identical** to ``reference_intervene`` on
random tensors across every dtype. Because the method has no learned
parameters, that proof needs no GPU, no LLaVA checkpoint, and none of the
released repo's vendored transformers tree.

What this module adds over the reference
----------------------------------------
* **Five intervention modes as flags, not code paths** -- so a run file's
  provenance fully determines what was computed. See ``INTERVENTION_MODES``
  in ``storage.py`` for what each arm tests and why.
* **An explicit logit storage dtype.** The reference hardcodes fp16 in three
  places. The paper's effect lives in the rounding of *this* tensor, so it has
  to be a parameter: varying the weight dtype instead moves two things at once
  and yields a null result that is correct but uninformative.
* **No Python loop over the batch**, so the observational and interventional
  branches can run as two rows of one forward pass. The reference runs them as
  two sequential passes, which reads the 13.16 GiB of weights twice per token;
  batching halves decode wall-clock on a bandwidth-bound GPU.
* **On-device JSD.** The reference calls ``.float().cpu().numpy()`` on a
  32064-wide logit vector once per generated token, forcing a GPU->CPU
  synchronisation. Numerically identical, verified in the equivalence test.
* **Per-row JSD.** The reference computes it from batch row 0 only and applies
  that scalar to every row, which silently breaks under batching.

Deliberately NOT fixed
----------------------
Two defects are preserved under ``mode="as_implemented"`` because fixing them
would stop this being a reproduction:

1. The double normalisation inside the entropy helper (the caller normalises,
   then ``_entropy_from_probs`` normalises again). It changes the last bits.
2. The greedy/sampling decision-rule asymmetry, which lives in ``fuse`` and is
   exposed as ``mask_greedy`` so both can be reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "TORCH_DTYPES",
    "resolve_logit_dtype",
    "ulp_at",
    "quantize_to_grid",
    "head_scores",
    "select_heads",
    "intervene",
    "jsd",
    "fuse",
    "contrastive_fuse",
    "add_diffusion_noise",
    "ct2s_keep",
    "ct2s_dropped_keys",
    "cf_shuffle_attention",
    "cf_weight_features",
]

#: Canonical dtype spelling -> torch dtype. ``"native"`` is resolved against
#: the tensor being operated on, so it means "leave the dtype alone".
TORCH_DTYPES: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_logit_dtype(spelling: str, like: torch.Tensor) -> torch.dtype:
    """Map a dtype spelling to a torch dtype, resolving ``"native"``."""
    if spelling == "native":
        return like.dtype
    try:
        return TORCH_DTYPES[spelling]
    except KeyError:
        raise ValueError(
            f"unknown logit dtype {spelling!r}; want 'native' or one of "
            f"{sorted(TORCH_DTYPES)}"
        ) from None


def ulp_at(magnitude: float, dtype: torch.dtype) -> float:
    """Spacing between representable values of ``dtype`` near ``magnitude``.

    This is the quantity the paper's mechanism actually rides on. For fp16
    (10 mantissa bits) at ``|x| ~ 1e4``, ``|x|`` sits in the binade
    ``[2^13, 2^14)``, so the spacing is ``2^(13-10) = 8.0`` -- and since real
    Llama attention logits span O(1)-O(30), subtracting 1e4 and casting back
    collapses the row onto a grid eight units apart.

    Measured with ``nextafter`` rather than derived, so it is right for any
    dtype including subnormals, and it returns ``inf`` where ``magnitude``
    overflows the dtype (fp16 tops out at 65504, which is why the gamma sweep
    stops at 6e4).
    """
    t = torch.tensor([abs(magnitude)], dtype=dtype)
    if not bool(torch.isfinite(t).all()):
        return float("inf")
    nxt = torch.nextafter(t, torch.tensor([float("inf")], dtype=dtype))
    return float((nxt - t).item())


def quantize_to_grid(x: torch.Tensor, step: float) -> torch.Tensor:
    """Round ``x`` onto a uniform grid of spacing ``step``.

    The ``ulp_control`` arm. If replacing the shift-and-cast round trip with
    this reproduces the paper's gain, then the mechanism is requantization and
    the causal framing adds nothing -- which is the single most informative
    result this reproduction can produce.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    return torch.round(x / step) * step


# --------------------------------------------------------------------------- #
# diagnosis: the per-head risk score
# --------------------------------------------------------------------------- #
def head_scores(
    attn_probs: torch.Tensor,
    img_start: int,
    img_end: int,
    *,
    use_m_sys: bool = True,
    eps: float = 1e-12,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-head ``(m_sys, H_vis)`` at the last query position.

    Args:
        attn_probs: ``[B, H, Q, K]`` post-softmax, causal mask already applied.
        img_start: first image-token key index. System keys are ``[0, img_start)``.
        img_end: last image-token key index, inclusive. Visual keys are
            ``[img_start, img_end + 1)``.
        use_m_sys: ``False`` reproduces the InstructBLIP copy of the released
            code, which scores on ``H_vis`` alone and so does not implement
            Eq. 4 of the paper at all.

    Returns ``None`` if the visual span is empty, which is the released
    early-return that makes the caller skip the intervention.

    ``m_sys`` is the attention mass on the system prompt -- the paper's claimed
    anchor for the language prior. ``H_vis`` is the entropy of the attention
    *renormalised over image tokens only*, so it measures dispersion of visual
    focus independently of how much total mass went to the image. The product
    is a conjunctive filter: it fires only where visual grounding has collapsed
    AND the prior path is active.
    """
    B, H, Q, K = attn_probs.shape
    v_s = max(0, min(img_start, K))
    v_e = max(v_s, min(img_end + 1, K)) if img_end >= 0 else 0
    if v_e <= v_s:
        return None

    a = attn_probs[:, :, -1, :]                                  # [B, H, K]
    m_sys = a[..., 0:img_start].sum(dim=-1)                      # [B, H]

    p_v = a[..., v_s:v_e]
    p_v = p_v / (p_v.sum(dim=-1, keepdim=True) + eps)
    # The released code normalises again inside its entropy helper. Preserved:
    # it changes the last bits, and bit-equality is the point of this port.
    p_v = p_v / (p_v.sum(dim=-1, keepdim=True) + eps)
    h_vis = -(p_v * (p_v + eps).log()).sum(dim=-1)               # [B, H]
    return m_sys, h_vis


def select_heads(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Top-k heads per batch row, as a boolean ``[B, H]`` mask.

    ``k = clamp(round(H * ratio), 1, H // 2)``. The upper clamp is in the
    released code and caps any ratio above 0.5 at half the heads: for 32 heads,
    the released ratio 0.35 gives 11, and the paper's 0.45 would give 14.

    Vectorised over the batch; the released code loops.
    """
    B, H = scores.shape
    k = int(round(H * ratio))
    k = max(1, min(k, H // 2))
    idx = torch.topk(scores, k=k, dim=-1, largest=True).indices
    mask = torch.zeros((B, H), dtype=torch.bool, device=scores.device)
    return mask.scatter_(1, idx, True)


# --------------------------------------------------------------------------- #
# the intervention
# --------------------------------------------------------------------------- #
def intervene(
    attn_logits: torch.Tensor,
    img_start: int,
    img_end: int,
    *,
    mode: str = "as_implemented",
    gamma: float = 10000.0,
    ratio: float = 0.35,
    logit_dtype: str = "native",
    attention_mask: Optional[torch.Tensor] = None,
    rows: Optional[torch.Tensor] = None,
    use_m_sys: bool = True,
    noise_halfwidth: Optional[float] = None,
    grid_step: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """Apply one arm of the Fox intervention to pre-softmax attention logits.

    Args:
        attn_logits: ``[B, H, Q, K]`` logits, already scaled by
            ``1/sqrt(head_dim)``, causal mask NOT yet added. **Mutated in
            place**, matching the released code.
        img_start / img_end: image-token key span, ``img_end`` inclusive.
        mode: one of ``off``, ``as_implemented``, ``as_described``,
            ``ulp_control``, ``noise_control``.
        gamma: the shift magnitude. Never stated in the paper; 10000.0 in the
            released code, where it is a local literal. Its effect is
            quantised: what matters is which ULP bucket ``|gamma|`` lands in,
            so a sweep over ``{1e2, 1e3, 1e4, 6e4}`` spans ULP
            ``{0.0625, 0.5, 8, 32}`` in fp16.
        logit_dtype: storage dtype for the round trip. ``"native"`` keeps the
            incoming dtype and is bit-equivalent to the released code.
        rows: batch rows to intervene on. ``None`` means all. Pass
            ``tensor([1])`` to run observational and interventional branches as
            rows 0 and 1 of a single forward pass.
        noise_halfwidth: for ``noise_control``; defaults to half the ULP at
            ``gamma`` in the target dtype, so the perturbation matches the
            quantization in magnitude but has no grid structure.
        grid_step: for ``ulp_control``; defaults to the ULP at ``gamma``.

    Returns ``(attn_logits, diagnostics)`` where ``diagnostics`` carries
    ``m_sys``, ``h_vis``, ``score`` and ``head_mask`` for the risk-heatmap
    panel, or ``None`` when the visual span was empty or ``mode="off"``.
    """
    if mode == "off":
        return attn_logits, None

    # Metrics come from a separate fp32, mask-applied, softmaxed copy -- never
    # from the tensor being mutated. The released code does the same.
    base = attn_logits.to(torch.float32)
    if attention_mask is not None:
        base = base + attention_mask.to(torch.float32)
        base = torch.max(
            base,
            torch.tensor(torch.finfo(torch.float32).min, device=base.device),
        )
    with torch.no_grad():
        probs = torch.softmax(base, dim=-1)
        stats = head_scores(
            probs, img_start, img_end, use_m_sys=use_m_sys, eps=eps
        )
    if stats is None:
        return attn_logits, None
    m_sys, h_vis = stats
    score = (m_sys * h_vis) if use_m_sys else h_vis

    head_mask = select_heads(score, ratio)                       # [B, H] bool
    B, H, Q, K = attn_logits.shape
    row_sel = (
        torch.arange(B, device=attn_logits.device) if rows is None
        else rows.to(attn_logits.device)
    )

    store_dtype = resolve_logit_dtype(logit_dtype, attn_logits)
    orig_dtype = attn_logits.dtype

    for b in row_sel.tolist():
        heads = head_mask[b]
        if not bool(heads.any()):
            continue
        # The released slice: ALL key positions, the LAST query only.
        target = attn_logits[b, heads, -1:, :]

        if mode == "as_implemented":
            # fp32 round trip, then back to storage dtype. The shift cancels in
            # the stable fp32 softmax downstream; the cast's rounding does not.
            y = (target.to(torch.float32) - gamma).to(store_dtype)
        elif mode == "boost_visual":
            # The experiment the paper argues from but never runs.
            #
            # Its motivating claim is that amplifying visual attention cannot
            # help, because Figure 1(a) shows faithful and hallucinated samples
            # carry the same global visual attention mass. But that is an
            # OBSERVATIONAL comparison, and the conclusion drawn from it is
            # about an INTERVENTION -- exactly the P(Y|X) vs P(Y|do(X))
            # distinction the paper takes as its theoretical foundation. A
            # quantity uncorrelated with an outcome can still change it when
            # intervened upon.
            #
            # PAI, named in the introduction as the representative of this
            # family, appears twice in the paper and is in none of its tables.
            # So: add a positive bias to the image-token keys at the same heads
            # and query positions Fox suppresses, leaving the system-prompt keys
            # untouched, and measure hallucination. Same slice as
            # `as_described`, opposite sign.
            y = target.clone()
            v_s = max(0, min(img_start, K))
            v_e = max(v_s, min(img_end + 1, K))
            y[..., v_s:v_e] = y[..., v_s:v_e] + gamma
        elif mode == "as_described":
            # What Appendix B describes: an additive bias on the image-token
            # keys only, no round trip. Not shift-invariant, so this has a real
            # effect independent of precision.
            y = target.clone()
            v_s = max(0, min(img_start, K))
            v_e = max(v_s, min(img_end + 1, K))
            y[..., v_s:v_e] = y[..., v_s:v_e] - gamma
        elif mode == "ulp_control":
            step = grid_step if grid_step is not None else ulp_at(gamma, store_dtype)
            if not math.isfinite(step):
                raise ValueError(
                    f"gamma={gamma} overflows {store_dtype}; ULP is undefined. "
                    "fp16 tops out at 65504."
                )
            y = quantize_to_grid(target.to(torch.float32), step).to(store_dtype)
        elif mode == "noise_control":
            step = grid_step if grid_step is not None else ulp_at(gamma, store_dtype)
            hw = noise_halfwidth if noise_halfwidth is not None else step / 2.0
            if not math.isfinite(hw):
                raise ValueError(f"gamma={gamma} overflows {store_dtype}")
            t32 = target.to(torch.float32)
            # uniform_() requires the generator on the tensor's device, same as
            # torch.multinomial. Drop a mismatched generator rather than
            # raising: losing per-step reproducibility of the noise is a far
            # smaller problem than a crash 300 samples into a run, and the
            # mismatch is reported so it cannot pass unnoticed.
            g = generator
            if g is not None and g.device != t32.device:
                import warnings

                warnings.warn(
                    f"noise_control: generator is on {g.device} but the logits "
                    f"are on {t32.device}; falling back to the global RNG, so "
                    f"this arm is not seed-reproducible for this run",
                    RuntimeWarning,
                    stacklevel=2,
                )
                g = None
            noise = torch.empty_like(t32).uniform_(-hw, hw, generator=g)
            y = (t32 + noise).to(store_dtype)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        attn_logits[b, heads, -1:, :] = y.to(orig_dtype)

    return attn_logits, {
        "m_sys": m_sys, "h_vis": h_vis, "score": score, "head_mask": head_mask,
    }


# --------------------------------------------------------------------------- #
# conflict-gated fusion
# --------------------------------------------------------------------------- #
def jsd(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    clip_value: float = 5e-8,
    candidate_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Jensen-Shannon divergence between two logit rows, per batch row, on-device.

    Args:
        p_logits / q_logits: ``[B, V]``.
        candidate_mask: optional ``[B, V]`` boolean restricting the support.
            The paper's Eq. 20 defines the JSD over a truncated candidate set
            and justifies it explicitly; the released code computes it over the
            full vocabulary regardless. Leave ``None`` to reproduce the code,
            pass a mask to test what the paper describes.

    Returns ``[B]``. Accumulated in float64 to match the released numpy path.
    """
    p_logits = p_logits.double()
    q_logits = q_logits.double()
    if candidate_mask is not None:
        neg = torch.finfo(torch.float64).min
        p_logits = p_logits.masked_fill(~candidate_mask, neg)
        q_logits = q_logits.masked_fill(~candidate_mask, neg)

    def _sm(x: torch.Tensor) -> torch.Tensor:
        x = x / temperature
        x = x - x.max(dim=-1, keepdim=True).values
        e = x.exp()
        return e / e.sum(dim=-1, keepdim=True)

    p = _sm(p_logits).clamp(clip_value, 1.0)
    q = _sm(q_logits).clamp(clip_value, 1.0)
    m = 0.5 * (p + q)
    kl_p = (p * (p / m).log()).sum(dim=-1)
    kl_q = (q * (q / m).log()).sum(dim=-1)
    return 0.5 * (kl_p + kl_q)


def contrastive_fuse(
    obs_logits: torch.Tensor,
    cd_logits_in: torch.Tensor,
    *,
    cd_alpha: float = 1.0,
    cd_beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The contrastive rule VCD, ICD and SID share.

    ``diffs = (1 + a) * z_obs - a * z_cd``, then the adaptive plausibility
    constraint keeps only tokens the ORIGINAL branch found plausible:
    ``z_obs >= log(beta) + max(z_obs)``.

    This is a **subtraction**, and Fox's :func:`fuse` is an addition. They
    are not variants of one another: this one amplifies where the two
    branches disagree, Fox's amplifies where they agree.

    Proven bit-identical to :func:`foxrepro.fox.reference.reference_contrastive_fuse`
    on random tensors across dtypes, so a number produced here is the number
    the baselines' own code produces.

    Args:
        obs_logits: next-token logits from the unperturbed branch.
        cd_logits_in: next-token logits from the perturbed branch.
        cd_alpha: contrast strength. 1.0 for VCD, 0.5 for SID -- the formula
            is shared, the tuning is not, so the caller states it.
        cd_beta: plausibility threshold, 0.1 in all three.

    Returns:
        ``(diffs, masked)``. Greedy argmaxes ``diffs`` unmasked; sampling
        draws from ``masked``. The released code has that asymmetry and so
        does Fox, which is why both are returned rather than one.
    """
    cutoff = math.log(cd_beta) + obs_logits.max(dim=-1, keepdim=True).values
    diffs = (1 + cd_alpha) * obs_logits - cd_alpha * cd_logits_in
    masked = diffs.masked_fill(obs_logits < cutoff, -float("inf"))
    return diffs, masked


def add_diffusion_noise(
    image_tensor: torch.Tensor,
    noise_step: int = 500,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """VCD's image corruption: forward diffusion on a **sigmoid** schedule.

    Not the linear DDPM schedule. ``linspace(-6, 6)`` through a sigmoid,
    scaled into ``[1e-5, 0.5e-2]``, keeps ``sqrt(alpha_bar) = 0.863`` of the
    image at step 500; the familiar ``linspace(1e-4, 0.02)`` keeps 0.280 and
    would be contrasting against near-white noise. An earlier version of this
    repo used the linear one, which is why the schedule is transcribed in
    ``reference.py`` and asserted equal rather than written from memory.

    Args:
        image_tensor: pixel values.
        noise_step: index into the 1000-step schedule; 500 in both releases.
        generator: for reproducible noise. The released code uses global
            RNG; threading a generator here is what lets a seeded run be
            repeatable without touching global state.
    """
    num_steps = 1000
    betas = torch.linspace(-6, 6, num_steps, device=image_tensor.device)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas_prod = torch.cumprod(1.0 - betas, dim=0)
    a_t = torch.sqrt(alphas_prod[noise_step]).to(image_tensor.dtype)
    a_1m_t = torch.sqrt(1.0 - alphas_prod[noise_step]).to(image_tensor.dtype)

    noise = torch.empty_like(image_tensor).normal_(generator=generator)
    return a_t * image_tensor + a_1m_t * noise


def fuse(
    obs_logits: torch.Tensor,
    do_logits: torch.Tensor,
    js: torch.Tensor,
    *,
    alpha: float = 2.0,
    tau_js: float = 0.15,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Conflict-gated additive fusion, per batch row.

    ``lambda = alpha`` where ``js < tau_js`` (consensus), else ``lambda = js``
    (conflict), then ``z_final = z_obs + lambda * z_do``.

    Returns ``(fused, masked, lam)``:
      * ``fused`` -- raw. The released GREEDY path argmaxes this, unmasked.
      * ``masked`` -- ``fused`` with VCD's plausibility constraint applied,
        keeping only tokens where ``z_obs >= log(beta) + max(z_obs)``. The
        released SAMPLING path uses this.

    Returning both is deliberate: that asymmetry is why the paper's Table 9
    (greedy) is not a clean de-randomisation of Table 1 (sampling), and
    reporting both decision rules is how this reproduction separates the two.

    Unlike the released code, ``js`` is per-row, so the two branches can be
    batched. The released version computes it from row 0 and applies that
    scalar everywhere.
    """
    js = js.to(obs_logits.dtype).reshape(-1, 1)
    lam = torch.where(js < tau_js, torch.full_like(js, alpha), js)
    fused = obs_logits + lam * do_logits

    cutoff = math.log(beta) + obs_logits.max(dim=-1, keepdim=True).values
    masked = fused.masked_fill(obs_logits < cutoff, -float("inf"))
    return fused, masked, lam.reshape(-1)


# --------------------------------------------------------------------------- #
# SID: CT2S vision-token selection
# --------------------------------------------------------------------------- #
def ct2s_keep(
    probs: torch.Tensor,
    img_start: int,
    img_end: int,
    *,
    rank: int = 100,
    row: int = 0,
) -> torch.Tensor:
    """The vision keys SID's negative branch keeps: the LEAST attended ones.

    ``probs`` is ``[B, H, Q, K]`` post-softmax attention from layer
    ``agg_layer - 1``, in the model dtype -- the released mean over heads runs
    in fp16, and averaging in fp32 instead would reorder near-ties and change
    which tokens survive.

    Returns absolute key positions, sorted by attention rather than by
    position; ``ct2s_dropped_keys`` does not care about the order.

    Asserted equal to ``reference.reference_ct2s_keep_indices``.
    """
    avg = torch.mean(probs, dim=1)[row]          # over heads, then one row
    last = avg[-1]                               # the last query position
    img = last[img_start:img_end + 1]
    k = min(int(rank), int(img.shape[0]))
    return img.topk(k, largest=False).indices + img_start


def ct2s_dropped_keys(
    keep: Optional[torch.Tensor],
    img_start: int,
    img_end: int,
    n_keys: int,
    device: Any = None,
) -> torch.Tensor:
    """Boolean ``[n_keys]``: True where a vision key must be unattendable.

    The released code builds the complement -- all ones, image span off, keep
    set back on (``reference_ct2s_attention_mask``) -- which is the same set.
    This form is returned instead because it is what gets added to an
    additive attention mask, and because it stays correct when ``n_keys``
    grows past ``img_end`` during decode.
    """
    drop = torch.zeros(n_keys, dtype=torch.bool, device=device)
    hi = min(int(img_end), n_keys - 1)
    if hi < img_start:
        return drop
    drop[img_start:hi + 1] = True
    if keep is not None and keep.numel():
        k = keep[keep <= hi]
        drop[k] = False
    return drop


# --------------------------------------------------------------------------- #
# CausalMM: the vision counterfactual
# --------------------------------------------------------------------------- #
def cf_shuffle_attention(
    attn: torch.Tensor, *, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Permute each head's attention entries, independently per head.

    ``attn`` is ``[B, H, X, Y]``. The permutation is over ``X * Y`` and is
    shared across the batch within a head, which is what the released
    ``gather(1, perm.expand(batch, -1))`` does.

    The generator is this repo's addition: the released code calls
    ``torch.randperm`` with no generator, so the counterfactual is drawn from
    global RNG and the one component whose purpose is randomness does not
    repeat per sample in a seeded run. Same deviation, same reason, as
    :func:`add_diffusion_noise`.

    Asserted equal to ``reference.reference_edit_attention(..., 'shuffle')``.
    """
    out = attn.clone()
    b, h, x, y = attn.shape
    for i in range(h):
        perm = torch.randperm(x * y, device=attn.device, generator=generator)
        out[:, i] = out[:, i].view(b, -1).gather(
            1, perm.expand(b, -1)).view(b, x, y)
    return out


def cf_weight_features(
    features: torch.Tensor, attn: torch.Tensor
) -> torch.Tensor:
    """Rescale each visual token by a head-averaged, resized attention value.

    ``features`` is ``[B, N, D]`` with ``N`` a square (576 on LLaVA-1.5) and
    is the **pre-projector** feature, matching the released order: weight
    first, project after. The projector is a two-layer GELU MLP, so the two
    orders are not the same transform.

    The dtype sequence is the released one and is load-bearing: interpolate
    and mean run in ``attn``'s dtype, the multiply in ``features``' dtype,
    and the result is cast to fp16 unconditionally.

    Asserted equal to ``reference.reference_apply_attention``.
    """
    n = int(math.sqrt(features.shape[1]))
    resized = F.interpolate(attn, size=(n, n), mode="bilinear",
                            align_corners=False)
    avg = torch.mean(resized, dim=1)
    avg = avg.view(avg.size(0), -1, 1).expand(-1, -1, features.shape[-1])
    return (features * avg).to(torch.float16)
