"""Literal transcriptions of the released Fox code. The numerical oracle.

**Do not clean this file up. Do not optimise it. Do not fix its bugs.**

Every function here is a line-for-line transcription of the code shipped with
the paper, lifted out of its 3370-file vendored-transformers context so it can
be executed on bare tensors. Its only job is to be the thing
``fox/kernel.py`` is proven bit-identical to.

Why this file exists
--------------------
The paper's mechanism is a *numerical* effect. Fox subtracts a large constant
from the pre-softmax attention logits and casts the result back to the model
dtype; because the softmax that follows is numerically stable and accumulates
in fp32, the uniform shift cancels exactly, and the only surviving effect is
the rounding of that tensor onto its dtype's ULP grid.

That makes a reimplementation dangerous in the usual way: if our port produced
different numbers from the paper, we could not tell "the paper is wrong" from
"our port is wrong". But the method has **no learned parameters** -- it is
arithmetic on tensors -- so the ambiguity is avoidable. Transcribe their code,
re-implement it cleanly, and assert bit-equality on random inputs across every
dtype. No GPU, no LLaVA checkpoint, no vendored transformers required.

Provenance of each transcription
--------------------------------
``reference_intervene``
    ``transformers/src/transformers/models/llama/modeling_llama.py``
    lines 336-392 of the released repo (the LLaVA-1.5 / Shikra code path).
    Note there is a SECOND, divergent copy at
    ``minigpt4/models/modeling_llama.py`` lines 197-243 used by InstructBLIP
    and MiniGPT-4, which drops ``m_sys`` from the score entirely and uses
    layers 4-10 with ratio 0.4. That copy is transcribed separately as
    ``reference_intervene_instructblip`` because the two disagree, and the
    paper does not say which produced which table.

``reference_span_visual_text`` / ``reference_metrics``
    ``Uncertain_head.py`` lines 4-128, whole file.

``reference_contrastive_fuse`` / ``reference_add_diffusion_noise``
    The compared methods, not Fox: ``vcd_sample.py`` lines 176-184 and
    ``vcd_add_noise.py``, whole file, taken from **huofushuo/SID** rather
    than from the Fox fork. SID is the unmodified parent -- the Fox repo is
    built on it, keeps the same two filenames, and replaced the contrastive
    branch with its own additive fusion while assigning ``use_icd = False``
    above the ICD branch. Transcribing from the parent gets the formula the
    compared baselines actually used.

``reference_prompt`` / ``reference_pope_predict``
    The evaluation harness, not the method: ``pope_eval.py`` lines 59-64 and
    232-242, ``char_eval.py`` lines 31-36 and 157,
    ``eval_configs/llava-1.5_eval.yaml``, and
    ``minigpt4/models/llava.py`` lines 228 and 344. They are transcribed here
    for the same reason as the kernel -- the harness decides what the numbers
    mean, and a paraphrase of it is a silent deviation.

``reference_fuse``
    ``vcd_sample.py`` lines 174-223.

Deviations from the released code, all forced and all recorded
--------------------------------------------------------------
1. ``reference_fuse`` takes ``js`` as an argument instead of computing it, so
   the JSD implementation can be tested on its own. ``reference_jsd`` is the
   transcription of their numpy routine.
2. Hardcoded constants that the released code buries as local literals
   (``SELECT_RATIO = 0.35``, ``PENALTY_VALUE = 10000.0``, the layer window
   ``2 <= layer_idx <= 15``) are exposed as keyword arguments *with the
   released values as defaults*. The defaults are the contract; passing
   anything else is no longer "the reference".
3. ``reference_metrics`` returns only the keys the intervention actually
   reads. The released function also computes ``mV_tail``, ``risk_tail``,
   ``mV_last``, ``mSys_last``, ``HVis_last`` and ``risk_last``, none of which
   is ever consumed by any caller.

Released hyperparameters, for the record
----------------------------------------
The code, the paper body, and a leftover author log path in a comment at
``vcd_sample.py:170`` disagree with each other:

=================  ==========  ==============  ====================
value              code        paper (C.2)     author's log comment
=================  ==========  ==============  ====================
layer window       2-15        2-13            2-15 ("2到15")
head ratio k       0.35        0.45            --
tau_JS             0.15        0.2             0.1
gamma              10000.0     never stated    --
alpha              2           2               --
beta               0.1         0.1             --
=================  ==========  ==============  ====================

This file implements the **code** column.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "RELEASED_SELECT_RATIO",
    "RELEASED_PENALTY_VALUE",
    "RELEASED_LAYER_WINDOW",
    "RELEASED_TAU_JS",
    "RELEASED_ALPHA",
    "RELEASED_BETA",
    "reference_span_visual_text",
    "reference_metrics",
    "reference_select_heads",
    "reference_intervene",
    "reference_intervene_instructblip",
    "reference_jsd",
    "reference_contrastive_fuse",
    "reference_add_diffusion_noise",
    "reference_fuse",
]

#: Local literals in the released ``modeling_llama.py`` patch. Not configurable
#: there, not printed in the paper.
RELEASED_SELECT_RATIO = 0.35       # modeling_llama.py:366
RELEASED_PENALTY_VALUE = 10000.0   # modeling_llama.py:367
RELEASED_LAYER_WINDOW = (2, 15)    # modeling_llama.py:337, inclusive
RELEASED_TAU_JS = 0.15             # vcd_sample.py:177
RELEASED_ALPHA = 2.0               # vcd_sample.py:176
RELEASED_BETA = 0.1                # vcd_sample.py:174


# --------------------------------------------------------------------------- #
# Uncertain_head.py
# --------------------------------------------------------------------------- #
def reference_span_visual_text(
    key_position: Dict[str, int], K: int
) -> Tuple[int, int, int, int, int]:
    """Transcription of ``_span_visual_text``.

    Returns ``(v_s, v_e, t_s, t_e, img_s)``: half-open visual key span, text
    key span, and the image start index. System keys are ``[0, img_s)``.

    Note the Shikra branch identifies the model by the magic numbers
    ``img_s == 33 and img_e == 290``; any prompt-length change silently
    selects the wrong visual span. Transcribed as-is.
    """
    img_s = int(key_position.get("image_start", 0))
    img_e = int(key_position.get("image_end", -1))

    is_shikra = (img_s == 33 and img_e == 290)

    if img_e >= 0:
        if is_shikra:
            v_s = max(0, min(img_s + 1, K))
            v_e = max(v_s, min(img_e, K))
        else:
            v_s = max(0, min(img_s, K))
            v_e = max(v_s, min(img_e + 1, K))
    else:
        v_s, v_e = 0, 0

    t_s = max(0, min((img_e + 1) if img_e >= 0 else 0, K))
    t_e = K
    return v_s, v_e, t_s, t_e, img_s


def _reference_entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Transcription of ``_entropy_from_probs``.

    Note it re-normalises internally *and* the caller already normalised, so
    the normalisation happens twice. Kept, because it changes the last bits.
    """
    p = p / (p.sum(dim=-1, keepdim=True) + eps)
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def reference_metrics(
    attn_probs: torch.Tensor,
    key_position: Dict[str, int],
    tail_n: int = 1,
    eps: float = 1e-12,
) -> Optional[Dict[str, torch.Tensor]]:
    """Transcription of ``compute_fusion_stage_metrics_vaf``, reduced to the
    two statistics the intervention reads: ``mSys_tail`` and ``HVis_tail``.

    Args:
        attn_probs: ``[B, H, Q, K]`` POST-softmax probabilities, fp32, with the
            causal mask already applied. The released code builds this from a
            separate fp32 copy of the logits, not from the tensor it mutates.
        key_position: ``{"image_start": int, "image_end": int}``.
        tail_n: released callers pass 1.

    Returns ``None`` when the visual span is empty, matching the released
    early-return (the caller then skips the intervention entirely).
    """
    if attn_probs is None or attn_probs.dim() != 4:
        return None

    B, H, Q, K = attn_probs.shape
    v_s, v_e, t_s, t_e, img_s = reference_span_visual_text(key_position, K)

    if v_e <= v_s:
        return None

    q_tail_s = max(t_s, t_e - tail_n)
    q_tail_e = t_e
    if q_tail_e <= q_tail_s:
        q_tail_s = max(0, Q - 1)
        q_tail_e = Q

    def _agg(qs: int, qe: int):
        qs, qe = int(qs), int(qe)
        qs = max(0, min(qs, Q))
        qe = max(0, min(qe, Q))
        if qe <= qs:
            qs = max(0, Q - 1)
            qe = Q
        if qs >= Q:
            return None

        A = attn_probs[:, :, qs:qe, :]
        mSys = A[:, :, :, 0:img_s].sum(dim=-1).mean(dim=-1)
        pV = A[:, :, :, v_s:v_e]
        mV_q = pV.sum(dim=-1, keepdim=True)
        pV_norm = pV / (mV_q + eps)
        HVis = _reference_entropy_from_probs(pV_norm, eps=eps).mean(dim=-1)
        return mSys, HVis

    res_tail = _agg(q_tail_s, q_tail_e)
    if res_tail is None:
        return None
    mSys_tail, HVis_tail = res_tail
    return {"mSys_tail": mSys_tail, "HVis_tail": HVis_tail}


# --------------------------------------------------------------------------- #
# modeling_llama.py: head selection + the intervention
# --------------------------------------------------------------------------- #
def reference_select_heads(
    score_b: torch.Tensor, select_ratio: float = RELEASED_SELECT_RATIO
) -> torch.Tensor:
    """Transcription of the per-batch-element top-k head selection.

    Note the clamp ``k = max(1, min(k, num_heads // 2))``: for 32 heads and
    ratio 0.35 this gives 11 heads, and any ratio above 0.5 is capped at 16.
    The paper's k = 0.45 would give 14.
    """
    num_heads = score_b.numel()
    k = int(round(num_heads * select_ratio))
    k = max(1, min(k, num_heads // 2))
    _topk_values, topk_indices = torch.topk(score_b, k=k, largest=True)
    mask_heads = torch.zeros(num_heads, dtype=torch.bool, device=score_b.device)
    mask_heads[topk_indices] = True
    return mask_heads


def reference_intervene(
    attn_weights: torch.Tensor,
    key_position: Optional[Dict[str, int]],
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    select_ratio: float = RELEASED_SELECT_RATIO,
    penalty_value: float = RELEASED_PENALTY_VALUE,
    layer_window: Tuple[int, int] = RELEASED_LAYER_WINDOW,
    tail_n: int = 1,
) -> torch.Tensor:
    """Transcription of the released intervention. **Mutates in place.**

    Args:
        attn_weights: ``[B, H, Q, K]`` PRE-softmax logits, already divided by
            ``sqrt(head_dim)``, in the model dtype (fp16 as released). The
            causal mask has NOT yet been added -- the released code adds it
            after this block.
        key_position: ``None`` selects the observational branch (no
            intervention). This is how the released code distinguishes the two
            forward passes: ``vcd_sample.py:136`` passes ``None``, and
            ``vcd_sample.py:165`` passes the real dict.
        attention_mask: additive float mask, used only to build the fp32 copy
            the metrics are computed from.

    Returns the same tensor it was given, mutated.

    The three lines that are the entire mechanism are at the bottom. Observe:
      * key dimension is ``:`` -- ALL key positions, not just system keys.
        A uniform additive shift is exactly cancelled by a stable softmax.
      * query dimension is ``-1:`` -- exactly ONE query, the last of this
        forward pass. Never the two-element set Q = {x_last, y_{t-1}} the
        paper describes; the union over steps covers both only because
        ``use_cache=True`` makes ``-1`` mean x_last at prefill and y_{t-1} at
        each decode step. With POPE's ``max_new_tokens=1``, only x_last is
        ever touched.
      * the fp32 round trip followed by ``.to(dtype_orig)`` is what survives
        the softmax. In fp16 this snaps the row onto an ULP-8.0 grid.
    """
    if key_position is None:
        return attn_weights
    lo, hi = layer_window
    if not (lo <= layer_idx <= hi):
        return attn_weights

    bsz = attn_weights.shape[0]

    attn_logits_base = attn_weights.to(torch.float32)
    if attention_mask is not None:
        attn_logits_base = attn_logits_base + attention_mask.to(torch.float32)
        attn_logits_base = torch.max(
            attn_logits_base,
            torch.tensor(torch.finfo(torch.float32).min, device=attn_logits_base.device),
        )

    with torch.no_grad():
        attn_probs_temp = torch.softmax(attn_logits_base, dim=-1)
        metrics = reference_metrics(
            attn_probs=attn_probs_temp, key_position=key_position, tail_n=tail_n
        )

    if metrics is None:
        return attn_weights

    mSys_all = metrics["mSys_tail"]
    HVis_all = metrics["HVis_tail"]
    score_all = mSys_all * HVis_all

    for b in range(bsz):
        mask_heads = reference_select_heads(score_all[b], select_ratio)
        if int(mask_heads.sum().item()) > 0:
            target_slice = attn_weights[b, mask_heads, -1:, :]
            dtype_orig = attn_weights.dtype
            y = target_slice.to(torch.float32) - penalty_value
            attn_weights[b, mask_heads, -1:, :] = y.to(dtype_orig)

    return attn_weights


def reference_intervene_instructblip(
    attn_weights: torch.Tensor,
    key_position: Optional[Dict[str, int]],
    layer_idx: int,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    select_ratio: float = 0.4,
    penalty_value: float = RELEASED_PENALTY_VALUE,
    layer_window: Tuple[int, int] = (4, 10),
    tail_n: int = 1,
) -> torch.Tensor:
    """The SECOND, divergent copy, from ``minigpt4/models/modeling_llama.py``.

    Used by InstructBLIP and MiniGPT-4. It differs from the LLaVA/Shikra copy
    in three ways that matter, and the paper does not say which copy produced
    which table:

      * the score is ``HVis_tail`` **alone** -- ``m_sys`` is dropped, so Eq. 4
        of the paper is not what runs for InstructBLIP;
      * the layer window is 4-10, not 2-15;
      * the ratio is 0.4, not 0.35.

    It also contains an unguarded ``print()`` inside the attention forward,
    firing 7x per generated token. Not transcribed.
    """
    if key_position is None:
        return attn_weights
    lo, hi = layer_window
    if not (lo <= layer_idx <= hi):
        return attn_weights

    bsz = attn_weights.shape[0]

    attn_logits_base = attn_weights.to(torch.float32)
    if attention_mask is not None:
        attn_logits_base = attn_logits_base + attention_mask.to(torch.float32)
        attn_logits_base = torch.max(
            attn_logits_base,
            torch.tensor(torch.finfo(torch.float32).min, device=attn_logits_base.device),
        )

    with torch.no_grad():
        attn_probs_temp = torch.softmax(attn_logits_base, dim=-1)
        metrics = reference_metrics(
            attn_probs=attn_probs_temp, key_position=key_position, tail_n=tail_n
        )

    if metrics is None:
        return attn_weights

    score_all = metrics["HVis_tail"]  # <-- m_sys dropped. Not Eq. 4.

    for b in range(bsz):
        mask_heads = reference_select_heads(score_all[b], select_ratio)
        if int(mask_heads.sum().item()) > 0:
            target_slice = attn_weights[b, mask_heads, -1:, :]
            dtype_orig = attn_weights.dtype
            y = target_slice.to(torch.float32) - penalty_value
            attn_weights[b, mask_heads, -1:, :] = y.to(dtype_orig)

    return attn_weights


# --------------------------------------------------------------------------- #
# vcd_sample.py: JSD + conflict-gated fusion
# --------------------------------------------------------------------------- #
def reference_jsd(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    temperature: float = 1.0,
    clip_value: float = 5e-8,
) -> float:
    """Transcription of ``cal_JS_div``, which the released code runs in numpy
    on the CPU, for the **full 32064-token vocabulary**, once per generated
    token.

    Two things to note, both confirmed by reading the source:

    1. The paper's Eq. 20 defines the JSD over a *truncated* candidate set
       ``V_t = {y : P_obs(y) >= beta * max P_obs}``, and justifies the
       truncation by saying the full-vocabulary divergence "is dominated by
       the long tail". The released code computes it over the full vocabulary
       anyway. ``beta`` is instead used later, as VCD's plausibility mask on
       the fused logits.
    2. ``.float().cpu().numpy()`` forces a GPU->CPU synchronisation per token.
       Our port does this in torch on-device instead; that is a performance
       deviation with no numerical content, and it is verified bit-equal here.

    This transcription uses torch rather than numpy so it can be compared
    without a numpy dependency; the arithmetic is identical in fp64, which is
    what ``.float()`` -> numpy float32 -> float64 accumulation amounts to for
    a sum of this size. The equivalence test pins it.
    """
    p_logits = p_logits.double()
    q_logits = q_logits.double()

    def _softmax_with_temperature(logits: torch.Tensor) -> torch.Tensor:
        scaled = logits / temperature
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        exps = scaled.exp()
        return exps / exps.sum(dim=-1, keepdim=True)

    p = _softmax_with_temperature(p_logits).clamp(clip_value, 1.0)
    q = _softmax_with_temperature(q_logits).clamp(clip_value, 1.0)
    m = 0.5 * (p + q)
    kl_p_m = (p * (p / m).log()).sum(dim=-1)
    kl_q_m = (q * (q / m).log()).sum(dim=-1)
    return float((0.5 * (kl_p_m + kl_q_m)).item())


def reference_fuse(
    next_token_logits: torch.Tensor,
    next_token_logits_cd: torch.Tensor,
    js: float,
    *,
    alpha: float = RELEASED_ALPHA,
    tau_js: float = RELEASED_TAU_JS,
    beta: float = RELEASED_BETA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Transcription of the conflict-gated fusion, ``vcd_sample.py:208-215``.

    Returns ``(diffs, cd_logits)``:
      * ``diffs`` is the raw fused logits. The released GREEDY path argmaxes
        **this**, unmasked.
      * ``cd_logits`` is ``diffs`` with VCD's plausibility mask applied. The
        released SAMPLING path uses this, after a logits processor and warper.

    That asymmetry is why Table 9 (greedy) is not a clean de-randomisation of
    Table 1 (sampling): the two paths apply different decision rules, not just
    different samplers.

    Note the fusion is additive on UNNORMALISED logits with ``alpha = 2``, so
    ``diffs`` has roughly 3x the scale of ``next_token_logits``. Under
    sampling that sharpens the distribution (an implicit temperature drop);
    under argmax it does not, which is why the greedy control does rule out
    the sharpening explanation even though it introduces this other asymmetry.
    """
    cutoff = torch.log(torch.tensor(beta, dtype=next_token_logits.dtype,
                                    device=next_token_logits.device))
    cutoff = cutoff + next_token_logits.max(dim=-1, keepdim=True).values

    if js < tau_js:
        diffs = next_token_logits + alpha * next_token_logits_cd
    else:
        diffs = next_token_logits + js * next_token_logits_cd

    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
    return diffs, cd_logits


# --------------------------------------------------------------------------- #
# the evaluation harness
# --------------------------------------------------------------------------- #
#: ``eval_configs/llava-1.5_eval.yaml``, the ``model.system_message`` value,
#: transcribed. ``minigpt4/models/llava.py:344`` reads it as
#: ``cfg.get("system_message", "")`` -- the model-level config in
#: ``configs/models/llava-1.5_vicuna7b.yaml`` does not set the key, the eval
#: config does, and the eval config wins the merge.
RELEASED_SYSTEM_MESSAGE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's "
    "questions."
)

#: ``pope_eval.py:59-64`` and ``char_eval.py:31-36``, the ``llava-1.5`` entry.
RELEASED_INSTRUCTION_TEMPLATE = "USER: <ImageHere> <question> ASSISTANT:"

#: ``char_eval.py:157``. POPE substitutes the annotation's question verbatim.
RELEASED_CHAIR_QUESTION = "Please describe this image in detail."

#: ``pope_eval.py:402`` and ``char_eval.py:191``.
#:
#: This is the FOX fork's value and the three forks disagree: SID's
#: ``pope_eval.py:317`` and OPERA's ``pope_eval.py:204`` both pass 10. Which
#: one a compared method should be run at is therefore a real choice, not a
#: detail -- see OPERA_POPE_MAX_NEW_TOKENS.
RELEASED_POPE_MAX_NEW_TOKENS = 1

#: OPERA's own ``pope_eval.py:204``, which SID's fork also carries.
#:
#: It matters for OPERA and not for the others, because OPERA is a beam
#: search: it scores whole sequences, so the first token it emits depends on
#: how many follow. VCD, ICD, SID and CausalMM decide each token on its own,
#: so for them the length only changes how much text the parser has to read.
#:
#: Fox re-ran its compared rows in its OWN harness (docs/DEVIATIONS.md Sec
#: 3.9), so the number in Fox's Table 1 is a 1-token run. That makes 1 the
#: right length for the row that is put against Fox's table, and 10 the right
#: length for the row that is put against OPERA's paper. The plan runs both.
OPERA_POPE_MAX_NEW_TOKENS = 10
RELEASED_CHAIR_MAX_NEW_TOKENS = 512

#: ``pope_eval.py:233``.
RELEASED_NEG_WORDS = ["No", "not", "no", "NO"]


def reference_prompt(question: str,
                     system_message: str = RELEASED_SYSTEM_MESSAGE) -> str:
    """``pope_eval.py:367-368`` then ``llava.py:228``, in that order.

    Two lines of released code::

        qu = [template.replace("<question>", q) for q in qu]      # pope_eval
        instruction = [self.system_message + p for p in instruction]  # llava

    The second is a bare ``+``. There is no separator, and that is not an
    oversight to be corrected here -- it is what produced the paper's numbers,
    so the string it builds runs ``questions.USER:`` together.

    Args:
        question: the POPE question, or :data:`RELEASED_CHAIR_QUESTION`.
        system_message: defaults to the released value; ``""`` is the ablation.

    Returns:
        The prompt with ``<ImageHere>`` still in place of the image token.
    """
    qu = RELEASED_INSTRUCTION_TEMPLATE.replace("<question>", question)
    return system_message + qu


def reference_pope_predict(out):
    """``pope_eval.py:232-242``, ``recorder``, transcribed.

    Note what it is not: there is no unparsable class, and no check that the
    answer is an answer. Any generation without a negation token scores as a
    positive -- ``""``, ``"There"`` and ``"Maybe"`` all become 1. Under the
    released prompt and ``max_new_tokens = 1`` that is a load-bearing choice,
    because the single generated token is frequently neither "Yes" nor "No".

    Args:
        out: iterable of generated strings.

    Returns:
        List of ``1`` (yes) / ``0`` (no), one per string.
    """
    NEG_WORDS = ["No", "not", "no", "NO"]
    pred_list = []
    for line in out:
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    return pred_list


# --------------------------------------------------------------------------- #
# the compared baselines: VCD, ICD, SID
# --------------------------------------------------------------------------- #
# These three are one decoding rule with three ways of building the negative
# branch, which is why they share a file in every repository that implements
# them:
#
#   VCD  perturbs the IMAGE        -- forward diffusion to ``noise_step``
#   ICD  perturbs the INSTRUCTION  -- prepends a negative role description
#   SID  perturbs the VISION TOKENS INSIDE the model -- CT2S keeps only the
#        least important ones after the early decoder layers
#
# The rule itself is a SUBTRACTION. Fox's fusion is ``z_obs + lambda*z_do``,
# an addition with lambda >= 0, so it amplifies agreement; this amplifies
# disagreement. They are not variants of one another.

#: ``vcd_sample.py`` defaults, per repository. The formula is identical; the
#: strength is not, and each project tuned its own.
SID_CD_ALPHA = 0.5      # huofushuo/SID
VCD_CD_ALPHA = 1.0      # the VCD release, and the Fox fork's dead branch
CD_BETA = 0.1           # all three agree

#: ``pope_eval.py:291`` in both SID and the Fox fork, verbatim.
ICD_NEGATIVE_INSTRUCTION = "You are a confused object detector."

#: ``--noise-step`` default in both.
VCD_NOISE_STEP = 500


def reference_contrastive_fuse(
    next_token_logits: torch.Tensor,
    next_token_logits_cd: torch.Tensor,
    cd_alpha: float = SID_CD_ALPHA,
    cd_beta: float = CD_BETA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``vcd_sample.py`` lines 176-184 of huofushuo/SID, transcribed.

    Two things to read off it rather than assume:

    The cutoff uses ``next_token_logits.max()`` alone -- the ORIGINAL
    branch's maximum -- not the minimum of the two branches. The Fox fork's
    dead ICD branch uses ``min(...)`` instead, so the fork had already
    diverged here too. SID's is the one the baselines were published with.

    The mask is applied on ``next_token_logits``, the original branch, while
    the values masked are ``diffs``. So the plausibility constraint asks
    "would the unperturbed model have considered this token at all", and only
    then lets the contrast reorder what survives.

    Returns ``(diffs, cd_logits)``: the released GREEDY path argmaxes
    ``diffs`` unmasked and the SAMPLING path samples ``cd_logits``, the same
    asymmetry Fox has.
    """
    cutoff1 = torch.log(torch.tensor(cd_beta, dtype=next_token_logits.dtype,
                                     device=next_token_logits.device))
    cutoff = cutoff1 + next_token_logits.max(dim=-1, keepdim=True).values

    diffs = (1 + cd_alpha) * next_token_logits - cd_alpha * next_token_logits_cd

    cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
    return diffs, cd_logits


def reference_add_diffusion_noise(
    image_tensor: torch.Tensor, noise_step: int
) -> torch.Tensor:
    """``vcd_add_noise.py``, whole file, transcribed.

    The schedule is **sigmoid**, not the linear DDPM one::

        betas = torch.linspace(-6, 6, 1000)
        betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    and it matters. At ``noise_step = 500`` this keeps ``sqrt(alpha_bar) =
    0.863`` of the image, while a linear ``linspace(1e-4, 0.02)`` schedule
    keeps 0.280 -- nearly white noise. An implementation that reached for the
    familiar DDPM schedule would be contrasting against a different image
    than VCD does, and would not be VCD.

    Note also ``q_x(noisy_image, noise_step)`` indexes the schedule at
    ``noise_step`` directly, not ``noise_step - 1``.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0, t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image, noise_step)

    return image_tensor_cd


# --------------------------------------------------------------------------- #
# SID -- CT2S, the vision-token selection on the negative branch
# --------------------------------------------------------------------------- #
# Transcribed from huofushuo/SID, ``minigpt4/models/modeling_llama.py``, the
# ``elif USE_FAST_V:`` branch of ``LlamaModel.forward`` (the mask
# implementation; ``fast_v_inplace`` defaults to False and its sibling branch
# physically drops tokens, which the file itself marks "KV-Cache not
# supported").
#
# The defaults come from ``pope_eval.py:86-98``:
#
#   --use-cd                 True    the negative branch exists
#   --use-vcd                False   and is NOT a noised image
#   --use-icd                False   and NOT a disturbed instruction
#   --use-fast-v             True    it is CT2S
#   --fast-v-inplace         False   via the attention mask
#   --fast-v-attention-rank  100     keep 100 of 576 vision tokens
#   --fast-v-agg-layer       2       from layer 2 onward
#   --sample-greedy          True
#
# Three things about this are easy to get wrong, and all three change the
# number:
#
# 1. ``largest=False``. The kept tokens are the ones the last query attends
#    to LEAST. The ``largest=True`` line is still there, commented out, one
#    line above (``modeling_llama.py:618-619``) -- that is FastV, which keeps
#    the most-attended tokens to save compute. SID inverts it: the negative
#    branch is made to read the image badly on purpose. Keeping the top-100
#    instead would build a *better* branch and the contrast would subtract
#    the wrong thing.
#
# 2. The ranking is read from layer ``AGG_LAYER - 1``, not ``AGG_LAYER``.
#    ``last_layer_attention = layer_outputs[1]`` is the previous loop
#    iteration's output, i.e. layer 1's attention, and the mask it produces is
#    applied from layer 2 onward.
#
# 3. The negative branch sees the SAME image. ``pope_eval.py:302`` sets
#    ``image_cd = image`` when ``--use-cd`` (the default) -- no noise, no
#    prompt change. CT2S is the entire perturbation. That is what makes SID
#    "self-introspective" and what makes it cheap.
#
# The clean branch runs with ``use_fast_v = False`` and ``key_position=None``
# (``vcd_sample.py:110-120``), so it is the unmodified model; CT2S is switched
# on only for the negative forward (``vcd_sample.py:141-143``).

#: ``--fast-v-agg-layer``. The mask applies from this layer onward and the
#: ranking is read from the layer below it.
SID_AGG_LAYER = 2

#: ``--fast-v-attention-rank``: how many vision tokens survive, out of 576.
SID_ATTENTION_RANK = 100

#: ``--fast-v-attention-rank-add``. Dead code: only the ``vad=True`` branch
#: reads it, and the forward that would pass ``vad=True`` is commented out at
#: ``vcd_sample.py:157-166``. Recorded so the next reader does not have to
#: re-derive that it never runs.
SID_ATTENTION_RANK_ADD = 100


def reference_ct2s_keep_indices(
    last_layer_attention,
    sys_length: int,
    image_token_length: int,
    attention_rank: int = SID_ATTENTION_RANK,
):
    """``modeling_llama.py:652-661``, verbatim down to the variable names.

    ``last_layer_attention`` is ``[B, H, Q, K]`` post-softmax attention from
    layer ``AGG_LAYER - 1``, in the model dtype -- the mean over heads is
    taken in fp16 in the released code, not fp32.

    ``sys_length`` is ``key_position['image_start']`` and
    ``image_token_length`` is ``image_end - image_start + 1``, both built in
    ``llava.py:260-265``.
    """
    # compute average attention over different head
    last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
    # generate new attention mask based on the average attention, sample the
    # top ATTENTION_RANK tokens with highest attention
    last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
    # get the attention in image token
    last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
        sys_length:sys_length + image_token_length
    ]
    # get the indexs of the top ATTENTION_RANK tokens
    top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(
        attention_rank, largest=False
    ).indices + sys_length
    return top_attention_rank_index


def reference_ct2s_attention_mask(
    batch_size: int,
    seq_length_with_past: int,
    sys_length: int,
    image_token_length: int,
    top_attention_rank_index,
    device=None,
):
    """``modeling_llama.py:663-666``, verbatim.

    Note the order: the whole image span is switched off and only then are
    the kept indices switched back on. Every vision key outside the keep-set
    is unattendable for every query, at every layer from ``AGG_LAYER``
    onward -- it is a *key* mask, so nothing is removed, position ids are
    untouched and the KV cache keeps its full length.
    """
    gen_attention_mask = torch.ones(
        (batch_size, seq_length_with_past), dtype=torch.bool, device=device
    )
    gen_attention_mask[:, sys_length:sys_length + image_token_length] = False
    gen_attention_mask[:, top_attention_rank_index] = True
    return gen_attention_mask


# --------------------------------------------------------------------------- #
# CausalMM -- the vision counterfactual
# --------------------------------------------------------------------------- #
# Transcribed from The-Martyr/CausalMM,
# ``llava-1.5/experiments/llava/model/multimodal_encoder/cf_encoder.py`` and
# ``causalmm_cf/causalmm_sm.py``.
#
# Two facts about this release decide what "reproducing CausalMM" means:
#
# 1. The POPE script imports ``causalmm_sm``, the SINGLE-counterfactual
#    sampler, and its rule is the one VCD, ICD and SID already share --
#    ``(1 + gamma) z_obs - gamma z_cf`` with ``log(epsilon)`` as the cutoff.
#    ``gamma`` sits in ``cd_alpha``'s slot and ``epsilon`` in ``cd_beta``'s.
#    So four of the five compared methods are one formula and differ only in
#    how the negative branch is built.
#
# 2. ``causalmm_mm.py``, the two-counterfactual version
#    (``(1 + 2g) z_obs - g z_cf_v - g z_cf_l``), is dead code in this
#    release: it unpacks a 3-tuple from a forward that returns 2, and no
#    shipped model provides a third output. The published POPE numbers are
#    the single-branch version.
#
# Defaults: ``llava1.5_eval.bash`` passes ``gamma=1.0`` and ``epsilon=0.6``,
# overriding the ``0.5`` / ``0.1`` in the sampler. The eval script runs
# ``do_sample=True, temperature=1.0, top_p=1, max_new_tokens=1024, seed=55``
# -- note ``top_p=1``, i.e. no nucleus truncation at all, against 0.9
# everywhere else in this comparison.

CAUSALMM_GAMMA = 1.0          # llava1.5_eval.bash; the sampler default is 0.5
CAUSALMM_EPSILON = 0.6        # ditto; the sampler default is 0.1
CAUSALMM_EDIT_METHOD = "shuffle"   # cf_encoder.edit_attention's own default


def reference_edit_attention(attention_maps, method: str = "shuffle"):
    """``cf_encoder.py:50-73``, verbatim.

    ``attention_maps`` is whatever ``get_attentions`` returned. On LLaVA that
    is CLIP's ``[B, H, 577, 577]`` attention **matrix**, not a spatial map --
    see :func:`reference_get_attentions` for why the reshape that would have
    made it one does not fire.

    Only ``"shuffle"`` is reachable from the shipped code: it is the default
    of this function and the call site passes nothing. The other three are
    the paper's ablation and are kept here because the paper reports them.

    ``torch.randperm`` is called without a generator, so the counterfactual
    is drawn from global RNG -- the one component whose entire purpose is
    randomness is the part of a seeded run that does not repeat per sample.
    Same defect as VCD's noise (:func:`reference_add_diffusion_noise`).
    """
    import random  # noqa: F401  (unused; kept to mirror the file's imports)

    batch_size, num_heads, height, width = attention_maps.shape

    if method == 'random':
        edited_attention_maps = torch.rand(
            batch_size, num_heads, height, width,
            device=attention_maps.device) * 2

    elif method == 'uniform':
        avg_value = torch.mean(attention_maps, dim=(2, 3), keepdim=True)
        edited_attention_maps = avg_value.expand(
            batch_size, num_heads, height, width)

    elif method == 'reversed':
        max_value_height, _ = torch.max(attention_maps, dim=2, keepdim=True)
        max_value, _ = torch.max(max_value_height, dim=3, keepdim=True)

        edited_attention_maps = max_value - attention_maps

    elif method == 'shuffle':
        edited_attention_maps = attention_maps.clone()
        for i in range(num_heads):
            edited_attention_maps[:, i] = edited_attention_maps[:, i].view(
                batch_size, -1).gather(
                    1,
                    torch.randperm(height * width,
                                   device=attention_maps.device).expand(
                                       batch_size, -1)).view(
                                           batch_size, height, width)

    else:
        raise ValueError(
            "Invalid method. Choose from "
            "['random', 'uniform', 'reversed', 'shuffle']")

    return edited_attention_maps


def reference_get_attentions(image_forward_outs):
    """``cf_encoder.py:76-94``, verbatim -- including the branch that never fires.

    The reshape to a square grid is guarded by ``len(attentions[0].shape) == 3``.
    CLIP's attentions are ``[B, H, 577, 577]``, which is 4-D, so the guard is
    false and ``attention_maps`` stays the full attention **matrix**. What
    :func:`reference_apply_attention` then bilinearly resizes to 24x24 is
    therefore a 577x577 matrix, not the spatial attention map the paper's
    figures show. This is transcribed rather than corrected because it is
    what produced the published numbers.
    """
    attentions = image_forward_outs.attentions

    # (batch_size, num_heads, seq_length)
    if len(attentions[0].shape) == 3:
        seq_length = attentions[0].shape[-1]
        grid_size = int(seq_length ** 0.5)  # Assuming it's a square grid
        attentions = [
            attn.view(attn.shape[0], attn.shape[1], grid_size, grid_size)
            for attn in attentions
        ]

    return attentions


def reference_apply_attention(image_features, attention_maps):
    """``cf_encoder.py:96-121``, verbatim.

    Resize to the feature grid, average over heads, broadcast along the
    hidden dimension and multiply. So the counterfactual branch sees the
    image features rescaled per visual token by a shuffled attention value
    -- a per-token gain, not a different image.

    The multiply happens on the **pre-projector** features
    (``llava_arch_cf.py`` projects ``weighted_features`` afterwards). The
    projector is a two-layer GELU MLP, so scaling before it is not the same
    as scaling after.

    The ``torch.float16`` cast at the end is unconditional, which is why this
    is kept verbatim: it pins the dtype of the counterfactual branch's
    features regardless of what the rest of the model is running in.
    """
    import math

    import torch.nn.functional as F

    # Resize
    height_img, width_img = (int(math.sqrt(image_features.shape[1])),
                             int(math.sqrt(image_features.shape[1])))
    attention_maps_resized = F.interpolate(
        attention_maps, size=(height_img, width_img), mode='bilinear',
        align_corners=False)

    # Average over all attention heads to get a single attention map
    averaged_attention_map = torch.mean(attention_maps_resized, dim=1)

    # match the features dimension
    averaged_attention_map = averaged_attention_map.view(
        averaged_attention_map.size(0), -1, 1)
    averaged_attention_map = averaged_attention_map.expand(
        -1, -1, image_features.shape[-1])

    # Apply attention map to image features
    weighted_features = image_features * averaged_attention_map
    weighted_features = weighted_features.to(torch.float16)

    return weighted_features
