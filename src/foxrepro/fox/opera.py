"""OPERA (Huang et al., CVPR 2024), ported onto this project's stack.

Why this is a separate module rather than a flag on ``fox_generate``: OPERA
is not a decoding *rule* layered over one sequence, it is a different search.
Five beams, five speculative forward passes per step to score candidates that
do not exist yet, and -- on long outputs -- a rollback that rewinds the
search. None of that fits a single-sequence loop.

Why a port is possible at all. The released implementation is 559 lines
inside a vendored, patched ``transformers-4.29.2``: it had to rewrite
``GenerationMixin.beam_search`` because that is where the loop lived. This
project already replaced the decode loop with its own, so there is nothing to
patch -- the search is ours to write, and three things that were hard in 4.29
are now library calls:

* the last layer's attention comes from our registered attention function
  (``FoxState.opera_capture``), not from ``output_attentions=True``, which on
  this stack returns an empty tuple under SDPA and would retain all 32 layers;
* ``DynamicCache.crop`` undoes a speculative forward in place;
* the hypothesis bookkeeping is 60 auditable lines (:class:`BeamHypotheses`,
  :class:`BeamScorer`) rather than a dependency. ``transformers``' own
  ``BeamSearchScorer`` is deprecated and removed in v4.62.0, and this project
  has to keep running on one environment across Kaggle sessions, so it is
  reimplemented and asserted equivalent to the library's while that still
  exists (``tests/test_opera.py``).

What the method actually is, which is NOT "beam search plus an attention
bonus on the log-probs". Per step, per beam::

    l            = logits[:, -1, :]                  # RAW, no log_softmax yet
    (l5, t5)     = topk(l, num_attn_candidates)      # top-5 by raw logit
    l5          += penalty_weights * attn_image(t5)  # bonus, in LOGIT space
    vec          = full(REJECT_FLOOR); vec[t5] = l5  # the rest of the vocab goes
    scores       = log_softmax(vec) + beam_scores    # normalised over the 5

Three consequences, each of which changes which sequences are reachable:

1. the bonus is added to logits, where top-5 LLaMA gaps are O(1)-O(5), not to
   log-probs;
2. ``log_softmax`` normalises over each beam's own five candidates, so every
   beam gets a different additive constant -- and beams are then compared in
   one global ``topk``, so a per-beam constant is not harmless;
3. the candidate set is hard-restricted to five per beam. Stock beam search
   picks the best ``2 * num_beams`` over ``num_beams * vocab_size``, and can
   take ten tokens from one beam; OPERA cannot see past each beam's fifth.

``REJECT_FLOOR`` is ``-999.0`` and not ``-inf``, which is load-bearing: after
``log_softmax`` the 31995 discarded tokens sit at a finite ~-1009, so when
``process`` banks an EOS candidate and walks further down the ``2 * num_beams``
list, a discarded token can be promoted into a live beam. With ``-inf`` it
could not be.

``attn_image`` is the candidate's own attention over the image span: the last
layer, max-pooled over heads (not averaged -- that is SID), row-normalised,
summed over ``img_start..img_end``. It cannot come from the main forward,
because the row needed is the row of a token that has not been emitted yet.
Hence five extra one-token forwards per step.

Not implemented here, deliberately: the retrospection-rollback half. On POPE
it is unreachable -- ``max_new_tokens`` there is at most 10, and both
``cur_response_lens <= 10`` (which selects the attention penalty instead of
the rollback score) and the ``> threshold`` repeat count over a history with
one column per step (15 required, 10 available) exclude it, as does the
``rollback_loc < 10`` guard in the body. So on POPE this module is OPERA, not
an approximation of it. See ``docs/DEVIATIONS.md`` Sec 3.13 for the rollback
audit and why reproducing it needs a decision about fp16 saturation first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from foxrepro.fox.attention import FOX, locate_image_span

__all__ = [
    "OPERA_DEFAULTS",
    "REJECT_FLOOR",
    "BeamHypotheses",
    "BeamScorer",
    "opera_generate",
    "rollback_is_reachable",
]

#: ``pope_eval.py``'s argparse block, which is also what ``chair_eval.py``
#: and the README's command line pass.
OPERA_DEFAULTS: Dict[str, Any] = {
    "num_beams": 5,
    "num_attn_candidates": 5,
    "penalty_weights": 1.0,
    "scale_factor": 50.0,   # rollback path only -- dead whenever rollback is
    "threshold": 15,        # unreachable, which on POPE it always is
}

#: ``next_token_logits.fill_(-999.)``. Finite on purpose; see the module
#: docstring.
REJECT_FLOOR = -999.0


def rollback_is_reachable(max_new_tokens: int, threshold: int = 15) -> bool:
    """Can OPERA's rollback fire at all for this generation length?

    Two independent gates, both from ``opera_beam_search``:

    * the rollback score is only *used* when ``cur_response_lens > 10``, and
      that counter is the number of response tokens emitted so far;
    * the trigger needs the modal column to repeat more than ``threshold``
      times in a history with exactly one column per step.

    So anything up to ``max(10, threshold)`` response tokens cannot roll back.
    Exposed as a function rather than a comment because it is the thing that
    makes this module's POPE output *equal* to OPERA rather than close to it,
    and a future change to ``max_new_tokens`` must trip over it.
    """
    return max_new_tokens > max(10, threshold)


class BeamHypotheses:
    """One batch item's finished hypotheses, ranked by length-penalised score.

    A transcription of ``transformers.generation.beam_search.BeamHypotheses``
    rather than an improvement on it: this project needs the library's
    behaviour, and the library's class is deprecated. Asserted equivalent in
    ``tests/test_opera.py`` for as long as the original is importable.
    """

    def __init__(self, num_beams: int, length_penalty: float,
                 early_stopping: bool) -> None:
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.beams: List[Any] = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float,
            generated_len: int) -> None:
        """Length-penalise and insert, evicting the worst past ``num_beams``."""
        score = sum_logprobs / (generated_len ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                worst_i = min(enumerate(self.beams),
                              key=lambda x: x[1][0])[0]
                del self.beams[worst_i]
                self.worst_score = min(s for s, _ in self.beams)
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """Can no remaining candidate beat the worst kept hypothesis?

        ``early_stopping=False`` -- the released setting, since OPERA passes
        ``generation_config.early_stopping`` and never sets it -- compares the
        worst kept score against the best score still attainable at this
        length.
        """
        if len(self) < self.num_beams:
            return False
        if self.early_stopping:
            return True
        highest_attainable = best_sum_logprobs / (
            cur_len ** self.length_penalty)
        return bool(self.worst_score >= highest_attainable)


class BeamScorer:
    """``process`` / ``finalize`` for one batch item.

    Batch size is fixed at 1 because both released eval scripts run
    ``--batch_size 1`` (``pope_eval.py:68``, ``chair_eval.py:80``) and because
    OPERA's rollback trigger is global across the batch dimension -- batching
    it would not be the same method, so pretending to support it would be
    worse than refusing.
    """

    def __init__(self, num_beams: int, *, length_penalty: float = 1.0,
                 early_stopping: bool = False, max_length: Optional[int] = None):
        self.num_beams = num_beams
        self.max_length = max_length
        self.hyps = BeamHypotheses(num_beams, length_penalty, early_stopping)
        self.done = False

    @property
    def is_done(self) -> bool:
        return self.done

    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_id: List[int],
        decoder_prompt_len: int,
    ) -> Dict[str, torch.Tensor]:
        """Fill the next ``num_beams`` live beams, banking any EOS candidate.

        The ``rank >= num_beams`` skip is the library's and is why
        ``REJECT_FLOOR`` being finite matters: an EOS in the top
        ``num_beams`` is banked and the walk continues down the
        ``2 * num_beams`` list, so a discarded-vocabulary token sitting at a
        finite ~-1009 can be promoted into a live beam.
        """
        cur_len = int(input_ids.shape[-1])
        device = input_ids.device
        n = self.num_beams
        scores = torch.zeros(1, n, dtype=next_scores.dtype, device=device)
        tokens = torch.zeros(1, n, dtype=torch.long, device=device)
        idx = torch.zeros(1, n, dtype=torch.long, device=device)

        if self.done:
            scores[0, :] = 0.0
            tokens[0, :] = pad_token_id
            return {"next_beam_scores": scores.view(-1),
                    "next_beam_tokens": tokens.view(-1),
                    "next_beam_indices": idx.view(-1)}

        filled = 0
        for rank in range(int(next_tokens.shape[-1])):
            token = int(next_tokens[0, rank])
            score = next_scores[0, rank]
            beam = int(next_indices[0, rank])
            if token in eos_token_id:
                if rank >= n:
                    continue
                self.hyps.add(
                    input_ids[beam].clone(), float(score),
                    generated_len=cur_len - decoder_prompt_len)
            else:
                scores[0, filled] = score
                tokens[0, filled] = token
                idx[0, filled] = beam
                filled += 1
            if filled == n:
                break
        if filled < n:
            raise ValueError(
                f"only {filled} of {n} beams could be filled from "
                f"{int(next_tokens.shape[-1])} candidates; at most "
                f"{n} of them may be eos"
            )
        self.done = self.hyps.is_done(
            float(next_scores[0].max()), cur_len - decoder_prompt_len)
        return {"next_beam_scores": scores.view(-1),
                "next_beam_tokens": tokens.view(-1),
                "next_beam_indices": idx.view(-1)}

    def finalize(
        self,
        input_ids: torch.Tensor,
        final_beam_scores: torch.Tensor,
        *,
        pad_token_id: int,
        eos_token_id: List[int],
        decoder_prompt_len: int,
    ) -> torch.Tensor:
        """Bank the unfinished beams, then return the single best sequence."""
        if not self.done:
            cur_len = int(input_ids.shape[-1])
            for beam in range(self.num_beams):
                self.hyps.add(
                    input_ids[beam].clone(), float(final_beam_scores[beam]),
                    generated_len=cur_len - decoder_prompt_len)
        best = max(self.hyps.beams, key=lambda x: x[0])[1]
        return best


def _resolve_eos(model: Any) -> List[int]:
    cfg = getattr(model.config, "text_config", model.config)
    eos = getattr(cfg, "eos_token_id", None)
    if eos is None:
        raise ValueError("the checkpoint declares no eos_token_id")
    return [int(eos)] if isinstance(eos, int) else [int(e) for e in eos]


def opera_generate(
    model: Any,
    inputs: Dict[str, torch.Tensor],
    *,
    max_new_tokens: int = 10,
    num_beams: int = 5,
    num_attn_candidates: int = 5,
    penalty_weights: float = 1.0,
    length_penalty: float = 1.0,
    early_stopping: bool = False,
    collect_trace: bool = False,
    allow_unreachable_rollback: bool = False,
    restrict_candidates: bool = True,
) -> Dict[str, Any]:
    """Beam search with OPERA's over-trust penalty. No rollback.

    Args:
        max_new_tokens: refused above the length at which rollback would
            become reachable, unless ``allow_unreachable_rollback``. Running
            longer than that without rollback is *not* OPERA, and producing
            a number labelled OPERA from it would be wrong in a way nobody
            downstream could see.
        num_attn_candidates: the hard per-beam candidate cap. Not a tuning
            knob -- see the module docstring.
        restrict_candidates: ``False`` keeps the penalty but drops the
            discard-the-vocabulary step, so ``log_softmax`` runs over the
            whole vocabulary as stock beam search does. A validation arm, not
            a variant: OPERA is three things at once -- a beam search, a hard
            five-candidate restriction, and an attention bonus -- and with
            ``penalty_weights=0`` this separates all three.

    Returns ``{"token_ids", "n_prompt_tokens", "n_gen_tokens", "image_span",
    "trace"?}``, the same shape ``fox_generate`` returns, so the benchmark
    harness does not have to know which search ran.
    """
    if num_beams < 2:
        # opera_greedy_search in the release is `assert False, "OPERA does not
        # support beam=1 in the current version."` -- the greedy variant was
        # never written, so there is nothing to reproduce at one beam.
        raise ValueError(
            f"OPERA needs num_beams >= 2 ({num_beams} given). The release's "
            "greedy path is an unconditional assert, so a one-beam OPERA has "
            "no reference behaviour"
        )
    if num_attn_candidates < 1:
        raise ValueError(
            f"Num of candidates must be larger than 0, but it is currently "
            f"{num_attn_candidates}."
        )
    if rollback_is_reachable(max_new_tokens) and not allow_unreachable_rollback:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} is long enough for OPERA's "
            "rollback to fire, and this module does not implement it. The "
            "result would be OPERA with half its method missing, under "
            "OPERA's name. Use <= 15 tokens (POPE), or pass "
            "allow_unreachable_rollback=True and label the arm accordingly"
        )

    device = next(model.parameters()).device
    image_token_id = model.config.image_token_index
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    img_start, img_end = locate_image_span(input_ids, image_token_id)
    n_prompt = int(input_ids.shape[1])

    eos_ids = _resolve_eos(model)
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_ids[0]

    n_layers = int(getattr(
        getattr(model.config, "text_config", model.config),
        "num_hidden_layers",
    ))

    ids = input_ids.repeat(num_beams, 1)
    mask = attn_mask.repeat(num_beams, 1)

    scorer = BeamScorer(
        num_beams, length_penalty=length_penalty,
        early_stopping=early_stopping,
        max_length=n_prompt + max_new_tokens,
    )
    # Only beam 0 is live at step 0: every beam row holds the same prompt, so
    # without this the first step would return the same token five times.
    beam_scores = torch.full((num_beams,), -1e9, dtype=torch.float32,
                             device=device)
    beam_scores[0] = 0.0

    cache = None
    trace: List[Dict[str, float]] = []
    top_scores = next_tokens = next_indices = None

    with FOX.configured(
        enabled=False, mode="off",
        img_start=img_start, img_end=img_end,
        # Off by default and switched on only around the speculative
        # forwards: the main forward's last row is never used, so capturing
        # it would be a wasted QK^T and would leave FOX.opera_attn holding a
        # row that belongs to no candidate.
        opera_capture=False, opera_layer=n_layers - 1, opera_attn=None,
    ):
        for step in range(max_new_tokens):
            if step == 0:
                # ONE row, then fan the cache out to num_beams. At step 0
                # every beam holds the same prompt, so this is numerically
                # identical -- and it is not an optimisation, it is what
                # makes the arm run at all. A 5-row 625-token prefill OOMs
                # on a T4: device_map="auto" packs the first GPU to ~14.0 of
                # its 14.56 GiB with weights, and five rows of activations
                # do not fit in what is left. One row also costs a fifth of
                # the prefill compute, which on POPE is most of the job.
                out = model(input_ids=input_ids, attention_mask=attn_mask,
                            pixel_values=pixel_values, use_cache=True)
                cache = out.past_key_values
                # index_select on the batch dim with an all-zeros index:
                # five copies of the one prefilled row. reorder_cache moves
                # the index to each layer's own device, so this is correct
                # on a sharded model.
                cache.reorder_cache(
                    torch.zeros(num_beams, dtype=torch.long))
                z = out.logits[:, -1, :].repeat(num_beams, 1)
            else:
                out = model(input_ids=ids[:, -1:], attention_mask=mask,
                            past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                # RAW logits, in the model dtype: the penalty is added here,
                # not after log_softmax, and the dtype is the released one.
                z = out.logits[:, -1, :]
            cache_len = int(cache.get_seq_length())
            cand_scores, cand_tokens = torch.topk(
                z, num_attn_candidates, dim=-1, largest=True, sorted=True)

            attn_cols: List[torch.Tensor] = []
            spec_mask = torch.cat(
                [mask, mask.new_ones(num_beams, 1)], dim=1)
            for c in range(num_attn_candidates):
                FOX.opera_attn = None
                FOX.opera_capture = True
                try:
                    model(
                        input_ids=cand_tokens[:, c:c + 1],
                        attention_mask=spec_mask,
                        past_key_values=cache, use_cache=True,
                    )
                finally:
                    FOX.opera_capture = False
                row = FOX.opera_attn
                if row is None:
                    raise RuntimeError(
                        f"no attention captured at layer {n_layers - 1}. The "
                        "model must be loaded with attn_implementation='fox' "
                        "for OPERA's penalty to exist at all"
                    )
                # Undo the speculative step. On transformers 4.29 the KV cache
                # was an immutable tuple and torch.cat built new tensors, so
                # the release could simply discard it; DynamicCache.update
                # mutates in place, so without this crop the five candidate
                # passes would corrupt the search's own cache.
                cache.crop(cache_len)
                # Row-normalise AFTER the max over heads, as the release does
                # (attn_last / attn_last.sum(-1)); a max over heads does not
                # sum to 1, so the order is observable.
                row = row / row.sum(dim=-1, keepdim=True)
                attn_cols.append(row[:, img_start:img_end + 1].sum(-1))

            # The `cur_response_lens <= 10` regime: penalty_scores is
            # -attn_scores, and `scores -= w * penalty` therefore ADDS it.
            # The other regime is the rollback score, which this module
            # refuses to reach.
            attn_scores = torch.stack(attn_cols, dim=1)
            # Recorded BEFORE the penalty lands, because the question these
            # two numbers answer is whether the penalty can change anything:
            # it is added in LOGIT space, so what matters is its spread across
            # the five candidates against the spread of their raw logits. If
            # the first is orders of magnitude smaller than the second, then
            # OPERA's over-trust penalty cannot reorder the candidates and the
            # arm is beam search with a truncated candidate set. That is a
            # measurement, not an assumption -- the trace carries both.
            logit_spread = float(
                (cand_scores[:, 0] - cand_scores[:, -1]).float().mean())
            penalty_spread = float(
                (attn_scores.max(dim=1).values
                 - attn_scores.min(dim=1).values).float().mean()
            ) * penalty_weights
            cand_scores = cand_scores + (
                penalty_weights * attn_scores).to(cand_scores.dtype)
            reordered = bool(
                (cand_scores.argmax(dim=-1) != 0).any())

            # Discard the rest of the vocabulary and renormalise over the
            # survivors. REJECT_FLOOR is finite on purpose.
            if restrict_candidates:
                vec = torch.full_like(z, REJECT_FLOOR)
                vec.scatter_(-1, cand_tokens, cand_scores)
            else:
                vec = z.clone()
                vec.scatter_(-1, cand_tokens, cand_scores)
            scores = F.log_softmax(vec, dim=-1)
            scores = scores + beam_scores[:, None].expand_as(scores)

            vocab = int(scores.shape[-1])
            flat = scores.reshape(1, num_beams * vocab)
            top_scores, top_flat = torch.topk(
                flat, 2 * num_beams, dim=1, largest=True, sorted=True)
            next_indices = torch.div(top_flat, vocab, rounding_mode="floor")
            next_tokens = top_flat % vocab

            beam_out = scorer.process(
                ids, top_scores, next_tokens, next_indices,
                pad_token_id=pad_id, eos_token_id=eos_ids,
                decoder_prompt_len=n_prompt,
            )

            beam_scores = beam_out["next_beam_scores"]
            beam_tokens = beam_out["next_beam_tokens"]
            beam_idx = beam_out["next_beam_indices"]

            ids = torch.cat(
                [ids[beam_idx, :], beam_tokens.unsqueeze(-1)], dim=-1)
            mask = torch.cat(
                [mask[beam_idx], mask.new_ones(num_beams, 1)], dim=1)
            cache.reorder_cache(beam_idx)

            if collect_trace:
                trace.append({
                    "step": step,
                    "attn_image_mean": float(attn_scores.mean()),
                    "penalty_spread": penalty_spread,
                    "logit_spread": logit_spread,
                    # True when the penalty moved the top candidate off the
                    # raw-logit argmax on at least one beam -- i.e. the step
                    # where OPERA's mechanism actually did something.
                    "penalty_changed_top": reordered,
                    "best_beam_score": float(beam_scores.max()),
                })

            if scorer.is_done:
                break

    best = scorer.finalize(
        ids, beam_scores, pad_token_id=pad_id, eos_token_id=eos_ids,
        decoder_prompt_len=n_prompt,
    )
    out_ids = [int(t) for t in best[n_prompt:].tolist()]

    result: Dict[str, Any] = {
        "token_ids": out_ids,
        "n_prompt_tokens": n_prompt,
        "n_gen_tokens": len(out_ids),
        "image_span": (img_start, img_end),
        "n_forwards_per_step": 1 + num_attn_candidates,
        "restrict_candidates": restrict_candidates,
        "penalty_weights": penalty_weights,
    }
    if collect_trace:
        result["trace"] = trace
    return result
