# How this implementation differs from the released Fox code

Every difference between this repository and the code published with
*"Dismantling Pathological Shortcuts: A Causal Framework for Faithful LVLM
Decoding"* (ICML 2026), classified by whether it can change a number.

Written for the reproduction report: when a result here disagrees with the
paper, this is the list of candidate causes, ordered so the ones that cannot be
the cause are ruled out first.

Reference: `github.com/Cc2021start/Fox` at the state audited 2026-10-12
(3 commits; initial commit `94d671f`, authored "anonymous").

---

## 1. Proven bit-identical

Asserted by `tests/test_kernel_equivalence.py` against
`foxrepro/fox/reference.py`, a verbatim transcription of the released code
lifted out of its vendored-transformers context. Random tensors, fp32 / fp16 /
bf16, prefill (Q=600) and decode (Q=1), batched and not. No GPU or checkpoint
needed, because the method has no learned parameters.

| Component | Released location | Ours |
|---|---|---|
| The intervention | `transformers/.../modeling_llama.py:389-392` | `kernel.intervene(mode="as_implemented")` |
| `m_sys`, `H_vis` | `Uncertain_head.py:compute_fusion_stage_metrics_vaf` | `kernel.head_scores` |
| Top-k head selection | `modeling_llama.py:371-380` | `kernel.select_heads` |
| JSD | `vcd_sample.py:189-206` | `kernel.jsd` (12 decimal places) |
| Conflict-gated fusion | `vcd_sample.py:208-215` | `kernel.fuse` |

Two released defects are **preserved deliberately**, because removing them
would stop this being a reproduction:

- the double normalisation inside the entropy helper (the caller normalises,
  then `_entropy_from_probs` normalises again) — it changes the last bits;
- the greedy/sampling asymmetry: the released greedy path argmaxes the
  **unmasked** fused logits while its sampling path uses the β-masked ones. We
  reproduce that by default and expose `mask_greedy=True` to run the clean
  de-randomisation the paper should have used. Report both.

## 2. Different code, verified same result

| Change | Why | Verification |
|---|---|---|
| JSD computed on-device in torch | The release calls `.float().cpu().numpy()` on a 32064-wide vector **once per generated token**, forcing a GPU→CPU sync. 512 syncs per CHAIR caption. | bit-equal to 12 dp |
| Head selection vectorised (`scatter_`) | The release loops over the batch in Python | bit-equal per row |
| Top-p via HF's `TemperatureLogitsWarper` + `TopPLogitsWarper` | The release applies `logits_warper` to the fused logits. A hand-rolled version kept one token fewer than HF in ~5% of rows: HF sorts ascending and drops the tail with cumulative mass `<= 1 - top_p`, the hand-rolled one sorted descending and kept while exclusive cumulative mass was `<= top_p` — different strictness, and opposite accumulation order. | 300/300 identical tokens |

## 3. Deviations that can change a number

Ordered by expected size of effect.

### 3.1 Reduced-precision accumulation is pinned off — **this one will show**

`pin_precision_flags()` sets `allow_fp16_reduced_precision_reduction = False`
before the model loads. The release leaves it at its default, **True**.

That flag governs accumulation *inside* the fp16 `Q @ Kᵀ` matmul — the exact
tensor Fox manipulates and this project studies. Leaving it on stacks a second,
uncontrolled precision effect on the one under test, so pinning it is the right
call scientifically, but it means our attention logits are not bit-identical to
the release's even before the intervention runs.

Measured on the host: the release would run with
`fp16_reduced=True, bf16_reduced=True, cudnn_tf32=True`. On a T4 (sm_75) the
TF32 flags are inert — TF32 is Ampere+ — so only the fp16 one bites there.

### 3.2 transformers 5.16.1, not the vendored 4.29.2

The release ships a patched HuggingFace 4.29.2 (3370 files) with the
intervention welded into `modeling_llama.py`. We register a custom attention
function through `AttentionInterface` on the host's own transformers.

The kernel is bit-equal, but everything around it — RoPE application, layer
norms, the matmul kernels chosen, the KV-cache implementation — is a different
release. Any of those can move the last bits.

`fox/reference.py` plus Track A (`requirements-trackA.lock`) exist to settle
this by cross-checking token-for-token on a few dozen samples. **Not yet run.**

### 3.3 `llava-hf/llava-1.5-7b-hf`, not `liuhaotian/llava-v1.5-7b`

Weight-identical, but the original stores the language model under
`model.layers.*` where transformers expects `model.language_model.layers.*` and
ships no `preprocessor_config.json`. Loading it as a
`LlavaForConditionalGeneration` reports the vision tower and projector as
MISSING and **initialises them randomly** without erroring — so
`load_llava` refuses that repo by name, and refuses any load that leaves
parameters uninitialised.

The converted repo expands `<image>` into 576 ids in `input_ids` via the
processor; the pre-4.44 path spliced embeddings inside the model instead. Image
span measured on the host: `[5, 580]` with no system prompt, width exactly 576.

### 3.4 Two branches batched, not run sequentially

The release runs the observational and interventional passes as two separate
`self(...)` calls, each with its own `past_key_values`. We run them as rows 0
and 1 of one forward pass, which reads the 13.16 GiB of weights once instead of
twice — roughly 2× faster decode, since decode is bandwidth-bound.

Mathematically identical: both rows process the same token sequence, and
`FOX.branch_rows` restricts the write to row 1. **Not yet verified bit-equal on
GPU**, where batched and unbatched matmuls can use different kernels.

Consequence: the release computes the JSD from batch row 0 and applies that
scalar to every row, which is harmless at batch 1 but silently wrong when
batched. Ours is per-row.

### 3.5 The system prompt

The release builds prompts through the original LLaVA repo's
`conv_templates["vicuna_v1"]`, which prepends a ~35-token system string. The
HuggingFace chat template injects none — verified by reading
`chat_template.jinja`, which emits `role.upper() + ': '` and nothing else.

We send a **transcription** of `conv_vicuna_v1`'s system prompt. Byte-identity
with what the release emits at runtime is **unverified**; the id is recorded in
every record so a correction stays traceable.

This matters more than formatting: `m_sys` is the attention mass on the keys
*before* the image, which the paper calls the anchor of the shortcut it claims
to sever. Omitting the system prompt leaves only `<s>`, `USER`, `:` there.
`--system-prompt none` is therefore a direct test of the paper's premise, not a
convenience.

### 3.6 Hyperparameter defaults come from the code, not the paper

They disagree. We default to the code, and record which source was used.

| | code (ours) | paper §C.2 | author's leftover log path |
|---|---|---|---|
| layer window | **2–15** | 2–13 | 2–15 (`2到15`) |
| head ratio `k` | **0.35** | 0.45 | — |
| `τ_JS` | **0.15** | 0.2 | 0.1 |
| `γ` | **10000.0** | never stated | — |
| `α` | 2 | 2 | — |
| `β` | 0.1 | 0.1 | — |

`γ` is a hardcoded local literal, absent from the paper's own list of "four key
hyperparameters" despite Algorithm 1 taking it as an input.

### 3.7 Two released defects we do fix

- `pope_eval.py:394` wraps generation in `torch.enable_grad()`. The cache then
  retains `grad_fn`, so memory grows linearly in generated tokens — survivable
  for POPE's 1 token, an OOM for CHAIR's 512 on any GPU in our range. We use
  `inference_mode()`.
- `--use-cd` is `store_true` with `default=True`, so it can never be turned
  off and the `--use-vcd` branch is unreachable. Our modes are explicit flags.

## 4. In the release, absent here

| | Why |
|---|---|
| InstructBLIP path | It is a **second, divergent copy** of the intervention (`minigpt4/models/modeling_llama.py`) that drops `m_sys` from Eq. 4 entirely and uses layers 4–10 with k=0.4. A different method, not a backbone swap. |
| Shikra path | No maintained HF conversion, and the release identifies its visual span by the magic numbers `img_s == 33 and img_e == 290`. |
| FastV / OPERA plumbing | Present in the release as argparse flags, all `default=False`, with no reweighting behind the SID one. |
| HF `logits_processor` | We apply the warpers (temperature, top-p) but not the processor list, which is empty by default for this configuration. |
| `use_phj` | Dead code: set on every layer, read nowhere. |
| A `print()` in the attention forward | Fires 7× per generated token in the InstructBLIP copy. |

## 5. Here, absent in the release

| | |
|---|---|
| A CHAIR evaluator | **The release has none.** `char_eval.py` only dumps captions; there is no `chair.py` and no synonym list. Table 2 is not reproducible from the release at all. |
| A frozen CHAIR image list | The release globs `--data-path` for `*.jpg` and takes the first `--test-sample 500`, so which 500 images CHAIR scores depends on what happens to be in the directory. We ship an explicit list of 500 val2014 ids so the set is fixed and recorded. This is a deviation in *selection*, not in protocol: the paper does not state which 500 it used. |
| Six intervention arms as flags | `off`, `as_described`, `ulp_control`, `noise_control` — controls the paper never ran — plus `boost_visual` (below) |
| **`boost_visual`** | The experiment the paper's motivation rests on and never performs. It argues from Figure 1(a), an *observational* comparison showing faithful and hallucinated samples carry the same global visual attention mass, that *amplifying* visual attention cannot help. That is the `P(Y\|X)` vs `P(Y\|do(X))` distinction the paper itself takes as its foundation: a quantity uncorrelated with an outcome can still change it when intervened upon. PAI, named in the introduction as the representative of that family, appears **twice in the whole paper** and is in **none of its tables**. This mode adds a positive bias to the image-token keys at the same heads and query positions Fox suppresses, leaving the system keys untouched — the same slice as `as_described`, opposite sign. |
| Three independent dtype axes | weights / pre-softmax logit storage / softmax accumulation. The mechanism lives in the second |
| Resume | Keyed on the run file; tested by actually crashing the process |
| 33 provenance fields per record | including `transformers_path`, `tf32_flags`, `torch_version`, `compute_capability` |
| Wilson / Newcombe intervals, paired bootstrap, McNemar | the release reports no variance at all |
| One shared sampler for baseline and method | so the only difference between them is the intervention |
| A session-resumable 53-job plan | the release has no MME script either, despite reporting MME |

## 6. Unverified, and what would settle it

| Open question | How to close it |
|---|---|
| Does transformers 5.16.1 give the same numbers as the vendored 4.29.2? | Track A on ~50 samples, greedy, fixed seed; diff token ids |
| Is batching the two branches bit-equal to two sequential passes? | Same prompt both ways on GPU, compare logits |
| Is our system prompt byte-identical to the release's? | Print `conv_templates["vicuna_v1"].get_prompt()` in Track A |
| Is `chair_synonyms.txt` identical to Rohrbach's upstream? | Diff against the upstream file; our SHA-256 is in every metrics file. **Do this before quoting any CHAIR number externally.** |
| Does `--system-prompt none` still show Fox's gain? | It is a planned job; a yes undermines the paper's premise |
