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
span measured on the host: `[5, 580]` with `--system-prompt none`, width
exactly 576; the released default pushes it to `[~40, ~616]`.

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

### 3.5 The prompt

Resolved on 2026-10-18 by reading the released scripts rather than inferring
from the paper. The release does **not** build prompts through
`conv_templates`; `pope_eval.py` and `char_eval.py` never import it. The
assembly is, in order:

1. `pope_eval.py:59-64` / `char_eval.py:31-36` —
   `INSTRUCTION_TEMPLATE["llava-1.5"] = "USER: <ImageHere> <question> ASSISTANT:"`.
   A **space** before the question, not a newline, and no added instruction:
   the POPE question goes in verbatim (`pope_v0_raw`) and CHAIR asks
   `"Please describe this image in detail."` (`char_eval.py:157`, our
   `chair_v1`).
2. `eval_configs/llava-1.5_eval.yaml` sets, under `model:`, a
   `system_message` equal to `conv_vicuna_v1`'s system string.
3. `minigpt4/models/llava.py:344` reads it as `cfg.get("system_message", "")`
   and `llava.py:228` does `instruction = [self.system_message + p for p in
   instruction]`.

Step 3 is a bare `+`. There is **no separator**, so the released prompt reads

```
...polite answers to the user's questions.USER: <image> Is there a dog in the image? ASSISTANT:
```

with the period running straight into `USER`. `--system-prompt llava_v1`
(the default) reproduces that byte for byte; the string is asserted in
`TestSystemPrompt.test_default_reproduces_the_released_concatenation_verbatim`
so a later cleanup cannot quietly insert the space.

Three arms are kept, each isolating one difference from this default:
`--system-prompt llava_v1_spaced` (adds the space), `--system-prompt none`
(drops the system message), `--wrapper newline` (HF's separator),
`--prompt-template-id pope_v1` (appends "Answer the question using a single
word or phrase").

Those are not cosmetic. `m_sys` is the attention mass on the keys *before* the
image, which the paper calls the anchor of the shortcut it claims to sever.
Under `none` those keys are only `<s>`, `USER`, `:` — measured image span
`[5, 580]` instead of `[~40, ~616]` — so `--system-prompt none` is a direct
test of the paper's premise, not a convenience. And `pope_v1` is the arm most
likely to explain session 1's inflated POPE baseline (81.90 against the
paper's 76.77): it keeps answers to one parseable token.

**Correction.** Session 1 ran with all four differences present at once —
system prompt joined with a space, `\n` separator, and the "single word or
phrase" suffix. An intermediate reading of `pope_eval.py` alone concluded the
release sends no system prompt at all; `eval_configs/` shows otherwise. The
session-1 run files are kept, under their own variant hashes, as the
measurement of that combination.

### 3.5.1 Generation settings, verified against the release

| | release | ours | matters? |
|---|---|---|---|
| POPE `max_new_tokens` | `1` (`pope_eval.py:402`) | `1` | — |
| CHAIR `max_new_tokens` | `512` (`char_eval.py:191`) | `512` | — |
| `num_beams` | `1` (`--beam` default) | `1` | — |
| sampling | `do_sample=True`, `top_p=0.9`, `temperature=1` (`llava.py:150,154,272`) | same | — |
| `use_cache` | `pope_eval.py` passes `False`, but `llava.py:272` **hardcodes `True`** in the inner `generate`, so the argument is dead | `True` | no |
| POPE scoring | see below | `--pope-parser released` | **yes** |

The released POPE scorer (`pope_eval.py:232-242`) has no unparsable class:

```python
NEG_WORDS = ["No", "not", "no", "NO"]
if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
    pred = 0
else:
    pred = 1
```

Anything without a negation token — including `""`, `"There"`, `"Maybe"` — is
scored as **yes**. Our strict parser calls those unparsable and
`--unparsed-policy negative` then scores them as **no**, the opposite. With
the "single word or phrase" suffix the two agree on every sample (measured:
3000/3000 on both session-1 POPE runs, so the committed numbers are unchanged
by this). Without it, `max_new_tokens=1` produces first tokens that are
neither `Yes` nor `No` and the rules diverge. `released` is therefore the
default, and `metrics["pope_parser"]` records which rule produced any number.

### 3.8 CHAIR's 500 images are not recoverable, and that is measurable

This is the one difference that cannot be closed by being more careful, so it
is worth stating exactly.

The release picks its CHAIR images like this (`char_eval.py:136-137`):

```python
img_files = os.listdir(args.data_path)      # default: /data-disk/only_test
random.shuffle(img_files)
...                                          # keeps the first --test-sample 500
```

Three independent problems, any one of which would be enough:

1. `/data-disk/only_test` is not in the repository and is not described
   anywhere. It is not full `val2014` — the name suggests a pre-filtered
   subset — so even its population is unknown.
2. `os.listdir` returns entries in filesystem order, which varies by
   filesystem and by the order files were written. The seeded `random.shuffle`
   that follows is therefore shuffling a machine-dependent list.
3. The paper never names the set either. It says "500 COCO samples" in five
   places (§5, Tables 2/6/9, C.3) and never which 500, never a selection
   rule, and never "following [prior work]" — which some CHAIR papers do, and
   which would have made the set recoverable.

So our CHAIR numbers are computed on a *different* 500 images from the
paper's, and no amount of care closes that. It also explains the shape of the
reproduction: POPE uses POPE's fixed official annotation files and our
baselines land within 0.05 points of the paper's, while the CHAIR baseline
sits 3.88 points away.

What can be done is to **measure how much it matters**. `CHAIR_DRAWS` adds
four more independent 500-image draws from val2014 (`chair_images_d1..d4.json`
in `assets/`, sampled with seeds 1–4), each selected by split name so every
run records which 500 it scored. Running the baseline and Fox over all five
draws turns "we could not reproduce CHAIR" into a number: how far CHAIR_S
moves when nothing changes but the image set. If that spread covers 3.88
points, the discrepancy is explained and the finding becomes one about the
benchmark protocol rather than about the method.

Those draw files are **ours, not the release's** — the released repository
contains 22 non-vendored files and none of them is an image list. (`pope/coco/426.json`
and `vase.json` look like candidates and are not: four POPE questions each,
left over from debugging, and two of `vase.json`'s four are malformed —
`"Waht are they doing?"` labelled `yes` and `"There are two"` labelled `no`.
`popo_err.jsonl` is zero bytes.)

### 3.10 A pre-cast fp16 checkpoint is not a deviation

If a run points at `kietnt0603/instructblip-vicuna-7b-fp16` rather than
`Salesforce/instructblip-vicuna-7b`, nothing about the numbers changes, and
this records why so the provenance is not read as a confound later.

`Salesforce/instructblip-vicuna-7b` stores its weights in **fp32**: 29.5 GiB
of safetensors to occupy about 15.8 GiB of VRAM once
`from_pretrained(dtype=torch.float16)` casts them at load. Every Kaggle
session pays that transfer again, because every session starts from an empty
disk. `scripts/cast_checkpoint_fp16.py` performs the same cast once and keeps
the result, so a session downloads ~14.7 GiB instead.

The cast is the identical operation `from_pretrained` already performs on
every load -- round-to-nearest fp32 to fp16 on the same inputs -- so the
weights are bit-identical, and `--verify` demonstrates that tensor by tensor
rather than asserting it. The run records `model_id` in every record, so a
pre-cast checkpoint is visible in the archive and cannot be mistaken for the
original.

**Quantization remains ruled out and this is not an exception to that.** The
quantity under study is a precision effect, so an int8 or int4 checkpoint
would alter the thing being measured; `Yhyu13/instructblip-vicuna-7b-gptq-4bit`
and `Mediocreatmybest/instructblip-vicuna-7b_8bit` exist and must not be used
here. fp16 is the dtype the runs already execute in.

**Licensing.** The copy is **private**. `Salesforce/instructblip-vicuna-7b` is
marked `license: other` and uses Vicuna-7b as its language model, which
derives from LLaMA-1 -- research-only, with redistribution of derived weights
restricted. A private repository is a copy for its owner rather than a public
redistribution. Do not make it public.

### 3.11 How the release runs three backbones with one patch

Worth understanding, because it was got wrong here first and because it
explains a hardcoded constant that otherwise looks arbitrary.

**Fox patches LLaMA attention**, in
`transformers/src/transformers/models/llama/modeling_llama.py`. Anything whose
decoder *is* LLaMA inherits the intervention without a second implementation.
That is the whole mechanism by which `pope_eval.py` offers a `shikra` entry in
`INSTRUCTION_TEMPLATE` beside `llava-1.5`.

`shikras/shikra-7b-delta-v1-0708/config.json` makes it concrete:

```json
"architectures": ["ShikraLlamaForCausalLM"],  "model_type": "shikra",
"hidden_size": 4096, "intermediate_size": 11008,
"num_attention_heads": 32, "num_hidden_layers": 32,   <- LLaMA-7B exactly
"mm_hidden_size": 1024, "use_mm_proj": true,
"mm_vision_tower": "openai/clip-vit-large-patch14",   <- stock, not bundled
"vocab_size": 32003                                    <- 32000 + 3 markers
```

and its weight index carries 325 tensors of which exactly **two** sit outside
a plain LLaMA decoder: `model.mm_projector.weight` and
`model.mm_projector.bias` — a single linear 1024 -> 4096. The vision tower is
downloaded separately from HuggingFace.

**The correction.** An earlier note here concluded Shikra needs the release's
`modeling_shikra.py`, therefore transformers 4.29.2, therefore Python 3.10,
therefore not Kaggle. The first step does not follow. A decoder that is LLaMA
loads as `LlamaForCausalLM` on any modern transformers; what has to be written
is the projector and the embedding splice, which is two tensors and a CLIP
forward. The real costs are the delta merge (`huggyllama/llama-7b` plus the
delta, ~27 GB down, ~40 GB transient) and the port, not the environment.

**And the magic numbers are explained.** The release detects Shikra's visual
span by testing `img_s == 33 and img_e == 290`. CLIP ViT-L/14 at 224 px gives
a 16x16 grid = 256 patches; `mm_use_im_start_end: true` wraps them in
`<im_start>` and `<im_end>`; 256 + 2 = **258**, and 290 - 33 + 1 = 258. The
33 is the token offset of the system message plus `USER: `. It is not an
arbitrary constant — it is this checkpoint's layout, hardcoded instead of
derived, which is why it would silently mis-locate the image on any other
prompt.

**Where Shikra's pixels are prepared.** The checkpoint ships no
`preprocessor_config.json`, which looks like a gap in the port and is not.
`modeling_shikra.py:49-81` loads the vision tower as
`CLIPVisionModel.from_pretrained(vision_tower)` — stock, at runtime, which is
why no vision weights are in the checkpoint — and the image processor comes
from the same repo. The release routes it through LAVIS, but
`minigpt4/processors/clip_processors.py`'s `clip_image_eval` adds nothing at
all:

```python
self.transform = CLIPImageProcessor.from_pretrained(proc_type)
return self.transform.preprocess(item, return_tensors='pt')['pixel_values'][0]
```

So the whole visual path is reachable from the modern stack:

```python
proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
vit  = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
h    = vit(pixel_values, output_hidden_states=True).hidden_states[-2][:, 1:]  # drop CLS -> 256
emb  = nn.Linear(1024, 4096)(h)        # the two mm_projector tensors
#      spliced between <im_start> and <im_end>
```

`mm_vision_select_layer: -2` is `hidden_states[-2]`, and `[:, 1:]`
(`modeling_shikra.py:136`) drops the CLS token, leaving the 256 patches.

`scripts/probe_backbone.py` reports all of this from `config.json` and the
weight index — a few hundred kilobytes rather than 13-30 GiB — for any
checkpoint, and is the thing to run before arguing about feasibility again.

### 3.9 Why only `sampling` and `fox` are re-run

The paper's tables carry eight rows. This project runs two of them, and the
reason is not cost.

**The contribution being tested is Fox.** Every claim the paper makes has the
shape "Fox improves on the baseline by X", so checking it needs exactly those
two rows. The other six place Fox in a ranking; they are context, not the
claim. And the baseline is the row that has to be right for any of it to
mean anything — ours matches the paper's to 0.05 points on all three POPE
splits, so the floor every gap is measured from is confirmed.

**The other six were re-run by the authors, not copied.** This was checked
rather than assumed, because the opposite assumption is easy to make and
would be wrong. VCD's own paper (Leng et al., 2024, Table 1, MSCOCO,
LLaVA-1.5) reports:

| split | VCD paper: Regular -> VCD | Fox paper: Sampling -> VCD |
|---|---|---|
| random | 83.29 -> 87.73 (+4.44) | 85.13 -> 86.84 (+1.71) |
| popular | 81.88 -> 85.38 (+3.50) | 82.53 -> 80.87 (**-1.66**) |
| adversarial | 78.96 -> 80.88 (+1.92) | 76.77 -> 75.33 (**-1.44**) |

Not one number matches, and the sign flips on two splits: VCD helps in its own
paper and hurts in Fox's. Numbers that disagree that strongly cannot have been
copied. The harness agrees — `pope_eval.py` carries `--use-vcd`, `--use-icd`
and `--opera` flags, and `vcd_sample.py:153` comments its shared path
`# for VCD and SID`, with a leftover author path pointing at
`/home/kdzlys/sid2/log/`.

The one exception is **CausalMM**, which appears nowhere in the released
repository and cannot be run from it. Whether its row was re-run elsewhere or
taken from its own paper is not determinable from what was published.

**Re-running them is also not available to us.** ICD, SID, OPERA and CausalMM
each need their own codebase; what this repo can run is VCD's noised branch
inside Fox's fusion, which is not VCD (see the VCD-noise note above). So
"reproduce the baselines too" would be three or four separate reproductions,
each answering a question this project is not asking.

Worth recording for the write-up: VCD's paper reports a standard deviation
beside every number (`83.29(±0.35)`). Fox claims "a two-sided t-test
(p < 0.05)" over "10 independent runs" (C.2) and prints variance nowhere. A
paper in the same line of work, three years earlier, met the reporting
standard this one skips.

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
| Shikra path | **Correction, see §3.11.** Earlier notes here called this a Track A job needing transformers 4.29.2. That was wrong: `ShikraLlamaForCausalLM` has no transformers class, but its decoder is a plain LLaMA-7B and the checkpoint carries exactly two tensors beyond one — `model.mm_projector.{weight,bias}`. It is a port on the modern stack, not a vendored environment. It is still a delta needing `huggyllama/llama-7b` merged in. |
| InstructBLIP path, practically | `Salesforce/instructblip-vicuna-7b` is public and transformers ships `InstructBlipForConditionalGeneration`, so unlike Shikra it can run on the modern stack. Costs: 31.7 GB to download (stored fp32; ~15.8 GB in VRAM once cast, which 2x T4 holds sharded). The work is porting, not hosting — see the row above on the second intervention copy, and note the visual span is **32 Q-Former query tokens**, not 576 patch tokens, so `H_vis` is an entropy over 32 positions instead of 576. |
| FastV / OPERA plumbing | Present in the release as argparse flags, all `default=False`, with no reweighting behind the SID one. |
| HF `logits_processor` | We apply the warpers (temperature, top-p) but not the processor list, which is empty by default for this configuration. |
| `use_phj` | Dead code: set on every layer, read nowhere. |
| A `print()` in the attention forward | Fires 7× per generated token in the InstructBLIP copy. |

## 5. Here, absent in the release

| | |
|---|---|
| A CHAIR evaluator | **The release has none.** `char_eval.py` only dumps captions; there is no `chair.py` and no synonym list. Table 2 is not reproducible from the release at all. |
| A frozen CHAIR image list | `char_eval.py:136` does `os.listdir(args.data_path)`, shuffles it, and keeps the first `--test-sample 500`. `data_path` defaults to `/data-disk/only_test` — a directory that was never published — and `os.listdir` returns filesystem order, so the set depends both on that directory's contents and on the author's filesystem. We ship explicit lists of val2014 ids instead, selected by split name and recorded in every run's provenance. See §3.8. |
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
| Is `chair_synonyms.txt` identical to Rohrbach's upstream? | Diff against the upstream file; our SHA-256 is in every metrics file. **Do this before quoting any CHAIR number externally.** |
| Does `--system-prompt none` still show Fox's gain? | It is a planned job; a yes undermines the paper's premise |
