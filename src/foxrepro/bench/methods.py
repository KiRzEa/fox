"""LLaVA-1.5 decoding methods. Imported lazily so ``--dry-run`` needs no torch.

Every method here runs through one shared sampling implementation
(:func:`foxrepro.fox.attention.fox_generate`), which is the point: unfair
baselines are the commonest silent error in this literature, and two different
top-p implementations between the baseline and the method under test would be
exactly that. The only thing that differs between ``sampling`` and any Fox arm
is the intervention.

Three distinct configurations, often confused:

``sampling``
    one branch, no fusion, sample from ``z_obs``. The baseline every table is
    measured against.
``fox --mode off``
    two branches, fusion kept, intervention disabled. Isolates the additive
    logit fusion ``z_obs + alpha * z_do`` from the intervention -- a control
    the paper never ran, and the one that matters most, because with
    ``alpha = 2`` the fused logits carry roughly 3x the scale of ``z_obs`` and
    sharpened sampling alone is known to move POPE.
``fox --mode as_implemented``
    two branches, fusion, intervention. Bit-equal to the released code.

Prompt format
-------------
``data.loaders`` supplies only the instruction; the LLaVA-1.5 conversation
wrapper is added here, because it is a property of the checkpoint rather than
of the benchmark. LLaVA-1.5 expects::

    USER: <image>\\n{instruction} ASSISTANT:

A mismatch here is the single most common silent source of a 2-4 point POPE
gap in reproductions, and it would be mistaken for a method effect. The exact
string used is recorded in ``describe()`` so a later run can be compared
against it.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional

from foxrepro.bench.generate import (
    GenConfig, Method, MethodOutput, register_method,
)
from foxrepro.data.loaders import Sample
from foxrepro.fox.reference import SID_AGG_LAYER as _REF_SID_AGG_LAYER
from foxrepro.fox.reference import SID_ATTENTION_RANK as _REF_SID_ATTENTION_RANK
from foxrepro.fox.reference import CAUSALMM_EDIT_METHOD as _REF_CAUSALMM_EDIT_METHOD
from foxrepro.fox.reference import CAUSALMM_EPSILON as _REF_CAUSALMM_EPSILON
from foxrepro.fox.reference import CAUSALMM_GAMMA as _REF_CAUSALMM_GAMMA
from foxrepro.storage import SYSTEM_PROMPT_IDS, WRAPPER_IDS

#: Conversation wrappers, selected by ``--wrapper``.
#:
#: ``released`` is verbatim from the Fox repo's ``INSTRUCTION_TEMPLATE``
#: (``pope_eval.py:64``, ``char_eval.py:36``)::
#:
#:     "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"
#:
#: with ``<ImageHere>`` standing in for the image token. A SPACE separates the
#: image from the question, not a newline.
#:
#: ``newline`` is the HuggingFace chat template's separator. It is kept as an
#: ablation because switching to it is one of the prompt differences being
#: measured.
LLAVA_WRAPPERS: Dict[str, str] = {
    "released": "USER: <image> {instruction} ASSISTANT:",
    "newline": "USER: <image>\n{instruction} ASSISTANT:",
}

#: Default. Fidelity to the code that produced the paper's numbers.
DEFAULT_WRAPPER_ID = "released"

#: System prompts, selected by ``--system-prompt``.
#:
#: The released scripts do send one, but not through ``conv_templates``, which
#: is why reading ``pope_eval.py`` alone suggests otherwise. The path is:
#:
#: 1. ``eval_configs/llava-1.5_eval.yaml`` sets, under ``model:``::
#:
#:        system_message: "A chat between a curious user and an artificial
#:        intelligence assistant. The assistant gives helpful, detailed, and
#:        polite answers to the user's questions."
#:
#: 2. ``minigpt4/models/llava.py:344`` reads it as
#:    ``cfg.get("system_message", "")`` and stores it on the model;
#: 3. ``llava.py:228``, inside ``generate``, does
#:    ``instruction = [self.system_message + p for p in instruction]``.
#:
#: Step 3 is a bare ``+``. There is no separator, so the released prompt reads
#: ``...user's questions.USER: <image> ...`` with the period running straight
#: into ``USER``. ``llava_v1`` reproduces that exactly; ``llava_v1_spaced``
#: inserts the space a reader would expect, and exists only to measure whether
#: that one character matters. Each entry is concatenated with the wrapper
#: verbatim, so the separator lives in the string rather than in the renderer.
#:
#: The choice is not cosmetic. Fox scores heads by ``S = m_sys * H_vis``, where
#: ``m_sys`` is the attention mass on the keys BEFORE the image. Under
#: ``llava_v1`` those keys are the ~40 tokens of the system message, which is
#: what the paper's "primary anchor for latent priors" refers to. Under
#: ``none`` they are only ``<s>``, ``USER`` and ``:`` -- measured image span
#: ``[5, 580]`` instead of ``[~40, ~616]`` -- which leaves the paper's causal
#: story without its premise. ``none`` is kept as that ablation.
_LLAVA_V1_SYSTEM = (
    "A chat between a curious user and an artificial intelligence "
    "assistant. The assistant gives helpful, detailed, and polite answers "
    "to the user's questions."
)

SYSTEM_PROMPTS: Dict[str, str] = {
    "none": "",
    "llava_v1": _LLAVA_V1_SYSTEM,            # released: no separator
    "llava_v1_spaced": _LLAVA_V1_SYSTEM + " ",
}

#: Default. Fidelity to the config that produced the paper's numbers.
DEFAULT_SYSTEM_PROMPT_ID = "llava_v1"

# ``GenConfig`` validates these names and the run-file variant tag is built
# from them, so a name known to one module and not the other would either
# reject a valid run or let two different prompts share a run file.
assert set(SYSTEM_PROMPTS) == set(SYSTEM_PROMPT_IDS), (
    "SYSTEM_PROMPTS and storage.SYSTEM_PROMPT_IDS disagree: "
    f"{sorted(set(SYSTEM_PROMPTS) ^ set(SYSTEM_PROMPT_IDS))}"
)
assert set(LLAVA_WRAPPERS) == set(WRAPPER_IDS), (
    "LLAVA_WRAPPERS and storage.WRAPPER_IDS disagree: "
    f"{sorted(set(LLAVA_WRAPPERS) ^ set(WRAPPER_IDS))}"
)
assert DEFAULT_SYSTEM_PROMPT_ID in SYSTEM_PROMPTS
assert DEFAULT_WRAPPER_ID in LLAVA_WRAPPERS


def render_full_prompt(
    instruction: str, system_prompt_id: str = DEFAULT_SYSTEM_PROMPT_ID,
    wrapper_id: str = DEFAULT_WRAPPER_ID,
) -> str:
    """Assemble the complete string the processor receives.

    The system prompt is concatenated with no separator, reproducing
    ``llava.py:228``. Any separator is part of the ``SYSTEM_PROMPTS`` entry.

    Args:
        instruction: the question (POPE) or caption request (CHAIR).
        system_prompt_id: key into :data:`SYSTEM_PROMPTS`.
        wrapper_id: key into :data:`LLAVA_WRAPPERS`.

    Returns:
        The full prompt string.
    """
    try:
        system = SYSTEM_PROMPTS[system_prompt_id]
    except KeyError:
        raise ValueError(
            f"unknown system_prompt_id {system_prompt_id!r}; "
            f"known: {sorted(SYSTEM_PROMPTS)}"
        ) from None
    try:
        wrapper = LLAVA_WRAPPERS[wrapper_id]
    except KeyError:
        raise ValueError(
            f"unknown wrapper_id {wrapper_id!r}; known: {sorted(LLAVA_WRAPPERS)}"
        ) from None
    return system + wrapper.format(instruction=instruction)


#: Released Fox defaults for LLaVA-1.5, from the CODE, not the paper. They
#: disagree -- see ``foxrepro.fox.reference`` for the full table -- and the
#: code is what produced the released behaviour.
FOX_DEFAULTS: Dict[str, Any] = {
    "gamma": 10000.0,
    "ratio": 0.35,          # paper says 0.45
    "tau_js": 0.15,         # paper says 0.2
    "alpha": 2.0,
    "beta": 0.1,
    "layers": (2, 15),      # paper says 2-13
}

#: VCD's published noise schedule (Leng et al., CVPR 2024).
VCD_NOISE_STEP_DEFAULT = 500

#: SID's CT2S defaults, from ``pope_eval.py``'s argparse block. Imported from
#: the reference module rather than restated, so a correction to the
#: transcription cannot leave the method running the old value.
SID_AGG_LAYER = _REF_SID_AGG_LAYER
SID_ATTENTION_RANK = _REF_SID_ATTENTION_RANK

#: CausalMM's published values, from ``llava1.5_eval.bash`` -- not the
#: sampler's own defaults, which are 0.5 and 0.1.
CAUSALMM_GAMMA = _REF_CAUSALMM_GAMMA
CAUSALMM_EPSILON = _REF_CAUSALMM_EPSILON
CAUSALMM_EDIT_METHOD = _REF_CAUSALMM_EDIT_METHOD


def _refuse_untracked_override(name: str, value: Any, default: Any) -> Any:
    """Refuse a ``cfg.extra`` hyperparameter that the run key cannot see.

    ``storage.VARIANT_FIELDS`` is what the run filename is built from, and
    ``noise_step`` / ``sid_rank`` / ``sid_agg_layer`` are not in it. Two runs
    that differ only in one of them would therefore land in the SAME file and
    resume from each other -- half one sweep point, half another, with no
    error anywhere and a plausible number at the end.

    So these are usable at their published values and refused otherwise. To
    sweep one, add it to ``VARIANT_FIELDS`` and to ``Job`` first; that is a
    deliberate change with a filename migration attached, which is the right
    size for the hazard.
    """
    if value == default:
        return value
    raise ValueError(
        f"{name}={value!r} differs from the published default {default!r}, "
        f"but {name} is not in storage.VARIANT_FIELDS -- so this run would "
        f"share a file with the default one and the two would silently merge. "
        f"Add it to VARIANT_FIELDS and to plan.Job before sweeping it."
    )


def parse_layer_window(spec: Optional[str]) -> frozenset:
    """``"2-15"`` or ``"2,3,4"`` -> a frozenset of layer indices.

    ``None`` gives the released window. Raises on anything unparseable rather
    than silently intervening on the wrong layers.
    """
    if not spec:
        lo, hi = FOX_DEFAULTS["layers"]
        return frozenset(range(lo, hi + 1))
    spec = spec.strip()
    try:
        if "-" in spec and "," not in spec:
            lo_s, hi_s = spec.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError
            return frozenset(range(lo, hi + 1))
        return frozenset(int(x) for x in spec.split(",") if x.strip())
    except ValueError:
        raise ValueError(
            f"cannot parse intervention_layers {spec!r}; want '2-15' or '2,3,4'"
        ) from None


class _LlavaMethod(Method):
    """Shared loading, prompting and image handling for every LLaVA method.

    Subclasses override :meth:`_decode_kwargs` only. Keeping the loop itself in
    one place is what makes the baseline comparable to the methods.
    """

    name: ClassVar[str] = "abstract-llava"
    requires_gpu: ClassVar[bool] = True

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        self.model: Any = None
        self.processor: Any = None
        self._attn: Any = None
        self._eager_layers: frozenset = frozenset()
        self._pin_report: Dict[str, Any] = {}

    # ---- lifecycle -------------------------------------------------------- #
    def load(self) -> None:
        from foxrepro.bench.generate import pin_precision_flags
        from foxrepro.fox import attention as attn

        # Before any weights are loaded. On the T4 the TF32 flags are inert
        # (TF32 is Ampere+), but allow_fp16_reduced_precision_reduction
        # defaults to True and governs accumulation inside the fp16 Q@K.T
        # matmul -- the exact tensor this project studies.
        self._pin_report = pin_precision_flags()

        self._attn = attn
        self._eager_layers = parse_layer_window(self.cfg.intervention_layers)
        self.model, self.processor = attn.load_llava(
            self.cfg.model_id,
            dtype=self.cfg.dtype,
            device_map=self.cfg.device_map,
            eager_layers=self._eager_layers,
            allow_cpu=bool(getattr(self.cfg, "allow_cpu", False)),
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

    def _system_prompt_id(self) -> str:
        return getattr(self.cfg, "system_prompt_id", DEFAULT_SYSTEM_PROMPT_ID)

    def _wrapper_id(self) -> str:
        return getattr(self.cfg, "wrapper_id", DEFAULT_WRAPPER_ID)

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        spid = self._system_prompt_id()
        info.update({
            "wrapper_id": self._wrapper_id(),
            "wrapper": LLAVA_WRAPPERS[self._wrapper_id()],
            "system_prompt_id": spid,
            "system_prompt": SYSTEM_PROMPTS[spid],
            "eager_layers": sorted(self._eager_layers),
            "pinned_precision_flags": self._pin_report.get("tf32_flags"),
        })
        if self._attn is not None:
            info["attention_env"] = self._attn.describe_environment()
        return info

    # ---- per sample ------------------------------------------------------- #
    def _encode(self, sample: Sample, text: str) -> Dict[str, Any]:
        """Image read plus processor call, for an already-rendered prompt.

        Split out from :meth:`_prepare` because ICD needs a second encode of
        a *different* prompt against the same image, and because this is the
        one place in a method that touches the filesystem -- so a test can
        stub exactly this and still run the real prompt rendering.
        """
        from PIL import Image

        with Image.open(sample.image_path) as im:
            image = im.convert("RGB")
        return self.processor(images=image, text=text, return_tensors="pt")

    def _prepare(self, sample: Sample, prompt: str) -> Dict[str, Any]:
        return self._encode(
            sample,
            render_full_prompt(
                prompt, self._system_prompt_id(), self._wrapper_id()),
        )

    def _decode_kwargs(self) -> Dict[str, Any]:
        """Per-method arguments to ``fox_generate``."""
        raise NotImplementedError

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        inputs = self._prepare(sample, prompt)
        cfg = self.cfg
        # Seed per sample so a resumed run reproduces the same tokens for any
        # sample it redoes, and so two methods see the same randomness.
        seed = None if cfg.seed is None else (cfg.seed * 1_000_003 + hash(sample.sample_id) % 1_000_003)

        result = self._attn.fox_generate(
            self.model,
            dict(inputs),
            max_new_tokens=cfg.max_new_tokens,
            greedy=(cfg.decoding == "greedy"),
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            logit_dtype=cfg.logit_dtype,
            softmax_dtype=cfg.softmax_dtype,
            eager_layers=self._eager_layers,
            seed=seed,
            collect_trace=True,
            **self._decode_kwargs(),
        )

        text = self.processor.tokenizer.decode(
            result["token_ids"], skip_special_tokens=True
        ).strip()

        trace = result.get("trace") or []
        extra: Dict[str, Any] = {
            "image_span": list(result["image_span"]),
            "n_steps": len(trace),
        }
        if trace:
            d = [t["d_t"] for t in trace]
            lam = [t["lambda_t"] for t in trace]
            extra.update({
                "d_t_mean": sum(d) / len(d),
                "d_t_max": max(d),
                "lambda_mean": sum(lam) / len(lam),
                "n_diverged_tokens": sum(1 for t in trace if t["diverged"]),
            })
        return MethodOutput(
            text=text,
            n_prompt_tokens=result["n_prompt_tokens"],
            n_gen_tokens=result["n_gen_tokens"],
            extra=extra,
        )


@register_method
class SamplingMethod(_LlavaMethod):
    """Plain nucleus sampling. One branch, no fusion, no intervention.

    The reference every other number is compared against. ``--decoding greedy``
    gives the de-randomised variant used for the Table 9 control.
    """

    name = "sampling"

    def _decode_kwargs(self) -> Dict[str, Any]:
        return {"single_branch": True, "mode": "off"}


@register_method
class FoxMethod(_LlavaMethod):
    """Fox: diagnose risky heads, intervene pre-softmax, fuse two branches.

    ``--mode`` selects the arm. ``as_implemented`` is bit-equal to the released
    code (proven in ``tests/test_kernel_equivalence.py``); ``off``,
    ``as_described``, ``ulp_control`` and ``noise_control`` are the controls.

    Hyperparameters default to the CODE's values, not the paper's. Any omitted
    on the command line are filled from ``FOX_DEFAULTS`` and recorded, so a run
    is never ambiguous about which of the two disagreeing sources it used.
    """

    name = "fox"

    def _decode_kwargs(self) -> Dict[str, Any]:
        c = self.cfg
        return {
            "single_branch": False,
            "mode": c.mode,
            "gamma": c.gamma if c.gamma is not None else FOX_DEFAULTS["gamma"],
            "ratio": c.k if c.k is not None else FOX_DEFAULTS["ratio"],
            "tau_js": c.tau_js if c.tau_js is not None else FOX_DEFAULTS["tau_js"],
            "alpha": c.alpha if c.alpha is not None else FOX_DEFAULTS["alpha"],
            "beta": c.beta if c.beta is not None else FOX_DEFAULTS["beta"],
        }

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info["resolved_hyperparameters"] = self._decode_kwargs()
        info["hyperparameter_source"] = (
            "released code (layers 2-15, k=0.35, tau_js=0.15, gamma=1e4); the "
            "paper states 2-13, 0.45, 0.2 and never states gamma"
        )
        return info


@register_method
class VcdFoxFusionMethod(_LlavaMethod):
    """**Not VCD.** VCD's noised image inside Fox's ADDITIVE fusion.

    Kept because it is what the Fox release's own reachable code path does --
    `--use-vcd` is unreachable there (`--use-cd` is `store_true, default=True`
    and cannot be unset), so its `images_cd` is the clean image and its fusion
    is Fox's. Running the same shape here makes that behaviour measurable
    instead of only described.

    It is NOT the VCD baseline, and the name says so now that
    :class:`VCDMethod` exists beside it. VCD subtracts; this adds. Earlier
    results in the archive carry the old ``vcd`` name -- see
    docs/DEVIATIONS.md.
    """

    name = "vcd_fox_fusion"

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        self.noise_step = int(_refuse_untracked_override(
            "noise_step",
            int((cfg.extra or {}).get("noise_step", VCD_NOISE_STEP_DEFAULT)),
            VCD_NOISE_STEP_DEFAULT))

    def _noised(self, pixel_values: Any, generator: Any = None) -> Any:
        """VCD's corruption, from the verified kernel.

        This used to carry its own linear DDPM schedule, which is not VCD's.
        The released schedule is sigmoid and keeps 0.863 of the image at step
        500 where the linear one keeps 0.280. The kernel version is asserted
        bit-identical to ``vcd_add_noise.py``; nothing here reimplements it.
        """
        from foxrepro.fox.kernel import add_diffusion_noise

        out = add_diffusion_noise(
            pixel_values.float(), self.noise_step, generator=generator
        )
        return out.to(pixel_values.dtype)

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        inputs = dict(self._prepare(sample, prompt))
        inputs["pixel_values_do"] = self._noised(inputs["pixel_values"])
        cfg = self.cfg
        seed = None if cfg.seed is None else (
            cfg.seed * 1_000_003 + hash(sample.sample_id) % 1_000_003
        )
        px_do = inputs.pop("pixel_values_do")
        result = self._attn.fox_generate(
            self.model, inputs,
            max_new_tokens=cfg.max_new_tokens,
            greedy=(cfg.decoding == "greedy"),
            top_p=cfg.top_p, temperature=cfg.temperature,
            logit_dtype=cfg.logit_dtype, softmax_dtype=cfg.softmax_dtype,
            eager_layers=self._eager_layers, seed=seed, collect_trace=True,
            single_branch=False, mode="off",
            pixel_values_do=px_do,
            alpha=cfg.alpha if cfg.alpha is not None else FOX_DEFAULTS["alpha"],
            tau_js=cfg.tau_js if cfg.tau_js is not None else FOX_DEFAULTS["tau_js"],
            beta=cfg.beta if cfg.beta is not None else FOX_DEFAULTS["beta"],
        )
        text = self.processor.tokenizer.decode(
            result["token_ids"], skip_special_tokens=True
        ).strip()
        trace = result.get("trace") or []
        extra: Dict[str, Any] = {
            "variant": "vcd-noise-fox-fusion",
            "noise_step": self.noise_step,
            "image_span": list(result["image_span"]),
            "n_steps": len(trace),
        }
        if trace:
            d = [t["d_t"] for t in trace]
            extra["d_t_mean"] = sum(d) / len(d)
        return MethodOutput(
            text=text,
            n_prompt_tokens=result["n_prompt_tokens"],
            n_gen_tokens=result["n_gen_tokens"],
            extra=extra,
        )

    def _decode_kwargs(self) -> Dict[str, Any]:  # pragma: no cover - unused
        return {"single_branch": False, "mode": "off"}

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info["variant"] = "vcd-noise-fox-fusion"
        info["caveat"] = (
            "VCD's noised branch inside Fox's additive gated fusion, NOT VCD's "
            "own contrastive formula. A faithful VCD row needs the upstream "
            "DAMO-NLP-SG/VCD repo."
        )
        info["noise_step"] = self.noise_step
        return info


# --------------------------------------------------------------------------- #
# the compared baselines: one decoding rule, two perturbations
# --------------------------------------------------------------------------- #
class _ContrastiveMethod(_LlavaMethod):
    """Shared body for VCD and ICD.

    They differ in exactly one thing -- what the negative branch is shown --
    so they share everything else, including the decoding rule
    ``(1 + a) * z_obs - a * z_neg`` with the adaptive plausibility constraint.
    Writing them as two independent classes would invite them to drift, and
    the drift would be invisible: both would still produce numbers.

    Subclasses implement :meth:`_negative`, which returns the keyword
    arguments that build the negative branch.
    """

    #: Contrast strength. VCD publishes 1.0; SID's fork of the same file
    #: defaults to 0.5. Stated per subclass rather than shared, because the
    #: formula is shared and the tuning is not.
    cd_alpha: ClassVar[float] = 1.0
    cd_beta: ClassVar[float] = 0.1

    #: Validation arms this method understands. Checked at construction, not
    #: ignored: a control aimed at the wrong method would otherwise produce a
    #: run file NAMED for a control that never happened, which is the worst
    #: outcome available to a validation arm.
    CONTROLS: ClassVar[frozenset] = frozenset({"negative_only"})

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        if cfg.control is not None and cfg.control not in self.CONTROLS:
            raise ValueError(
                f"control {cfg.control!r} does not apply to {self.name}; "
                f"known here: {sorted(self.CONTROLS)}"
            )

    def _negative(self, inputs: Dict[str, Any], sample: Sample,
                  prompt: str, generator: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def _decode_kwargs(self) -> Dict[str, Any]:
        """Exactly the keywords ``fox_generate`` takes -- nothing else.

        Kept separate from :meth:`_record` because a subclass hyperparameter
        that is not a decode-loop argument (VCD's ``noise_step``) belongs in
        the provenance and would be a ``TypeError`` here.
        """
        return {
            "cd_alpha": float(
                self.cfg.alpha if self.cfg.alpha is not None else self.cd_alpha),
            "cd_beta": float(
                self.cfg.beta if self.cfg.beta is not None else self.cd_beta),
        }

    def _record(self) -> Dict[str, Any]:
        """Every resolved hyperparameter, for ``extra`` and ``describe``."""
        d = dict(self._decode_kwargs())
        if self.cfg.control is not None:
            d["control"] = self.cfg.control
        return d

    def _as_negative_branch(self, inputs: Dict[str, Any],
                            neg: Dict[str, Any]):
        """Move the perturbation onto the branch that will actually be run.

        An external perturbation (VCD's image, ICD's prompt) replaces the
        clean input. An internal one (SID's CT2S, CausalMM's counterfactual)
        stays a keyword and ``fox_generate`` applies it to row 0 instead of
        row 1, because under ``negative_only`` there is no row 1.
        """
        inputs, neg = dict(inputs), dict(neg)
        if "pixel_values_do" in neg:
            inputs["pixel_values"] = neg.pop("pixel_values_do")
        elif "input_ids_do" in neg:
            inputs["input_ids"] = neg.pop("input_ids_do")
            mask = neg.pop("attn_mask_do", None)
            if mask is not None:
                inputs["attention_mask"] = mask
        return inputs, neg

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        import torch

        inputs = dict(self._prepare(sample, prompt))
        cfg = self.cfg
        seed = None if cfg.seed is None else (
            cfg.seed * 1_000_003 + hash(sample.sample_id) % 1_000_003
        )
        # One generator for the perturbation and the sampler, seeded per
        # sample: the released code draws the image noise from global RNG,
        # which makes the one component whose purpose is randomness the one
        # part of a seeded run that does not repeat.
        dev = getattr(self.model, "device", None)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=dev) if dev is not None else torch.Generator()
            gen.manual_seed(seed)

        neg = self._negative(inputs, sample, prompt, gen)
        shared = dict(
            max_new_tokens=cfg.max_new_tokens,
            greedy=(cfg.decoding == "greedy"),
            top_p=cfg.top_p, temperature=cfg.temperature,
            logit_dtype=cfg.logit_dtype, softmax_dtype=cfg.softmax_dtype,
            eager_layers=self._eager_layers, seed=seed, collect_trace=True,
            # The intervention is OFF: the perturbation IS the method. Leaving
            # Fox's head edit on would make this Fox-plus-a-noisy-image.
            mode="off",
        )
        if cfg.control == "negative_only":
            # No contrast at all: run what the method SUBTRACTS, on its own.
            # Every one of these four says the negative branch is a worse
            # model; this is the arm that checks it, and none of the four
            # papers reports it.
            inputs, neg = self._as_negative_branch(inputs, neg)
            result = self._attn.fox_generate(
                self.model, inputs, single_branch=True, negative_only=True,
                **shared, **neg,
            )
        else:
            result = self._attn.fox_generate(
                self.model, inputs, single_branch=False,
                fusion="contrastive",
                **shared, **self._decode_kwargs(), **neg,
            )
        text = self.processor.tokenizer.decode(
            result["token_ids"], skip_special_tokens=True
        ).strip()
        trace = result.get("trace") or []
        extra: Dict[str, Any] = {
            "variant": self.name,
            "image_span": list(result["image_span"]),
            "n_steps": len(trace),
            **self._record(),
        }
        if trace and self.cfg.control != "negative_only":
            # Did the contrastive rule change the decision, and how often? A
            # method that never diverges from its own observational branch is
            # not running; one that diverges on every token is broken. Both
            # are invisible in an accuracy number.
            n_div = sum(1 for t in trace if t.get("diverged"))
            extra["n_diverged"] = n_div
            extra["diverged_frac"] = n_div / len(trace)
        if "sid_n_kept" in result:
            # Per-step, because the keep-set is recomputed per step. A run
            # whose min and max are not both `sid_rank` either hit the clamp
            # or never ran CT2S at all, and would look like a finished SID.
            kept = result["sid_n_kept"] or [0]
            extra["sid_n_kept_min"] = min(kept)
            extra["sid_n_kept_max"] = max(kept)
        return MethodOutput(
            text=text,
            n_prompt_tokens=int(result["n_prompt_tokens"]),
            n_gen_tokens=int(result["n_gen_tokens"]),
            extra=extra,
        )

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info["resolved_hyperparameters"] = self._record()
        info["fusion"] = "contrastive"
        # Recorded, not implied: these rows must be readable as "no head edit
        # happened here" without re-reading the class. Fox's intervention
        # being off is what makes them baselines.
        #
        # Deliberately NOT called "mode": the record's own `mode` field is
        # cfg.mode, which stays "as_implemented" for these arms, and two keys
        # with one name and two values is how a provenance field stops being
        # evidence.
        info["fox_intervention"] = "off"
        info["intervention"] = "none"
        return info


@register_method
class VCDMethod(_ContrastiveMethod):
    """VCD (Leng et al., CVPR 2024): contrast against a diffusion-noised image.

    The negative branch sees the same prompt and a corrupted image, so the
    contrast isolates what the model was reading off the picture.

    The noise schedule is the released sigmoid one, from the verified kernel
    -- not the linear DDPM schedule, which at step 500 keeps 0.280 of the
    image against the correct 0.863 and would be contrasting against noise.
    """

    name = "vcd"
    cd_alpha: ClassVar[float] = 1.0

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        self.noise_step = int(_refuse_untracked_override(
            "noise_step",
            int((cfg.extra or {}).get("noise_step", VCD_NOISE_STEP_DEFAULT)),
            VCD_NOISE_STEP_DEFAULT))

    def _negative(self, inputs, sample, prompt, generator):
        from foxrepro.fox.kernel import add_diffusion_noise

        px = inputs["pixel_values"]
        noised = add_diffusion_noise(
            px.float(), self.noise_step, generator=generator
        ).to(px.dtype)
        return {"pixel_values_do": noised}

    def _record(self) -> Dict[str, Any]:
        d = super()._record()
        d["noise_step"] = self.noise_step
        return d


@register_method
class ICDMethod(_ContrastiveMethod):
    """ICD (Wang et al., ACL 2024): contrast against a disturbed instruction.

    The negative branch sees the same image and a prompt carrying a negative
    role description, so the contrast isolates what the instruction was
    contributing rather than what the image was.

    The string and its insertion point are transcribed from
    ``pope_eval.py:291-295``, which is identical in SID and in the Fox fork::

        text_cd = 'You are a confused object detector.'
        prompt_cd = qu.split("<ImageHere>")[0] + "<ImageHere>" + ' ' + text_cd
                    + qu.split("<ImageHere>")[-1]

    so it lands immediately after the image marker and before the question,
    with a single space on each side. ``<ImageHere>`` is ``<image>`` in the
    converted checkpoint.

    The two branches cannot be batched -- the sequences differ in length --
    so ``fox_generate`` runs them as two passes with two caches, which is
    what the released code does anyway.
    """

    name = "icd"
    cd_alpha: ClassVar[float] = 1.0

    #: ``pope_eval.py:291``, verbatim.
    negative_instruction: ClassVar[str] = "You are a confused object detector."

    def disturbed_prompt(self, prompt: str) -> str:
        """The negative prompt string, exposed so a test can assert on it."""
        text = render_full_prompt(
            prompt, self._system_prompt_id(), self._wrapper_id())
        marker = "<image>"
        if marker not in text:
            raise ValueError(
                f"the rendered prompt has no {marker!r} to insert the "
                f"negative instruction after: {text[:80]!r}"
            )
        head, _, tail = text.partition(marker)
        return f"{head}{marker} {self.negative_instruction}{tail}"

    def _negative(self, inputs, sample, prompt, generator):
        enc = self._encode(sample, self.disturbed_prompt(prompt))
        dev = inputs["input_ids"].device
        return {
            "input_ids_do": enc["input_ids"].to(dev),
            "attn_mask_do": enc.get("attention_mask", None).to(dev)
            if enc.get("attention_mask", None) is not None else None,
        }

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info["negative_instruction"] = self.negative_instruction
        return info


@register_method
class SIDMethod(_ContrastiveMethod):
    """SID (Huo et al., ICLR 2025): contrast against the model's own bad read.

    Same image, same prompt. The negative branch is built *inside* the model
    by CT2S: from layer ``agg_layer`` onward, the only vision keys left
    visible are the ``rank`` the last query attends to **least**. So the
    contrast is between the model reading the picture and the model reading
    the part of the picture it had already decided was irrelevant -- no
    preprocessing, no second prompt, and the cheapest negative branch of the
    three compared methods.

    Three defaults here differ from VCD's and are not interchangeable:

    * ``cd_alpha = 0.5``, not 1.0. ``pope_eval.py`` parses ``--cd-alpha 1``
      and then does not pass it (``pope_eval.py:327-328`` are commented out),
      so ``vcd_sample.py:168``'s own default is what ran.
    * ``rank = 100`` of 576, ``largest=False``. The ``largest=True`` line
      directly above it in the source is FastV, a compute optimisation that
      keeps the *most*-attended tokens. Flipping it would build a better
      negative branch and subtract the wrong thing.
    * ``agg_layer = 2``, with the ranking read from layer 1.

    ``--sample-greedy`` also defaults to True in the released harness, which
    argmaxes the *unmasked* ``diffs`` and so bypasses both the beta cutoff
    and the temperature/top-p warpers -- SID's published POPE numbers are
    greedy whatever ``--sample`` says. Run it both ways and report both.
    """

    name = "sid"
    cd_alpha: ClassVar[float] = 0.5

    #: ``--fast-v-agg-layer``.
    agg_layer: ClassVar[int] = SID_AGG_LAYER
    #: ``--fast-v-attention-rank``.
    rank: ClassVar[int] = SID_ATTENTION_RANK

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        extra = cfg.extra or {}
        self.agg_layer = int(_refuse_untracked_override(
            "sid_agg_layer",
            int(extra.get("sid_agg_layer", type(self).agg_layer)),
            type(self).agg_layer))
        self.rank = int(_refuse_untracked_override(
            "sid_rank",
            int(extra.get("sid_rank", type(self).rank)),
            type(self).rank))

    CONTROLS: ClassVar[frozenset] = frozenset(
        {"negative_only", "sid_keep_most", "sid_random"})

    #: ``control`` -> which vision tokens CT2S keeps.
    SELECT_BY_CONTROL: ClassVar[Dict[Optional[str], str]] = {
        None: "lowest",
        "negative_only": "lowest",
        "sid_keep_most": "highest",
        "sid_random": "random",
    }

    def select(self) -> str:
        """``"lowest"`` is SID; the others are validation arms."""
        try:
            return self.SELECT_BY_CONTROL[self.cfg.control]
        except KeyError:
            raise ValueError(
                f"control {self.cfg.control!r} does not apply to sid; "
                f"known here: {sorted(k for k in self.SELECT_BY_CONTROL if k)}"
            ) from None

    def _negative(self, inputs, sample, prompt, generator):
        """Nothing external to build: the perturbation is inside the model.

        Returning the CT2S switches rather than a tensor is the whole point
        of SID, and it is why this method needs no second encode and no
        second image.
        """
        return {
            "sid": True,
            "sid_agg_layer": self.agg_layer,
            "sid_rank": self.rank,
            "sid_select": self.select(),
        }

    def _record(self) -> Dict[str, Any]:
        d = super()._record()
        d["sid_agg_layer"] = self.agg_layer
        d["sid_rank"] = self.rank
        d["sid_select"] = self.select()
        return d


@register_method
class CausalMMMethod(_ContrastiveMethod):
    """CausalMM (Zhou et al., ICLR 2025): contrast against a shuffled read.

    Same image, same prompt, same model. The negative branch is built in the
    **vision tower**: CLIP's layer -2 attention is shuffled, resized to the
    24x24 feature grid, averaged over heads and multiplied into the
    pre-projector visual features. So the contrast is between the model
    reading the image and the model reading the image with its own visual
    attention scrambled.

    The decoding rule is the one VCD, ICD and SID already share. CausalMM
    calls the coefficients ``gamma`` and ``epsilon``, but
    ``(1 + gamma) z_obs - gamma z_cf`` with ``log(epsilon)`` as the cutoff is
    ``(1 + cd_alpha) z_obs - cd_alpha z_cd`` with ``log(cd_beta)``, term for
    term. Four of the five compared methods are one formula.

    Its published values are NOT the sampler's defaults:
    ``llava1.5_eval.bash`` passes ``gamma=1.0`` (default 0.5) and
    ``epsilon=0.6`` (default 0.1). ``epsilon=0.6`` is a much tighter
    plausibility cutoff than the 0.1 the other three use, and unlike SID
    this harness never argmaxes the unmasked logits, so here the cutoff
    actually binds.

    Two protocol differences to carry into any table built from this arm:
    the released script runs ``top_p=1`` -- no nucleus truncation at all,
    against 0.9 everywhere else in this comparison -- and
    ``max_new_tokens=1024`` on POPE. Neither is changed here; this class
    sets the method, and the protocol stays the shared one so the arms remain
    comparable. Both are recorded in docs/DEVIATIONS.md Sec 3.13.
    """

    name = "causalmm"
    cd_alpha: ClassVar[float] = CAUSALMM_GAMMA
    cd_beta: ClassVar[float] = CAUSALMM_EPSILON

    #: ``edit_attention``'s own default, and the only value the released call
    #: site can produce. The other three are the paper's ablation.
    edit: ClassVar[str] = CAUSALMM_EDIT_METHOD

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        self.edit = str(_refuse_untracked_override(
            "cf_edit",
            str((cfg.extra or {}).get("cf_edit", type(self).edit)),
            type(self).edit))

    CONTROLS: ClassVar[frozenset] = frozenset(
        {"negative_only", "cf_constant", "cf_blind"})

    #: ``control`` -> which edit the counterfactual branch gets.
    EDIT_BY_CONTROL: ClassVar[Dict[Optional[str], str]] = {
        None: CAUSALMM_EDIT_METHOD,
        "negative_only": CAUSALMM_EDIT_METHOD,
        "cf_constant": "constant",
        "cf_blind": "blind",
    }

    def resolved_edit(self) -> str:
        try:
            return self.EDIT_BY_CONTROL[self.cfg.control] or self.edit
        except KeyError:
            raise ValueError(
                f"control {self.cfg.control!r} does not apply to causalmm; "
                f"known here: {sorted(k for k in self.EDIT_BY_CONTROL if k)}"
            ) from None

    def _negative(self, inputs, sample, prompt, generator):
        """Nothing external: the counterfactual is built inside the encoder."""
        return {"cf_vision": True, "cf_edit": self.resolved_edit()}

    def _record(self) -> Dict[str, Any]:
        d = super()._record()
        d["cf_edit"] = self.resolved_edit()
        return d

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        # Not a choice, and not hideable: output_attentions=True puts CLIP on
        # its eager path and the hidden states move (3.9e-3 in fp16 on a
        # stand-in). CausalMM derives both branches from that one forward, so
        # this arm's clean branch is not bit-identical to the other arms'.
        info["clean_branch_bit_identical_to_other_arms"] = False
        info["vision_tower_attn"] = "eager (forced by output_attentions)"
        return info


@register_method
class OPERAMethod(_LlavaMethod):
    """OPERA (Huang et al., CVPR 2024): beam search with an over-trust penalty.

    Structurally unlike every other arm here. The rest are one sequence with
    a decoding rule; this is a five-beam search that runs five extra
    speculative forwards per step to score candidate tokens that have not
    been emitted yet. The search lives in :mod:`foxrepro.fox.opera`, which
    also carries the audit of what the released 559-line version does.

    **Labelled greedy, and the paper agrees.** Beam search is deterministic,
    so OPERA has no sampling variant -- and indeed the paper prints the
    *identical* OPERA numbers in Table 1 (its sampling table) and Table 9
    (its greedy one). One run per seed is therefore the whole arm, and
    ``decoding="sampling"`` is refused rather than silently accepted.

    **POPE only, and that is a property of the method rather than a budget
    decision.** OPERA's second mechanism -- retrospection-reallocation --
    cannot fire below 16 response tokens: the rollback score is only selected
    when more than 10 tokens have been emitted, and the trigger needs a modal
    column repeated more than 15 times in a history with one column per step.
    ``opera_generate`` refuses a length where rollback would become
    reachable, so this arm cannot silently become "OPERA minus half its
    method" on CHAIR.

    Generation length is 10, not our POPE default of 1, because that is what
    ``pope_eval.py`` passes. With beam search that is not cosmetic: the
    search scores whole sequences, so the first token depends on how many
    follow it.
    """

    name = "opera"

    #: ``pope_eval.py``'s argparse defaults, and the README's command line.
    num_beams: ClassVar[int] = 5
    num_attn_candidates: ClassVar[int] = 5
    penalty_weights: ClassVar[float] = 1.0

    #: ``control`` -> (restrict the candidate set?, penalty weight).
    #:
    #: OPERA is three things at once. With the plain arm these two rows
    #: separate them: whatever the restriction alone buys is
    #: ``opera_no_penalty`` minus a plain beam search, and whatever the
    #: penalty alone buys is ``opera_unrestricted`` minus the same.
    CONTROL_SETTINGS: ClassVar[Dict[Optional[str], Any]] = {
        None: (True, penalty_weights),
        "opera_unrestricted": (False, penalty_weights),
        "opera_no_penalty": (True, 0.0),
    }

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        if cfg.decoding == "sampling":
            raise ValueError(
                "OPERA is beam search and has no sampling variant -- the "
                "paper prints the same numbers in its sampling and greedy "
                "tables. Run it with --decoding greedy"
            )
        try:
            self.restrict, self.weight = self.CONTROL_SETTINGS[cfg.control]
        except KeyError:
            raise ValueError(
                f"control {cfg.control!r} does not apply to opera; known "
                f"here: {sorted(k for k in self.CONTROL_SETTINGS if k)}"
            ) from None

    def _decode_kwargs(self) -> Dict[str, Any]:  # pragma: no cover - unused
        """Never called: this method does not go through ``fox_generate``."""
        raise NotImplementedError

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        from foxrepro.fox.opera import opera_generate

        inputs = dict(self._prepare(sample, prompt))
        cfg = self.cfg
        result = opera_generate(
            self.model, inputs,
            max_new_tokens=cfg.max_new_tokens,
            num_beams=self.num_beams,
            num_attn_candidates=self.num_attn_candidates,
            penalty_weights=self.weight,
            restrict_candidates=self.restrict,
            collect_trace=True,
        )
        text = self.processor.tokenizer.decode(
            result["token_ids"], skip_special_tokens=True
        ).strip()
        trace = result.get("trace") or []
        extra: Dict[str, Any] = {
            "variant": "opera",
            "image_span": list(result["image_span"]),
            "n_steps": len(trace),
            "n_forwards_per_step": result["n_forwards_per_step"],
            **self._record(),
        }
        if trace:
            # The two numbers that decide whether the penalty does anything.
            # It is added in LOGIT space, so what matters is its spread across
            # the five candidates against the spread of their raw logits.
            extra["penalty_spread_mean"] = sum(
                t["penalty_spread"] for t in trace) / len(trace)
            extra["logit_spread_mean"] = sum(
                t["logit_spread"] for t in trace) / len(trace)
            extra["n_steps_penalty_changed_top"] = sum(
                1 for t in trace if t["penalty_changed_top"])
        return MethodOutput(
            text=text,
            n_prompt_tokens=int(result["n_prompt_tokens"]),
            n_gen_tokens=int(result["n_gen_tokens"]),
            extra=extra,
        )

    def _record(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "num_beams": self.num_beams,
            "num_attn_candidates": self.num_attn_candidates,
            "penalty_weights": self.weight,
            "restrict_candidates": self.restrict,
            "rollback": False,
        }
        if self.cfg.control is not None:
            d["control"] = self.cfg.control
        return d

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info["resolved_hyperparameters"] = self._record()
        info["search"] = f"beam-{self.num_beams}"
        info["fusion"] = "none"
        info["fox_intervention"] = "off"
        info["intervention"] = "none"
        # Not a deviation but a proved-unreachable omission; see the class
        # docstring and docs/DEVIATIONS.md Sec 3.13.
        info["rollback_implemented"] = False
        info["rollback_reachable_at_this_length"] = bool(
            self.cfg.max_new_tokens > 15)
        return info
