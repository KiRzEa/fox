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

#: The conversation wrapper LLaVA-1.5 was instruction-tuned with.
LLAVA_V1_5_WRAPPER = "USER: <image>\n{instruction} ASSISTANT:"

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
        )

    def unload(self) -> None:
        self.model = None
        self.processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass

    def describe(self) -> Dict[str, Any]:
        info = dict(super().describe())
        info.update({
            "wrapper": LLAVA_V1_5_WRAPPER,
            "eager_layers": sorted(self._eager_layers),
            "pinned_precision_flags": self._pin_report.get("tf32_flags"),
        })
        if self._attn is not None:
            info["attention_env"] = self._attn.describe_environment()
        return info

    # ---- per sample ------------------------------------------------------- #
    def _prepare(self, sample: Sample, prompt: str) -> Dict[str, Any]:
        from PIL import Image

        text = LLAVA_V1_5_WRAPPER.format(instruction=prompt)
        with Image.open(sample.image_path) as im:
            image = im.convert("RGB")
        return self.processor(images=image, text=text, return_tensors="pt")

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
class VCDMethod(_LlavaMethod):
    """VCD (Leng et al., CVPR 2024): contrast against a diffusion-noised image.

    Same dual-branch machinery, but row 1 sees a corrupted image instead of a
    modified attention pattern, and no head intervention runs.

    A caveat worth recording: this reproduces VCD's *noised second branch*
    inside Fox's additive gated fusion, not VCD's own contrastive formula
    ``(1 + a) * logits - a * logits_cd``. The released Fox repo has the same
    property -- its reachable ``use_cd`` path runs VCD noise through Fox's
    fusion -- so a faithful VCD baseline needs the upstream VCD repo. This is
    labelled ``vcd-noise-fox-fusion`` in ``describe()`` so no table silently
    presents it as VCD.
    """

    name = "vcd"

    def __init__(self, cfg: GenConfig) -> None:
        super().__init__(cfg)
        self.noise_step = int(
            (cfg.extra or {}).get("noise_step", VCD_NOISE_STEP_DEFAULT)
        )

    def _noised(self, pixel_values: Any) -> Any:
        """VCD's forward-diffusion corruption at ``noise_step``."""
        import torch

        steps = 1000
        betas = torch.linspace(1e-4, 0.02, steps)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)[self.noise_step - 1]
        noise = torch.randn_like(pixel_values.float())
        out = alpha_bar.sqrt() * pixel_values.float() + (1 - alpha_bar).sqrt() * noise
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
