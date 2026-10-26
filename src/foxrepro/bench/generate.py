"""GPU driver: run one (method, benchmark, split, seed, dtype, variant) to JSONL.

This program owns **everything except the model**. It loads the benchmark,
resumes an interrupted run, times each sample, writes one flushed JSONL line
per sample, and shuts down cleanly when the session is killed. The actual
decoding lives behind :class:`Method`; the implementations in this file are
deliberately stubs.

Contract with the method implementations
---------------------------------------
A method is a class deriving from :class:`Method` (or any object satisfying
:class:`MethodProtocol`) registered with :func:`register_method`::

    @register_method
    class MyMethod(Method):
        name = "mine"
        def load(self) -> None: ...                      # model goes here
        def generate(self, sample, prompt) -> MethodOutput: ...

``load()`` is called once, before the first sample; ``generate()`` is called
once per sample and must return the decoded text plus token counts. Nothing
in this file imports torch or transformers at module scope, so
``--dry-run`` works on a laptop.

Separation of concerns (Decision 1)
-----------------------------------
This program computes **no metrics**. It writes raw outputs with full
provenance, and ``bench/evaluate.py`` turns them into numbers. A metric bug is
then a CPU-side fix, never a re-run of the GPU.

Crash safety
------------
* every record is ``json.dumps``-ed, written, flushed and ``fsync``-ed before
  the next sample starts, so a hard kill loses at most the in-flight sample;
* SIGTERM / SIGINT set a stop flag, the current sample finishes, the file is
  closed and the process exits with code 7 ("stopped early, resume me");
* a trailing partial line from a previous hard kill is detected at startup and
  truncated away, so the file stays valid JSONL.

Resume key (Decision 4)
-----------------------
``(method, benchmark, split, seed, dtype, sample_id)``. The first five are
fixed per file (and are asserted against every record read back), so resume
reduces to "skip the ``sample_id``s already present".
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType
from typing import (
    Any, ClassVar, Dict, List, NamedTuple, Optional, Protocol, Sequence, Set,
    Type, runtime_checkable,
)


from foxrepro.data.loaders import (  # noqa: E402
    Sample, load_benchmark, render_prompt, template_benchmark,
)
from foxrepro.data.loaders import DEFAULT_TEMPLATE_IDS  # noqa: E402
from foxrepro.storage import (  # noqa: E402
    BENCHMARK_SPLITS, DTYPES, INTERVENTION_MODES, LOGIT_DTYPES,
    MAX_NEW_TOKENS_DEFAULT, PROVENANCE_FIELDS, SOFTMAX_DTYPES,
    SYSTEM_PROMPT_IDS, VARIANT_FIELDS, WRAPPER_IDS,
    RunKey, git_commit_hash, paths,
)

__all__ = [
    "GenConfig", "MethodOutput", "Method", "MethodProtocol", "register_method",
    "get_method", "list_methods", "JsonlWriter", "scan_run_file", "run",
]

#: Exit code used when the run stopped early and should be resumed.
EXIT_STOPPED_EARLY = 7


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenConfig:
    """Full configuration of one generation run.

    Hyperparameter defaults follow the paper for LLaVA-1.5 (Appendix C.2):
    nucleus sampling ``top_p = 0.9``, ``T = 1.0``, no repetition or length
    penalty; Fox uses ``alpha = 2``, ``beta = 0.1``, ``k = 0.45``,
    ``tau_js = 0.2`` and intervention layers 2-13.

    ``gamma`` has **no default from the paper**: it is described only as "a
    large intervention constant" and its value is never printed, which is
    exactly why this project sweeps it. It is therefore required for any
    method that uses it, and always recorded.
    """

    method: str
    benchmark: str
    split: str
    seed: int
    dtype: str
    model_id: str
    prompt_template_id: str
    max_new_tokens: int
    # method hyperparameters (Fox); ``None`` means "not applicable"
    gamma: Optional[float] = None
    k: Optional[float] = None
    tau_js: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    intervention_layers: Optional[str] = None
    # --- precision axes of the central experiment ------------------------- #
    # ``dtype`` above is the weight/activation dtype. The paper's effect does
    # NOT live there: it lives in the storage dtype of the pre-softmax
    # attention logits, because a uniform shift cancels in a numerically
    # stable fp32 softmax and only the cast's rounding survives. Varying
    # ``dtype`` alone therefore moves two things at once and shows a null
    # result that is correct but uninformative.
    #: Which system prompt precedes the turn. A first-class field rather
    #: than an `extra` key because the run-file variant tag is built from
    #: GenConfig attributes, and two values that change the output MUST NOT
    #: share a file -- resume would otherwise merge them silently.
    system_prompt_id: str = "llava_v1"
    #: Conversation wrapper. "released" is verbatim from the Fox repo's
    #: INSTRUCTION_TEMPLATE; "newline" uses the HF chat separator.
    wrapper_id: str = "released"
    #: Deliberate CPU run. Not recorded in the variant hash -- the host is
    #: provenance (n_gpus, gpu_name, torch_version already are), not an
    #: experimental arm.
    allow_cpu: bool = False
    logit_dtype: str = "native"     # "native" == same as ``dtype``
    softmax_dtype: str = "fp32"     # what every HF version actually does
    mode: str = "as_implemented"    # see storage.INTERVENTION_MODES
    # --- sharding (Kaggle T4x2 is the primary host) ----------------------- #
    # A single T4 cannot hold the model in fp16 (13.16 GiB of weights against
    # ~14.3 GiB usable), but two can when the layers are split across them.
    device_map: str = "auto"
    # decoding
    decoding: str = "sampling"
    top_p: float = 0.9
    temperature: float = 1.0
    # bookkeeping
    limit: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}")
        if self.logit_dtype not in LOGIT_DTYPES:
            raise ValueError(f"logit_dtype must be one of {LOGIT_DTYPES}")
        if self.softmax_dtype not in SOFTMAX_DTYPES:
            raise ValueError(f"softmax_dtype must be one of {SOFTMAX_DTYPES}")
        if self.mode not in INTERVENTION_MODES:
            raise ValueError(f"mode must be one of {INTERVENTION_MODES}")
        if self.system_prompt_id not in SYSTEM_PROMPT_IDS:
            raise ValueError(
                f"system_prompt_id must be one of {SYSTEM_PROMPT_IDS}"
            )
        if self.wrapper_id not in WRAPPER_IDS:
            raise ValueError(f"wrapper_id must be one of {WRAPPER_IDS}")
        if self.benchmark not in BENCHMARK_SPLITS:
            raise ValueError(f"unknown benchmark {self.benchmark!r}")
        if self.split not in BENCHMARK_SPLITS[self.benchmark]:
            raise ValueError(
                f"split {self.split!r} not valid for {self.benchmark!r}"
            )
        if self.decoding not in ("sampling", "greedy"):
            raise ValueError("decoding must be 'sampling' or 'greedy'")
        if template_benchmark(self.prompt_template_id) != self.benchmark:
            raise ValueError(
                f"prompt template {self.prompt_template_id!r} does not belong "
                f"to benchmark {self.benchmark!r}"
            )
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    def run_key(self) -> RunKey:
        """The :class:`RunKey` identifying this run's JSONL file."""
        return RunKey(
            method=self.method, benchmark=self.benchmark, split=self.split,
            seed=self.seed, dtype=self.dtype,
            variant_fields={n: getattr(self, n, None) for n in VARIANT_FIELDS},
        )

    def provenance(self) -> Dict[str, Any]:
        """Config half of a JSONL record (everything but the per-sample part)."""
        return {
            "method": self.method, "benchmark": self.benchmark,
            "split": self.split, "seed": self.seed, "dtype": self.dtype,
            "gamma": self.gamma, "k": self.k, "tau_js": self.tau_js,
            "alpha": self.alpha, "beta": self.beta,
            "max_new_tokens": self.max_new_tokens,
            "prompt_template_id": self.prompt_template_id,
            "model_id": self.model_id,
            "decoding": self.decoding, "top_p": self.top_p,
            "temperature": self.temperature,
            "intervention_layers": self.intervention_layers,
            "logit_dtype": self.logit_dtype,
            "softmax_dtype": self.softmax_dtype,
            "mode": self.mode,
            "system_prompt_id": self.system_prompt_id,
            "wrapper_id": self.wrapper_id,
            "device_map": self.device_map,
            **host_fingerprint(),
        }


def host_fingerprint() -> Dict[str, Any]:
    """GPU count/name and the TF32 precision flags, for the record.

    Two reasons this is per-record rather than per-session. A sharded T4x2 run
    and a single-GPU run are not interchangeable, and TF32 silently reduces an
    fp32 matmul to roughly fp16 precision in the QK product on Ampere+, which
    would contaminate the fp32 arm of the experiment in a way that is
    invisible unless the flags were recorded at the time.

    Never raises: torch may be absent (CPU-side tests, ``echo`` method).
    """
    out: Dict[str, Any] = {
        "n_gpus": 0, "gpu_name": None, "tf32_flags": None,
        "torch_version": None, "transformers_version": None,
        "compute_capability": None,
    }
    # Track B deliberately does NOT pin torch: the host (Kaggle) ships a recent
    # build with a native sm_75 cubin, and forcing a downgrade costs minutes per
    # session for no scientific gain. The price of not pinning is that the
    # resolved versions must be recorded, so a run stays reproducible after the
    # fact. That is what these fields are for.
    try:
        import transformers  # noqa: PLC0415

        out["transformers_version"] = getattr(transformers, "__version__", "unknown")
    except BaseException:  # noqa: BLE001
        pass
    try:
        import torch  # noqa: PLC0415  (intentionally lazy)

        out["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            out["n_gpus"] = torch.cuda.device_count()
            out["gpu_name"] = torch.cuda.get_device_name(0)
            p = torch.cuda.get_device_properties(0)
            out["compute_capability"] = f"{p.major}.{p.minor}"
        flags = {
            "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
            "fp16_reduced": bool(
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
        }
        try:
            flags["bf16_reduced"] = bool(
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            )
        except AttributeError:
            pass
        out["tf32_flags"] = flags
    except BaseException:  # noqa: BLE001 - provenance must never break a run
        pass
    return out


def pin_precision_flags() -> Dict[str, Any]:
    """Turn off every reduced-precision shortcut, then report what was set.

    Call once before loading the model. ``cudnn.allow_tf32`` defaults to
    **True**, so without this the fp32 arm of the dtype experiment runs at
    roughly fp16 mantissa width in the QK product and the experiment silently
    measures nothing.
    """
    try:
        import torch  # noqa: PLC0415

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        except AttributeError:
            pass
        # torch >= 2.9 exposes an explicit fp32 precision mode.
        try:
            torch.backends.cuda.matmul.fp32_precision = "ieee"  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError, ValueError):
            pass
    except BaseException:  # noqa: BLE001
        pass
    return host_fingerprint()


def resolve_transformers_path() -> str:
    """Resolved ``transformers.__file__``, or ``"unavailable:<reason>"``.

    The Fox reference implementation patches a **vendored transformers 4.29.2**.
    If stock HuggingFace transformers is on ``sys.path`` instead, the
    intervention silently does nothing while the run appears to succeed, and
    the resulting JSONL is indistinguishable from a real Fox run. Recording the
    resolved path per record makes that failure detectable after the fact.
    """
    try:
        import transformers  # noqa: PLC0415  (intentionally lazy)

        return str(getattr(transformers, "__file__", "unknown"))
    except BaseException as exc:  # pragma: no cover - environment dependent
        return f"unavailable:{type(exc).__name__}"


# --------------------------------------------------------------------------- #
# method interface
# --------------------------------------------------------------------------- #
class MethodOutput(NamedTuple):
    """What a method returns for one sample.

    Attributes:
        text: the decoded answer / caption, already stripped of any template.
        n_prompt_tokens: prompt length in tokens *including* image tokens.
        n_gen_tokens: number of generated tokens (1 for POPE).
        extra: optional per-sample diagnostics (e.g. mean JSD, gate
            activations). Stored under ``"method_extra"`` in the record.
    """

    text: str
    n_prompt_tokens: int
    n_gen_tokens: int
    extra: Dict[str, Any] = {}


@runtime_checkable
class MethodProtocol(Protocol):
    """Structural interface a decoding method must satisfy."""

    name: str

    def load(self) -> None:
        """Load whatever the method needs (model, processor, hooks)."""

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        """Produce one output for ``sample`` given the rendered ``prompt``."""

    def describe(self) -> Dict[str, Any]:
        """Return extra provenance about the loaded implementation."""

    def unload(self) -> None:
        """Release resources. Must be safe to call twice."""


class Method(ABC):
    """Base class for decoding methods.

    Subclasses implement :meth:`load` and :meth:`generate`. Everything else
    (benchmark iteration, resume, JSONL, timing, signals) is handled by this
    module, so a method implementation never touches the filesystem.
    """

    #: Registry key, also the value written to the ``method`` provenance field.
    name: ClassVar[str] = "abstract"
    #: Set to ``False`` for CPU-only test doubles.
    requires_gpu: ClassVar[bool] = True

    def __init__(self, cfg: GenConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def load(self) -> None:
        """Load the model and anything else needed. Called once."""

    @abstractmethod
    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        """Generate one output. Must be deterministic given the run seed."""

    def describe(self) -> Dict[str, Any]:
        """Extra provenance about the implementation (overridable)."""
        return {"class": type(self).__name__, "requires_gpu": self.requires_gpu}

    def unload(self) -> None:
        """Release resources. Default is a no-op."""


_REGISTRY: Dict[str, Type[Method]] = {}


def register_method(cls: Type[Method]) -> Type[Method]:
    """Class decorator registering a method under ``cls.name``.

    Raises:
        ValueError: on a missing or duplicate name.
    """
    name = getattr(cls, "name", None)
    if not name or name == "abstract":
        raise ValueError(f"{cls.__name__} must set a class-level 'name'")
    if name in _REGISTRY and _REGISTRY[name] is not cls:
        raise ValueError(f"method name {name!r} already registered")
    _REGISTRY[name] = cls
    return cls


#: Methods whose implementations live in ``bench/methods.py`` and need torch.
#:
#: This list is a routing hint for the lazy import, nothing more. It used to
#: be unioned into :func:`list_methods`, which made the advertised list a
#: superset of what :func:`get_method` could return -- renaming ``vcd`` to
#: ``vcd_fox_fusion`` left ``vcd`` advertised and unfetchable, and the error
#: message helpfully listed the name it had just refused. A name here that is
#: not in ``bench/methods.py`` is now a test failure rather than a phantom
#: entry in ``--list-methods``.
_GPU_METHODS = frozenset({
    "sampling", "fox", "vcd", "icd", "sid", "causalmm", "vcd_fox_fusion",
})


def _load_gpu_methods() -> None:
    """Import ``bench.methods`` on demand, registering the real implementations.

    Deliberately lazy. ``bench/generate.py`` imports no torch at module scope,
    so ``--dry-run`` (which uses the CPU ``echo`` double) and the whole
    evaluate/report path keep working on a laptop with neither torch nor a GPU.
    """
    if _GPU_METHODS <= set(_REGISTRY):
        return
    try:
        from foxrepro.bench import methods as _m  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            f"the real method implementations need torch and transformers "
            f"({exc}). Install requirements-trackB.lock, or use --dry-run, "
            f"which runs the CPU 'echo' double instead."
        ) from exc

    # Merge explicitly rather than trusting the decorator to have written into
    # *this* module's registry. Under `python -m foxrepro.bench.generate` this
    # module is in sys.modules as "__main__", so methods.py's
    # `from foxrepro.bench.generate import register_method` imports a SECOND
    # copy with its own _REGISTRY, and the decorator populates that one. The
    # console scripts and ordinary imports do not hit this, which is exactly
    # what makes it the kind of bug that only shows up in one invocation path.
    # Duck-typed on purpose: under the "__main__" double-import the `Method` in
    # this namespace is a DIFFERENT class object from the one methods.py
    # subclassed, so issubclass() would return False for every real
    # implementation. Structural checks are the only thing that survives it.
    for obj in vars(_m).values():
        if (
            isinstance(obj, type)
            and isinstance(getattr(obj, "name", None), str)
            and obj.name not in ("abstract", "abstract-llava")
            and callable(getattr(obj, "load", None))
            and callable(getattr(obj, "generate", None))
            and obj.name not in _REGISTRY
        ):
            _REGISTRY[obj.name] = obj


def get_method(name: str) -> Type[Method]:
    """Look up a registered method class.

    Raises:
        KeyError: with the list of known names, which is the error a typo in
            ``--method`` should produce.
        ImportError: if a GPU method is requested without torch installed.
    """
    if name in _GPU_METHODS:
        _load_gpu_methods()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; known: {sorted(list_methods())}") from None


def list_methods() -> List[str]:
    """Names of all registered methods.

    Imports the GPU implementations first when it can, so the list is what
    :func:`get_method` will actually return. Falls back to the registry alone
    when torch is absent, which is the laptop/evaluate path -- a shorter
    honest list beats a longer one containing names that cannot be fetched.
    """
    try:
        _load_gpu_methods()
    except ImportError:
        pass
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
# remaining stub: opera. The rest are real, in bench/methods.py
# --------------------------------------------------------------------------- #
class _NotImplementedMethod(Method):
    """Shared body for the not-yet-implemented methods."""

    #: One-line description of what the implementation must do.
    todo: ClassVar[str] = ""

    def load(self) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} ({self.name}) is a stub.\n"
            f"TODO: {self.todo}\n"
            "Implement load() (model + processor, fp16/bf16/fp32 per "
            "cfg.dtype, NO quantization) and generate(). Everything else "
            "(resume, JSONL, timing, signals) is already handled by "
            "bench/generate.py -- do not re-implement it here."
        )

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        raise NotImplementedError(f"{self.name}.generate: see load()'s message")


@register_method
class OPERAMethod(_NotImplementedMethod):
    """OPERA (Huang et al., 2024): beam search with an over-trust penalty.

    Optional baseline. Note it is incompatible with the shared nucleus-sampling
    protocol: the paper evaluates it with its own ``num_beams=5``
    configuration, so a run of this method must set ``decoding="greedy"``
    or an OPERA-specific decoding label, never ``"sampling"``.
    """

    name = "opera"
    todo = "beam search (num_beams=5) with the attention over-trust penalty and rollback."


@register_method
class EchoMethod(Method):
    """CPU-only test double: deterministic fake outputs, no model.

    Used by ``--dry-run`` and by ``tests/test_resume.py`` to exercise the whole
    pipeline (resume, JSONL, signals, evaluation) without a GPU. Its outputs
    are a deterministic function of the sample id, so the test suite can assert
    on them. Never use it for science: its ``method`` field is ``"echo"``,
    which no report table recognises.
    """

    name = "echo"
    requires_gpu = False

    def load(self) -> None:
        """No-op; nothing to load."""
        self._loaded = True

    def generate(self, sample: Sample, prompt: str) -> MethodOutput:
        """Yes/no for POPE (alternating, deterministic), a stock caption for CHAIR."""
        digest = sum(ord(c) for c in sample.sample_id)
        if sample.benchmark == "pope":
            text = "Yes" if digest % 2 == 0 else "No"
            return MethodOutput(text, len(prompt.split()) + 576, 1, {"digest": digest})
        text = (
            "The image shows a person riding a horse on a beach. "
            "There is a dog nearby and a surfboard on the sand."
        )
        return MethodOutput(text, len(prompt.split()) + 576, len(text.split()), {"digest": digest})


# --------------------------------------------------------------------------- #
# JSONL I/O and resume
# --------------------------------------------------------------------------- #
@dataclass
class ScanResult:
    """What :func:`scan_run_file` found in an existing run file.

    Attributes:
        done_ids: ``sample_id``s already completed.
        n_records: records parsed.
        n_duplicates: records whose ``sample_id`` had already been seen.
        truncated_bytes: bytes removed from a trailing partial line.
        mismatched: records whose run key disagreed with the expected one.
    """

    done_ids: Set[str] = field(default_factory=set)
    n_records: int = 0
    n_duplicates: int = 0
    truncated_bytes: int = 0
    mismatched: List[str] = field(default_factory=list)


def scan_run_file(
    path: Path, key: Optional[RunKey] = None, repair: bool = True
) -> ScanResult:
    """Read an existing run file and return the resume state.

    A hard kill can leave a partial final line. Appending after it would
    produce one unparsable record *and* corrupt the following one, so the
    partial line is truncated away (when ``repair``) before anything is
    appended.

    Args:
        path: the run JSONL.
        key: expected run key; every record's ``(method, benchmark, split,
            seed, dtype)`` is checked against it. A mismatch means two
            different runs are sharing a file, which is a bug, not a warning.
        repair: truncate a trailing partial line in place.

    Returns:
        A :class:`ScanResult`.
    """
    result = ScanResult()
    path = Path(path)
    if not path.is_file():
        return result

    good_offset = 0
    with open(path, "rb") as fh:
        for raw in fh:
            if not raw.endswith(b"\n"):
                # trailing partial line: the process died mid-write
                result.truncated_bytes = len(raw)
                break
            good_offset += len(raw)
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # a corrupt line in the middle: refuse to guess
                raise ValueError(
                    f"{path} contains a corrupt record before offset "
                    f"{good_offset}; inspect it by hand rather than resuming."
                ) from None
            result.n_records += 1
            if key is not None:
                observed = (
                    record.get("method"), record.get("benchmark"),
                    record.get("split"), record.get("seed"), record.get("dtype"),
                )
                expected = (key.method, key.benchmark, key.split, key.seed, key.dtype)
                if observed != expected:
                    result.mismatched.append(str(record.get("sample_id")))
            sid = record.get("sample_id")
            if sid is None:
                raise ValueError(f"{path}: record without a sample_id")
            if sid in result.done_ids:
                result.n_duplicates += 1
            result.done_ids.add(str(sid))

    if result.mismatched:
        raise ValueError(
            f"{path} holds {len(result.mismatched)} records from a different run "
            f"(e.g. sample {result.mismatched[0]}). Two runs are sharing one "
            "file; check the run-key variant tag."
        )
    if result.truncated_bytes and repair:
        os.truncate(path, good_offset)
    return result


class JsonlWriter:
    """Append-only JSONL writer that flushes and fsyncs every record.

    The fsync is the point: Colab disconnects are abrupt, and a buffered line
    is a lost sample. One fsync per sample costs milliseconds against
    generation latency of hundreds of milliseconds (POPE) to tens of seconds
    (CHAIR).
    """

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self._fsync = fsync
        self.n_written = 0

    def write(self, record: Dict[str, Any]) -> None:
        """Validate the provenance schema, then append one record."""
        missing = [f for f in PROVENANCE_FIELDS if f not in record]
        if missing:
            raise ValueError(f"record is missing provenance fields: {missing}")
        self._fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self.n_written += 1

    def close(self) -> None:
        """Flush, fsync and close. Safe to call more than once."""
        if not self._fh.closed:
            try:
                self._fh.flush()
                if self._fsync:
                    os.fsync(self._fh.fileno())
            finally:
                self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# signal handling
# --------------------------------------------------------------------------- #
class _StopFlag:
    """Cooperative stop flag driven by SIGTERM / SIGINT."""

    def __init__(self) -> None:
        self.stop = False
        self.signal: Optional[int] = None
        self._previous: Dict[int, Any] = {}

    def install(self) -> None:
        """Install handlers for SIGINT and SIGTERM (idempotent)."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[sig] = signal.signal(sig, self._handle)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    def restore(self) -> None:
        """Put the previous handlers back."""
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover
                pass
        self._previous.clear()

    def _handle(self, signum: int, frame: Optional[FrameType]) -> None:
        if self.stop:
            # second signal: give up immediately, the writer has already
            # fsynced every completed sample
            print(f"\n[generate] second signal {signum}; exiting now", flush=True)
            raise KeyboardInterrupt
        self.stop = True
        self.signal = signum
        print(
            f"\n[generate] signal {signum} received: finishing the current "
            "sample, then stopping. Re-run the same command to resume.",
            flush=True,
        )


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def _maybe_test_crash(n_written: int) -> None:
    """Testing hook: simulate a hard kill after N records.

    Driven by ``FOX_TEST_CRASH_AFTER`` (an int) and ``FOX_TEST_CRASH_MODE``
    (``"hard"`` kills the process, ``"truncate"`` first appends a partial
    line). Used only by ``tests/test_resume.py``; a normal run never sets them.
    """
    target = os.environ.get("FOX_TEST_CRASH_AFTER")
    if not target or n_written < int(target):
        return
    mode = os.environ.get("FOX_TEST_CRASH_MODE", "hard")
    if mode == "truncate":
        path = os.environ["FOX_TEST_CRASH_PATH"]
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"method": "echo", "sample_id": "partia')
            fh.flush()
            os.fsync(fh.fileno())
    sys.stderr.write(f"[generate] TEST CRASH after {n_written} records\n")
    sys.stderr.flush()
    os._exit(99)


def run(
    cfg: GenConfig,
    samples: Sequence[Sample],
    run_path: Path,
    progress_every: int = 25,
    fsync: bool = True,
) -> Dict[str, Any]:
    """Execute a run, resuming from ``run_path`` if it already exists.

    Args:
        cfg: the run configuration.
        samples: benchmark items in deterministic order.
        run_path: JSONL output path.
        progress_every: print a progress line every N samples (0 disables).
        fsync: fsync after every record. Only ever disable for tests.

    Returns:
        A summary dict: counts, timings and the resume state found at start.
    """
    key = cfg.run_key()
    scan = scan_run_file(run_path, key)
    pending = [s for s in samples if s.sample_id not in scan.done_ids]
    if cfg.limit is not None:
        pending = pending[: max(0, cfg.limit - len(scan.done_ids))]

    print(
        f"[generate] run file : {run_path}\n"
        f"[generate] samples  : {len(samples)} total, {len(scan.done_ids)} done, "
        f"{len(pending)} pending"
        + (f", repaired {scan.truncated_bytes}B partial line" if scan.truncated_bytes else "")
        + (f", {scan.n_duplicates} duplicates in file" if scan.n_duplicates else ""),
        flush=True,
    )

    summary: Dict[str, Any] = {
        "run_file": str(run_path),
        "n_samples_total": len(samples),
        "n_done_at_start": len(scan.done_ids),
        "n_pending_at_start": len(pending),
        "truncated_bytes_repaired": scan.truncated_bytes,
        "duplicates_in_file": scan.n_duplicates,
        "n_written": 0,
        "stopped_early": False,
        "stop_signal": None,
        "wall_seconds": 0.0,
    }
    if not pending:
        print("[generate] nothing to do; run already complete", flush=True)
        return summary

    method_cls = get_method(cfg.method)
    method = method_cls(cfg)

    stop = _StopFlag()
    stop.install()
    t_start = time.perf_counter()
    commit = git_commit_hash()
    transformers_path = resolve_transformers_path()
    writer = JsonlWriter(run_path, fsync=fsync)
    base = cfg.provenance()
    base["commit_hash"] = commit
    base["transformers_path"] = transformers_path

    try:
        method.load()
        summary["method_describe"] = method.describe()
        # The CONFIGURED hyperparameters are None whenever the released
        # defaults were used, and None is what lands in the provenance and in
        # the variant hash -- correctly, since "default" is one arm. But a
        # record that cannot say which gamma produced it is not much of an
        # archive, so the RESOLVED values ride along in method_extra, which is
        # free-form and outside the variant hash.
        resolved = summary["method_describe"].get("resolved_hyperparameters")
        for i, sample in enumerate(pending, start=1):
            if stop.stop:
                break
            prompt = render_prompt(cfg.prompt_template_id, sample.question)
            t0 = time.perf_counter()
            out = method.generate(sample, prompt)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            record = dict(base)
            record.update({
                "sample_id": sample.sample_id,
                "image_id": sample.image_id,
                "prompt": prompt,
                "output": out.text,
                "latency_ms": latency_ms,
                "n_prompt_tokens": out.n_prompt_tokens,
                "n_gen_tokens": out.n_gen_tokens,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
            extra = dict(out.extra) if out.extra else {}
            if resolved:
                extra["resolved_hyperparameters"] = resolved
            if extra:
                record["method_extra"] = extra
            if sample.label is not None:
                record["label"] = sample.label
            writer.write(record)
            summary["n_written"] = writer.n_written
            _maybe_test_crash(writer.n_written)

            if progress_every and (i % progress_every == 0 or i == len(pending)):
                elapsed = time.perf_counter() - t_start
                rate = i / elapsed if elapsed else 0.0
                eta = (len(pending) - i) / rate if rate else float("nan")
                print(
                    f"[generate] {i}/{len(pending)} "
                    f"({rate:.2f} samples/s, eta {eta / 60:.1f} min, "
                    f"last {latency_ms:.0f} ms)",
                    flush=True,
                )
    except KeyboardInterrupt:
        summary["stopped_early"] = True
        summary["stop_signal"] = "KeyboardInterrupt"
    finally:
        writer.close()
        try:
            method.unload()
        except Exception as exc:  # pragma: no cover - best effort teardown
            print(f"[generate] unload failed: {exc!r}", flush=True)
        stop.restore()

    if stop.stop:
        summary["stopped_early"] = True
        summary["stop_signal"] = stop.signal
    summary["wall_seconds"] = time.perf_counter() - t_start
    n = summary["n_written"]
    print(
        f"[generate] wrote {n} records in {summary['wall_seconds']:.1f}s"
        + (" (stopped early -- re-run to resume)" if summary["stopped_early"] else ""),
        flush=True,
    )
    return summary


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the ``generate.py`` argument parser."""
    p = paths()
    ap = argparse.ArgumentParser(
        description="Run one method on one benchmark split and write JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--method", default="sampling", help=f"one of {list_methods()}")
    ap.add_argument("--benchmark", choices=sorted(BENCHMARK_SPLITS), default="pope")
    ap.add_argument("--split", default="random", help="benchmark split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", choices=DTYPES, default="fp16",
                    help="fp16/bf16/fp32; quantization is deliberately not an option")
    ap.add_argument(
        "--model-id", default="llava-hf/llava-1.5-7b-hf",
        help="MUST be the HuggingFace-converted repo. The original\n"
             "liuhaotian/llava-v1.5-7b uses a different key layout "
             "(model.layers.* vs model.language_model.layers.*) and ships "
             "no preprocessor_config.json, so transformers loads it with a "
             "randomly initialised vision tower and projector. load_llava() "
               "refuses that, but do not rely on the guard -- pass the right id.",
    )
    ap.add_argument(
        "--prompt-template-id", default=None,
        help="default: the template the release sends -- "
             + ", ".join(f"{b}={t}" for b, t in sorted(DEFAULT_TEMPLATE_IDS.items()))
             + ". pope_v1 appends \"Answer the question using a single word or "
               "phrase\", which the release does not and which is worth about "
               "four points of POPE-adversarial accuracy; it is kept as an "
               "ablation.")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="default: 1 for POPE, 512 for CHAIR")
    ap.add_argument(
        "--allow-cpu", action="store_true",
        help="run without a GPU. Refused by default: r4 sessions 1 and 2 "
             "burned to the ten-hour wall on a Kaggle notebook whose "
             "accelerator was never attached, at 724 s per caption against "
             "7.8 s on two T4s, and produced ~50 of 500 samples each.")
    ap.add_argument("--decoding", choices=("sampling", "greedy"), default="sampling")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=None,
                    help="Fox intervention constant; unspecified in the paper, so it is swept")
    ap.add_argument("--k", type=float, default=None, help="Fox per-layer head ratio (0.45)")
    ap.add_argument("--tau-js", type=float, default=None, help="Fox JSD threshold (0.2)")
    ap.add_argument("--alpha", type=float, default=None, help="Fox consensus gain (2)")
    ap.add_argument("--beta", type=float, default=None, help="Fox truncation ratio (0.1)")
    # --- precision axes of the central experiment ------------------------- #
    ap.add_argument(
        "--logit-dtype", choices=LOGIT_DTYPES, default="native",
        help="storage dtype of the PRE-SOFTMAX attention logits -- the tensor "
             "the paper's mechanism actually acts on. 'native' keeps the "
             "weight dtype (bit-equivalent to the reference code). Varying "
             "--dtype instead moves weights and logits together and yields a "
             "null result that is correct but uninformative.",
    )
    ap.add_argument(
        "--softmax-dtype", choices=SOFTMAX_DTYPES, default="fp32",
        help="softmax accumulation dtype. 'fp32' is what every HuggingFace "
             "version does (4.29.2 through 5.17.0), and is why a uniform "
             "-gamma shift cancels exactly. 'native' is the ablation that "
             "tests the paper's claimed fp16-saturation story.",
    )
    ap.add_argument(
        "--mode", choices=INTERVENTION_MODES, default="as_implemented",
        help="which intervention arm: off (fusion only -- the control the "
             "paper never ran), as_implemented (uniform shift over all keys, "
             "as released), as_described (bias on image-token keys only, as "
             "Appendix B describes), ulp_control (explicit quantization), "
             "noise_control (matched-magnitude uniform noise).",
    )
    ap.add_argument(
        "--system-prompt", choices=SYSTEM_PROMPT_IDS, default="llava_v1",
        help="llava_v1 (default) is what eval_configs/llava-1.5_eval.yaml "
             "sets and llava.py:228 concatenates onto the turn -- with no "
             "separator, so the prompt reads '...questions.USER:'. ~35 tokens, "
             "pushing the image span from index ~5 to ~40. "
             "llava_v1_spaced adds the missing space, to measure whether that "
             "one character matters. 'none' omits the system prompt, which "
             "directly tests the paper's claim that the shortcut is anchored "
             "there: with none, m_sys measures attention over <s>/USER/: and "
             "nothing semantic. If Fox still gains, that story fails.",
    )
    ap.add_argument(
        "--wrapper", choices=("released", "newline"), default="released",
        help="conversation wrapper. 'released' is verbatim from the Fox "
             "repo's INSTRUCTION_TEMPLATE -- a SPACE between the image "
             "and the question, no system prompt. 'newline' uses the HF "
             "chat separator and is kept as an ablation.",
    )
    ap.add_argument(
        "--device-map", default="auto",
        help="HuggingFace device_map. 'auto' shards layers across all visible "
             "GPUs, which is what makes Kaggle's 2x T4 16GB viable: a single "
             "T4 cannot hold 13.16 GiB of fp16 weights with headroom, two can. "
             "Use 'cuda:0' to force a single device.",
    )
    ap.add_argument("--intervention-layers", default=None,
                    help="e.g. '2-13' for LLaVA-1.5 (paper Sec 5.2)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N completed samples")
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--runs-dir", type=Path, default=p.runs_root)
    ap.add_argument("--images-dir", type=Path, default=p.coco_images_dir)
    ap.add_argument("--pope-dir", type=Path, default=p.pope_ann_dir)
    ap.add_argument(
        "--chair-image-list", type=Path, default=None,
        help="default: the list for --split. Each CHAIR draw has its own "
             "file, so the split name alone selects which 500 images are "
             "scored and the run's provenance records it.")
    ap.add_argument("--dry-run", action="store_true",
                    help="use the CPU-only 'echo' method (pipeline smoke test)")
    ap.add_argument("--no-fsync", action="store_true", help="tests only")
    ap.add_argument("--list-methods", action="store_true")
    return ap


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_methods:
        for name in list_methods():
            cls = get_method(name)
            doc = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
            print(f"{name:10s} {doc}")
        return 0

    method = "echo" if args.dry_run else args.method
    template = args.prompt_template_id or DEFAULT_TEMPLATE_IDS[args.benchmark]
    max_new = args.max_new_tokens or MAX_NEW_TOKENS_DEFAULT[args.benchmark]

    cfg = GenConfig(
        method=method, benchmark=args.benchmark, split=args.split, seed=args.seed,
        dtype=args.dtype, model_id=args.model_id, prompt_template_id=template,
        max_new_tokens=max_new, gamma=args.gamma, k=args.k, tau_js=args.tau_js,
        alpha=args.alpha, beta=args.beta,
        intervention_layers=args.intervention_layers, decoding=args.decoding,
        top_p=args.top_p, temperature=args.temperature, limit=args.limit,
        logit_dtype=args.logit_dtype, softmax_dtype=args.softmax_dtype,
        mode=args.mode, device_map=args.device_map,
        system_prompt_id=args.system_prompt,
        wrapper_id=args.wrapper,
        allow_cpu=args.allow_cpu,
    )
    # Must happen before any model is loaded: cudnn.allow_tf32 defaults to
    # True, and on Ampere+ a TF32 matmul has ~10 mantissa bits, which would
    # make the fp32 arm of the dtype experiment indistinguishable from fp16
    # for a reason that has nothing to do with the method.
    if not args.dry_run:
        pin_precision_flags()
    samples = load_benchmark(
        benchmark=cfg.benchmark, split=cfg.split, pope_dir=args.pope_dir,
        images_dir=args.images_dir,
        chair_image_list=(args.chair_image_list
                          or paths().chair_image_list_for(cfg.split)
                          if cfg.benchmark == "chair"
                          else args.chair_image_list),
        prompt_template_id=cfg.prompt_template_id,
        require_images=not args.dry_run,
    )
    run_path = Path(args.runs_dir) / cfg.run_key().filename()
    summary = run(
        cfg, samples, run_path, progress_every=args.progress_every,
        fsync=not args.no_fsync,
    )
    return EXIT_STOPPED_EARLY if summary["stopped_early"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
