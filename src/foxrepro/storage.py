"""Single source of truth for every filesystem path used by the Fox reproduction.

Nothing else in this repository may contain an absolute path. Modules import
:func:`paths` (or :data:`PATHS`) and ask for the directory they need.

Environment model
-----------------
The code has to survive three places:

``kaggle``
    Kaggle Notebooks. **The project's primary host** (2x Tesla T4 16 GB).
    ``/kaggle/working`` is the persistent root: it survives as the notebook's
    output, but it is capped at 20 GB, so the 14 GB checkpoint must NOT go
    there. ``/kaggle/temp`` is the fast, larger, ephemeral scratch root and is
    where the HuggingFace cache and the extracted COCO images belong.
    ``/kaggle/input/<name>`` is read-only and is the right home for the model:
    upload it once as a private Kaggle Dataset, attach it to the notebook, and
    point ``FOX_HF_HOME`` at it to avoid re-downloading 14 GB every session.
    Sessions are capped at 12 h with a 30 GPU-h weekly quota, which is why
    resume is not optional here.
``colab``
    Google Colab. Ephemeral VM, Google Drive mounted at ``/content/drive``.
    Drive is the *persistent* root; ``/content`` is the *fast local* root.
    Drive hates many small files, so only large sequential artefacts
    (the COCO subset tar, run JSONL files, reports) live there.
    Note: stock Colab runs Python 3.12+, which cannot install Track A
    (transformers 4.29.2 needs tokenizers 0.13.3, wheels cp37-cp311 only).
``runpod``
    A rented RunPod / Vast.ai box with a persistent network volume at
    ``/workspace`` and fast container-local disk at ``/root`` (or ``$HOME``).
    The persistent volume plays the role Drive plays on Colab.
``local``
    A laptop. Persistence and "local" are the same disk; defaults live under
    ``~/.cache/fox-repro`` unless overridden.

Every root can be overridden by an environment variable, which is the escape
hatch for setups this detector does not recognise:

=============================  =====================================
``FOX_ENV``                    force ``kaggle`` / ``colab`` / ``runpod`` / ``local``
``FOX_PERSIST_ROOT``           persistent root (Drive, network volume)
``FOX_LOCAL_ROOT``             fast scratch root (extracted images, HF cache)
``FOX_RUNS_ROOT``              where run JSONL files are written
``FOX_HF_HOME``                HuggingFace cache root
=============================  =====================================

Run-file naming
---------------
A run is identified by the provenance tuple. Decision 4 of the project brief
fixes the *resume key* at ``(method, benchmark, split, seed, dtype, sample_id)``.
The remaining method hyperparameters (``gamma``, ``k``, ``tau_js``, ``alpha``,
``beta``, ``max_new_tokens``, ``prompt_template_id``, ``model_id``) are **not**
part of that key, yet the project deliberately sweeps ``gamma``. To keep both
true at once, the hyperparameters that are not in the resume key are folded
into the *file name* as a variant tag (see :meth:`RunKey.filename`): two runs
that differ only in ``gamma`` land in different files, and inside one file the
resume key is exactly the one the brief specifies.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

__all__ = [
    "Environment",
    "detect_environment",
    "Paths",
    "paths",
    "RunKey",
    "git_commit_hash",
    "sha256_file",
    "readiness_report",
]

Environment = Literal["kaggle", "colab", "runpod", "local"]

#: Canonical dtype spellings accepted everywhere in the repo.
DTYPES = ("fp32", "fp16", "bf16")

#: Dtype of the **pre-softmax attention logit tensor**. This is a separate axis
#: from :data:`DTYPES` (which is the weight/activation dtype) because the
#: paper's mechanism lives here and nowhere else: Fox subtracts a large
#: constant from this tensor and casts it back, and since the softmax that
#: follows is numerically stable and computed in fp32, the uniform shift
#: cancels exactly. The only surviving effect is the rounding of this tensor
#: onto its dtype's ULP grid. ``"native"`` means "leave it in the weight
#: dtype", i.e. bit-equivalent to the reference implementation.
LOGIT_DTYPES = ("native", "fp32", "fp16", "bf16")

#: Dtype the softmax accumulates in. HuggingFace hard-upcasts to fp32 in every
#: version checked (4.29.2, 4.37.2, 4.57.6, 5.17.0) -- ``"fp32"`` therefore
#: reproduces the reference behaviour, and ``"native"`` is the ablation that
#: tests whether the paper's "finite-precision saturation" story needs an
#: fp16 softmax that no HuggingFace code path actually has.
SOFTMAX_DTYPES = ("fp32", "native")

#: What the intervention does. These are **flag values, not code paths**: every
#: arm of the central experiment must be selectable without touching code, so
#: that a run file's provenance fully determines what was computed.
#:
#: ``off``            no intervention; fusion still runs. The control the paper
#:                    never ran, which separates the intervention's effect from
#:                    the additive logit fusion ``z_obs + alpha * z_do``.
#: ``as_implemented`` what the released code does: a uniform ``-gamma`` shift
#:                    across ALL key positions at one query position, then a
#:                    cast back to the logit dtype. Bit-equivalent to the
#:                    vendored patch.
#: ``as_described``   what the paper's Appendix B describes: an additive bias
#:                    restricted to the 576 image-token key indices, no cast
#:                    round trip. If this also reproduces the gain, the
#:                    described mechanism stands on its own.
#: ``ulp_control``    replace the shift round trip with an explicit
#:                    ``quantize_to_grid(logits, step)``. If this reproduces
#:                    the gain, the mechanism is requantization, full stop.
#: ``boost_visual``   the inverse of ``as_described``: a POSITIVE bias on the
#:                    image-token keys, leaving the system keys alone. This is
#:                    the experiment the paper's motivation rests on and never
#:                    runs -- it argues from an observational comparison
#:                    (Figure 1a) that amplifying visual attention cannot help,
#:                    and never intervenes to check. PAI, its named
#:                    representative, is in none of the paper's tables.
#: ``noise_control``  add i.i.d. uniform noise of the same magnitude as the ULP
#:                    grid, with no quantization structure. If this also
#:                    reproduces the gain, the method is noise injection and
#:                    the head-selection machinery is decorative.
INTERVENTION_MODES = (
    "off", "as_implemented", "as_described", "boost_visual",
    "ulp_control", "noise_control",
)

#: Prompt arms. The strings themselves live in ``bench.methods``, which cannot
#: be imported here (it imports ``bench.generate``, which imports this module),
#: so the *names* live here and ``methods`` asserts its dicts match. Keeping
#: them in one place is not tidiness: ``GenConfig`` validates against this
#: tuple and the run-file variant tag is built from it, so a name known to one
#: module and not the other would either reject a valid run or let two
#: different prompts share a file.
#:
#: ``llava_v1`` is the default -- it is what ``eval_configs/llava-1.5_eval.yaml``
#: sets and ``llava.py:228`` concatenates. ``none`` and ``llava_v1_spaced`` are
#: ablations.
SYSTEM_PROMPT_IDS = ("llava_v1", "llava_v1_spaced", "none")

#: Turn wrappers. ``released`` is the Fox repo's ``INSTRUCTION_TEMPLATE``.
WRAPPER_IDS = ("released", "newline")

#: Every provenance field a JSONL record must carry. ``generate.py`` asserts
#: that each written record has exactly these keys, so a schema drift shows up
#: at write time instead of at analysis time.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "method", "benchmark", "split", "seed", "dtype", "gamma", "k", "tau_js",
    "alpha", "beta", "max_new_tokens", "prompt_template_id", "model_id",
    "commit_hash", "transformers_path", "sample_id", "image_id", "prompt",
    "output", "latency_ms", "n_prompt_tokens", "n_gen_tokens", "timestamp",
    # --- the precision axes of the central experiment -------------------- #
    # The paper's effect is a dtype effect, so dtype cannot be one field.
    # These three are independent and all three must be explicit in every
    # record, or a run is not interpretable after the fact.
    # The system prompt is what Fox's m_sys term measures attention over, so
    # it is part of the method, not presentation. Omitting it leaves only
    # <s>/USER/: before the image and m_sys loses its subject entirely.
    "system_prompt_id",
    "wrapper_id",
    "logit_dtype",      # pre-softmax attention logit storage -- the mechanism
    "softmax_dtype",    # softmax accumulation (HF hard-upcasts to fp32)
    "mode",             # which intervention arm; see INTERVENTION_MODES
    # --- host / sharding ------------------------------------------------- #
    # On Kaggle T4x2 the model is sharded across devices, which changes where
    # each layer runs. Recorded so a multi-GPU run is never silently compared
    # against a single-GPU one.
    "device_map",
    "n_gpus",
    "gpu_name",
    # Track B does not pin torch (the host ships a usable build and a
    # downgrade costs minutes per session), so the resolved versions are
    # recorded instead. Without these a run cannot be reproduced later.
    "torch_version",
    "transformers_version",
    "compute_capability",
    # TF32 silently reduces an fp32 matmul to ~10 mantissa bits on Ampere+,
    # which would contaminate the fp32 arm into looking like fp16. The flags
    # are pinned at startup and recorded here so that contamination is
    # detectable rather than invisible.
    "tf32_flags",
)

#: Hyperparameters that are NOT part of the resume key but do change what the
#: GPU computes; they are folded into the run file name (see :class:`RunKey`).
VARIANT_FIELDS: tuple[str, ...] = (
    "gamma", "k", "tau_js", "alpha", "beta", "max_new_tokens",
    "prompt_template_id", "model_id",
    # not in the brief's list, but they change the output and would otherwise
    # alias two different experiments into one file: Table 9 is the *greedy*
    # control of Table 1 at identical method/benchmark/split/seed/dtype.
    "decoding", "top_p", "temperature", "intervention_layers",
    # the precision/arm axes. These MUST be here: the whole experiment is a
    # sweep over them at otherwise identical settings, so without them every
    # arm would resume from -- and overwrite -- the previous arm's file.
    "logit_dtype", "softmax_dtype", "mode", "system_prompt_id", "wrapper_id",
)

#: Default generation length per benchmark, matching the reference code:
#: POPE answers are a single token (yes/no), CHAIR captions are 512 tokens.
MAX_NEW_TOKENS_DEFAULT: Dict[str, int] = {"pope": 1, "chair": 512}

#: Benchmarks and their legal splits.
#: Extra CHAIR splits, each an independent 500-image draw from val2014.
#:
#: The released ``char_eval.py`` does ``os.listdir(args.data_path)``, shuffles
#: it, and keeps the first 500 -- where ``data_path`` defaults to
#: ``/data-disk/only_test``, a directory that was never published. Which 500
#: images produced the paper's CHAIR table therefore depends on that
#: directory's contents and on the author's filesystem ordering, and cannot be
#: recovered. Our CHAIR baseline sits 3.88 points from the paper's; our POPE
#: baseline, which uses POPE's fixed official annotation files, sits 0.05 from
#: it. These draws measure how much of that 3.88 is just the image set.
CHAIR_DRAWS: tuple[str, ...] = (
    "val2014_500_d1", "val2014_500_d2", "val2014_500_d3", "val2014_500_d4",
)

BENCHMARK_SPLITS: Dict[str, tuple[str, ...]] = {
    "pope": ("random", "popular", "adversarial"),
    "chair": ("val2014_500",) + CHAIR_DRAWS,
}


# --------------------------------------------------------------------------- #
# environment detection
# --------------------------------------------------------------------------- #
def detect_environment() -> Environment:
    """Return the execution environment, honouring ``FOX_ENV`` if it is set.

    Detection order: explicit override, Kaggle markers, Colab markers,
    RunPod/Vast markers, then ``local``. Detection never touches the network
    and never imports torch.

    Kaggle is checked before Colab because a Kaggle notebook also has some
    Google-ish environment variables, and its own markers are unambiguous.
    """
    forced = os.environ.get("FOX_ENV", "").strip().lower()
    if forced:
        if forced not in ("kaggle", "colab", "runpod", "local"):
            raise ValueError(
                f"FOX_ENV must be kaggle/colab/runpod/local, got {forced!r}"
            )
        return forced  # type: ignore[return-value]

    # Kaggle: the kernel sets these, and /kaggle/working always exists.
    if any(
        k in os.environ
        for k in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "KAGGLE_DATA_PROXY_TOKEN")
    ):
        return "kaggle"
    if Path("/kaggle/working").is_dir():
        return "kaggle"

    # Colab: the google.colab module exists, or the VM layout is unmistakable.
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if Path("/content").is_dir() and Path("/usr/local/lib/python3/dist-packages").exists():
        return "colab"
    if Path("/content").is_dir() and Path("/opt/bin/nvidia-smi").exists():
        return "colab"

    # RunPod / Vast.ai: pod env vars, or a /workspace volume on a Linux box.
    if any(k in os.environ for k in ("RUNPOD_POD_ID", "RUNPOD_API_KEY", "VAST_CONTAINERLABEL")):
        return "runpod"
    if Path("/workspace").is_dir() and not Path("/content").is_dir():
        return "runpod"

    return "local"


def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c.exists():
            return c
    return None


def drive_is_mounted() -> bool:
    """True when Google Drive looks mounted (Colab only, cheap check)."""
    return Path("/content/drive/MyDrive").is_dir()


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Paths:
    """Resolved roots and the derived directories the pipeline uses.

    All directory properties are pure functions of the three roots; nothing is
    created until :meth:`ensure` is called, so importing this module is free of
    side effects.
    """

    env: Environment
    repo_root: Path
    #: Directory of the installed ``foxrepro`` package. Shipped data files
    #: (the CHAIR synonym list, the POPE annotations) live under
    #: ``pkg_root/assets`` so they travel with the package rather than
    #: depending on the checkout layout.
    pkg_root: Path
    persist_root: Path
    local_root: Path
    runs_root: Path
    hf_home: Path

    # ---- persistent (Drive / network volume): few, large, sequential files ----
    @property
    def archive_dir(self) -> Path:
        """Holds the COCO subset tar + manifest. One-time build, read-only after."""
        return self.persist_root / "archives"

    @property
    def coco_subset_tar(self) -> Path:
        return self.archive_dir / "coco_val2014_subset.tar"

    @property
    def coco_subset_manifest(self) -> Path:
        return self.archive_dir / "coco_val2014_subset.manifest.json"

    @property
    def reports_dir(self) -> Path:
        return self.persist_root / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.persist_root / "logs"

    # ---- fast local disk: many small files (images) live here only ----
    @property
    def data_root(self) -> Path:
        """Where the subset tar is extracted at session start."""
        return self.local_root / "data"

    @property
    def coco_images_dir(self) -> Path:
        return self.data_root / "images"

    @property
    def coco_ann_dir(self) -> Path:
        return self.data_root / "annotations"

    def coco_annotation(self, stem: str) -> Path:
        """``instances_val2014`` / ``captions_val2014`` -> the file to read.

        Two of this project's own tools disagreed on the name.
        ``build_coco_subset.py`` writes ``<stem>_subset.json`` -- the 500-image
        slice packed into the Kaggle tar -- while
        ``fetch_images --with-annotations`` keeps the full
        ``<stem>.json`` straight out of COCO's zip. Either is a valid input to
        CHAIR, since the scorer indexes by image id and ignores the rest, so
        this prefers the subset and falls back to the full file rather than
        making the caller know which tool last ran.

        Args:
            stem: the file stem, without ``_subset`` and without ``.json``.

        Returns:
            The subset path if it exists, else the full path. The full path is
            also what comes back when neither exists, so the caller's own
            "missing file" error names something a user can act on.
        """
        subset = self.coco_ann_dir / f"{stem}_subset.json"
        full = self.coco_ann_dir / f"{stem}.json"
        return subset if subset.is_file() else full

    @property
    def pope_ann_dir(self) -> Path:
        """Directory holding ``coco_{random,popular,adversarial}.jsonl``.

        The POPE annotation files are tiny (a few hundred kB) and are tracked
        inside this repository at ``data/pope/``, so they are available even
        before the image tar is extracted. The copy frozen into the subset tar
        (``<data_root>/pope``) is used only as a fallback, which keeps old
        archives working.
        """
        repo_copy = self.pkg_root / "assets" / "pope"
        if any(repo_copy.glob("*.json*")):
            return repo_copy
        tar_copy = self.data_root / "pope"
        if any(tar_copy.glob("*.json*")) if tar_copy.is_dir() else False:
            return tar_copy
        return repo_copy

    @property
    def chair_image_list(self) -> Path:
        """The 500 CHAIR image ids.

        Prefers a copy under ``data_root`` (written by ``build_coco_subset``
        when the full val2014 is available) and otherwise falls back to the
        list shipped with the package.

        Shipping it matters: deriving the ids needs ``instances_val2014.json``,
        which is only distributed inside a 241 MB zip. Downloading that in
        every Kaggle session to recover 500 integers would be absurd, and
        without it the CHAIR jobs fail at startup -- which is exactly what
        happened on the first real session.
        """
        return self.chair_image_list_for("val2014_500")

    def chair_image_list_for(self, split: str) -> Path:
        """The image-id list for one CHAIR split.

        ``val2014_500`` is the original draw; ``val2014_500_dN`` are the
        image-set sensitivity draws (see :data:`CHAIR_DRAWS`). Each has its own
        file, and the split name is what selects it -- so a run's provenance
        records which 500 images it scored, which the released code cannot.
        """
        if split not in BENCHMARK_SPLITS["chair"]:
            raise ValueError(
                f"unknown CHAIR split {split!r}; "
                f"known: {BENCHMARK_SPLITS['chair']}"
            )
        suffix = "" if split == "val2014_500" else split[len("val2014_500"):]
        name = f"chair_images{suffix}.json"
        local = self.data_root / name
        if local.is_file():
            return local
        return self.pkg_root / "assets" / name

    @property
    def tmp_dir(self) -> Path:
        return self.local_root / "tmp"

    # ---- repository assets ----
    @property
    def chair_synonyms(self) -> Path:
        """CHAIR synonym list. See :mod:`bench.chair` for provenance rules."""
        return self.pkg_root / "assets" / "chair_synonyms.txt"

    @property
    def requirements_lock(self) -> Path:
        return self.repo_root / "requirements-cpu.lock"

    # ---- helpers ----
    def run_file(self, key: "RunKey") -> Path:
        """Absolute path of the JSONL file for one run."""
        return self.runs_root / key.filename()

    def eval_file(self, key: "RunKey") -> Path:
        """Absolute path of the per-run metrics JSON produced by evaluate.py."""
        return self.reports_dir / (key.filename().replace(".jsonl", ".metrics.json"))

    def ensure(self, which: Iterable[str] = ()) -> "Paths":
        """Create the writable directories. ``which`` restricts the set by name."""
        targets = {
            "archive_dir": self.archive_dir,
            "reports_dir": self.reports_dir,
            "logs_dir": self.logs_dir,
            "data_root": self.data_root,
            "tmp_dir": self.tmp_dir,
            "runs_root": self.runs_root,
            "hf_home": self.hf_home,
        }
        names = list(which) or list(targets)
        for name in names:
            targets[name].mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> Dict[str, str]:
        """Flat ``name -> path`` mapping, for the readiness report and logs."""
        out: Dict[str, str] = {"env": self.env}
        for name in (
            "repo_root", "pkg_root", "persist_root", "local_root", "runs_root", "hf_home",
            "archive_dir", "coco_subset_tar", "coco_subset_manifest", "reports_dir",
            "logs_dir", "data_root", "coco_images_dir", "coco_ann_dir", "pope_ann_dir",
            "chair_image_list", "tmp_dir", "chair_synonyms",
        ):
            out[name] = str(getattr(self, name))
        return out


_CACHED: Optional[Paths] = None


def paths(refresh: bool = False) -> Paths:
    """Resolve (and memoise) the :class:`Paths` for this process.

    Args:
        refresh: recompute even if a cached value exists (used by tests that
            monkeypatch the environment variables).
    """
    global _CACHED
    if _CACHED is not None and not refresh:
        return _CACHED

    env = detect_environment()
    # src-layout: this file is <repo>/src/foxrepro/storage.py, so the repo
    # root is three levels up. ``pkg_root`` is where the shipped assets live.
    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parents[1]

    if env == "kaggle":
        # /kaggle/working persists as the notebook's output but is capped at
        # 20 GB, so it holds only run JSONL + reports (megabytes). /kaggle/temp
        # is larger and faster and holds the HF cache and extracted images,
        # both of which are reconstructible.
        default_persist = Path("/kaggle/working/fox-repro")
        default_local = Path("/kaggle/temp/fox-local")
    elif env == "colab":
        default_persist = Path("/content/drive/MyDrive/fox-repro")
        default_local = Path("/content/fox-local")
    elif env == "runpod":
        default_persist = Path("/workspace/fox-repro")
        default_local = Path(os.environ.get("HOME", "/root")) / "fox-local"
    else:
        default_persist = Path.home() / ".cache" / "fox-repro"
        default_local = Path.home() / ".cache" / "fox-repro" / "local"

    persist_root = Path(os.environ.get("FOX_PERSIST_ROOT", str(default_persist))).expanduser()
    local_root = Path(os.environ.get("FOX_LOCAL_ROOT", str(default_local))).expanduser()
    runs_root = Path(
        os.environ.get("FOX_RUNS_ROOT", str(persist_root / "runs"))
    ).expanduser()
    hf_home = Path(os.environ.get("FOX_HF_HOME", str(local_root / "hf"))).expanduser()

    _CACHED = Paths(
        env=env,
        repo_root=repo_root,
        pkg_root=pkg_root,
        persist_root=persist_root,
        local_root=local_root,
        runs_root=runs_root,
        hf_home=hf_home,
    )
    return _CACHED


#: Convenience module-level handle. Prefer ``paths()`` in library code.
PATHS = paths()


# --------------------------------------------------------------------------- #
# run identity
# --------------------------------------------------------------------------- #
def _slug(value: Any) -> str:
    """Filesystem-safe rendering of a scalar config value."""
    if value is None:
        return "na"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        text = f"{value:g}"
    else:
        text = str(value)
    return "".join(ch if ch.isalnum() or ch in "-._" else "-" for ch in text)


@dataclass(frozen=True)
class RunKey:
    """Identity of one generation run.

    The first five fields plus ``sample_id`` form the resume key fixed by the
    project brief. ``variant_fields`` holds everything else that changes what
    the GPU actually computes; it only affects the *file name*, so that a
    ``gamma`` sweep cannot alias two runs into one JSONL file.
    """

    method: str
    benchmark: str
    split: str
    seed: int
    dtype: str
    variant_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {self.dtype!r}")
        if self.benchmark not in BENCHMARK_SPLITS:
            raise ValueError(
                f"benchmark must be one of {tuple(BENCHMARK_SPLITS)}, got {self.benchmark!r}"
            )
        legal = BENCHMARK_SPLITS[self.benchmark]
        if self.split not in legal:
            raise ValueError(f"split for {self.benchmark} must be one of {legal}, got {self.split!r}")

    # ---- identity ----
    def resume_key(self, sample_id: str) -> tuple[str, str, str, int, str, str]:
        """The exact tuple Decision 4 keys resume on."""
        return (self.method, self.benchmark, self.split, self.seed, self.dtype, sample_id)

    def variant_tag(self) -> str:
        """Short deterministic tag for the non-resume-key hyperparameters.

        Readable for the fields a human sweeps (``gamma``, ``k``, ``tau_js``),
        hashed for the rest so names stay bounded.
        """
        if not self.variant_fields:
            return "default"
        readable_order = ("decoding", "gamma", "k", "tau_js")
        prefixes = {"decoding": "", "gamma": "g", "k": "k", "tau_js": "t"}
        bits = [
            f"{prefixes[name]}{_slug(self.variant_fields[name])}"
            for name in readable_order
            if self.variant_fields.get(name) is not None
        ]
        payload = json.dumps(self.variant_fields, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
        return ("-".join(bits) + "-" + digest) if bits else digest

    def filename(self) -> str:
        """JSONL file name carrying the whole run identity."""
        return (
            f"m-{_slug(self.method)}__b-{_slug(self.benchmark)}__s-{_slug(self.split)}"
            f"__seed-{self.seed}__dt-{_slug(self.dtype)}__v-{self.variant_tag()}.jsonl"
        )

    def as_dict(self) -> Dict[str, Any]:
        """Provenance fields for a JSONL record."""
        out: Dict[str, Any] = {
            "method": self.method,
            "benchmark": self.benchmark,
            "split": self.split,
            "seed": self.seed,
            "dtype": self.dtype,
        }
        out.update(self.variant_fields)
        return out

    @classmethod
    def from_record(cls, record: Dict[str, Any], variant_names: Iterable[str]) -> "RunKey":
        """Rebuild a key from a JSONL record (used by evaluate.py / report.py)."""
        return cls(
            method=record["method"],
            benchmark=record["benchmark"],
            split=record["split"],
            seed=int(record["seed"]),
            dtype=record["dtype"],
            variant_fields={n: record.get(n) for n in variant_names},
        )


# --------------------------------------------------------------------------- #
# misc helpers
# --------------------------------------------------------------------------- #
def git_commit_hash(repo: Optional[Path] = None) -> str:
    """Short git commit of the repo, or ``"nogit"`` when unavailable.

    Appends ``"-dirty"`` when the working tree has uncommitted changes, because
    an ambiguous commit hash in a provenance record is worse than no hash.
    """
    root = repo or paths().repo_root
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (subprocess.SubprocessError, OSError):
        return "nogit"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (used for tar/manifest verification)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def kaggle_input_dirs() -> list[Path]:
    """Attached Kaggle Datasets, newest-looking first (empty off Kaggle).

    The 14 GB checkpoint should be uploaded once as a private Kaggle Dataset
    and attached read-only, rather than re-downloaded every session. Point
    ``FOX_HF_HOME`` (or the loader's model path) at the relevant entry here.
    """
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    try:
        return sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)
    except OSError:
        return []


def readiness_report(p: Optional[Paths] = None) -> Dict[str, Any]:
    """Machine-readable session readiness summary (used by the setup scripts)."""
    p = p or paths()
    rep: Dict[str, Any] = {"paths": p.as_dict(), "checks": {}}
    checks = rep["checks"]
    checks["drive_mounted"] = drive_is_mounted() if p.env == "colab" else None
    if p.env == "kaggle":
        checks["kaggle_input_datasets"] = [d.name for d in kaggle_input_dirs()]
        # /kaggle/working is capped at 20 GB and becomes the notebook output,
        # so warn before a 14 GB checkpoint ends up in it.
        checks["hf_home_outside_working"] = "/kaggle/working" not in str(p.hf_home)
    checks["persist_root_exists"] = p.persist_root.is_dir()
    checks["subset_tar_present"] = p.coco_subset_tar.is_file()
    checks["manifest_present"] = p.coco_subset_manifest.is_file()
    n_images = len(list(p.coco_images_dir.glob("*.jpg"))) if p.coco_images_dir.is_dir() else 0
    checks["n_local_images"] = n_images
    checks["pope_files"] = sorted(x.name for x in p.pope_ann_dir.glob("*.json")) if p.pope_ann_dir.is_dir() else []
    checks["chair_list_present"] = p.chair_image_list.is_file()
    checks["synonyms_present"] = p.chair_synonyms.is_file()
    checks["n_run_files"] = len(list(p.runs_root.glob("*.jsonl"))) if p.runs_root.is_dir() else 0
    rep["commit"] = git_commit_hash(p.repo_root)
    rep["ready_for_generation"] = bool(n_images > 0 and checks["pope_files"])
    return rep


def _main(argv: Optional[list[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Print resolved Fox storage layout.")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--ensure", action="store_true", help="create writable directories")
    args = ap.parse_args(argv)

    p = paths()
    if args.ensure:
        p.ensure()
    rep = readiness_report(p)
    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0
    print(f"environment     : {p.env}")
    for name, value in p.as_dict().items():
        if name == "env":
            continue
        print(f"  {name:22s}: {value}")
    print("checks:")
    for name, value in rep["checks"].items():
        print(f"  {name:22s}: {value}")
    print(f"commit          : {rep['commit']}")
    print(f"ready           : {rep['ready_for_generation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
