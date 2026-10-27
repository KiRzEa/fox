#!/usr/bin/env bash
# Session bootstrap for Kaggle Notebooks (2x Tesla T4 16GB) -- the project's
# primary host. Idempotent: safe to re-run after the 12-hour session cap or an
# unexpected disconnect.
#
# Run from the repo root inside a notebook cell:
#     !bash setup_kaggle.sh
#
# Why Kaggle rather than Colab:
#   * 2x T4 means the model can be SHARDED. A single T4 cannot hold 13.16 GiB
#     of fp16 weights with headroom (~14.3 GiB usable), but split across two
#     devices it needs only ~8.3 GiB each. Quantization is ruled out for this
#     project -- the mechanism under study IS a numerical-precision effect --
#     so sharding is the only way a 16 GB card participates.
#   * torch 2.1.2+cu121 ships a NATIVE cubin for sm_75 (T4). L4/RTX 4090 are
#     sm_89 and have no native cubin in that wheel; they rely on CUDA
#     minor-version binary compatibility. Track A is therefore safer here.
#   * Kaggle's Python is 3.11 at time of writing, which Track A can use
#     (transformers 4.29.2 needs tokenizers 0.13.3, wheels cp37-cp311 only).
#     Stock Colab is 3.12+ and cannot install Track A at all.
#
# Known Kaggle constraints this script works around:
#   * /kaggle/working persists as the notebook output but is capped at 20 GB,
#     so it holds ONLY run JSONL and reports. The 14 GB checkpoint must not.
#   * /kaggle/temp is larger, faster and ephemeral: HF cache + extracted images.
#   * 12 h max per session, 30 GPU-h per week. Resume is not optional.
#   * No bf16 on T4 (sm_75). See the note at the end.

set -uo pipefail

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
die() { printf '\033[31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT" || die "cannot cd to repo root"

# Honour a caller-supplied FOX_ENV so this script can be smoke-tested off
# Kaggle; otherwise assume Kaggle and fail early with something readable
# rather than a traceback from mkdir on a read-only /kaggle.
if [ -z "${FOX_ENV:-}" ]; then
  if [ ! -d /kaggle/working ]; then
    die "/kaggle/working not found -- this is the Kaggle bootstrap.
       On Kaggle: enable Accelerator = 'GPU T4 x2', then re-run.
       Elsewhere: use scripts/setup_colab.sh, or set FOX_ENV=local|runpod
       to smoke-test this script's logic."
  fi
  export FOX_ENV=kaggle
fi

SKIP_PIP=0
SKIP_MODEL=0
TRACK=B
for arg in "$@"; do
  case "$arg" in
    --skip-pip)   SKIP_PIP=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --track-a)    TRACK=A ;;
    --track-b)    TRACK=B ;;
    *) die "unknown flag: $arg (want --skip-pip|--skip-model|--track-a|--track-b)" ;;
  esac
done

# --------------------------------------------------------------- 1. probe ---
say "1/7  Host probe"
if [ -f scripts/probe_host.py ]; then
  python scripts/probe_host.py || warn "probe reported problems -- read them before continuing"
else
  warn "scripts/probe_host.py missing; skipping"
fi

# ---------------------------------------------------------------- 2. gpus ---
say "2/7  GPU inventory"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv
  N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  if [ "$N_GPU" -lt 2 ]; then
    warn "only $N_GPU GPU visible. A single T4 16GB CANNOT hold the model in"
    warn "fp16 with headroom. In the notebook sidebar set Accelerator = 'GPU T4 x2'."
  else
    echo "  $N_GPU GPUs -- device_map='auto' will shard the layers across them."
  fi
else
  warn "nvidia-smi not found; is the accelerator enabled?"
fi

# ------------------------------------------------------------- 3. python ----
say "3/7  Interpreter"
python -c 'import sys; print("python", ".".join(map(str,sys.version_info[:3])))'
python - <<'PY'
import sys
ok = (3, 8) <= sys.version_info[:2] <= (3, 11)
print(f"  track A installable: {ok}")
if not ok:
    print("  -> Track A needs <=3.11 (tokenizers 0.13.3 has no cp312 wheel).")
    print("     Track B is unaffected. Use `uv venv --python 3.10` for Track A.")
PY

# ---------------------------------------------------------------- 4. deps ---
say "4/7  Dependencies (track $TRACK)"
if [ "$SKIP_PIP" -eq 1 ]; then
  echo "  skipped (--skip-pip)"
else
  pip install -q --no-cache-dir -e . || die "editable install failed"
  if [ "$TRACK" = "A" ]; then
    warn "Track A pins torch 2.1.2 and needs the Fox repo's VENDORED"
    warn "transformers 4.29.2. Kaggle preinstalls a newer torch, so this"
    warn "downgrade takes several minutes and must be redone each session."
    pip install -q --no-cache-dir -r requirements-trackA.lock \
      || die "track A deps failed"
    echo "  NOW RUN, pointing at your clone of the Fox repo:"
    echo "      pip install -e /kaggle/input/<your-fox-dataset>/transformers"
    echo "  Without that, LLaVA-1.5 runs with ZERO intervention and the run"
    echo "  still reports success."
  else
    pip install -q --no-cache-dir -r requirements-trackB.lock \
      || die "track B deps failed"
  fi
fi

# -------------------------------------------------------------- 5. layout ---
say "5/7  Storage layout"
python -m foxrepro.storage --ensure || die "storage layout failed"

# ---------------------------------------------------------------- 6. data ---
say "6/7  COCO subset + POPE annotations"
python - <<'PY'
import json, sys
from foxrepro.storage import paths, readiness_report
p = paths()
rep = readiness_report(p)
c = rep["checks"]
print(f"  subset tar present : {c['subset_tar_present']}")
print(f"  local images       : {c['n_local_images']}")
print(f"  POPE files         : {c['pope_files'] or 'MISSING'}")
if c.get("kaggle_input_datasets") is not None:
    print(f"  attached datasets  : {c['kaggle_input_datasets'] or 'none'}")
    print(f"  hf cache off /working: {c['hf_home_outside_working']}")
if not c["subset_tar_present"]:
    print()
    print("  The COCO subset tar is not on the persistent side yet. Build it")
    print("  ONCE from a full val2014 download, then upload the ~150-200 MB tar")
    print("  as a private Kaggle Dataset and attach it read-only:")
    print("      python -m foxrepro.data.build_coco_subset build \\")
    print("          --val2014-dir <val2014> --annotations-dir <annotations> \\")
    print("          --pope-dir data/pope")
    print("  You do NOT need all 40k val2014 images: CHAIR uses 500 and POPE")
    print("  uses 500, so the union is <=1000 images.")
PY

if python -c "
import sys
from foxrepro.storage import paths
raise SystemExit(0 if paths().coco_subset_tar.is_file() else 1)" 2>/dev/null; then
  python -m foxrepro.data.build_coco_subset extract || warn "extract failed"
  python -m foxrepro.data.build_coco_subset verify --level fast || warn "verify failed"
else
  # No attached tar. Fetch exactly the images the plan needs straight from
  # COCO rather than leaving the session NOT READY with a build instruction
  # the operator has to go and find: the union of POPE's and CHAIR's image
  # sets is under 3000 files and ~460 MB, which is about six minutes, against
  # the documented 6.2 GB val2014 zip. Idempotent -- already-present files
  # are skipped, so a re-run after a dead session costs nothing.
  if python -c "
import sys
from foxrepro.storage import paths
raise SystemExit(0 if any(paths().coco_images_dir.glob('*.jpg')) else 1)" 2>/dev/null; then
    echo "  images already present; skipping the fetch"
  else
    warn "no subset tar attached -- fetching the needed images from COCO."
    warn "Attach the checksummed tar as a read-only Kaggle Dataset to make"
    warn "later sessions cost nothing here."
    python -m foxrepro.data.fetch_images --what union \
      || warn "image fetch failed -- the readiness gate below will say so"
  fi
fi

# --------------------------------------------------------------- 7. model ---
say "7/7  Model"
if [ "$SKIP_MODEL" -eq 1 ]; then
  echo "  skipped (--skip-model)"
else
  echo "  Preferred: upload the checkpoint ONCE as a private Kaggle Dataset and"
  echo "  attach it, then point the loader at /kaggle/input/<name>. Kaggle gives"
  echo "  no persistent volume between sessions, so otherwise you re-download"
  echo "  14 GB every time."
  echo
  echo "  To fetch into /kaggle/temp instead (ephemeral, ~5 min):"
  if [ "$TRACK" = "A" ]; then
    echo "      huggingface-cli download liuhaotian/llava-v1.5-7b   # Track A only"
  else
    echo "      huggingface-cli download llava-hf/llava-1.5-7b-hf"
  fi
  echo "  Do NOT download into /kaggle/working: it is capped at 20 GB and"
  echo "  becomes the notebook's committed output."
fi

# ------------------------------------------------------------- readiness ----
say "Readiness"
python - <<'PY'
import sys
from foxrepro.storage import paths, readiness_report
rep = readiness_report(paths())
ready = rep["ready_for_generation"]
print("  READY" if ready else "  NOT READY")
if not ready:
    c = rep["checks"]
    if not c["pope_files"]:
        print("   - POPE annotations missing from data/pope/ (see its README)")
    if not c["n_local_images"]:
        print("   - no COCO images locally. Step 6 tries to fetch them; if it")
        print("     failed, run it directly and read the error:")
        print("       python -m foxrepro.data.fetch_images --what union")
    sys.exit(1)
PY
RC=$?

cat <<'NOTE'

------------------------------------------------------------------------
T4-specific note on the dtype experiment
------------------------------------------------------------------------
T4 is sm_75 (Turing) and has no bf16 COMPUTE. The experiment only needs a
bf16 cast-and-add on the attention logits, not a bf16 matmul, so it may still
work -- scripts/probe_host.py measures this and reports the bf16 arm as available or
not.

It does not block the project either way. The decisive experiment is the gamma
magnitude sweep, which spans the fp16 ULP grid entirely within fp16:

    gamma=1e2 -> ULP 0.0625      gamma=1e4 -> ULP 8   (the paper's value)
    gamma=1e3 -> ULP 0.5         gamma=6e4 -> ULP 32

That is four orders of magnitude of dose-response on a T4. Only ULP 64 (the
bf16 regime, where attention collapses to uniform) is out of reach, because
fp16 overflows at 65504 -- and that regime is the least informative one.

A monotone response in ULP rather than in gamma magnitude is the signature of
the requantization account of the paper's mechanism.

Resume after a session dies: re-run the exact same generate.py command. The
run file is keyed on the full configuration and completed sample_ids are
skipped.
------------------------------------------------------------------------
NOTE

exit $RC
