#!/usr/bin/env bash
# Idempotent session bootstrap for the Fox reproduction.
#
# Safe to re-run after a disconnect: every step checks whether it has already
# been done. Nothing here is destructive, and nothing here touches the run
# JSONL files.
#
#   bash setup_colab.sh                      # full bootstrap
#   bash setup_colab.sh --skip-pip           # deps already installed
#   bash setup_colab.sh --skip-model         # CPU-only session (evaluate/report)
#   bash setup_colab.sh --verify-level full  # re-hash every extracted file
#
# Steps
#   1. detect the environment (Colab / RunPod / local) via storage.py
#   2. mount Google Drive when on Colab and not yet mounted
#   3. pip install -r requirements.lock (skipped when the lock file is unchanged)
#   4. create the directory layout
#   5. extract the COCO val2014 subset tar from persistent storage to fast
#      local disk, verifying the tar's SHA-256 against the manifest first
#   6. pre-download the LLaVA-1.5-7B checkpoint into the *local* HF cache
#      (~14 GB; never into Drive -- it would be re-uploaded and is slower to
#      read than re-downloading)
#   7. print a readiness report
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
# Track B (the default) needs the HF-CONVERTED repo. The original
# liuhaotian/llava-v1.5-7b has a different key layout and no image
# processor, and transformers will load it with random vision weights.
MODEL_ID="${FOX_MODEL_ID:-llava-hf/llava-1.5-7b-hf}"
VERIFY_LEVEL="fast"
SKIP_PIP=0
SKIP_MODEL=0
FORCE_EXTRACT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-pip)       SKIP_PIP=1 ;;
    --skip-model)     SKIP_MODEL=1 ;;
    --force-extract)  FORCE_EXTRACT=1 ;;
    --verify-level)   VERIFY_LEVEL="${2:?}"; shift ;;
    --model-id)       MODEL_ID="${2:?}"; shift ;;
    -h|--help)        sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n=== %s\n' "$*"; }
warn() { printf '  !! %s\n' "$*" >&2; }

# --------------------------------------------------------------- 1. environment
say "1/7 environment"
FOX_ENV_DETECTED="$("$PYTHON" -c 'import storage; print(storage.detect_environment())')"
echo "  detected: ${FOX_ENV_DETECTED}"
echo "  python  : $("$PYTHON" -c 'import sys; print(sys.version.split()[0])')"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  gpu     : /'
else
  warn "no nvidia-smi: this session can run evaluate.py / report.py but not generate.py"
fi

# -------------------------------------------------------------------- 2. Drive
say "2/7 persistent storage"
if [[ "$FOX_ENV_DETECTED" == "colab" ]]; then
  if [[ -d /content/drive/MyDrive ]]; then
    echo "  Drive already mounted at /content/drive"
  else
    echo "  mounting Google Drive (approve the prompt) ..."
    "$PYTHON" - <<'PYMOUNT'
from google.colab import drive
drive.mount("/content/drive")
PYMOUNT
  fi
  [[ -d /content/drive/MyDrive ]] || { warn "Drive mount failed"; exit 1; }
fi
"$PYTHON" storage.py --ensure >/dev/null
PERSIST_ROOT="$("$PYTHON" -c 'import storage; print(storage.paths().persist_root)')"
LOCAL_ROOT="$("$PYTHON" -c 'import storage; print(storage.paths().local_root)')"
HF_CACHE="$("$PYTHON" -c 'import storage; print(storage.paths().hf_home)')"
echo "  persistent: ${PERSIST_ROOT}"
echo "  local     : ${LOCAL_ROOT}"

# ----------------------------------------------------------------------- 3. pip
say "3/7 python dependencies"
PIP_STAMP="${LOCAL_ROOT}/.pip-installed.sha256"
LOCK_HASH="$("$PYTHON" -c 'import hashlib,sys;print(hashlib.sha256(open("requirements.lock","rb").read()).hexdigest())')"
if [[ "$SKIP_PIP" == "1" ]]; then
  echo "  skipped (--skip-pip)"
elif [[ -f "$PIP_STAMP" && "$(cat "$PIP_STAMP")" == "$LOCK_HASH" ]]; then
  echo "  requirements.lock unchanged since the last install; skipping"
else
  grep -qE '^\s*(torch|transformers)==' requirements.lock \
    || warn "torch/transformers are still TODO in requirements.lock -- generate.py will not run yet"
  "$PYTHON" -m pip install --no-cache-dir --quiet -r requirements.lock
  mkdir -p "$(dirname "$PIP_STAMP")"
  printf '%s' "$LOCK_HASH" > "$PIP_STAMP"
  echo "  installed"
fi

# ----------------------------------------------------------------- 4. directories
say "4/7 directory layout"
"$PYTHON" storage.py --ensure | sed -n '1,12p'

# --------------------------------------------------------------- 5. COCO subset
say "5/7 COCO val2014 subset"
TAR_PATH="$("$PYTHON" -c 'import storage; print(storage.paths().coco_subset_tar)')"
if [[ ! -f "$TAR_PATH" ]]; then
  warn "no subset tar at ${TAR_PATH}"
  warn "build it once on a machine that has the full val2014 download:"
  warn "  python -m data.build_coco_subset build --val2014-dir ... --annotations-dir ... --pope-dir ..."
  warn "then copy the tar + manifest into ${PERSIST_ROOT}/archives/"
else
  EXTRACT_ARGS=()
  [[ "$FORCE_EXTRACT" == "1" ]] && EXTRACT_ARGS+=(--force)
  "$PYTHON" -m data.build_coco_subset extract "${EXTRACT_ARGS[@]}"
  if "$PYTHON" -m data.build_coco_subset verify --level "$VERIFY_LEVEL"; then
    echo "  subset verified (${VERIFY_LEVEL})"
  else
    warn "subset verification FAILED -- re-run with --force-extract"
    exit 1
  fi
fi

# ----------------------------------------------------------------- 6. checkpoint
say "6/7 model checkpoint"
if [[ "$SKIP_MODEL" == "1" ]]; then
  echo "  skipped (--skip-model)"
else
  export HF_HOME="$HF_CACHE"
  export HUGGINGFACE_HUB_CACHE="${HF_CACHE}/hub"
  mkdir -p "$HUGGINGFACE_HUB_CACHE"
  echo "  HF_HOME=${HF_HOME}  (local disk on purpose: ~14 GB, re-downloaded each session)"
  if "$PYTHON" -c 'import huggingface_hub' 2>/dev/null; then
    MODEL_ID="$MODEL_ID" "$PYTHON" - <<'PYFETCH'
import os
from huggingface_hub import snapshot_download

repo = os.environ["MODEL_ID"]
path = snapshot_download(
    repo_id=repo,
    allow_patterns=[
        "*.json", "*.model", "*.safetensors", "*.bin", "*.txt",
    ],
    ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.tflite"],
    resume_download=True,
)
total = sum(f.stat().st_size for f in __import__("pathlib").Path(path).rglob("*") if f.is_file())
print(f"  snapshot: {path}")
print(f"  size    : {total / 1e9:.1f} GB")
PYFETCH
  else
    warn "huggingface_hub is not installed; install requirements.lock first"
  fi
fi

# ------------------------------------------------------------------- 7. readiness
say "7/7 readiness report"
"$PYTHON" storage.py
echo
if "$PYTHON" - <<'PYREADY'
import sys, storage
sys.exit(0 if storage.readiness_report()["ready_for_generation"] else 1)
PYREADY
then
  echo "READY: generate.py can run."
else
  echo "NOT READY for generation (images and/or POPE files missing)."
  echo "evaluate.py / report.py / the test suite still work."
fi
