#!/usr/bin/env bash
# Pack what the paper's tables need out of a Kaggle session into ONE zip:
# the run JSONL files, the per-job logs, and the calibration TSV.
#
# The run JSONL is included DELIBERATELY rather than computed metrics. It holds
# the raw generated text with full provenance, so if an evaluation bug is found
# later -- a CHAIR synonym mismatch, a yes/no parse edge case, a wrong
# denominator -- the metric is recomputed locally in seconds with no GPU. NEVER
# reduce this to metrics only.
#
# No model weights, no HF cache: those live in /kaggle/temp by design and are
# reconstructible. The zip stays in the low megabytes.
#
# Called after every finished job by `plan run`, so a cut session still leaves
# a downloadable zip of everything that completed.
#   bash scripts/pack_runs.sh [tag]   -> /kaggle/working/fox_runs_<tag>.zip
# (falls back to ./fox_runs_<tag>.zip when not on Kaggle)
set -uo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-$(date +%Y%m%d_%H%M)}"
DEST="/kaggle/working"; [ -d "$DEST" ] || DEST="."
OUT="${DEST}/fox_runs_${TAG}.zip"

# storage.py is the only module allowed to know absolute paths, so ask it.
read -r RUNS LOGS <<<"$(python3 - <<'PY'
from foxrepro.storage import paths
p = paths()
print(p.runs_root, p.logs_dir)
PY
)" || { echo "[pack] cannot resolve the storage layout"; exit 1; }

[ -d "$RUNS" ] || { echo "[pack] no runs directory yet ($RUNS); nothing to pack"; exit 0; }

command -v zip >/dev/null 2>&1 || { echo "[pack] zip not installed"; exit 1; }

rm -f "$OUT"
zip -q -j "$OUT" "$RUNS"/*.jsonl 2>/dev/null
[ -d "$LOGS" ] && zip -q -j "$OUT" "$LOGS"/*.log "$LOGS"/_job_times.tsv 2>/dev/null

if [ -f "$OUT" ]; then
  ls -lh "$OUT"
  echo "[pack] $(unzip -l "$OUT" | tail -1)"
else
  echo "[pack] nothing matched; no zip written"
fi
