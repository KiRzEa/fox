#!/usr/bin/env bash
# Kaggle (GPU T4 x2, Internet on): one session's share of the Fox reproduction.
# Sessions are disjoint and cost-balanced, so several can run concurrently --
# on separate accounts, or one after another on the same one.
#
#   local:   python3 -m foxrepro.bench.plan plan --sessions 9
#            python3 -m foxrepro.bench.plan plan --sessions 9 \
#                    --calibrate <logs>/_job_times.tsv --round r2   (after session 1)
#   Kaggle:  bash run_kaggle_fox.sh <N>        N = session number (1..9)
#            DRY=1 bash run_kaggle_fox.sh <N>  print the commands only
#            BUDGET=10 bash run_kaggle_fox.sh <N>     hours, default from the plan
#
# Run files are re-packed to /kaggle/working/fox_runs_<tag>.zip after EVERY
# finished job, so a cut session loses at most the job in flight. Download each
# zip into runs_inbox/ locally, then merge them with one command:
#   python3 -m foxrepro.bench.plan ingest
#   python3 -m foxrepro.bench.evaluate --all && python3 -m foxrepro.bench.report
#
# Resumable: relaunch the identical command. Completed samples are skipped, and
# a job the previous session left half-done is continued, not restarted,
# because resume is keyed on the run file rather than on the session.
set -uo pipefail
cd "$(dirname "$0")"

N="${1:?usage: bash run_kaggle_fox.sh <session number>}"
export FOX_ENV="${FOX_ENV:-kaggle}"

if [ "${DRY:-0}" = "1" ]; then
  python3 -m foxrepro.bench.plan run --session "$N" --dry-run
  exit 0
fi

pip install -q -e . || { echo "editable install failed"; exit 1; }

# Fail before 87 GPU-hours of work discovers it: the checkpoint must be the
# HF-converted repo. The original liuhaotian/llava-v1.5-7b loads with a
# randomly initialised vision tower and produces fluent captions unrelated to
# the image, which no metric downstream would catch.
python3 scripts/probe_host.py || echo "[warn] probe reported problems; read them"

python3 - <<'PY'
import sys
from foxrepro.storage import paths, readiness_report
p = paths()
c = readiness_report(p)["checks"]
if not c["pope_files"]:
    sys.exit("POPE annotations missing from the package -- reinstall")
if not p.chair_image_list.is_file():
    sys.exit("the CHAIR image list is missing -- reinstall; it ships in assets/")
PY

# --what auto reads the frozen plan and fetches exactly what THIS session's
# jobs need. Hardcoding --what pope here is what made every CHAIR job in
# session 1 die within seconds: the data step and the run step disagreed.
python3 -m foxrepro.data.fetch_images --what auto --session "$N" \
    --workers 8 --progress-every 100

# Kaggle gives a session no memory of the last one: /kaggle/working starts
# empty, so resume would see no run files and redo work already paid for.
# Attach previous sessions' result zips as a Dataset and they are picked up
# here. Without this, re-running session 1 to finish its CHAIR jobs would
# also redo its two POPE jobs -- 3.5 GPU-hours for nothing.
if [ -d /kaggle/input ]; then
  python3 -m foxrepro.bench.plan ingest /kaggle/input || true
fi

python3 -m foxrepro.bench.plan run --session "$N" --budget-hours "${BUDGET:-10}"
RC=$?

bash scripts/pack_runs.sh "$(python3 -m foxrepro.bench.plan run --session "$N" --tag)"
exit $RC
