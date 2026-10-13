"""Copy-paste cells for a Kaggle notebook. Each CELL block is one notebook cell.

Not meant to be run as a script -- it is a reference you paste from, so the
cell boundaries stay visible and you can stop between steps.

Before anything: notebook sidebar -> Accelerator = **GPU T4 x2**. One T4 cannot
hold LLaVA-1.5-7B in fp16 with headroom (13.16 GiB of weights against ~14.3 GiB
usable); two can, sharded. Quantization is ruled out for this project because
the mechanism under study *is* a numerical-precision effect, so sharding is the
only way a 16 GB card participates.

Session budget: 12 h per session, 30 GPU-h per week. Every generate command
resumes, so a session dying mid-run costs nothing but the current sample.
"""

CELL_1_CLONE = r"""
# --- CELL 1: clone and install -------------------------------------------
# Replace the URL with your repo. Kaggle has no persistent volume between
# sessions, so this runs every session (~20 s).
!git clone -q https://github.com/<you>/<your-repo>.git /kaggle/working/fox
%cd /kaggle/working/fox
!pip install -q -e .

# Editable install puts `foxrepro` on sys.path without copying anything, so
# edits you make in the notebook's file editor take effect immediately.
"""

CELL_2_PROBE = r"""
# --- CELL 2: probe the host BEFORE downloading 14 GB ---------------------
# Answers four things that cannot be determined off-host:
#   1. Kaggle's Python version (Track A needs <= 3.11)
#   2. whether bf16 cast+add works on sm_75 (T4 has no bf16 *compute*)
#   3. whether both GPUs are visible and the model will fit sharded
#   4. the actual fp16 ULP at each candidate gamma -- the mechanism itself
!python scripts/probe_host.py
"""

CELL_3_SETUP = r"""
# --- CELL 3: bootstrap ---------------------------------------------------
# Idempotent. Re-run after any disconnect.
!bash kaggle_setup.sh --track-b
"""

CELL_4_SELFTEST = r"""
# --- CELL 4: prove the port is bit-identical to the released code --------
# This is the most important check in the repo and it needs no GPU, no
# checkpoint and no COCO download: the method has no learned parameters, so
# `foxrepro.fox.kernel` is asserted bit-equal to the verbatim transcription in
# `foxrepro.fox.reference` on random tensors across fp32 / fp16 / bf16.
#
# If this passes, every other experiment arm is a *deliberate* deviation
# measured against a trusted baseline. If it fails, stop -- nothing downstream
# is interpretable.
!python -m unittest discover -s tests -t . -v 2>&1 | tail -25
"""

CELL_5_DATA = r"""
# --- CELL 5: data --------------------------------------------------------
# POPE annotations ship inside the package (3000 questions x 3 splits, 500
# images, balanced 1500 yes / 1500 no), so POPE needs IMAGES ONLY.
#
# COCO serves images individually, so skip the 6.2 GB val2014 zip entirely and
# fetch exactly the 500 POPE references -- about 100 MB, a couple of minutes.
# Resumable and idempotent: re-run it after any disconnect and it skips what is
# already on disk. Truncated files are detected and refetched, which matters
# because a half-downloaded JPEG has a valid header and PIL will open it.
import os
os.environ["FOX_ENV"] = "kaggle"
!python -m foxrepro.storage --ensure
!python -m foxrepro.data.fetch_images --what pope --workers 8

# For CHAIR you additionally need the ground-truth annotations (~241 MB zip,
# the only way those two json files are distributed) and a CHAIR image list:
#   !python -m foxrepro.data.fetch_images --what union --with-annotations
#
# Once a session has fetched them, upload /kaggle/temp/fox-local/data as a
# private Kaggle Dataset and attach it read-only; later sessions then cost
# nothing. data/build_coco_subset.py builds a checksummed tar for exactly that.
"""

CELL_6_SMOKE = r"""
# --- CELL 6: pipeline smoke test, no GPU --------------------------------
# The `echo` method is a CPU test double: it exercises loading, the prompt
# template, JSONL writing, resume and the metric path without touching a
# model. Run this before burning GPU time.
!python -m foxrepro.bench.generate --method echo --benchmark pope \
    --split random --dry-run --limit 20 --progress-every 5
!python -m foxrepro.bench.evaluate --all
!python -m foxrepro.bench.report
"""

CELL_7_BASELINE = r"""
# --- CELL 7: the first real run -----------------------------------------
# Plain nucleus sampling, no intervention. This is the baseline every table is
# measured against, and it is also the run that proves the sharded 2x T4 load
# works end to end.
#
# ~1 h for all 3 POPE splits on T4x2 (prefill-dominated: 600-token prefill,
# 1-token answer). If the session dies, re-run the identical command.
!python -m foxrepro.bench.generate \
    --method sampling --benchmark pope --split random \
    --dtype fp16 --device-map auto --seed 42
"""

CELL_8_FOX = r"""
# --- CELL 8: Fox, as released -------------------------------------------
# Released hyperparameters, NOT the paper's -- they disagree, and the code is
# what produced the released behaviour:
#     layers 2-15 (paper says 2-13), k=0.35 (paper 0.45),
#     tau_js=0.15 (paper 0.2), gamma=10000 (paper never states it)
!python -m foxrepro.bench.generate \
    --method fox --benchmark pope --split adversarial \
    --mode as_implemented --gamma 10000 --k 0.35 --tau-js 0.15 \
    --alpha 2 --beta 0.1 --intervention-layers 2-15 \
    --dtype fp16 --logit-dtype native --device-map auto --seed 42
"""

CELL_9_THE_EXPERIMENT = r"""
# --- CELL 9: the gamma sweep -- the decisive experiment -----------------
# The claim under test: Fox's "causal intervention" is a uniform shift across
# all key positions, which a numerically stable fp32 softmax cancels exactly.
# What survives is the cast back to fp16, which snaps the logit row onto that
# dtype's ULP grid. If that is the mechanism, the response should track the
# ULP, not gamma itself:
#
#     gamma=1e2 -> ULP 0.0625      gamma=1e4 -> ULP 8   (the released value)
#     gamma=1e3 -> ULP 0.5         gamma=6e4 -> ULP 32
#
# Four orders of magnitude, entirely inside fp16, so a T4 can run all of it.
# gamma=1e5 is excluded: fp16 overflows at 65504.
for gamma in (1e2, 1e3, 1e4, 6e4):
    !python -m foxrepro.bench.generate \
        --method fox --benchmark chair --split val2014_500 \
        --mode as_implemented --gamma {gamma} --k 0.35 --tau-js 0.15 \
        --dtype fp16 --logit-dtype native --device-map auto --seed 42
"""

CELL_10_CONTROLS = r"""
# --- CELL 10: the three controls the paper never ran --------------------
# Each is one flag. Run them on CHAIR, where the headline gain lives.
#
#   off            fusion only, intervention disabled. Separates the additive
#                  logit fusion (z_obs + 2*z_do) from the intervention.
#   ulp_control    replaces the shift+cast with explicit quantization. If this
#                  reproduces the gain, the mechanism is requantization, full
#                  stop, and the causal framing adds nothing.
#   noise_control  matched-magnitude uniform noise, no grid structure. If this
#                  ALSO reproduces the gain, the method is noise injection and
#                  the head-selection machinery is decorative.
#   as_described   the bias restricted to image-token keys, as Appendix B
#                  describes rather than as the code does. If this works on its
#                  own, the described mechanism stands and the finding is
#                  "code != paper" rather than "paper is wrong".
for mode in ("off", "ulp_control", "noise_control", "as_described"):
    !python -m foxrepro.bench.generate \
        --method fox --benchmark chair --split val2014_500 \
        --mode {mode} --gamma 10000 --k 0.35 --tau-js 0.15 \
        --dtype fp16 --device-map auto --seed 42
"""

CELL_11_REPORT = r"""
# --- CELL 11: evaluate and report ---------------------------------------
# CPU only. Re-runnable without touching the GPU, which is the whole point of
# keeping generation and evaluation in separate programs: fixing a metric bug
# never costs a GPU hour.
#
# Every table carries Wilson 95% intervals and flags gaps that are NOT
# statistically resolvable. For reference, at the paper's own sample sizes:
#   POPE  n=3000 -> +/-1.37 points
#   CHAIR n=500  -> +/-4.35 points on CHAIR_S
# So several gaps the paper reports between Fox / OPERA / CausalMM are inside
# sampling noise, and the paper reports no variance at all despite stating a
# two-sided t-test protocol in Appendix C.2.
!python -m foxrepro.bench.evaluate --all
!python -m foxrepro.bench.report | tee /kaggle/working/report.md
"""

CELL_12_SAVE = r"""
# --- CELL 12: keep the results ------------------------------------------
# /kaggle/working persists as the notebook's committed output, but it is capped
# at 20 GB -- which is why the HF cache lives in /kaggle/temp instead. Run
# JSONL is a few MB and is the irreplaceable artefact, so check it is there
# before the session ends.
!ls -la /kaggle/working/fox-repro/runs/ | head -20
!du -sh /kaggle/working/fox-repro
"""

if __name__ == "__main__":
    import re as _re
    cells = [(k, v) for k, v in sorted(globals().items()) if k.startswith("CELL_")]
    cells.sort(key=lambda kv: int(_re.match(r"CELL_(\d+)", kv[0]).group(1)))
    for name, body in cells:
        print(f"{'=' * 72}\n# {name}\n{'=' * 72}{body}")
