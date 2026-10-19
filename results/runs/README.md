# Committed run archive

Every finished job's raw JSONL, tracked in git.

## Why these are in the repository

They look like large binary-ish artefacts and are neither. 63% of every record
is provenance repeated identically on all 3000 lines, so git compresses them
about 23x: the two POPE runs are 7.7 MB on disk and 0.34 MB in `.git`. All 53
jobs come to roughly 9 MB.

What that buys:

- **A clone already contains the results.** A fresh Kaggle session needs no
  attached Dataset and no manual step for resume to see what is already done;
  `run_kaggle_fox.sh` ingests this directory at startup.
- **Provenance for free.** Git history records exactly when each result landed
  and at which commit of the code that produced it. Every record also carries
  its own `commit_hash`, so the two can be cross-checked.
- **Re-scoring without a GPU.** These are raw generations, not metrics. When an
  evaluation bug turns up -- a CHAIR synonym mismatch, a yes/no parse edge
  case, a wrong denominator -- the metric is recomputed from here in seconds.
  That has already paid for itself in other projects.

## Workflow

After a Kaggle session, download its `fox_runs_*.zip`, then:

```bash
python -m foxrepro.bench.plan ingest <zip>      # merge into the live runs dir
python -m foxrepro.bench.plan archive           # copy into results/runs/
git add results/runs && git commit -m "session N results"
```

`archive` merges by `sample_id` union, so re-archiving a partially finished job
after it completes adds the new records rather than replacing the file.

The live run directory (`foxrepro.storage.Paths.runs_root`) stays outside the
repo and gitignored: it is working state, and on Kaggle it is ephemeral.

## Two rounds of session 1 are archived, on purpose

The variant hash in each filename is derived from the fields that change the
output, so the same job under a different prompt is a different file and both
survive here. Session 1 exists twice:

| hash | prompt | POPE-adv baseline / Fox |
|---|---|---|
| `0a53e43e` (POPE), `ccf25f3f` / `2ff108f2` (CHAIR) | **r1** — system prompt joined with a space, `\n` before the question, and `"Answer the question using a single word or phrase."` appended | 81.90 / 83.33 |
| `2d07edcb` (POPE), `5c7c06df` / `4a534adb` (CHAIR) | **r2** — byte-faithful to the release (`docs/DEVIATIONS.md` §3.5) | 77.17 / 81.40 |

r1 is not a failed run to be deleted. It is the measurement of what those
three prompt differences do together, and it is the reason the paper's
POPE-adversarial baseline of 76.77 is reachable at all: r1 sat 5.13 points
above it, r2 sits 0.40 above. Quote r2 for any comparison against the paper.
