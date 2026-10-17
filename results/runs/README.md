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
