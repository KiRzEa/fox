# runs_inbox

Drop the zips downloaded from each Kaggle session here, then run:

```bash
python -m foxrepro.bench.plan ingest
```

With no arguments `ingest` reads this folder, expands every `*.zip`, and merges
the run JSONL into the runs root so `evaluate` and `report` see every session's
work at once. It is idempotent: re-ingesting the same zip adds nothing.

Partial files for the same job -- which happens when a plan edit moves a job
between sessions -- are merged by `sample_id` union rather than one session's
file overwriting the other's. Both halves are real GPU time.

The zips themselves are gitignored; the merged JSONL in the runs root is the
artefact worth keeping, and it is irreplaceable, so back it up deliberately.
