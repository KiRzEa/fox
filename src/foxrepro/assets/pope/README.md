# POPE annotation files (git-tracked)

The three COCO-POPE files belong **in this directory**, under the names the Fox
reference repository uses:

```
data/pope/coco_random.jsonl
data/pope/coco_popular.jsonl
data/pope/coco_adversarial.jsonl
```

They are a few hundred kB each, so they are kept in git rather than in the
image archive: the CPU-side evaluation then works on any laptop with no Drive
and no tar.

### Where they come from

They ship inside the Fox reference repo at `pope/coco/coco_{split}.jsonl`.
(The repo's own scripts point at the authors' absolute paths
`/home/kdzlys/Fox/pope/coco/*`; the files themselves are in the tree.)
The identical files are also in the official POPE release at
`POPE/output/coco/coco_pope_{split}.json` --- `storage.paths().pope_ann_dir`
and `data.loaders.pope_file_for_split` accept either naming.

Ignore the `426` / `427` keyed variants and `popo_err.jsonl`: empty / unused.

### Format

One JSON object per line:

```json
{"question_id": 1, "image": "COCO_val2014_000000310196.jpg",
 "text": "Is there a snowboard in the image?", "label": "yes"}
```

500 val2014 images x 6 questions = **3000 items per split**. `data/loaders.py`
sorts by `question_id` and derives `sample_id = pope/<split>/q<question_id>`.

### Checking what you dropped in

```bash
python -c "
from pathlib import Path; import sys; sys.path.insert(0,'.')
from storage import paths
from data.loaders import load_pope, describe_split
for s in ('random','popular','adversarial'):
    print(s, describe_split(load_pope(paths().pope_ann_dir, paths().coco_images_dir, s, require_images=False)))
"
```
