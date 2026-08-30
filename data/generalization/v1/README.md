# Frozen generalization sets v1

These three 800-session sets are deterministic, mutually target-disjoint, and
exclude every target in `data/public_set.jsonl`. They are evaluation data, not
training examples or runtime lookup tables.

| File | Purpose | Samples |
|---|---|---:|
| `iid_800.jsonl` | Random catalog sample after public-target exclusion | 800 |
| `long_tail_800.jsonl` | Category-balanced low-popularity products | 800 |
| `stress_800.jsonl` | Numeric, dimensional, width, and price-like language | 800 |

Every set contains 320 Browsing, 320 Buying, 120 Intent Override, and 40
Boundary sessions. Target overlap is zero between all pairs. The SHA-256 values
are recorded in `SHA256SUMS`.

Regenerate the files from the frozen catalog and public set with:

```bash
python evaluator/generate_generalization_sets.py \
  --catalog data/catalog.jsonl \
  --public-set data/public_set.jsonl \
  --output-dir data/generalization/v1 \
  --source-git "$(git rev-parse HEAD)"
```

Ground truth is permitted only in offline metric calculation. It must never be
passed into State, Retrieval, Reranking, Dialogue, or a production feature.
