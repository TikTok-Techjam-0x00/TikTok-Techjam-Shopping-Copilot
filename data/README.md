# Competition Data

These files support local development and evaluation. Production Agent code
must use only the catalog and observable session inputs, never target labels
or hidden evaluator fields.

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` and `SHA256SUMS` from the
[official Participant Kit release](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participantkit),
not from an assumed Release on this team's repository. Verify the archive
against the published checksum, then run from the repository root:

```bash
python scripts/prepare_catalog.py --archive /path/to/catalog.jsonl.gz
```

The script only reads a local archive: it does not download data. Without
`--archive`, it looks for `catalog.jsonl.gz` in the repository root. It writes
`data/catalog.jsonl` after validating the frozen catalog hash. Expected row
count: 50,000. The extracted catalog is ignored by Git; the full upstream
Amazon Reviews dataset is not required.

## `generalization/v1/`

Three generated 800-session development sets are included: IID, long-tail,
and stress. They exclude Public-200 targets and are mutually target-disjoint.
They are not the organizer's private 800-session final test set. See the
[generation manifest and data notes](generalization/v1/README.md) for their
construction and checksums.

Run a set with the official evaluator's `--dataset` option and save results
outside this directory, for example under `artifacts/release-validation/`.
See the [project README](../README.md) for installation, runtime-mode controls,
and complete evaluation commands.

Never place API keys, organizer-private evaluation data, or generated result
files in this directory. See [DATA_ATTRIBUTION.md](../DATA_ATTRIBUTION.md)
before using or redistributing the data.
