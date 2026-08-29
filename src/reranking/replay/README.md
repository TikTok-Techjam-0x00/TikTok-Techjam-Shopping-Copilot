# Reranking Replay Evaluator

Replay evaluation freezes the exact input to Module 3A (`ShoppingState` and
`Candidates100`) for every public evaluator turn. Multiple Rerankers can then be
compared without rerunning or changing State, Retrieval, or Dialogue.

## 1. Record a versioned dataset

```powershell
python -m src.reranking.replay.recorder
```

Every invocation creates a new directory under
`artifacts/reranking_replay/<timestamp>_<git-commit>/`. Existing runs are never
overwritten. Use `--run-id NAME` for a human-readable identifier, and `--limit`
plus `--max-turns` for smoke tests.

Each run contains:

- `cases.jsonl.gz`: state and compact Retrieval candidates; no target labels.
- `labels.jsonl`: `case_id -> target_parent_asin`, read only after ranking.
- `manifest.json`: Git revision, branch, dirty files, diff hash, per-component
  source hashes, Catalog/public-set hashes, runtime, command, policy, and counts.

The component section separately records the most recent commit that changed
State, Retrieval, Reranking, Dialogue, Pipeline, shared item/attribute
contracts, and the official local evaluator. It also stores current file hashes,
so a run remains identifiable even when generated from an uncommitted working
tree.

The recorder intentionally continues to turn 10 after a target would have been
hit. This avoids selecting later cases based on the baseline Reranker. It is
valid in the current Pipeline because Dialogue reads `Candidates100` rather than
the reranked Top 10. Intent Override cases before the override are recorded with
`scorable=false` and excluded from hit metrics.

By default, capture records the real pre-reranking input but uses Retrieval order
for the temporary response recommendations. Those recommendations do not affect
the next turn in the current Dialogue implementation, so this makes repeated
dataset generation much faster without changing recorded State or Candidates100.
Use `--execute-runtime-reranker` for the slower delegate mode; the selected mode
is always written to `manifest.json`.

## 2. Replay experiments

```powershell
python -m src.reranking.replay.evaluator `
  artifacts/reranking_replay/<dataset-run> `
  --experiment-id RR-001 `
  --experiment s1_rule_fuzzy
```

One experiment ID evaluates exactly one Reranker configuration. IDs use
`RR-000`, `RR-001`, ... and the result folder is always
`<dataset-run>/results/<experiment-id>/`. Run the Retrieval control and S1 as
separate IDs when a paired comparison is needed.

The CLI prints progress every 100 cases by default; use `--progress-every N` to
change it or `--progress-every 0` to disable progress output.

The evaluator verifies checksums, restores each `Candidate` from the exact
recorded Catalog version, ranks all candidates for diagnostics, and writes a new
result version under `<dataset-run>/results/<experiment-id>/`:

- `report.json`: complete aggregate metrics, human-readable configuration,
  total test duration, and both the dataset-generation and evaluation Git
  versions.
- `report.md`: overall comparison plus separate case-level and session-level
  tables for `browsing`, `buying`, `boundary`, and `intent_override`.
- `case_results.jsonl.gz`: ranks, promotions, demotions, constraint quality, and
  latency for every case and experiment.

Metrics include `coverage@100`, conditional Hit@10/MRR@10, promotions,
demotions, mean exact rank change, hard-constraint violation/unknown rates,
P50/P95 latency, and replay estimates of Hit Rate, MRR, MTTC, and TechnicalScore.
The same metrics are also grouped by `scenario_type`. Case metrics show what the
Reranker changed on individual turns; session metrics show whether and how soon
the target entered Top 10 across the complete dialogue. Each scenario receives
its own efficiency and Replay TechnicalScore, using the same formula as the
overall score.

The replay estimate is a reranking-only diagnostic. A selected configuration
must still be verified with `python -m evaluator.local_evaluator`.

## 3. Verify an equivalent optimization

S1-fast caches immutable Catalog product text, normalized tokens, and repeated
fuzzy comparisons. It does not change the S1 formula or fusion weights. Before
accepting a performance-only experiment, compare every observable rank output:

```powershell
python -m src.reranking.replay.equivalence `
  artifacts/reranking_replay/<dataset-run> `
  --output artifacts/reranking_replay/<dataset-run>/equivalence/RR-005.json
```

The checker compares all ranks, `parent_asin`, rounded rerank scores, matched
attributes, and violations between uncached S1 and S1-fast. It fails on the
first difference and records cache sizes, hit/miss counts, timings, input
checksums, and Git provenance when the comparison succeeds.
