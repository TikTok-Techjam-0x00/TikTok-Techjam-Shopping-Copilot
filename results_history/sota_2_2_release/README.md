# SOTA 2.2 release validation — 2026-08-31

## Provenance and scope

- Source repository: `TikTok-Techjam-0x00/shopping-copilot`.
- Source branch: `sota-2.2`.
- Source commit: `d5ac6e93d7b86986514a6f3ff85473f5e6b6ee92`.
- Final repository: `TikTok-Techjam-0x00/TikTok-Techjam-Shopping-Copilot`, `main`.
- The release commit containing this record preserves the source history.
- `src/`, `starter/`, and `evaluator/` are unchanged from the source commit.
  No ranking parameters, recommendation-width policies, labels, or evaluation
  logic were changed during release preparation.
- Packaging changes: project/data documentation, local catalog validation
  utility and five tests, ignored local archives/environment files, and this
  versioned validation record.

The root `catalog.jsonl.gz` and `results.json` were removed from the release
snapshot's tracked files, not erased from the local workspace or Git history.
The archive is obtained from the official participant release; the verified
result is retained here. Existing LFS caches and team experiment history are
preserved, including the four historical vector matrices.

## Runtime conditions

- Windows, CPython 3.11.4; new `.venv` installed from `requirements.txt` and
  `frontend/requirements.txt`.
- `DASHSCOPE_API_KEY` was explicitly set to an empty string inside each test
  process, before imports. No saved `.env` was copied into the release directory.
- The original SOTA 2.2 Agent entry point ran its **no-key BM25 fallback**;
  automatic Qwen State resolution was disabled. No model service was invoked.
- LFS matrices were present and checked, but Dense retrieval was not enabled.
- The optional API-enabled path is preserved, not revalidated against a live
  provider. It can still activate when credentials and compatible cache exist.

## Results

Core discovery: **244 tests passed in 7.628 s**. Frontend discovery:
**9 tests passed in 1.274 s**. `pip check` found no broken requirements.
Catalog preparation verified both frozen hashes; Git LFS object/pointer
validation passed for all four matrices.

Public evaluator: **200 sessions**, **199 hits**, TechnicalScore **0.961150**,
HitRate@10 **0.995000**, MRR **0.960833**, MTTC **2.23000**.

| Scenario | Sessions | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Browsing | 80 | 1.000000 | 0.963542 | 2.037500 |
| Buying | 80 | 0.987500 | 0.965625 | 1.875000 |
| Intent override | 30 | 1.000000 | 0.927778 | 3.633333 |
| Boundary | 10 | 1.000000 | 1.000000 | 2.400000 |

Measured wall time was **50.936 s**, covering imports, catalog/index loading,
evaluation, and result serialization. This is one local run, not a controlled
latency benchmark; frontend tests briefly ran concurrently. The full output
is [`results.json`](results.json), byte-identical to the source commit's frozen
root `results.json`. Reported model-token counters are zero. API-enabled runs
must separately account for embedding and State usage; the Agent's counters
do not currently aggregate them.

The three generated 800-session sets were not rerun for this packaging-only
change. Their existing historical results are documented in
[`docs/SOTA_2_2_RETRIEVAL.md`](../../docs/SOTA_2_2_RETRIEVAL.md). Matching the
public result does not establish hidden-test generalization.

## Reproduction

After installing dependencies and preparing `data/catalog.jsonl`, run from
the repository root with the environment activated:

```bash
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('unittest', run_name='__main__')" discover -s . -p "test*.py"
python -c "import os, runpy; os.environ['DASHSCOPE_API_KEY']=''; runpy.run_module('unittest', run_name='__main__')" discover -s frontend -p "test*.py"
python -c "from pathlib import Path; Path('artifacts/release-validation').mkdir(parents=True, exist_ok=True)"
python -c "import os, runpy, time; os.environ['DASHSCOPE_API_KEY']=''; started=time.perf_counter(); runpy.run_module('evaluator.local_evaluator', run_name='__main__'); print('RELEASE_WALL_SECONDS', round(time.perf_counter()-started, 3))" --output artifacts/release-validation/public200.json
```

[`requirements.verified.txt`](requirements.verified.txt) records the exact
Python packages in this Windows/Python-3.11 validation environment. It is a
verification snapshot, not a cross-platform dependency lock.

## SHA256 identity

| Artifact | SHA256 |
|---|---|
| Official `catalog.jsonl.gz` | `07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8` |
| Prepared `data/catalog.jsonl` | `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67` |
| `data/public_set.jsonl` | `571359a8a69014c43fc30d39c996c4a28e875dccc249dffc707358757beb16c0` |
| This directory's `results.json` | `5e7a225f70d18572a8bd7b3f5d0d57284a14451d51555e74d284ed7505f5d07d` |

The hashes identify the files used in this validation run. Text checkout
line-ending conversion can change byte hashes without changing JSON contents.
