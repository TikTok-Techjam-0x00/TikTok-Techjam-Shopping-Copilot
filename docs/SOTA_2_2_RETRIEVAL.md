# SOTA 2.2: lexical-gated semantic residual Retrieval

SOTA 2.2 retains routed BM25 as the primary retriever and adds one bounded
semantic exploration cohort on turn 8. It does not use ordinary score fusion:
a Dense candidate is eligible only when a deeper BM25 search independently
supports it.

## Frozen algorithm

- Product representation: `dense_needs_v1`, `text-embedding-v4`, 256 dimensions.
- Turns 1-7, 9, and 10: exact SOTA 2.1 BM25 behavior.
- Turn 8 lexical cohort: BM25 ranks 201-210.
- Turn 8 semantic cohort: the first 10 products that are both Dense Top 200 and
  BM25 ranks 301-1000.
- Maximum focused cohort: 20 candidates before the existing reranker.
- Missing credentials, missing/incompatible cache, provider error, or provider
  timeout: exact SOTA 2.1 BM25 fallback.

## Frozen end-to-end results

| Dataset | SOTA 2.1 BM25 | SOTA 2.2 | Delta | HitRate | MRR | MTTC | Gains / losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| Public-200 | 0.961150 | 0.961150 | 0.000000 | 0.995000 | 0.960833 | 2.23000 | 0 / 0 |
| IID-800 | 0.934085 | **0.935620** | **+0.001535** | 0.995000 | 0.904651 | 2.66375 | 2 / 0 |
| Long-tail-800 | 0.968592 | 0.968592 | 0.000000 | 1.000000 | 0.969975 | 2.12000 | 0 / 0 |
| Stress-800 | 0.981444 | 0.981444 | 0.000000 | 0.998750 | 0.997812 | 1.86375 | 0 / 0 |

Parameters were selected on IID-800. Long-tail-800, Stress-800, and Public-200
were held out for frozen validation. No configuration was changed after the
final validation runs.

## Leakage and generalization controls

- Production code contains no sample IDs, target ASINs, dataset branches, or
  ground-truth lookup.
- Ground truth is read only after ranking by offline evaluators.
- The three 800 sets exclude all Public-200 targets and are mutually disjoint.
- BM25 and Dense must independently support every residual candidate.
- Dense runs only for unresolved sessions reaching turn 8; the early MRR path
  is unchanged.
- Intent Override queries use the active constraint epoch, excluding stale soft
  preferences.

The sole remaining Public miss is already present at BM25 depth 284 after
clarification, making it a within-page reranking issue rather than a Retrieval
recall miss.
