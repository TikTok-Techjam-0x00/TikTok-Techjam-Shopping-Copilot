# SOTA 2.1: deterministic BM25 routing

SOTA 2.1 starts from SOTA 2.0 commit `6b5cad0` and changes only the default
Retrieval route and its internal candidate-pool configurability. The production
factory no longer selects Dense/Hybrid from environment configuration; Hybrid
factories remain available for isolated experiments.

## Frozen configuration

- Candidate pool: 100. Larger internal pools were rejected.
- Buying and later Browsing turns: `all_fields_v4` BM25, with Store weight 0.75
  and Description weight 0.5 to reduce low-signal matches.
- Browsing turn 1: title/category BM25 with weights 5.0/6.0, favoring the
  catalog category hierarchy for broad warm-start queries.
- Browsing returns to the detailed Buying route after turn 1.

## Candidate sweep

Each parameter was evaluated once on the same 800-session development set. Its
800 target ASINs have zero overlap with the 200 official public targets.

| Internal pool | Hit@10 | MRR | MTTC | Technical Score |
| ---: | ---: | ---: | ---: | ---: |
| **100** | **0.992500** | **0.901441** | **2.713750** | **0.932407** |
| 130 | 0.975000 | 0.894248 | 2.751250 | 0.920749 |
| 150 | 0.977500 | 0.896279 | 2.741250 | 0.922809 |
| 170 | 0.978750 | 0.897529 | 2.740000 | 0.923834 |
| 200 | 0.978750 | 0.897529 | 2.740000 | 0.923834 |

Larger pools underperformed the original paged/stratified Top-100 exploration
policy. The 170 and 200 results were identical, indicating saturation.

## Frozen final result

| Dataset | SOTA 2.0 BM25 | SOTA 2.1 | Delta |
| --- | ---: | ---: | ---: |
| 800-session development set | 0.932407 | **0.932620** | +0.000213 |
| Official public 200 | 0.961050 | **0.961150** | +0.000100 |

Hit@10 was unchanged on both datasets. On the official public set, MTTC moved
from 2.235 to 2.230. The primary benefit is deterministic BM25-only behavior
without embedding-service or API variability; the score gain is intentionally
reported as small.

No evaluator, dataset, target lookup, sample ID, target ASIN, or target-specific
production rule was changed or introduced.
