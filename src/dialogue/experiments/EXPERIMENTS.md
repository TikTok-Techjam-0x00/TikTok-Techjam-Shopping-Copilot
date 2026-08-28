# Module 3B benchmark variants

These files are independent replacements for the current production
`src/dialogue/three_b.py`. They preserve the latest public interfaces,
Top100 retrieval input, attribute extraction, and evaluator-aligned
`use_case` behavior. Only the scoring variables listed below differ.

The production baseline used to create this set is git commit `55f865b`.
Historical scoring behavior was recovered from git commit `b959324`.

| File | Profile | Turn | Base | λ | Coverage | Cardinality | Purpose |
|---|---|---|---|---:|---|---|---|
| `three_b_exp_baseline.py` | Off | Off | Current | 18 | Rank-weighted | None | Same-workflow baseline |
| `three_b_exp_profile_boost.py` | Historical +12/tag | Off | Current | 18 | Rank-weighted | None | Profile-only ablation |
| `three_b_exp_turn_boost.py` | Off | Historical +8/+5 | Current | 18 | Rank-weighted | None | Turn-only ablation |
| `three_b_exp_profile_turn.py` | Historical +12/tag | Historical +8/+5 | Current | 18 | Rank-weighted | None | Profile/turn interaction |
| `three_b_exp_legacy_base.py` | Off | Off | Historical | 18 | Rank-weighted | None | Base-only ablation |
| `three_b_exp_diversity_12.py` | Off | Off | Current | 12 | Rank-weighted | None | Conservative dynamic signal |
| `three_b_exp_diversity_24.py` | Off | Off | Current | 24 | Rank-weighted | None | Moderately stronger dynamic signal |
| `three_b_exp_diversity_38.py` | Off | Off | Current | 38 | Rank-weighted | None | Historical coefficient only |
| `three_b_exp_unweighted_coverage.py` | Off | Off | Current | 18 | Ordinary | None | Coverage-only ablation |
| `three_b_exp_cardinality_penalty.py` | Off | Off | Current | 18 | Rank-weighted | `min(1, 5/K)` | Cardinality-only ablation |
| `three_b_exp_historical_composite.py` | Historical +12/tag | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Whole historical policy reference |

## Loading

Each variant intentionally retains the production file's package-relative
imports so it can be copied over `src/dialogue/three_b.py` without editing.
For tests that leave files in this directory, load them by path under an
`src.dialogue.*` module name, as demonstrated in
`src/dialogue/test_three_b_experiments.py`.

The historical composite changes several variables at once. It is a reference
for whole-policy performance, not a causal single-variable ablation. The
answerability-prior intermediate experiment is intentionally excluded because
the repository history does not contain an `ANSWERABILITY_PRIOR` implementation.
