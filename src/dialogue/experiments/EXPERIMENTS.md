# Module 3B benchmark variants

These files are independent 3B benchmark snapshots. They preserve the latest
public interfaces, Top100 retrieval input, attribute extraction, and
evaluator-aligned `use_case` behavior. The original production policy remains
in `three_b_exp_baseline.py`; after the A/B benchmark, Historical Composite was
promoted to production `src/dialogue/three_b.py`.

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
| `three_b_exp_diversity_38_cardinality.py` | Off | Off | Current | 38 | Rank-weighted | `min(1, 5/K)` | Coefficient/cardinality interaction |
| `three_b_exp_diversity_38_legacy_base.py` | Off | Off | Historical | 38 | Rank-weighted | None | Coefficient/Base interaction |
| `three_b_exp_unweighted_coverage.py` | Off | Off | Current | 18 | Ordinary | None | Coverage-only ablation |
| `three_b_exp_cardinality_penalty.py` | Off | Off | Current | 18 | Rank-weighted | `min(1, 5/K)` | Cardinality-only ablation |
| `three_b_exp_historical_composite.py` | Historical +12/tag | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Promoted production policy; whole-policy reference |
| `three_b_exp_historical_minus_legacy_base.py` | Historical +12/tag | Historical +8/+5 | Current | 38 | Ordinary | `min(1, 5/K)` | Composite minus historical Base |
| `three_b_exp_historical_minus_profile.py` | Off | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Composite minus Profile |
| `three_b_exp_historical_minus_turn.py` | Historical +12/tag | Off | Historical | 38 | Ordinary | `min(1, 5/K)` | Composite minus Turn |
| `three_b_exp_historical_minus_diversity_38.py` | Historical +12/tag | Historical +8/+5 | Historical | 18 | Ordinary | `min(1, 5/K)` | Composite with lambda restored to 18 |
| `three_b_exp_historical_minus_ordinary_coverage.py` | Historical +12/tag | Historical +8/+5 | Historical | 38 | Rank-weighted | `min(1, 5/K)` | Composite with weighted coverage |
| `three_b_exp_historical_minus_cardinality.py` | Historical +12/tag | Historical +8/+5 | Historical | 38 | Ordinary | None | Composite minus cardinality |
| `three_b_exp_historical_override_no_profile.py` | Historical except override turn | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Pause Profile on detected override |
| `three_b_exp_historical_override_reask.py` | Historical +12/tag | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Re-open old asked attributes on override |
| `three_b_exp_historical_override_adaptive.py` | Historical except override turn | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Combined override-aware policy |
| `three_b_exp_historical_override_phase_no_profile.py` | Off after override | Historical +8/+5 | Historical | 38 | Ordinary | `min(1, 5/K)` | Persistent override-phase Profile suppression |
| `three_b_exp_scenario_adaptive.py` | Normalized semantic prior | Mode-specific Top20/Top100 | Replaced by normalized utility | N/A | Rank-weighted | Repeatability + option concentration | Composite scenario-adaptive policy |

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

## Scenario-adaptive composite

`three_b_exp_scenario_adaptive.py` is also a whole-policy experiment, not a
single-variable ablation. It routes each turn into `EXPLORE`, `CONSTRAIN`,
`BOUNDARY_RECOVER`, or `OVERRIDE_RECOVER`; combines Top20/Top100 weighted Gini
impact with coverage, repeated-value mass, top-three option concentration, and
a weak normalized semantic prior; and uses a conservatively filtered candidate
pool for buying turns. Override turns may revisit attributes from the previous
need, while `record_asked_attribute()` starts a new asked-attribute epoch.

All signals come from the current Module 2 State and Module 1 Top100 Retrieval.
The file contains no evaluator labels, target products, session-specific rules,
or public-set frequency priors. Production `src/dialogue/three_b.py` remains
unchanged.
