# 3B — Ask Attribute

3B has one responsibility: decide what to ask next.

```python
from src import decide_ask, record_asked_attribute

decision = decide_ask(state_from_module_2, ranking_result_from_module_3)
# {"ask_attribute": "material", "message": "Which material do you prefer: ...?"}

# 3B 本身无状态，所以主流程需要把本轮提问交还给模块 2 保存。
record_asked_attribute(state_from_module_2, decision["ask_attribute"])
```

## Inputs

- `state`: module 2's current state. `known_attributes` supports both
  `{"category": "shoes"}` and `["category", "color"]`. 3B also reads asked,
  unavailable/no-preference attributes, turn number, and optional profile tags.
- `ranking_result`: module 3's ranked candidates. Candidate metadata may be inline
  or nested under `product`, `item`, or `metadata`.

`decide_ask` never changes either input and stores no session state. After each
decision, the caller must record `ask_attribute` in module 2. The optional
`record_asked_attribute` helper performs this explicit write for mutable states.

## Decision policy

1. Exclude attributes already known, already asked, or marked no-preference.
2. Use entropy, metadata coverage, and value count to estimate which attribute best
   separates the Top 10. This is a candidate-diversity heuristic, not strict
   information gain based on simulated user answers and reranking.
3. Combine that diversity signal with turn stage and profile relevance.
4. Return one official `ask_attribute` and a deterministic question template.

The input readers accept a few common field aliases until modules 2 and 3 finalize
their contracts. Once fixed, the aliases can be replaced with exact TypedDicts
without changing the policy.
