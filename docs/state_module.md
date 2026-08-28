# Intent and Conversation State Module

Module 2 turns every user message into a deterministic, session-scoped shopping
state. It uses only the Python standard library and remains available when an
external model or another advanced pipeline component fails.

## Public interface

```python
from src.state import create_state, retrieval_query, update_state

state = create_state(session_id, user_profile)
update_state(state, user_message, turn=1, asked_attribute=None)

retrieval_text = retrieval_query(state)  # input for keyword/dense retrieval
pipeline_state = state.to_dict()         # JSON-safe input for other modules
```

`Agent.get_state(session_id)` exposes the same serialized contract after
`Agent.reset()` and one or more `Agent.respond()` calls.

## State contract

Important fields include:

- `intent` and `intent_confidence`
- `hard_constraint: AttributeMap`
- `soft_constraint: AttributeMap`
- `rejected_values: AttributeMap`
- `no_prefernce: list[AttributeName]`
- `asked_attributes`
- `session_id`, `user_message`, and `turn`
- `override_detected`
- `intent_transitions`
- `boundary_detected` and `boundary_attributes`
- `user_profile`

The object directly implements the protocols consumed by `src/retrieval`,
`src/reranking`, and `src/dialogue`. Compatibility imports remain in `starter/`
for callers that used the earlier module locations.

## Rule-first semantic fallback

The state manager always runs deterministic rules first. Rules own explicit,
high-confidence facts such as intent keywords, category phrases, budget, color,
material, size, use case, negative values, no-preference replies, and explicit
overrides.

An optional semantic resolver is considered only when the rule result indicates
one or more of these conditions:

- low intent confidence with no extracted attributes;
- an override that rules could not resolve;
- a pronoun or reference that depends on earlier context;
- an unresolved comparison such as `more formal` or `similar to the last one`;
- an unresolved alternative.

```python
from src.state import CallableSemanticResolver, update_state

resolver = CallableSemanticResolver(call_structured_llm)
update_state(
    state,
    user_message,
    turn=turn,
    semantic_resolver=resolver,
)
```

The provider callback receives a compact `SemanticRequest` and must return a
strict mapping with optional `intent`, `hard_constraint`, `soft_constraint`,
`no_preference`, `rejected_values`, `override`, and `confidence` fields. Its
output is normalized through the shared `AttributeMap` contract.

Rule-extracted values take precedence when rule and semantic outputs disagree;
semantic output may fill missing attributes. If the resolver is absent, fails,
times out, or returns invalid output, the rule result is still committed and the
pipeline continues. `semantic_fallback_used`, `semantic_fallback_count`, and
`semantic_fallback_reasons` provide runtime diagnostics.

## Override behavior

When a user changes category, the new category replaces the old one and stale
category-dependent `use_case` state is removed. Generic constraints such as
budget, size, color, and material remain unless the user replaces or rejects
them. Soft preferences are cleared on an explicit override before new
preferences are added.

An intent change is also an override even without an explicit correction word.
A transition from browsing to a concrete buying request replaces the exploratory
context. A transition from buying back to browsing removes the old
purchase-specific hard constraints before applying the new browsing target.

## Boundary behavior

When a clarification answer says that an attribute does not matter, the state
records it in `no_prefernce` and `boundary_attributes`, removes any existing
hard/soft value for that attribute, and exposes `boundary_detected` for
diagnostics. The Dialogue module reads `no_prefernce`, so it will not ask the
same attribute again.

Supported English boundary forms include `no preference`, `any color is fine`,
`it doesn't matter`, `I don't care`, `I do not want to consider size`, and
`use your judgment`. A concrete rejection such as `not leather` remains a
`rejected_values` constraint rather than being treated as a boundary.

## Verification

```bash
python3 -m unittest -v
python3 -m evaluator.local_evaluator --output results-attribute-map.json
```

The current test suite covers intent routing, slot extraction, state
accumulation, overrides, rejected/negative attributes, session isolation,
serialization, Agent integration, and evaluator compatibility.
