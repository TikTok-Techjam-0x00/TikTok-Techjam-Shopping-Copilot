# Intent and Conversation State Module

Module 2 turns every user message into a deterministic, session-scoped shopping
state. It uses only the Python standard library and remains available when an
external model or another advanced pipeline component fails.

## Public interface

```python
from starter.conversation import create_state, retrieval_query, update_state

state = create_state(user_profile)
update_state(state, user_message, turn=1, asked_attribute=None)

retrieval_text = retrieval_query(state)  # input for keyword/dense retrieval
pipeline_state = state.to_dict()         # JSON-safe input for other modules
```

`Agent.get_state(session_id)` exposes the same serialized contract after
`Agent.reset()` and one or more `Agent.respond()` calls.

## State contract

Important fields include:

- `intent` and `intent_confidence`
- `category`
- `hard_constraints`
- `soft_preferences`
- `negative_constraints`
- `known_attributes`
- `asked_attributes` and `rejected_attributes`
- `turn_count` / `turn`
- `override_detected`
- `user_profile`

The aliases `known_attributes`, `asked_attributes`, `rejected_attributes`, and
`turn` are directly consumable by the current 3B dialogue-policy helper in
`src/three_b.py`.

## Override behavior

When a user changes category, the new category replaces the old one and stale
category-dependent `use_case` state is removed. Generic constraints such as
budget, size, color, and material remain unless the user replaces or rejects
them. Soft preferences are cleared on an explicit override before new
preferences are added.

## Verification

```bash
python3 -m unittest -v
python3 -m evaluator.local_evaluator --output results-stateful.json
```

The current test suite covers intent routing, slot extraction, state
accumulation, overrides, rejected/negative attributes, session isolation,
serialization, Agent integration, and evaluator compatibility.
