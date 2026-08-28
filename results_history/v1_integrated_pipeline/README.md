# v1 Integrated Pipeline

This snapshot records the first end-to-end integration of the team modules behind the official `Agent` interface.

## Version identity

- Version: `v1_integrated_pipeline`
- Evaluation date: 2026-08-28
- Dataset: `data/public_set.jsonl`
- Sessions: 200
- Maximum turns: 10
- Recommendations per turn: Top 10
- Token usage: 0 (no LLM dependency)
- Result file: `results.json`

## Components used

The official evaluator calls `starter/agent.py`, which is kept as a thin compatibility adapter. It forwards session lifecycle and turn requests to `src/pipeline/pipeline.py`.

The integrated execution order is:

```text
Official Evaluator
    -> Agent
    -> Pipeline
    -> State
    -> Retrieval (Top 100)
    -> Reranking (Top 10)
    -> Dialogue decision
    -> Official AgentResponse
```

The version uses:

- State: creates isolated session state, updates intent and constraints across turns, and builds the state-aware retrieval query.
- Retrieval: loads the catalog and uses the team's BM25 retriever to produce up to 100 shared candidate objects.
- Reranking: applies the team's `SimpleReranker` to convert the retrieved candidates into the requested Top K.
- Dialogue: selects a permitted clarification attribute and records it in session state so questions are not repeated incorrectly.
- Integration: `Pipeline` coordinates the modules and returns `message`, `ask_attribute`, `recommendations`, and zero-token `usage` in the official schema.

## Overall comparison with v0 baseline

| Metric | v0 baseline | v1 integrated | Absolute change | Interpretation |
|---|---:|---:|---:|---|
| HitRate@10 | 0.125000 | 0.820000 | +0.695000 | Improved by 69.5 percentage points |
| MRR | 0.068034 | 0.529458 | +0.461424 | Relevant products rank much higher |
| MTTC | 9.810 | 6.175 | -3.635 turns | Improved because lower is better |
| Efficiency | 0.119000 | 0.482500 | +0.363500 | More sessions succeed earlier |
| TechnicalScore | 0.106710 | 0.665337 | +0.558627 | Large overall improvement |

Relative to the baseline, HitRate@10 increased by about 556%, MRR by about 678%, and TechnicalScore by about 524%. MTTC decreased by about 37%, meaning the target is found roughly 3.6 turns earlier on average.

## Scenario comparison

| Scenario | HitRate@10 (v0 -> v1) | MRR (v0 -> v1) | MTTC (v0 -> v1) |
|---|---:|---:|---:|
| Boundary | 0.000000 -> 1.000000 | 0.000000 -> 0.757500 | 11.000 -> 6.200 |
| Browsing | 0.025000 -> 0.900000 | 0.004514 -> 0.580809 | 10.750 -> 4.800 |
| Buying | 0.237500 -> 0.862500 | 0.126508 -> 0.504697 | 8.625 -> 6.3125 |
| Intent Override | 0.133333 -> 0.433333 | 0.104167 -> 0.382540 | 10.066667 -> 9.466667 |

No reported metric regressed against v0: every scenario improved in HitRate@10 and MRR, while every scenario reduced MTTC.

The largest gains are in Boundary and Browsing. Intent Override also improves, but remains the weakest v1 scenario: its HitRate@10 is 0.433333 and its MTTC is 9.466667. Future work should focus on how overridden constraints replace stale state and how the revised query is retrieved and reranked.

## Reproduction

From the repository root, run:

```powershell
.\.venv\Scripts\python.exe -m evaluator.local_evaluator
```

The command writes the latest run to the root `results.json`. Copy that output into this version directory only after verifying that all 200 sessions completed successfully.
