# v1 Integrated Pipeline / v1 集成流水线

This snapshot records the first end-to-end integration of the team modules behind the official `Agent` interface.

## Version identity / 版本标识

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

---

## 中文说明

该快照记录了团队模块首次通过官方 `Agent` 接口完成端到端集成的评测结果。

### 版本信息

- 版本：`v1_integrated_pipeline`
- 评测日期：2026-08-28
- 数据集：`data/public_set.jsonl`
- 会话数量：200
- 最大对话轮数：10
- 每轮推荐数量：Top 10
- Token 使用量：0（不依赖 LLM）
- 结果文件：`results.json`

### 使用的模块

官方 evaluator 调用 `starter/agent.py`。该文件保持为轻量兼容入口，并将 session 生命周期及每轮请求转发给 `src/pipeline/pipeline.py`。

```text
官方 Evaluator
    -> Agent 官方入口
    -> Pipeline 集成调度
    -> State 状态管理
    -> Retrieval 检索（Top 100）
    -> Reranking 重排序（Top 10）
    -> Dialogue 对话决策
    -> 官方 AgentResponse
```

- State：为不同 session 创建隔离状态，跨轮更新意图与约束，并构造基于状态的检索查询。
- Retrieval：加载商品目录，使用团队实现的 BM25 检索器生成最多 100 个共享 Candidate 对象。
- Reranking：使用团队实现的 `SimpleReranker`，将检索候选转换成请求所需的 Top K。
- Dialogue：选择官方允许的追问属性，并将其记录到 session state，避免错误地重复提问。
- Integration：由 `Pipeline` 串联所有模块，并按照官方 schema 返回 `message`、`ask_attribute`、`recommendations` 和 token 数为零的 `usage`。

### 与 v0 baseline 的总体对比

| 指标 | v0 baseline | v1 integrated | 绝对变化 | 说明 |
|---|---:|---:|---:|---|
| HitRate@10 | 0.125000 | 0.820000 | +0.695000 | 提升 69.5 个百分点 |
| MRR | 0.068034 | 0.529458 | +0.461424 | 相关商品的平均排名显著提高 |
| MTTC | 9.810 | 6.175 | -3.635 轮 | MTTC 越低越好，因此属于提升 |
| Efficiency | 0.119000 | 0.482500 | +0.363500 | 更多 session 能够更早命中目标 |
| TechnicalScore | 0.106710 | 0.665337 | +0.558627 | 总体技术分显著提高 |

相较 baseline，HitRate@10 约提升 556%，MRR 约提升 678%，TechnicalScore 约提升 524%。MTTC 约下降 37%，代表系统平均提前约 3.6 轮找到目标商品。

### 分场景对比

| 场景 | HitRate@10（v0 -> v1） | MRR（v0 -> v1） | MTTC（v0 -> v1） |
|---|---:|---:|---:|
| Boundary | 0.000000 -> 1.000000 | 0.000000 -> 0.757500 | 11.000 -> 6.200 |
| Browsing | 0.025000 -> 0.900000 | 0.004514 -> 0.580809 | 10.750 -> 4.800 |
| Buying | 0.237500 -> 0.862500 | 0.126508 -> 0.504697 | 8.625 -> 6.3125 |
| Intent Override | 0.133333 -> 0.433333 | 0.104167 -> 0.382540 | 10.066667 -> 9.466667 |

与 v0 相比，没有已报告指标出现退步：所有场景的 HitRate@10 和 MRR 均有提升，所有场景的 MTTC 均有下降。

提升最明显的是 Boundary 和 Browsing。Intent Override 同样有所改善，但仍然是 v1 中表现最弱的场景：HitRate@10 为 0.433333，MTTC 为 9.466667。后续应重点检查意图覆盖发生时，旧约束是否被正确替换，以及更新后的查询如何参与检索与重排序。

### 本地复现

在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m evaluator.local_evaluator
```

命令会将最新评测写入仓库根目录的 `results.json`。确认完整跑完 200 个 session 后，再将该文件复制到此版本目录中保存。
