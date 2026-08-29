# Retrieval 实验登记表

| 编号 | 状态 | 方法 / 唯一主变量 | 口径 | Metric@10 | Metric@50 | Metric@100 | 结论 |
|---|---|---|---|---:|---:|---:|---|
| RET-000 | 完成 | BM25 `all_fields_v4` | 严格第一轮 | 0.185 | 0.380 | 0.525 | 当前生产基线 |
| RET-001 | 完成 | BM25文本字段消融 | 严格第一轮 | 见README | 见README | 最高0.525 | 保持all_fields默认 |
| RET-002 | 完成 | Dense `all_fields_v4` symmetric | 严格第一轮 | 0.160 | 0.290 | 0.385 | 低于BM25 |
| RET-003 | 完成 | Dense Query Instruction | 严格第一轮 | 0.180 | 0.360 | 0.440 | 后续Dense默认查询模式 |
| RET-004 | 完成 | BM25 + Dense instruction | 严格RRF | 0.225 | 0.390 | 0.525 | Top100追平BM25 |
| RET-005 | 完成 | `dense_attributes_v2`标签对照 | 严格Dense | 0.135 | 0.335 | 0.475 | 标签优于无标签 |
| RET-006 | 完成 | BM25 + attributes Dense | 严格RRF | 0.195 | 0.430 | 0.535 | Top100提升、Top10回退 |
| RET-007 | 计划 | identity/needs多向量 | 严格第一轮 | — | — | — | 比较max与加权融合 |
| RET-008 | 计划 | RRF/Weighted/Source-K消融 | 严格第一轮 | — | — | — | 搜索最佳融合参数 |
| RET-009A | 失败 / 2026-08-29 | Browsing全程`title_category_v1` | 最终累计Session HitRate | 0.570 | 0.755 | 0.835 | Browsing后续轮次无法消费新属性，不进入基线 |
| RET-009B | 完成 / 2026-08-29 | Browsing仅首轮短文本，之后`all_fields_v4` | 最终累计Session HitRate | 0.795 | 0.920 | 0.965 | 可靠性门槛全通过，升级为新基线 |
| RET-010 | 完成 / 2026-08-29 | BM25完整多轮基线 | 最终累计Session HitRate | 0.790 | 0.920 | 0.965 | Top10命中即停；结果见`artifacts/retrieval/multiturn_bm25_v1.json` |
| RET-011 | 计划 | Metadata filtering/boosting | 严格第一轮/多轮 | — | — | — | 优先soft boost |
| RET-012 | 计划 | 最佳方案完整多轮确认 | 严格逐轮 + 累计 | — | — | — | 交给完整Pipeline复测 |

详细历史分析仍保留在上层 `README.md`。新实验完成后必须补充Git commit、完整配置、
耗时、结果JSON路径和失败分析；不能只登记最好的一项分数。

RET-010使用200个Session、最多10轮、`all_fields_v4` BM25。Top10首次命中158个
Session，平均首次Top10命中轮次4.392，按未命中记第11轮的MTTC为5.780。第一轮
严格Recall只计算170个已进入有效目标阶段的Session，分别为0.1176/0.3353/0.4941；
30个Intent Override Session在override前被正确排除。最终累计@50/@100是Retrieval
覆盖诊断，不是比赛的最终推荐HitRate；比赛停止条件仍为Top10。

RET-009B与RET-010使用相同200 Session严格对照。Top10命中由158增至159，MTTC
由5.780降至5.600；Top100覆盖保持193/200。Buying和Intent Override完全不变；
Browsing命中数不变但Top10 MTTC由4.1875降至3.875；Boundary Top10由8/10升至
9/10。该路由只在 `state.intent=browsing` 且Turn=1时使用`title_category_v1`，
第二轮起恢复`all_fields_v4`，避免RET-009A长时间丢失material/feature匹配。
