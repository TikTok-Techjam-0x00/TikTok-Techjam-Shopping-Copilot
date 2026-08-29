# Retrieval 实验规划

## 统一评测纪律

所有实验固定使用官方Catalog和Public Set，不修改标签或Evaluator。第一轮实验记录
严格 `Recall@10/50/100`；多轮实验同时记录每轮严格Recall和累计Session HitRate。
Union指标必须标注为非严格候选池口径，RRF/Weighted/BM25/Dense必须是最终列表不
超过K的严格口径。

每次只改变一个主要变量，并记录：

```text
experiment_id / git_commit / dataset / method / product_text / query_mode
source_k / fusion_parameters / metadata_policy / Recall / latency / cache_size
```

## 推荐执行顺序

### P0：建立多轮基线（RET-010）

- 方法：当前默认 `all_fields_v4` BM25。
- 目的：按官方Top10命中即停止规则得到逐轮严格Recall、累计Session HitRate、场景拆分和失败Session。
- 诊断补充：另跑 `--continue-after-hit`，命中后轮次只作为反事实结果，不混入正式指标。
- 判定：后续方案必须同时对比第一轮和多轮，避免只优化某一轮。

### P1：多向量Dense（RET-007）

- `dense_identity_v1`：Title、Product type、Brand。
- `dense_needs_v1`：Material、Color、Size、Style、Use case、Features。
- 对比：identity only、needs only、max fusion、加权融合。
- 重点：是否补回 `dense_attributes_v2` Top10较弱、Top100较强的问题。

### P2：融合参数与候选预算（RET-008）

- RRF常数：20、40、60、80。
- Weighted alpha：0.3–0.9。
- Source K：50、100、150、200，最终输出仍严格Top100。
- 判定：优先Recall@100，Top10不能出现不可接受回退。

### P3：Buying / Browsing路由（RET-009，已完成）

- Buying：使用`all_fields_v4`完整字段BM25。
- Browsing：仅首轮使用`title_category_v1`，第二轮起恢复`all_fields_v4`。
- Boundary：不把no-preference字段继续加入Query。
- Intent Override：只消费覆盖后的State，禁止旧偏好泄漏。

全程使用短Browsing文本的RET-009A失败；首轮warm-start的RET-009B通过总体、MTTC和
所有场景Top10/Top100无回退门槛，已经作为新Pipeline基线。

### P4：Metadata boosting（RET-011）

- Category、Brand、Price先做soft boost，不直接删除缺失字段商品。
- 只在State置信度高且Catalog字段存在时试验hard filter。
- 单独记录“目标被过滤”的数量，任何Recall@100下降都必须解释。

### P5：最终多轮确认（RET-012）

- 对P1–P4选出的最佳严格Top100策略运行完整200 Session × 10 Turn。
- 与RET-010使用同一代码口径对照。
- 再交由3A/3B运行官方端到端HitRate、MRR、MTTC和TechnicalScore。

## 可视化方案

每个方案保留三层视图：

1. 总览：方法的Recall@10/50/100与平均延迟。
2. 多轮曲线：横轴Turn 1–10，蓝色为该轮严格Recall，绿色为累计Session HitRate。
3. 失败明细：sample、scenario、query、target rank、Top-N结果和首次命中轮次。

第一、二层由 `visualize_results.py` 自动生成HTML；第三层保留在JSON的`sessions`
中，后续可按 `target_rank=null`、场景或轮次筛选。
