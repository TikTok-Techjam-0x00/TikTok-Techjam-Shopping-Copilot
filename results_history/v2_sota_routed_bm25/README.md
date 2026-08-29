# v2 SOTA Routed BM25 完整多轮结果

该快照记录 `sota` 提交 `e56b3d0` 上的完整 200-session 官方本地评测结果，
不是第一轮 Retrieval Recall。

## 固定配置

- Catalog：`data/catalog.jsonl`，50,000 个商品。
- Dataset：`data/public_set.jsonl`，200 个 Session，最多 10 轮。
- State：SOTA 多轮状态更新；为控制变量关闭可选 semantic resolver API。
- Retrieval：SOTA routed BM25。
  - Buying 使用 `all_fields_v4`。
  - Browsing 第一轮使用 `title_category_v1`，之后恢复 `all_fields_v4`。
  - 第 7、8 轮使用更深候选页。
- Reranking：`e56b3d0` 的 evidence-aware 确定性本地 fallback。
- Dialogue：SOTA high-information ask policy。
- Token usage：0。

## 最终指标

| 指标 | 结果 |
|---|---:|
| HitRate@10 | 0.995000 |
| MRR | 0.946000 |
| MTTC | 2.260000 |
| Efficiency | 0.874000 |
| TechnicalScore | 0.956100 |

分场景结果：

| 场景 | HitRate@10 | MRR | MTTC |
|---|---:|---:|---:|
| Boundary | 1.000000 | 1.000000 | 2.400000 |
| Browsing | 1.000000 | 0.940625 | 2.075000 |
| Buying | 0.987500 | 0.952500 | 1.875000 |
| Intent Override | 1.000000 | 0.925000 | 3.733333 |

完整逐 Session 结果见同目录 `results.json`。根目录 `results.json` 同步指向该
最新快照。此结果不代表 Qwen 线上 reranker 全量 A/B；线上冒烟的输入 token 很高，
因此这里使用可重复、零 token 的生产 fallback 口径。
