# Reranking 实验登记表

本文件是 Module 3A 的实验台账。每个编号只对应一套 Reranker 配置；同一编号不能
混合多个 Hard Constraint 策略、语义模型或融合权重。开始实验前填写配置，完成后
补齐指标、耗时、Git 版本和结果目录。

结果目录统一为：

```text
artifacts/reranking_replay/<replay-data-version>/results/<experiment-id>/
```

编号使用 `RR-000`、`RR-001`、`RR-002`……。已经使用或失败的编号不得复用。

## 实验总表

| 编号 | 状态 / 日期 | Replay 数据版本 | 数据 Git / 评测 Git | Hard Constraint 策略 | 语义相关度模型 | 文本序列化（Query / Product） | Score Fusion | 其他变量 | Cond. Hit / MRR@10 | Promotion / Demotion | Hard violation@10 | Replay TechnicalScore | P50 / P95 | 整体测试耗时 | 结果目录 / 结论 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `RR-000` | 完成 / 2026-08-29 | `public200-git0ad81a1` | `0ad81a1` / `d1fceef` | 无；保持 Retrieval 顺序 | 无 | 无 / 无 | 无 | Diversity=无；Profile=无 | 0.683959 / 0.506699 | 0 / 0 | 0.318366 | 0.631889 | 0.156 / 0.241 ms | **16.001 s** | `results/RR-000/`；Retrieval control |
| `RR-001` | 完成 / 2026-08-29 | `public200-git0ad81a1` | `0ad81a1` / `7731c15` | Buying=H2 Feasibility Tier；Browsing=H1 Soft Penalty | S1 local Rule/Fuzzy | 结构化 hard/soft，均为空时使用当前消息 / Item 属性与限长 observations | F1 intent-aware 手工线性融合 | D1 无多样性；Profile lexical | **0.791126 / 0.585244** | **171 / 14** | **0.147763** | **0.716910** | 378.363 / 1396.241 ms | 00:17:04（包含同次 RR-000 control） | `results/RR-001/`；整体提升，下一步检查 Intent Override 回退与延迟 |
| `RR-002` | 完成 / 2026-08-29 | `public200-git0ad81a1` | `0ad81a1` / `c319c18` | 旧版任一 violation 二元惩罚；无 H1/H2 | 初版 exact token overlap；无 fuzzy/model | 直接读取当前结构化 State / 属性字段 + title/categories/features/description | 初版 intent-aware 线性权重（`799e8c1`） | Diversity=无；Profile exact token | 0.703754 / 0.512632 | 30 / 1 | 0.323881 | 0.636383 | 39.710 / 68.472 ms | **97.346 s** | `results/RR-002/`；仅略优于 Retrieval，明显弱于 RR-001，且 hard violation 略升 |
| `RR-003` | 完成 / 2026-08-29 | `public200-git80e002e` | `80e002e` / `80e002e` | 无；保持 Retrieval 顺序 | 无 | 无 / 无 | 无 | Diversity=无；Profile=无；生产 3B baseline | 0.664122 / 0.484044 | 0 / 0 | 0.218054 | 0.646763 | 0.148 / 0.209 ms | **15.257 s** | `results/RR-003/`；新 State/Retrieval Replay control，Intent Override coverage 明显恢复 |
| `RR-004` | 完成 / 2026-08-29 | `public200-git80e002e` | `80e002e` / `fd18213` | Buying=H2 Feasibility Tier；Browsing=H1 Soft Penalty | S1 local Rule/Fuzzy | 结构化 hard/soft，均为空时使用当前消息 / Item 属性与限长 observations | F1 intent-aware 手工线性融合 | D1 无多样性；Profile lexical；生产 3B baseline | **0.770356 / 0.563447** | **180 / 13** | **0.033039** | **0.740699** | 374.494 / 1370.143 ms | **1016.591 s** | `results/RR-004/`；新 Replay 当前 S1 主基准，四类场景均优于 RR-003 |
| `RR-005` | 完成 / 2026-08-29 | `public200-git80e002e` | `80e002e` / `82cc4b3` | 与 RR-004 完全相同 | S1-fast；公式等价的有界 product/text/score cache | 与 RR-004 完全相同；缓存商品侧标准化结果 / 完全相同 | 与 RR-004 完全相同 | Cache=8192/131072/131072；200,000 候选严格等价 | **0.770356 / 0.563447** | **180 / 13** | **0.033039** | **0.740699** | **199.621 / 724.412 ms** | **522.013 s** | `results/RR-005/`；质量完全不变，总耗时下降 48.651%，作为新的 S1 性能基准 |

## 四类场景对照

以下数值由同一版 `public200-git0ad81a1` Replay 的已保存
`case_results.jsonl.gz` 重新汇总，不重新执行 Reranker。`RR-000` 是 Retrieval
control，`RR-002` 是最初版 SimpleReranker，`RR-001` 是当前 S1 Rule/Fuzzy。
这些数值只能在旧 Replay 内横向比较，不能直接与新 Replay 的 `RR-003` 判断
Reranker 优劣。

### Session 级结果

| 场景 | 编号 | Sessions | HitRate@10 | MRR | MTTC ↓ | Efficiency | Replay Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Browsing | `RR-000` | 80 | 0.875000 | 0.535432 | 4.187500 | 0.681250 | 0.734379 |
| Browsing | `RR-002` | 80 | 0.875000 | 0.512336 | 3.887500 | 0.711250 | 0.733451 |
| Browsing | `RR-001` | 80 | **0.975000** | **0.606334** | 3.162500 | **0.783750** | **0.826150** |
| Buying | `RR-000` | 80 | 0.762500 | 0.454683 | 6.575000 | 0.442500 | 0.606155 |
| Buying | `RR-002` | 80 | 0.775000 | 0.463537 | 6.512500 | 0.448750 | 0.616311 |
| Buying | `RR-001` | 80 | **0.900000** | **0.555179** | **5.562500** | **0.543750** | **0.725304** |
| Boundary | `RR-000` | 10 | 0.800000 | 0.575000 | 6.300000 | 0.470000 | 0.666500 |
| Boundary | `RR-002` | 10 | 0.800000 | 0.578571 | 6.300000 | 0.470000 | 0.667571 |
| Boundary | `RR-001` | 10 | **0.900000** | **0.595238** | **6.100000** | **0.490000** | **0.726571** |
| Intent Override | `RR-000` | 30 | **0.500000** | 0.427778 | **9.133333** | **0.186667** | 0.415667 |
| Intent Override | `RR-002` | 30 | **0.500000** | **0.444444** | **9.133333** | **0.186667** | **0.420667** |
| Intent Override | `RR-001` | 30 | 0.466667 | 0.433333 | 9.166667 | 0.183333 | 0.400000 |

### Case 级诊断

| 场景 | 编号 | Scorable cases | Coverage@100 | Cond. Hit / MRR@10 | Promotion / Demotion | Mean rank Δ | Hard violation@10 | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Browsing | `RR-000` | 800 | 0.888750 | 0.766526 / 0.569721 | 0 / 0 | 0.000000 | 0.246750 | 0.241 ms |
| Browsing | `RR-002` | 800 | 0.888750 | 0.800281 / 0.585006 | 25 / 1 | 0.511955 | 0.251000 | 70.159 ms |
| Browsing | `RR-001` | 800 | 0.888750 | **0.881857 / 0.668409** | **83 / 1** | **3.436006** | **0.036500** | 1512.236 ms |
| Buying | `RR-000` | 800 | 0.775000 | 0.570968 / 0.396829 | 0 / 0 | 0.000000 | 0.213000 | 0.240 ms |
| Buying | `RR-002` | 800 | 0.775000 | 0.579032 / 0.390700 | 5 / 0 | 1.877419 | 0.220750 | 68.178 ms |
| Buying | `RR-001` | 800 | 0.775000 | **0.690323 / 0.466207** | **86 / 12** | **7.241935** | **0.039750** | 1333.215 ms |
| Boundary | `RR-000` | 100 | 0.700000 | 0.671429 / 0.548214 | 0 / 0 | 0.000000 | 0.221000 | 0.258 ms |
| Boundary | `RR-002` | 100 | 0.700000 | 0.671429 / 0.550000 | 0 / 0 | 2.114286 | 0.231000 | 59.712 ms |
| Boundary | `RR-001` | 100 | 0.700000 | **0.700000 / 0.558503** | **2 / 0** | **2.528571** | **0.010000** | 1043.974 ms |
| Intent Override | `RR-000` | 222 | 0.288288 | **0.875000 / 0.825521** | 0 / 0 | 0.000000 | 1.000000 | 0.238 ms |
| Intent Override | `RR-002` | 222 | 0.288288 | **0.875000 / 0.848958** | 0 / 0 | 0.140625 | 1.000000 | 57.581 ms |
| Intent Override | `RR-001` | 222 | 0.288288 | 0.859375 / 0.843750 | 0 / 1 | -1.484375 | 1.000000 | 1129.079 ms |

### 参数调整指引

- **Browsing**：`RR-001` 的 HitRate、MRR、promotion 和 hard-constraint 质量均明显
  提升，说明 H1 Soft Penalty + S1 方向有效。下一轮可优先压低 fuzzy 计算成本，
  再小范围调整 semantic/retrieval/profile 权重，避免破坏现有提升。
- **Buying**：`RR-001` 提升最大，H2 Feasibility Tier 有效；但出现 12 次 demotion。
  下一轮应把这 12 个 case 单独检查，区分“正确执行 hard constraint”与“错误匹配/UNKNOWN
  分层”，再调 hard tier、unknown penalty 和层内 relevance 权重。
- **Boundary**：当前仅有 10 个 session，方向看起来改善但方差很大。参数选择不能主要
  依赖该场景；应重点检查空约束、`no_prefernce` 和缺失属性时是否保持 Retrieval 稳定。
- **Intent Override**：`RR-001` 比 control 和初版都回退，且三种方法的 Top 10 hard
  violation item rate 都是 1.0。应先修复/确认上游 State 对旧约束的清除，再调整
  Reranker；否则 H2 可能在更严格地执行已经过期的 hard constraint。
- **Coverage@100** 是冻结的 Retrieval 输入，同一场景下三个 Reranker 完全相同，不能
  通过调 Reranking 参数提升。Intent Override 的 0.288288 首先指向 Retrieval/State，
  不是 Top 10 排序本身。

## Replay 数据版本更新

`public200-git80e002e` 使用 commit `80e002e` 重新录制，共 200 个 session、2000 个
turn case，其中 1922 个可评分。Catalog 和 Public Set 的输入哈希与旧版完全一致；
变化来自 State `560b92b` 和 Retrieval `4eee3d6`。录制时 3B 使用生产入口
`src/dialogue/three_b.py`，即实验目录中的 behavior-identical baseline，不使用其他
3B 消融版本。

| Retrieval control | 旧 `RR-000` / `0ad81a1` | 新 `RR-003` / `80e002e` |
| --- | ---: | ---: |
| Overall Coverage@100 | 0.762227 | **0.817898** |
| Overall Session HitRate@10 | 0.770000 | **0.790000** |
| Overall Replay Score | 0.631889 | **0.646763** |
| Intent Override Coverage@100 | 0.288288 | **0.770270** |
| Intent Override Session HitRate@10 | 0.500000 | **0.633333** |
| Intent Override Hard violation@10 | 1.000000 | **0.131532** |
| Intent Override Replay Score | 0.415667 | **0.514825** |

Browsing、Buying、Boundary 的 Retrieval control 数值保持不变，主要变化集中在
Intent Override。之后的新 Reranker 实验应统一使用 `public200-git80e002e`，并从
`RR-006` 开始编号；旧 `RR-000`～`RR-002` 仅保留为旧 Replay 的历史对照。

### 新 Replay 上的 Reranker 增益

`RR-003` 与 `RR-004` 使用相同 Replay，下面的差异只来自 S1 Rule/Fuzzy Reranker。

| 场景 | RR-003 HitRate / MRR / MTTC / Score | RR-004 HitRate / MRR / MTTC / Score | Score Δ |
| --- | ---: | ---: | ---: |
| Browsing | 0.875000 / 0.535432 / 4.187500 / 0.734379 | **0.975000 / 0.606334 / 3.162500 / 0.826150** | **+0.091771** |
| Buying | 0.762500 / 0.454683 / 6.575000 / 0.606155 | **0.900000 / 0.555179 / 5.562500 / 0.725304** | **+0.119149** |
| Boundary | 0.800000 / 0.575000 / 6.300000 / 0.666500 | **0.900000 / 0.595238 / 6.100000 / 0.726571** | **+0.060071** |
| Intent Override | 0.633333 / 0.442751 / 7.733333 / 0.514825 | **0.666667 / 0.513095 / 7.433333 / 0.558595** | **+0.043770** |
| Overall | 0.790000 / 0.491208 / 5.780000 / 0.646763 | **0.895000 / 0.571331 / 4.910000 / 0.740699** | **+0.093936** |

相对旧 Replay 的 `RR-001`，`RR-004` 的 Browsing、Buying、Boundary 结果完全相同；
Intent Override Session HitRate 从 0.466667 升至 0.666667，Replay Score 从
0.400000 升至 0.558595。整体 Conditional Hit/MRR 受新增 Override 难例的分母影响
不宜直接比较，但 Top 10 绝对命中 case 从 1159 增至 1211，Overall Session
HitRate 从 0.865000 升至 0.895000。

### S1-fast 等价性能优化

`RR-005` 只缓存 Catalog 商品文本、标准化 token 和重复 fuzzy comparison；排序
公式、约束策略、融合权重及输入均与 `RR-004` 相同。独立等价检查覆盖 2000 个 case、
200,000 个候选，逐一比较完整 Top100 的 `parent_asin`、rank、六位小数 score、
matched 和 violation，结果零差异。

| 指标 | RR-004 S1 | RR-005 S1-fast | 变化 |
| --- | ---: | ---: | ---: |
| Mean latency | 501.412 ms | **254.069 ms** | **-49.329%** |
| P50 latency | 374.494 ms | **199.621 ms** | **-46.696%** |
| P95 latency | 1370.143 ms | **724.412 ms** | **-47.129%** |
| Max latency | 2380.907 ms | **1511.026 ms** | **-36.536%** |
| 整体测试耗时 | 1016.591 s | **522.013 s** | **-48.651%** |
| Replay TechnicalScore | 0.740699 | **0.740699** | 0 |

全量配对检查中，旧 S1 排序累计耗时 998.503 秒，S1-fast 为 498.212 秒，配对
speedup 为 **2.004×**。等价报告位于 Replay 目录的
`equivalence/RR-005.json`；之后的实验和生产默认使用 S1-fast。

## 字段填写规则

- **Replay 数据版本**：填写录制目录名，而不是只写“public set”。目录中的
  `manifest.json` 是 State、Retrieval、Dialogue、Catalog 和数据版本的依据。
- **数据 Git / 评测 Git**：前者是生成 Replay case 的 commit，后者是实际运行
  Reranker 的 commit。两者允许不同，但必须同时记录。
- **Hard Constraint 策略**：至少说明 Buying/Browsing 分别使用 H1 Soft Penalty、
  H2 Feasibility Tier、过滤或其他方案。
- **语义相关度模型**：例如 `None`、`S1 Rule/Fuzzy`、MiniLM、BGE，并记录模型
  checkpoint/版本；外部 API 还要记录服务模型名。
- **文本序列化**：同时记录 Query 和 Product。使用 Q1/Q2/Q3、P1/P2/P3/P4 时，
  应注明编号及具体字段，不能只写“默认”。
- **Score Fusion**：记录 F1/RRF/LTR 及权重。只写 `F1` 但不保存权重不算完整。
- **其他变量**：记录 diversity、user profile、候选截断数量、随机种子、硬件或
  其他本次发生变化的因素。
- **Cond. Hit/MRR、Promotion/Demotion、Hard violation**：使用 Replay report 中的
  case metrics；目标未进入 Candidates100 的 case 不进入 conditional 分母。
- **Replay TechnicalScore**：使用 report 的 session-level 反事实估计。它不是官方
  end-to-end 最终分，入选方案仍需运行官方 evaluator。
- **P50/P95**：单位必须写清楚，默认使用毫秒。
- **整体测试耗时**：使用 `report.json.total_elapsed_seconds`，包括读取 Replay、加载
  Catalog、完整重排、指标统计，不用 `P95 × case 数` 推算。

## 实验执行流程

1. 在本表中占用下一个编号并填写唯一变量、Replay 版本和预期配置。
2. 确认目标目录不存在；Evaluator 会拒绝覆盖已有编号。
3. 运行单个配置：

   ```powershell
   python -m src.reranking.replay.evaluator `
     artifacts/reranking_replay/public200-git0ad81a1 `
     --experiment-id RR-002 `
     --experiment s1_rule_fuzzy
   ```

4. 从 `report.json` 把结果和 `total_elapsed_seconds` 回填本表。
5. 检查 `case_results.jsonl.gz` 中的 promotion、demotion 和分场景回退。
6. 只有在 conditional MRR 提升、promotion 大于 demotion、关键场景无不可接受
   回退且延迟可接受时，才进入官方 end-to-end evaluator。

## 当前基准说明

`RR-001` 的旧报告最初同时运行了 Retrieval control 与 S1，因此整体耗时是两臂
合计。新接口已经限制一个编号只能运行一个配置，之后每行都会拥有独立、准确的
整体测试耗时。`RR-000` 已使用新接口重新运行并作为目录和时间记录的标准示例。
