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
| `RR-000` | 待重新登记 | `public200-git0ad81a1` | `0ad81a1` / 待运行 | 无；保持 Retrieval 顺序 | 无 | 无 / 无 | 无 | Diversity=无；Profile=无 | 0.683959 / 0.506699（旧报告） | 0 / 0 | 0.318366 | 0.631889 | 0.138 / 0.193 ms（旧报告） | 待独立运行 | 待生成 `results/RR-000/`；Retrieval control |
| `RR-001` | 完成 / 2026-08-29 | `public200-git0ad81a1` | `0ad81a1` / `7731c15` | Buying=H2 Feasibility Tier；Browsing=H1 Soft Penalty | S1 local Rule/Fuzzy | 结构化 hard/soft，均为空时使用当前消息 / Item 属性与限长 observations | F1 intent-aware 手工线性融合 | D1 无多样性；Profile lexical | **0.791126 / 0.585244** | **171 / 14** | **0.147763** | **0.716910** | 378.363 / 1396.241 ms | 00:17:04（包含同次 RR-000 control） | `results/RR-001/`；整体提升，下一步检查 Intent Override 回退与延迟 |

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
整体测试耗时。`RR-000` 将使用新接口重新运行后补齐独立目录与耗时。
