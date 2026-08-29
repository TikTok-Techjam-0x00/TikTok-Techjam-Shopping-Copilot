# Retrieval 实验区

本目录只存放可重复运行的实验脚本和实验台账；线上可复用的检索实现仍位于
`src/retrieval/` 根层。每次正式实验使用唯一编号 `RET-000`、`RET-001`……，并在
`EXPERIMENTS.md` 记录代码版本、数据口径、配置和结果。

## 文件说明

```text
text_ablation.py       BM25商品文本字段消融
query_instruction.py   Dense查询编码方式对照
dense_text.py          Dense商品文本版本对照
multivector_dense.py   identity/needs单向量与多向量融合对照
hybrid_comparison.py    BM25、Dense、Union、RRF、Weighted对照
intent_routing.py       Buying/Browsing多轮路由可靠性对照
visualize_results.py    将实验JSON转换为离线HTML报告
PLAN.md                 后续实验顺序、变量和判定标准
EXPERIMENTS.md          已完成与计划实验登记表
```

统一使用模块方式运行，避免工作目录和相对导入问题：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.text_ablation
.\.venv\Scripts\python.exe -m src.retrieval.experiments.query_instruction
.\.venv\Scripts\python.exe -m src.retrieval.experiments.dense_text
.\.venv\Scripts\python.exe -m src.retrieval.experiments.multivector_dense
.\.venv\Scripts\python.exe -m src.retrieval.experiments.hybrid_comparison
.\.venv\Scripts\python.exe -m src.retrieval.experiments.intent_routing
```

## 可视化

实验先输出机器可读JSON，再生成不依赖第三方前端库的HTML：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.visualize_results `
  artifacts\retrieval\multiturn_bm25_v1.json
```

HTML主视图按总体、Buying、Browsing、Boundary、Intent Override分别展示逐轮累计
Retrieval `Session Hit@100` 和剩余未进入Top100的数量。最终Hit@10与逐轮严格
Recall放在汇总或可展开诊断区；完整
Session明细保留在同名JSON中，避免HTML过大。JSON和HTML属于本地实验产物，默认
不提交Git；确定结论后把关键配置和指标写入 `EXPERIMENTS.md`。

## 最终实验流程

每次正式实验固定执行以下流程：

1. 固定 `public_set.jsonl`、对话构造方式、最大10轮和官方 `stop@10` 规则。
2. 先运行当前生产基线，避免用历史结果和新代码直接比较。
3. 每个实验只修改一个主要Retrieval变量。
4. 对完整200个Session运行总体及四场景多轮评估。
5. 主看累计`Session Hit@100`曲线和`MTTC@100`，再检查最终`Hit@10`；严格Recall只用于逐轮诊断。
6. 只有可靠性门槛通过并且完整测试通过，才能替换生产基线。

统一可靠性门槛：总体Hit@10和Hit@100不得下降，MTTC@10不得升高；Buying、Browsing、
Boundary、Intent Override任一场景的Hit@10或Hit@100不得减少。
