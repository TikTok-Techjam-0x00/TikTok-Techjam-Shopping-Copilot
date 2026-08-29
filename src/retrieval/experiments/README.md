# Retrieval 实验区

本目录只存放可重复运行的实验脚本和实验台账；线上可复用的检索实现仍位于
`src/retrieval/` 根层。每次正式实验使用唯一编号 `RET-000`、`RET-001`……，并在
`EXPERIMENTS.md` 记录代码版本、数据口径、配置和结果。

## 文件说明

```text
text_ablation.py       BM25商品文本字段消融
query_instruction.py   Dense查询编码方式对照
dense_text.py          Dense商品文本版本对照
hybrid_comparison.py    BM25、Dense、Union、RRF、Weighted对照
visualize_results.py    将实验JSON转换为离线HTML报告
PLAN.md                 后续实验顺序、变量和判定标准
EXPERIMENTS.md          已完成与计划实验登记表
```

统一使用模块方式运行，避免工作目录和相对导入问题：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.text_ablation
.\.venv\Scripts\python.exe -m src.retrieval.experiments.query_instruction
.\.venv\Scripts\python.exe -m src.retrieval.experiments.dense_text
.\.venv\Scripts\python.exe -m src.retrieval.experiments.hybrid_comparison
```

## 可视化

实验先输出机器可读JSON，再生成不依赖第三方前端库的HTML：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.experiments.visualize_results `
  artifacts\retrieval\multiturn_bm25_v1.json
```

HTML中的蓝色条表示该轮严格Recall，绿色条表示截至该轮的Session HitRate。原始
JSON被折叠保留，便于复核。JSON和HTML属于本地实验产物，默认不提交Git；确定结论
后把关键配置和指标写入 `EXPERIMENTS.md`。
