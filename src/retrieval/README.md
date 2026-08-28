# 模块 1 — 检索（Retrieval）

当前实验版本：`bm25_v0`。

## 数据流程

```text
catalog.jsonl(.gz) -> Catalog[parent_asin, Item] -> SQLite FTS5
当前查询 + ShoppingState -> 检索查询 -> BM25 -> list[Candidate]
```

每个 `Item` 还提供统一的 `item.attributes` 派生属性，供 Retrieval 后续
metadata 实验、3A 约束匹配和 3B 候选属性分析共同使用。属性在第一次访问时
才提取并缓存，因此加载 50,000 个商品和构建 BM25 时不会提前承担全部抽取开销。
`Item.to_dict()` 仍只返回官方 Catalog 字段。

如需把全部商品属性导出为独立 JSONL，可在项目根目录直接运行：

```powershell
.\.venv\Scripts\python.exe extract_attributes.py
```

默认输出 `data/catalog_attributes.jsonl`，该生成文件已被 Git 忽略。

模块的公共调用入口如下：

```python
from src.retrieval import Catalog, BM25Retriever

catalog = Catalog.load("data/catalog.jsonl")
retriever = BM25Retriever(catalog)
candidates_100 = retriever.retrieve(
    query=user_message,
    state=shopping_state,
    intent=shopping_state.intent,
    k=100,
)
```

## 输出约束

- 返回 `src/item.py` 中定义的完整共享 `Candidate` 对象；
- `parent_asin` 必须来自冻结 Catalog，并保持精确、唯一；
- 候选按照从优到劣排列，`retrieval_rank` 从 1 开始；
- `bm25_score` 和 `retrieval_score` 都遵循**分数越高越好**；
- 空的文本查询返回空列表；
- `k` 可配置，不强制只能返回 100 个候选。

SQLite 原生 BM25 分数是越低越好，因此本实现会先对原生分数取负，再将其作为对外分数。标题和类别的字段权重最高，与官方 starter 的基线设计保持一致。

Catalog Loader 会跳过格式错误的记录；如果出现重复 ASIN，则保留第一个有效商品，并通过 `catalog.stats` 提供加载诊断信息。

## ShoppingState 查询构造

如果 ShoppingState 中存在当前有效的类别或约束，查询构造器会把当前状态视为权威信息来源，而不是盲目拼接全部历史消息。

这样可以避免 Module 2 完成 intent override 后，旧偏好再次进入检索查询。例如用户从 `running shoes` 改为 `hiking boots`，只要 Module 2 已删除旧约束，Retrieval 就只会使用当前状态中的 `hiking boots`。

目前纯数字范围约束不会转换成 BM25 关键词。例如预算上限 `$100` 会继续保存在 ShoppingState 中，交给后续 metadata 实验或 3A 使用，避免无意义的数字给 BM25 带来噪声。

## 运行测试

使用项目虚拟环境运行全部测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

也可以在 VSCode 中直接打开并运行：

```text
src/retrieval/test_retrieval.py
src/retrieval/test_evaluate.py
```

`test_retrieval.py` 验证 Catalog、查询构造、BM25、Top-K、去重、分数方向和确定性；`test_evaluate.py` 验证 Recall 的命中数量、排名和场景拆分是否计算正确。

## 运行 Recall 评估

执行可重复的首轮 Recall 评估：

```powershell
.\.venv\Scripts\python.exe -m src.retrieval.evaluate `
  --catalog data/catalog.jsonl `
  --dataset data/public_set.jsonl `
  --ks 10 50 100 `
  --output artifacts/retrieval_bm25_v0.json `
  --include-sessions
```

也可以在 VSCode 中直接运行：

```text
src/retrieval/evaluate.py
```

评估程序复用官方 public simulator，为每个 session 生成首轮用户消息。隐藏目标 ASIN 只用于生成官方模拟消息和判断返回排名，绝不会作为输入传给 Retriever。

这个命令测量的是 Retrieval 的首轮 Recall，不是官方完整多轮评估中的 Hit Rate、MRR 或 MTTC。特别是 Intent Override 场景，在 override 正式发生前出现目标并不会被官方完整 evaluator 计为命中，因此首轮 Recall 只能作为检索诊断指标。

## 已测基线

`bm25_v0` 于 2026-08-27 在全部 50,000 个 Catalog 商品和 200 个公开 session 上完成测量。查询使用未修改的官方 evaluator 生成的首轮消息。

| 版本 | Recall@10 | Recall@50 | Recall@100 | 平均查询延迟 | P95 查询延迟 |
|---|---:|---:|---:|---:|---:|
| `bm25_v0` | 0.185 | 0.380 | 0.525 | 11.7 ms | 27.8 ms |

对应的目标命中数量为：

```text
Top 10：37 / 200
Top 50：76 / 200
Top 100：105 / 200
```

在当时的测量机器上，Catalog 解析耗时 0.681 秒，内存 FTS5 索引构建耗时 1.938 秒。延迟与机器环境有关，主要实验比较指标仍然是 Recall。这些结果只是 Retrieval 的首轮指标，不代表完整流水线的 Hit Rate、MRR 或 MTTC。

## 后续实验

在更改默认策略前，建议依次进行以下可测量实验：

1. 比较不同商品文本字段组合；
2. 比较不同 BM25 字段权重；
3. 分析 Top-100 miss 的主要原因；
4. 实现并评估 Dense Retrieval；
5. 比较 BM25、Dense、Candidate Union 和 RRF；
6. 重点记录 Recall@10、Recall@50、Recall@100 和查询延迟。
