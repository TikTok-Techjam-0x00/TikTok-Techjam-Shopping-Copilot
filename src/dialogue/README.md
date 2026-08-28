# Dialogue Clarification（Module 3B）

本目录负责根据 Module 2 的 `shopping_state` 和 Module 1 Retrieval 返回的最多
100 个 `Candidate`，判断是否需要继续 clarification、选择 `ask_attribute`，并用
确定性模板生成面向用户的 `message`。

3B 不修改商品顺序，也不依赖 3A 的具体排序公式。当前 clarification policy 是
可运行的启发式基线，后续可以优化属性价值估计而不改变模块输入输出边界。

## 1. 数据流

```text
Official evaluator
  reset(session_id, user_profile)
  respond(session_id, user_message, turn, top_k)
                 |
                 v
Module 2: shopping_state
                 |
                 +----------------------------+
                 |                            |
                 v                            v
Module 1: Retrieval                     Module 3B: Dialogue
output candidates_100  ---------------> shopping_state + candidates_100
                 |                            |
                 v                            v
Module 3A: Reranking                   ask_attribute + message
output candidates_10
                 |
                 v
Official recommendations
```

主要文件：

- `src/item.py`：跨模块共享的 `Item`、`Candidate`、`RankedCandidate`。
- `src/attribute.py`：Module 2 hard/soft constraint 使用的属性契约。
- `src/dialogue/three_b.py`：clarification 属性选择和问题生成。
- `src/dialogue/test_three_b.py`：3B 接口与行为测试。

推荐导入：

```python
from src.dialogue import decide_ask, record_asked_attribute
from src.item import Candidate, Item
```

## 2. 为什么直接读取 Candidate.item

最新版共享模型使用组合关系：

```text
Candidate.item ---------> Item
RankedCandidate.item ---> Item
```

`Candidate` 保存 Retrieval 阶段信息，`Item` 保存 catalog 商品字段。3B 的正式
商品读取路径是：

```text
Candidate
    ↓
Candidate.item
    ↓
Item.to_dict()
    ↓
统一 product dict
    ↓
属性分布分析
```

这使 3B 不依赖 `Candidate` 是否实现 Mapping，也不会把 Retrieval 分数混入商品
元数据。迁移阶段仍兼容嵌套 `item` 的 Mapping 和直接 catalog Mapping，但它们
不是正式数据路径。

## 3. 输入契约

公开入口：

```python
decision = decide_ask(
    shopping_state=shopping_state,
    candidates_100=candidates_100,
)
```

### 3.1 shopping_state

State 由 Module 2 创建和维护。3B 同时支持对象属性和普通 Mapping：

| 字段 | 含义 | 3B 用途 |
| --- | --- | --- |
| `session_id` | 当前会话 ID | 共享契约，不参与当前评分 |
| `user_profile` | 匿名聚合画像 | 保留共享契约；当前不参与 3B attribute scoring |
| `user_message` | 当前用户消息 | Module 2 应先解析为约束 |
| `turn` | 当前轮次 1～10 | 第 10 轮停止继续追问 |
| `intent` | `buying` 或 `browsing` | 共享契约，当前 3B 不直接使用 |
| `hard_constraint` | 用户硬约束 | 其中的属性不会再次询问 |
| `soft_constraint` | 用户软偏好 | 其中的属性不会再次询问 |
| `no_prefernce` | 用户不关心的属性 | 对应属性不会询问 |
| `asked_attributes` | 历史已问属性 | 防止重复 clarification |

`hard_constraint` 和 `soft_constraint` 可以使用团队的 `AttributeMap`：

```python
shopping_state.hard_constraint = {
    "category": AttributeValue(values=["running shoes"]),
    "budget": AttributeValue(maximum=100, unit="USD"),
}
shopping_state.soft_constraint = {
    "color": AttributeValue(values=["black"]),
}
shopping_state.no_prefernce = [AttributeName.BRAND]
shopping_state.asked_attributes = ["material"]
```

共享枚举现在就是官方的十个 `ask_attribute` 字段。3B 仅把旧版输入中的
`fit` 兼容映射为 `style`；新 State 应直接使用 `AttributeName.STYLE`。

3B 当前只需要判断 constraint value 是否存在，不解析 `AttributeValue` 的具体
语义。`no_prefernce` 是团队正式拼写；低成本保留 `no_preference` 兼容拼写。
`rejected_values={"material": ["leather"]}` 表示拒绝一个具体值，不等于用户不
关心整个 material 属性，因此不会被当成 no preference。

### 3.2 candidates_100

正式输入是 Module 1 按 Retrieval 顺序输出的 `list[Candidate]`：

```python
product = Item.from_dict(catalog_record)

retrieved = Candidate(
    item=product,
    bm25_score=9.2,
    dense_score=0.82,
    retrieval_score=0.91,
    retrieval_rank=1,
)

candidates_100 = [retrieved]
```

3B 最多读取列表前 100 个候选。当前算法只使用列表顺序作为相对 rank，不依赖
`retrieval_score` 的绝对阈值，也不要求 BM25/Dense 分数存在。

保留的 Mapping 兼容形式：

```python
mapping_candidate = {
    "item": {
        "parent_asin": "B001...",
        "title": "Black Running Shoes",
        "features": ["Lightweight"],
        "description": [],
        "price": 79.99,
        "categories": ["Shoes", "Running"],
        "details": {"Material": "Mesh", "Color": "Black"},
        "average_rating": 4.5,
        "rating_number": 120,
        "store": "Example Store",
    },
    "retrieval_rank": 1,
}
```

旧的 `ranking_result/results/ranked_products/recommendations` 外层包装不再属于
3B 正式接口。

## 4. 输出

`decide_ask()` 返回：

```python
{
    "ask_attribute": "material",
    "message": "Which material do you prefer: mesh or leather?",
}
```

`ask_attribute` 为官方允许的 clarification 属性或 `None`。第 10 轮不继续追问：

```python
{
    "ask_attribute": None,
    "message": "",
}
```

3B 本身无状态。Agent 得到决定后，必须让 Module 2 持久化本轮属性：

```python
record_asked_attribute(shopping_state, decision["ask_attribute"])
```

该 helper 支持可变 Mapping 和可写属性对象，并统一写回 `asked_attributes` 列表。

## 5. Clarification policy

### 5.1 排除不能再问的属性

```text
excluded
  = hard_constraint.keys()
  + soft_constraint.keys()
  + asked_attributes
  + no_prefernce
```

如果 category 尚未出现，3B 优先确认 category。其他属性再根据 Base Priority、
动态 Answerability 和 Ranking Impact 评分。

### 5.2 商品属性提取

`_product()` 统一解包 Candidate；`_values()` 按以下顺序提取属性：

```text
明确的 Item 顶层字段
        ↓
Item.details（key 大小写不敏感）
        ↓
categories / features / price / store 等结构化字段
        ↓
title / features / description / categories / details 文本
        ↓
有限正则 fallback
```

`use_case` 是一个严格例外：只从 `features` 和 `details` 搜索，并且只识别
`hiking / running / gym / winter / outdoor / work`。title、description、categories、
store 和顶层 `use_case` 字段都不会计入其候选 coverage；`wedding / travel / daily`
也不会被识别为 `use_case`。

例如以下 details 无需出现在正则词典中即可识别：

```python
{
    "Material": "Stainless Steel",
    "Color": "Silver",
    "Brand": "Acme",
}
```

Brand 优先级为：

```text
明确 brand → details["Brand"] → store → 文本 fallback
```

### 5.3 Answerability 与 Ranking Impact

对每个尚未排除的属性，当前策略计算：

- Base Priority：只提供稳定的通用语义层级，不使用测试集属性分布。
- rank weighting：`1 / sqrt(rank)`，排名靠前的 Candidate 权重更高。
- Answerability：当前 Top100 中具有该属性值的排名加权覆盖率。
- Ranking Impact：候选值分布的 normalized entropy。

Base Priority 使用少量语义分层：

```text
category=90
feature=65, material=65
use_case=60, size=60, style=60, color=60
budget=55, brand=35, other=5
```

最终评分公式为：

```text
rank_weight(r) = 1 / sqrt(r)
answerability = sum(rank_weight of covered candidates) / sum(all rank_weight)
ranking_impact = normalized_entropy
question_value = 18 × answerability × ranking_impact
score = base_priority + question_value
```

Answerability 和 Ranking Impact 都只从本轮 Module 1 candidates 计算，不读取
public ground truth、scenario label 或固定测试集统计。排名靠前的候选具有更大权重；
单个商品有多个值时，其权重仍平均分给这些值。

normalized entropy 已把不同候选值数量归一化到 `0～1`，因此不同 cardinality
不会被额外奖励或惩罚。系数 `18` 是动态 Question Value 的上限。`user_profile`
仍保留在 State 接口中，但不参与 3B 评分，避免与已经使用 Profile 的 3A 排名
重复计算。

Ranking Impact 仍是候选多样性启发式，不是模拟用户回答并重新 Retrieval 后的
严格排名增益。
现有权重保持 deterministic，3B 不读取 `rerank_score` 或 Retrieval score 阈值。

### 5.4 问题生成

问题通过固定 if 模板生成，不调用 LLM。存在稳定候选值时会把最多三个主要选项
加入问题，例如：

```text
What size or fit do you need: small, medium, or large?
```

## 6. 与其他模块的责任边界

### Module 1：Retrieval

- 从 catalog 召回最多 100 个 `Item`。
- 封装并输出有序的 `candidates_100: list[Candidate]`。
- 同一份 candidates 分别传给 3A 和 3B。

### Module 2：State

- 保存 session、画像、消息和轮次。
- 更新 hard/soft constraints、no preference 和 asked attributes。
- 处理 intent override；3B 只读取更新后的状态。

### Module 3A：Reranking

- 输入 `shopping_state + candidates_100`。
- 输出最多 10 个 `RankedCandidate` 用于最终 recommendations。
- 可以替换 scorer，而不改变 3B clarification policy。

### Module 3B：Dialogue

- 输入 `shopping_state + candidates_100`。
- 判断是否追问、询问哪个属性、如何表达问题。
- 不读取 `candidates_10`，不修改排名，不生成 recommendations。

### Agent / Orchestrator

- 调用 State、Retrieval、Reranking 和 Dialogue。
- 把 3A 的 `candidates_10` 转换成官方 recommendations。
- 保存 3B 本轮的 `asked_attributes`。
- 合并 `message`、`ask_attribute`、`recommendations` 和 `usage`。

## 7. 最小联调示例

```python
from src.dialogue import decide_ask, record_asked_attribute
from src.reranking import recommendations_from_ranking, rerank

# Module 1
candidates_100 = retrieval.search(
    shopping_state.user_message,
    top_k=100,
)

# Module 3A：只负责最终推荐排序
candidates_10 = rerank(
    shopping_state,
    candidates_100,
    top_k=10,
)

# Module 3B：直接分析 Module 1 Candidate，不把 3A Top 10 当成 Top 100
decision = decide_ask(
    shopping_state,
    candidates_100,
)
record_asked_attribute(shopping_state, decision["ask_attribute"])

response = {
    **decision,
    "recommendations": recommendations_from_ranking(candidates_10),
    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
}
```

## 8. 测试

运行 3B 测试：

```text
python -m unittest src.dialogue.test_three_b -v
```

测试覆盖：

- 非 Mapping `Candidate.item.to_dict()` 正式路径。
- 嵌套 item Mapping 和直接 catalog Mapping 兼容。
- Top 100 截断以及第 11～100 名对多样性计算的影响。
- hard/soft constraint、`AttributeValue`、no preference 和 asked attributes。
- dict State 与 object State。
- details 的 Material、Color、Brand 及文本 fallback。
- use_case 仅使用 features/details 和六个官方关键词。
- Fabric/Fit/Width 和 feature 排他分类与官方属性边界一致。
- 空候选、缺少所有分数字段和 turn >= 10。
- 语义分层 Base Priority、动态 Answerability、18 分 Question Value 上限。
- 排名靠前的属性覆盖对 Answerability 影响更大。
- normalized entropy、coverage、不同 cardinality 等权和问题模板行为。
- 不同 `user_profile` 不会改变 3B 的评分与选择结果。

## 9. 接口摘要

```python
from src.dialogue import decide_ask, record_asked_attribute
from src.item import Candidate

# Module 1 output
candidates_100: list[Candidate]

# Module 3B
decision = decide_ask(
    shopping_state=shopping_state,
    candidates_100=candidates_100,
)

# Module 2 / Agent persistence
record_asked_attribute(shopping_state, decision["ask_attribute"])
```
