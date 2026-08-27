# Reranking（Module 3A）

本目录负责将 Retrieval 返回的最多 100 个候选商品，结合模块 2 维护的
`shopping_state` 重新排序，并输出最多 10 个 `reranked_candidate`。

当前排序算法是可运行的占位基线。接口、对象和测试已经固定，后续可以将
占位打分替换为 Cross-Encoder、Learning-to-Rank、MMR 或更强的规则，而不必
修改 Retrieval、Dialogue 和官方 Agent 的数据边界。

## 1. 模块位置与数据流

```text
Official evaluator
  reset(session_id, user_profile)
  respond(session_id, user_message, turn, top_k)
                |
                v
Module 2: shopping_state
                |
                +-------------------------+
                |                         |
                v                         v
Module 1: Retrieval                 Module 3A: Reranking
output candidates_100  -----------> input candidates_100
                                      + shopping_state
                                            |
                                            v
                                 output candidates_10
                                            |
                         +------------------+------------------+
                         v                                     v
                 Module 3B: Dialogue                  Official response
                 clarification question               recommendations
```

主要文件：

- `src/item.py`：跨模块共用的商品对象。
- `src/reranking/reranker.py`：候选预处理、占位打分、排序和官方格式转换。
- `src/reranking/test_reranker.py`：对象接口与 Reranking 单元测试。
- `examples/reranker_demo.py`：可直接运行的模拟输入输出。

推荐使用显式导入：

```python
from src.item import item, candidate, reranked_candidate
from src.reranking import rerank, recommendations_from_ranking
```

## 2. 全局共享类

以下对象由多个模块共同使用，不属于某一个排序算法的私有实现。

### 2.1 `item`

`item` 表示官方 catalog 中的一件商品。它只包含 participant-visible 的 10 个
字段，字段顺序和 `data/catalog.jsonl` 保持一致。

| 字段 | Python 类型 | 说明 |
| --- | --- | --- |
| `parent_asin` | `str` | 商品唯一标识；官方最终按它评分，不能为空 |
| `title` | `str` | 商品标题 |
| `features` | `list[str]` | 商品卖点列表 |
| `description` | `list[str]` | 商品描述列表 |
| `price` | `float \| None` | 商品价格，数据缺失时为 `None` |
| `categories` | `list[str]` | 从大类到细类的分类路径 |
| `details` | `dict[str, Any]` | 材质、颜色、尺寸等非固定详情 |
| `average_rating` | `float \| None` | 平均评分 |
| `rating_number` | `int \| None` | 评分数量 |
| `store` | `str` | 店铺或品牌相关文本 |

从 catalog 字典创建对象：

```python
from src.item import item

product = item.from_dict(catalog_record)
```

转回可 JSON 序列化的官方数据形状：

```python
product_dict = product.to_dict()
```

`item` 同时实现了 `Mapping`，因此新旧代码都可以读取它：

```python
product.parent_asin
product["parent_asin"]
product.get("price")
```

### 2.2 `candidate`

`candidate` 继承 `item`，是 Retrieval 输出到 Reranking 的单个候选。

在所有 `item` 字段之外，它新增：

| 字段 | Python 类型 | 说明 |
| --- | --- | --- |
| `retrieval_score` | `float \| None` | Retrieval 相关性分数，约定越高越好 |
| `retrieval_rank` | `int \| None` | Retrieval 原始名次，从 1 开始 |

Retrieval 推荐这样创建对象：

```python
from src.item import candidate

retrieved = candidate.from_dict({
    "parent_asin": "B001...",
    "title": "Black Lightweight Running Shoes",
    "features": ["Lightweight", "Comfortable"],
    "description": ["Road running shoes"],
    "price": 79.99,
    "categories": ["Shoes", "Running"],
    "details": {"Material": "Mesh", "Color": "Black"},
    "average_rating": 4.5,
    "rating_number": 120,
    "store": "Example Store",
    "retrieval_score": 0.83,
    "retrieval_rank": 1,
})
```

### 2.3 `reranked_candidate`

`reranked_candidate` 继承 `item`，是 Reranking 的单个输出。它保留 Retrieval
信息，并添加模块 3A 的诊断信息。

| 字段 | Python 类型 | 说明 |
| --- | --- | --- |
| `retrieval_score` | `float \| None` | 原始 Retrieval 分数 |
| `retrieval_rank` | `int \| None` | 原始 Retrieval 名次 |
| `rank` | `int` | Reranking 后的新名次，从 1 开始 |
| `score` | `float` | 当前 Reranking 综合分数，不是概率 |
| `matched` | `list[str]` | 匹配到的约束/偏好属性名 |
| `violation` | `list[str]` | 违反的硬约束及原因 |

示例：

```python
reranked_candidate(
    parent_asin="B001...",
    # 其余 item 字段省略
    retrieval_score=0.83,
    retrieval_rank=2,
    rank=1,
    score=0.87,
    matched=["category", "material", "budget", "color"],
    violation=[],
)
```

兼容属性 `matched_attributes` 和 `violations` 暂时映射到 `matched` 和
`violation`，新代码应优先使用短名称。

### 2.4 列表类型别名

```python
from src.item import candidates_100, candidates_10

candidates_100 = list[candidate]
candidates_10 = list[reranked_candidate]
```

它们是类型别名，不是新的容器类。Reranking 在运行时只读取输入的前 100 个
有效候选，并强制 `top_k` 位于 1 到 10 之间。

## 3. Reranking 期待的输入

入口：

```python
candidates_10 = rerank(
    shopping_state=shopping_state,
    candidates_100=candidates_100,
    top_k=10,
)
```

### 3.1 `shopping_state`

真正的 `shopping_state` 类由模块 2 定义并维护。Reranking 不创建或修改它，
只按以下结构读取属性：

| 字段 | 来源 | 说明 | 当前 Reranking 是否使用 |
| --- | --- | --- | --- |
| `session_id` | 官方 `reset/respond` | 当前会话标识 | 否，只作为共享状态契约 |
| `user_profile` | 官方 `reset` | 匿名聚合用户画像 | 是，读取 `preference_tags` |
| `user_message` | 官方每次 `respond` | 当前用户消息 | 否，模块 2 应先将其解析为约束 |
| `turn` | 官方每次 `respond` | 当前轮次 1～10 | 否，预留给动态策略 |
| `intent` | 模块 2 推断 | 只能是 `buying` 或 `browsing` | 是，选择打分权重 |
| `hard_constraint` | 模块 2 提取 | 必须满足的属性和值 | 是 |
| `soft_constraint` | 模块 2 提取 | 满足则加分的偏好 | 是 |
| `no_prefernce` | 模块 2 提取 | 用户明确不关心的属性名 | 是，从排序条件中移除 |

示例对象的内容：

```python
shopping_state.session_id = "session-001"
shopping_state.user_profile = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.5,
    "rating_style": "usually positive",
    "preference_tags": ["comfort", "durability"],
    "summary": "Prior purchases emphasize comfort and durability.",
}
shopping_state.user_message = (
    "I want black lightweight running shoes under $100."
)
shopping_state.turn = 2
shopping_state.intent = "buying"
shopping_state.hard_constraint = {
    "category": "running shoes",
    "budget": {"max": 100},
    "material": "mesh",
}
shopping_state.soft_constraint = {
    "color": "black",
    "feature": ["lightweight", "comfortable"],
}
shopping_state.no_prefernce = ["brand"]
```

注意：当前团队接口使用的是 `no_prefernce` 这个拼写。Reranking 同时兼容
`no_preference`，但各模块联调时应统一使用同一个字段名。

`no_prefernce` 表示“不关心这个属性”，不是“拒绝某个值”：

```python
shopping_state.no_prefernce = ["brand"]
# 品牌不参与排序，也不会产生 brand:not_matched。
```

如果未来模块 2 增加结构化拒绝值，可以额外提供兼容字段：

```python
shopping_state.rejected_values = {
    "material": ["leather"],
}
```

### 3.2 `candidates_100`

标准输入是按 Retrieval 相关性从高到低排列的 `list[candidate]`：

```python
candidates_100 = [
    candidate(..., retrieval_score=0.95, retrieval_rank=1),
    candidate(..., retrieval_score=0.88, retrieval_rank=2),
    # 最多 100 个
]
```

要求：

- `parent_asin` 必须是非空且来自官方 catalog。
- 同一个 `parent_asin` 不应重复。
- `retrieval_score` 若提供，必须遵守“越高越好”。
- 最好同时提供 `retrieval_rank`，便于分析重排前后的名次变化。
- 迁移阶段仍兼容字典候选，但新代码应输出 `candidate` 对象。

Reranking 会再次校验、去重；无效候选会被忽略。若缺少可比较的
`retrieval_score`，占位实现会根据原始顺序生成递减分数。

## 4. 预期输出：`candidates_10`

标准输出是按新分数从高到低排列的 `list[reranked_candidate]`：

```python
candidates_10 = [
    reranked_candidate(
        parent_asin="A-MESH",
        title="Black Lightweight Mesh Running Shoes",
        features=["Lightweight", "Comfortable"],
        description=["Road running shoes"],
        price=79.99,
        categories=["Shoes", "Running"],
        details={"Material": "Mesh", "Color": "Black"},
        average_rating=4.5,
        rating_number=420,
        store="Example Store",
        retrieval_score=0.88,
        retrieval_rank=2,
        rank=1,
        score=0.65,
        matched=["category", "feature", "material", "budget", "color"],
        violation=[],
    ),
]
```

输出约束：

- 最多 10 个对象。
- `rank` 必须连续为 `1..len(candidates_10)`。
- 顺序就是最终推荐顺序。
- `matched` 和 `violation` 用于解释、调试及 Dialogue 决策。
- `score` 只用于内部排序；它不是置信概率，官方评分也会忽略它。
- Reranking 不修改传入的 `shopping_state` 或 `candidates_100`。

需要打印或写入 JSON 时：

```python
serializable = [value.to_dict() for value in candidates_10]
```

### 转换成官方 Agent 输出

```python
recommendations = recommendations_from_ranking(candidates_10)
```

得到：

```python
[
    {"parent_asin": "A-MESH"},
    {"parent_asin": "B-LEATHER"},
]
```

转换函数会过滤空 ID 和重复 ID。官方只评分前 10 个有效唯一
`parent_asin`；推荐项中的可选 `score` 即使提供也不会参与评分。

## 5. 当前占位排序

当前实现只用于跑通框架，不代表最终方案。

共同信号：

- 归一化后的 Retrieval 分数。
- 硬约束匹配率。
- 软约束匹配率。
- `user_profile.preference_tags` 的词语匹配率。
- 是否存在硬约束或明确拒绝值违规。

`buying` 权重：

```text
0.55 * retrieval
+ 0.25 * hard constraint match
+ 0.15 * soft constraint match
+ 0.05 * profile match
- 0.80 * violation penalty
```

`browsing` 权重：

```text
0.70 * retrieval
+ 0.10 * hard constraint match
+ 0.10 * soft constraint match
+ 0.10 * profile match
- 0.75 * violation penalty
```

普通属性当前使用英文 token 重合率；硬约束匹配率低于 `0.6` 时记录
`<attribute>:not_matched`。预算支持 `min/max`、数值和简单英文区间。

待优化项：

- 语义匹配、同义词和词形变化。
- Cross-Encoder 或 Learning-to-Rank。
- browsing 的 MMR/类目覆盖等真正多样性策略。
- 按属性类型设计匹配器。
- 更细粒度的违规严重度与可解释分项。
- 使用 public sessions 调整权重并做消融实验。

替换排序算法时，优先修改 `reranker.py` 中的 `_score_candidate()`，不要改变
`shopping_state + candidates_100 -> candidates_10` 的公共边界。

## 6. 与其他模块的责任边界

### 模块 1：Retrieval

- 输入 query/state 中可检索的信息。
- 从固定 catalog 召回最多 100 个商品。
- 创建 `candidate` 对象并输出有序 `candidates_100`。
- 保证 `retrieval_score` 越高越相关。

### 模块 2：State

- 在 `reset()` 时保存 `session_id` 和 `user_profile`。
- 每轮更新 `user_message` 和 `turn`。
- 从消息和历史推断 `intent`，官方不会直接提供 buying/browsing。
- 更新硬约束、软约束和无偏好属性。
- 处理 intent override；Reranking 只读取更新后的结果。

### 模块 3A：Reranking

- 不解析对话，不修改 `shopping_state`。
- 清洗并去重 `candidates_100`。
- 计算匹配、违规和综合分数。
- 输出稳定、有序的 `candidates_10`。

### 模块 3B：Dialogue

- 读取 `candidates_10` 的商品属性和排序结果。
- 判断是否需要追问以及应该询问哪个属性。
- 不负责修改商品排名。

### Agent / Orchestrator

- 管理 session 到 `shopping_state` 的映射。
- 依次调用 State、Retrieval、Reranking、Dialogue。
- 将 `candidates_10` 转换成官方 `recommendations`。
- 返回 `message`、`ask_attribute` 和 `recommendations`。

## 7. 运行示例与测试

在仓库根目录运行完整模拟：

```text
python examples/reranker_demo.py
```

运行全部测试：

```text
python -m unittest discover -v
```

Reranking 测试覆盖：

- `item` 是否与官方 catalog 字段一致。
- `candidate` / `reranked_candidate` 的继承关系。
- 硬约束能否改变 Retrieval 原始顺序。
- 输出是否为 `list[reranked_candidate]`。
- `no_prefernce` 是否真正从排序中排除属性。
- `intent` 是否只允许 buying/browsing。
- 重复和无效候选是否被删除。
- `top_k` 和官方格式转换是否正确。
- 输入对象是否保持不变。
- 空候选是否安全返回空列表。
- 是否禁止输出超过 10 个候选。
- `candidates_10` 是否能被 Dialogue 模块读取。

## 8. 当前接口摘要

```python
# Shared data classes
from src.item import item, candidate, reranked_candidate

# Module 1 output
candidates_100: list[candidate]

# Module 3A
from src.reranking import rerank
candidates_10: list[reranked_candidate] = rerank(
    shopping_state,
    candidates_100,
    top_k=10,
)

# Official output adapter
from src.reranking import recommendations_from_ranking
recommendations = recommendations_from_ranking(candidates_10)
```
