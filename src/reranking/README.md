# Reranking（Module 3A）

本目录负责将 Retrieval 返回的最多 100 个候选，结合模块 2 维护的
`shopping_state` 重新排序，并输出最多 10 个 `RankedCandidate`。

当前算法是可运行、可解释的规则基线。共享对象和模块边界已经固定，后续可以在
不改变上下游接口的前提下替换为 Cross-Encoder、Learning-to-Rank 或 MMR。

## 1. 数据流

```text
Official evaluator
  reset(session_id, user_profile)
  respond(session_id, user_message, turn, top_k)
                 |
                 v
Module 2: shopping_state
                 |
                 +---------------------------+
                 |                           |
                 v                           v
Module 1: Retrieval                    Module 3A: Reranking
output candidates_100  --------------> shopping_state + candidates_100
                                             |
                                             v
                                  output candidates_10
                                             |
                         +-------------------+-------------------+
                         v                                       v
                 Module 3B: Dialogue                    Official response
                 clarification question                 recommendations
```

主要文件：

- `src/item.py`：跨模块共享的数据类。
- `src/attribute.py`：hard/soft constraint 共用的标准属性契约。
- `src/reranking/constraint_matcher.py`：属性专用三态约束匹配。
- `src/reranking/feature_extractor.py`：将匹配结果转换为候选排序特征。
- `src/reranking/scorers/rule_scorer.py`：S1 Rule/Fuzzy 相关度基线。
- `src/reranking/reranker.py`：候选清洗、策略融合、排序和官方格式转换。
- `src/reranking/test_reranker.py`：对象契约及 Reranking 测试。
- `examples/reranker_demo.py`：完整模拟输入输出。

推荐显式导入：

```python
from src.item import Item, Candidate, RankedCandidate
from src.attribute import AttributeName, AttributeValue, normalize_attribute_map
from src.reranking import rerank, recommendations_from_ranking
```

## 2. 为什么使用组合

`Item` 是 catalog 商品；`Candidate` 和 `RankedCandidate` 是算法阶段的结果，
它们不是新的商品类型。因此使用 has-a 关系，而不是 is-a 继承：

```text
Candidate.item ---------> Item
RankedCandidate.item ---> Item
```

代码关系：

```python
class Candidate:
    item: Item

class RankedCandidate:
    item: Item
```

优点：

- catalog 商品字段只有一份，不会在不同阶段重复复制。
- BM25、Dense、Reranking 分数不会污染 `Item`。
- 各模块职责清晰，后续增加新模型分数只改包装类。
- `RankedCandidate.item` 可以直接复用输入 `Candidate.item`。

## 3. 全局共享类

这些类定义在 `src/item.py`，可能被 Retrieval、Reranking、Dialogue 和 Agent
共同使用。

### 3.1 `Item`

`Item` 对应官方 `data/catalog.jsonl` 中的一件商品，只包含 participant-visible
的 10 个字段。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `parent_asin` | `str` | 商品唯一 ID，不能为空；官方最终按它评分 |
| `title` | `str` | 商品标题 |
| `features` | `list[str]` | 商品卖点列表 |
| `description` | `list[str]` | 商品描述列表 |
| `price` | `float \| None` | 商品价格 |
| `categories` | `list[str]` | 从大类到细类的分类路径 |
| `details` | `dict[str, Any]` | 材质、颜色、尺寸等非固定详情 |
| `average_rating` | `float \| None` | 平均评分 |
| `rating_number` | `int \| None` | 评分数量 |
| `store` | `str \| None` | 店铺或品牌相关文本 |

从 catalog 记录创建：

```python
from src.item import Item

product = Item.from_dict(catalog_record)
```

转回可 JSON 序列化的 catalog 形状：

```python
product_dict = product.to_dict()
```

`Item` 实现了 `Mapping`，兼容对象和字典两种读取方式：

```python
product.parent_asin
product["parent_asin"]
product.get("price")
```

### 3.2 `Candidate`

`Candidate` 是 Retrieval 输出到 Reranking 的单个候选。它通过 `item` 组合
`Item`，并保存 Retrieval 阶段分数。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `item` | `Item` | catalog 商品对象 |
| `bm25_score` | `float \| None` | BM25 分数 |
| `dense_score` | `float \| None` | Dense Retrieval 分数 |
| `retrieval_score` | `float \| None` | Retrieval 融合分数，约定越高越好 |
| `retrieval_rank` | `int \| None` | Retrieval 原始名次，从 1 开始 |

推荐创建方式：

```python
from src.item import Candidate, Item

product = Item.from_dict(catalog_record)

retrieved = Candidate(
    item=product,
    bm25_score=9.2,
    dense_score=0.82,
    retrieval_score=0.91,
    retrieval_rank=1,
)
```

也可以从嵌套字典创建：

```python
retrieved = Candidate.from_dict({
    "item": {
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
    },
    "bm25_score": 9.2,
    "dense_score": 0.82,
    "retrieval_score": 0.91,
    "retrieval_rank": 1,
})
```

访问商品字段：

```python
retrieved.item.parent_asin
retrieved.item.title
retrieved.retrieval_score
```

### 3.3 `RankedCandidate`

`RankedCandidate` 是 Reranking 的单个输出。它同样组合 `Item`，保留 Retrieval
诊断分数，并增加重排结果。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `item` | `Item` | 与输入候选共享的商品对象 |
| `bm25_score` | `float \| None` | 原始 BM25 分数 |
| `dense_score` | `float \| None` | 原始 Dense 分数 |
| `retrieval_score` | `float \| None` | 原始 Retrieval 融合分数 |
| `retrieval_rank` | `int \| None` | Retrieval 原始名次 |
| `rerank_score` | `float` | Reranking 综合分数，不是概率 |
| `rerank_rank` | `int` | Reranking 后的新名次，从 1 开始 |
| `matched` | `list[str]` | 匹配到的约束/偏好属性 |
| `violation` | `list[str]` | 违反的硬约束及原因 |

示例：

```python
RankedCandidate(
    item=product,
    bm25_score=9.2,
    dense_score=0.82,
    retrieval_score=0.91,
    retrieval_rank=2,
    rerank_score=0.87,
    rerank_rank=1,
    matched=["category", "material", "budget", "color"],
    violation=[],
)
```

新代码应使用 `rerank_score` 和 `rerank_rank`。为方便旧代码迁移，当前还提供
只读别名 `score`、`rank`、`matched_attributes` 和 `violations`。

### 3.4 列表类型别名

```python
from src.item import Candidates100, Candidates10

Candidates100 = list[Candidate]
Candidates10 = list[RankedCandidate]
```

兼容期仍导出旧的小写别名，但新代码统一使用 `Item`、`Candidate`、
`RankedCandidate`、`Candidates100` 和 `Candidates10`。

### 3.5 `AttributeName`、`AttributeValue` 和 `AttributeMap`

模块 2 的 `hard_constraint` 与 `soft_constraint` 统一使用：

```python
AttributeMap = dict[AttributeName, AttributeValue]
```

`AttributeName` 的最终标准字段为：

```text
category, material, color, size, style, brand,
budget, feature, use_case, other
```

`AttributeValue` 统一表达三种常见值，避免不同字段分别使用 `list`、`dict`
和 `None`：

```python
AttributeValue(
    values=["black", "white"],       # 离散文本值
    minimum=50,                       # 可选数值下界
    maximum=100,                      # 可选数值上界
    unit="USD",                      # 可选单位
    details={"waist": ["32 inches"]},  # size/other 等子字段
)
```

建议模块 2 对 LLM 或规则提取结果统一调用：

```python
hard_constraint = normalize_attribute_map(raw_hard_constraint)
soft_constraint = normalize_attribute_map(raw_soft_constraint)
```

常见别名会自动归一化到官方十类，例如 `categories -> category`、
`Fabric Type -> material`、`Department -> style`、
`Occasion -> use_case`。不能归类的字段不会丢弃，而是保存为：

```python
AttributeName.OTHER: AttributeValue(
    details={"care_instructions": ["hand wash only"]}
)
```

## 4. Reranking 输入

入口：

```python
candidates_10 = rerank(
    shopping_state=shopping_state,
    candidates_100=candidates_100,
    top_k=10,
)
```

### 4.1 `shopping_state`

真正的 `shopping_state` 类由模块 2 定义和维护。Reranking 不创建、不修改它，
只读取以下结构：

| 字段 | 来源 | 含义 | 当前是否用于排序 |
| --- | --- | --- | --- |
| `session_id` | 官方 `reset/respond` | 当前会话 ID | 否，只作为共享契约 |
| `user_profile` | 官方 `reset` | 匿名聚合画像 | 是，读取 `preference_tags` |
| `user_message` | 官方每轮 `respond` | 当前消息 | 否，模块 2 应先解析成约束 |
| `turn` | 官方每轮 `respond` | 当前轮次 1～10 | 否，预留给动态策略 |
| `intent` | 模块 2 推断 | `buying` 或 `browsing` | 是，选择权重 |
| `hard_constraint` | 模块 2 提取 | 必须满足的属性和值 | 是 |
| `soft_constraint` | 模块 2 提取 | 满足则加分的偏好 | 是 |
| `no_prefernce` | 模块 2 提取 | 用户明确不关心的属性名 | 是，排除对应条件 |
| `rejected_values` | 模块 2 提取 | 用户明确排除的属性值 | 是，命中则标记违规 |

示例：

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
shopping_state.hard_constraint = normalize_attribute_map({
    "category": "running shoes",
    "budget": {"max": 100},
    "material": "mesh",
})
shopping_state.soft_constraint = normalize_attribute_map({
    "color": "black",
    "feature": ["lightweight", "comfortable"],
})
shopping_state.no_prefernce = [AttributeName.BRAND]
shopping_state.rejected_values = normalize_attribute_map({
    "material": "leather",
})
```

注意：团队当前接口使用 `no_prefernce` 这个拼写。Reranking 也兼容
`no_preference`，但联调时应统一字段名。

`no_prefernce=["brand"]` 表示品牌不影响排序，不表示拒绝某个品牌。

### 4.2 `candidates_100`

标准输入是按 Retrieval 相关性从高到低排列的 `list[Candidate]`：

```python
candidates_100 = [
    Candidate(
        item=product_a,
        bm25_score=9.2,
        dense_score=0.82,
        retrieval_score=0.95,
        retrieval_rank=1,
    ),
    Candidate(
        item=product_b,
        bm25_score=8.7,
        dense_score=0.91,
        retrieval_score=0.88,
        retrieval_rank=2,
    ),
]
```

要求：

- `Candidate.item.parent_asin` 必须非空并来自官方 catalog。
- 同一个 `parent_asin` 不应重复。
- `retrieval_score` 若提供，必须遵守“越高越好”。
- 建议同时提供各分项分数和 `retrieval_rank`，方便调试和消融。
- 最多传入 100 个候选；超过部分不会参与 Reranking。
- 迁移阶段兼容旧的字典候选，新代码应输出 `Candidate` 对象。

Reranking 会再次校验、去重。若没有可比较的 `retrieval_score`，占位实现会
根据输入顺序生成递减分数，但不会回写或修改输入对象。

## 5. Reranking 输出

标准输出是按 `rerank_score` 从高到低排列的 `list[RankedCandidate]`：

```python
candidates_10 = [
    RankedCandidate(
        item=product_b,
        bm25_score=8.7,
        dense_score=0.91,
        retrieval_score=0.88,
        retrieval_rank=2,
        rerank_score=0.65,
        rerank_rank=1,
        matched=["category", "feature", "material", "budget", "color"],
        violation=[],
    ),
]
```

输出约束：

- 最多 10 个对象。
- `rerank_rank` 连续为 `1..len(candidates_10)`。
- 列表顺序就是最终推荐顺序。
- `RankedCandidate.item` 复用输入中的 `Item`，不复制 catalog 实体。
- `matched` 和 `violation` 可供解释、调试和 Dialogue 使用。
- `rerank_score` 只用于内部排序，官方评分会忽略推荐项中的分数。
- Reranking 不修改 `shopping_state` 或 `candidates_100`。

转换为 JSON：

```python
serializable = [value.to_dict() for value in candidates_10]
```

得到嵌套结构：

```python
{
    "item": {
        "parent_asin": "B001...",
        "title": "...",
        # 其余 catalog 字段
    },
    "bm25_score": 8.7,
    "dense_score": 0.91,
    "retrieval_score": 0.88,
    "retrieval_rank": 2,
    "rerank_score": 0.65,
    "rerank_rank": 1,
    "matched": ["category", "material", "budget"],
    "violation": [],
}
```

### 转换成官方输出

```python
recommendations = recommendations_from_ranking(candidates_10)
```

结果：

```python
[
    {"parent_asin": "B001..."},
    {"parent_asin": "B002..."},
]
```

转换函数从 `RankedCandidate.item.parent_asin` 读取 ID，并过滤空值和重复值。
官方只评分前 10 个有效唯一 ID。

## 6. Constraint Matcher 与当前排序

`constraint_matcher.py` 已替换原先所有属性共用的 token-overlap 判断。单条约束
的正式内部输出为：

```python
ConstraintMatch(
    attribute=AttributeName.MATERIAL,
    status=MatchStatus.SATISFIED,
    score=1.0,
    requested_values=["mesh"],
    observed_values=["Mesh"],
    evidence=["requested 'mesh' matched observed 'Mesh' (1.00)"],
)
```

三种状态的含义：

| 状态 | 含义 | 当前排序处理 |
| --- | --- | --- |
| `SATISFIED` | 商品信息明确满足约束 | 计入 hard/soft 匹配分 |
| `UNKNOWN` | 商品信息缺失，或只能确认部分开放型要求 | 不记为违规，保留候选 |
| `VIOLATED` | 已知商品值与封闭型约束冲突，或命中 rejected value | 写入 `violation` 并施加惩罚 |

调用单条匹配：

```python
from src.reranking import match_constraint

result = match_constraint(
    product,
    AttributeName.BUDGET,
    AttributeValue(maximum=100, unit="USD"),
)
```

调用完整候选匹配：

```python
matcher = ConstraintMatcher()
matches = matcher.match_candidate(
    product,
    hard=shopping_state.hard_constraint,
    soft=shopping_state.soft_constraint,
    rejected=shopping_state.rejected_values,
)
```

第一版匹配规则：

- `budget` 使用数值上下界；商品缺少价格时返回 `UNKNOWN`。
- `category/material/color/size/style/brand` 属于封闭型属性：商品已有明确值但不匹配
  才返回 `VIOLATED`。
- `feature/use_case/other` 属于开放型属性：未找到值默认返回 `UNKNOWN`，因为商品
  metadata 很可能不完整。
- `category/material/color/size/style/brand` 的多个请求值默认按 `ANY` 解释。
- `feature/use_case/other` 默认按 `ALL` 解释；可通过
  `ConstraintMatcherConfig` 切换为 `ANY` 做消融实验。
- rejected value 使用反向语义：发现被拒绝值为 `VIOLATED`，明确存在其他值为
  `SATISFIED`，缺少信息仍为 `UNKNOWN`。
- 结构化属性优先；只有结构化信息不足时才使用与该属性对应的商品文本 fallback。

### 6.1 Candidate Feature Extractor

`feature_extractor.py` 将一个候选的全部 matcher 结果转换为
`CandidateSignals`，主要包括：

```text
normalized_retrieval_score / retrieval_rank_score
bm25_score / dense_score
hard_satisfied_count / hard_unknown_count / hard_violation_count
hard_match_score / hard_weighted_score
soft_satisfied_count / soft_unknown_count / soft_violation_count
soft_match_score / soft_weighted_score
rejected_match_count / rejected_unknown_count / rejected_weighted_score
profile_match_score / semantic_score
feasibility_tier / soft_penalty_adjustment
```

`constraint_matches` 原样保留在 `CandidateSignals` 中。因此更换权重、融合方式或
排序策略时不需要重新执行商品匹配，诊断证据也不会丢失。

### 6.2 S1：Rule/Fuzzy Relevance Scorer

`scorers/rule_scorer.py` 是完全本地、无模型和无新增第三方依赖的 S1 基线。
输入为当前有效的 hard/soft constraints 和一个 `Item`，输出统一的：

```python
RelevanceScore(
    score=0.86,
    attribute_scores={AttributeName.CATEGORY: 0.90},
    phrase_score=0.80,
    token_overlap_score=0.92,
    fuzzy_score=0.84,
    category_score=0.90,
    numeric_score=1.00,
    matched_terms=["running shoes", "mesh"],
    evidence=[...],
)
```

当前计算包含：

- phrase exact/substring match；
- token overlap；
- 基于标准库 `SequenceMatcher` 的 fuzzy similarity 和 typo rescue；
- 小型属性同义词归一化，例如 `water resistant -> waterproof`；
- category 层级与 title 的组合匹配；
- budget 数值范围与超界距离；
- hard constraint 权重 2、soft constraint 权重 1；
- 属性级分数与全局商品文本分数融合。

Scorer 不读取 rejected values，它们继续由 Constraint Matcher 独立处理。默认只读
State 中当前有效的 hard/soft constraints；只有两者都为空时，才使用当前
`user_message` 作为 fallback，不读取完整 history。

调用示例：

```python
scorer = RuleFuzzyScorer()
relevance = scorer.score(
    product,
    hard_constraints=shopping_state.hard_constraint,
    soft_constraints=shopping_state.soft_constraint,
    query_text=shopping_state.user_message,
)
```

`relevance.score` 写入 `CandidateSignals.semantic_score`。统一接口为
`RelevanceScorer`，后续的 Cross-Encoder 可以直接替换 S1。

### 6.3 Buying：Feasibility Tier

Buying 先按可行性分层，再在同一层内按当前相关度分数排序：

```text
Tier 0：所有 hard 明确满足，或者当前没有 hard
Tier 1：没有 hard violation，但至少一个 hard UNKNOWN
Tier 2：至少一个 hard VIOLATED
Tier 3：命中任意 rejected value
```

排序键等价于：

```python
(feasibility_tier, -relevance_score, original_order, parent_asin)
```

当前同层相关度为：

```text
0.20 * retrieval
+ 0.35 * S1 semantic relevance
+ 0.30 * hard match score
+ 0.10 * soft match score
+ 0.05 * profile match
```

这保证低 Retrieval 分但完整满足 hard 的商品，仍排在高 Retrieval 但 hard 未知或
明确冲突的商品前面。Tier 只改变顺序，不删除候选。

### 6.4 Browsing：Soft Penalty

Browsing 不使用 Tier，也不删除违反约束的候选，而是使用分数调整：

```text
0.60 * retrieval
+ 0.30 * S1 semantic relevance
+ 0.10 * profile match
+ soft_penalty_adjustment
```

第一版 adjustment 权重为：

```text
hard SATISFIED  +2.0
hard UNKNOWN     0.0
hard VIOLATED   -4.0
soft SATISFIED  +1.0
soft UNKNOWN     0.0
soft VIOLATED    0.0
rejected match  -6.0
```

这些权重集中在 `ConstraintFeatureWeights`，可以替换后直接做消融实验。

当前仍属于可解释的规则基线，后续应重点优化：

- 语义匹配、同义词和词形变化。
- Cross-Encoder 或 Learning-to-Rank。
- browsing 的 MMR/类目覆盖等多样性策略。
- 属性同义词表、尺寸单位换算和更细的类型专用 matcher。
- 更细粒度的违规严重度和可解释分项。
- 使用 public sessions 调整权重并做消融实验。

替换算法时应优先调整 `ConstraintFeatureWeights`、`RerankerStrategyConfig` 或
scorer，不要改变 `shopping_state + candidates_100 -> candidates_10` 的边界。

## 7. 模块责任边界

### Retrieval

- 从固定 catalog 召回最多 100 个 `Item`。
- 计算 BM25、Dense 和融合分数。
- 封装为 `Candidate` 并输出有序 `candidates_100`。

### State

- 在 `reset()` 保存 `session_id` 和 `user_profile`。
- 每轮更新 `user_message` 和 `turn`。
- 从消息与历史推断 `intent`；官方不会直接传 buying/browsing。
- 更新硬约束、软约束和无偏好属性。

### Reranking

- 不解析对话，不修改 `shopping_state`。
- 清洗并去重 `candidates_100`。
- 计算匹配、违规和综合分数。
- 输出稳定、有序的 `candidates_10`。

### Dialogue

- 读取 `RankedCandidate.item` 中的商品属性及排序诊断。
- 判断是否追问以及询问哪个属性。
- 不负责修改商品排名。

### Agent / Orchestrator

- 管理 session 到 `shopping_state` 的映射。
- 依次调用 State、Retrieval、Reranking、Dialogue。
- 把 `candidates_10` 转换成官方 `recommendations`。

## 8. 示例与测试

运行模拟：

```text
python examples/reranker_demo.py
```

运行全部测试：

```text
python -m unittest discover -v
```

Reranking 测试覆盖：

- `Item` 是否与官方 catalog 字段一致。
- `Candidate` / `RankedCandidate` 是否使用组合而不是继承。
- `RankedCandidate.item` 是否复用原始 `Item`。
- BM25、Dense、Retrieval 分数是否保留到输出。
- 硬约束能否改变 Retrieval 原始顺序。
- CandidateSignals 的计数、匹配分、Soft Penalty 和 Tier 是否正确。
- S1 的 phrase/token/fuzzy/synonym/category/budget 分项是否正确。
- S1 是否能在同一 Buying Tier 内以及 Browsing 总分中改变顺序。
- Buying 是否严格先按 Tier、再按同层相关度排序。
- Browsing 是否只调分、不删除候选，也不意外启用 Tier。
- 输出是否为 `list[RankedCandidate]` 且名次连续。
- `no_prefernce` 是否真正排除属性。
- `intent` 是否只允许 buying/browsing。
- 重复和无效候选是否被清理。
- `top_k`、空输入和官方格式转换是否正确。
- 输入对象是否保持不变。
- `candidates_10` 是否能被 Dialogue 读取。

## 9. Replay Evaluator

Reranking 专用离线评测位于 `src/reranking/replay/`。先录制固定的 State 与
Candidates100，再让不同 Reranker 重放同一批输入：

```text
python -m src.reranking.replay.recorder
python -m src.reranking.replay.evaluator artifacts/reranking_replay/<dataset-run>
```

每次录制都会创建不可覆盖的新版本目录。`manifest.json` 保存整体 Git commit、
branch、dirty 状态、各模块最近一次提交、当前文件哈希、Catalog/public set 哈希、
运行命令与生成策略。评测报告同时记录“数据生成代码版本”和“被测试 Reranker
代码版本”，避免用不同上游输入得出错误对比。

目标 ASIN 单独保存在 `labels.jsonl`，不会出现在 `cases.jsonl.gz` 或传给
Reranker。详细格式、命令和指标见 `src/reranking/replay/README.md`。

所有正式实验使用 `RR-000`、`RR-001` 等唯一编号，并登记在
`src/reranking/experiments/EXPERIMENTS.md`。结果目录必须以同一实验编号命名，
一个编号只允许对应一套 Reranker 配置。

## 10. 接口摘要

```python
from src.item import Item, Candidate, RankedCandidate
from src.reranking import rerank, recommendations_from_ranking

# Module 1
candidates_100: list[Candidate]

# Module 3A
candidates_10: list[RankedCandidate] = rerank(
    shopping_state,
    candidates_100,
    top_k=10,
)

# Agent output
recommendations = recommendations_from_ranking(candidates_10)
```
