# Reranking 优化与实验方案

本文档定义 Module 3A（Reranking）的下一阶段实现方案、实验变量、评测方法和
模块边界。目标是在不改变现有上下游接口的前提下，将当前占位
`SimpleReranker` 逐步升级为可解释、可评测、可替换的商品精排系统。

## 1. 范围与职责边界

### 1.1 Module 3A 负责

- 将当前有效的 `shopping_state` 序列化为排序需求。
- 判断每个候选对 hard constraint、soft constraint 和 rejected value 的满足情况。
- 提取候选级排序特征。
- 计算规则相关度和可选的神经模型相关度。
- 融合 Retrieval、约束、语义和用户画像信号。
- 为 Buying 和 Browsing 采用不同的最终选择策略。
- 输出最多 10 个 `RankedCandidate`。
- 提供 reranking 专用 replay evaluator 和消融实验报告。

### 1.2 Module 3A 不负责

- 不修改 `shopping_state`，失效约束由 State 模块清理。
- 不重新搜索 50,000 件商品，候选召回由 Retrieval 模块负责。
- 不决定下一轮具体追问字段，追问由 Dialogue 模块负责。
- 不在运行时读取目标 `parent_asin`。

### 1.3 Product Attribute View 由 1 号负责

商品字段统一、`details` 字段别名处理、商品属性值标准化和
`Product Attribute View` 的实现由 1 号负责。Reranking 不重复实现这一层，
但需要和 1 号确认至少能够读取以下信息：

```text
parent_asin
category
material
color
size
style
brand
price
feature
use_case
other
searchable_text
```

Reranking 对这层的最低语义约定：

- 离散属性使用标准化后的 `list[str]`。
- 数值属性使用 `float | None` 或明确的数值范围结构。
- 缺失字段表示为 `None` 或空列表，不能用空字符串伪装成已知值。
- `searchable_text` 仅作为结构化字段无法判断时的 fallback。
- 原始 `Item` 必须保留，最终 `RankedCandidate.item` 仍引用原商品对象。

在 1 号接口尚未稳定前，可以通过 `Protocol` 或 adapter 接入，不要在
Reranking 内复制另一套商品标准化逻辑。

## 2. 保持不变的公共接口

输入：

```python
shopping_state: ShoppingState
candidates_100: list[Candidate]
```

输出：

```python
candidates_10: list[RankedCandidate]
```

正式入口保持：

```python
candidates_10 = rerank(
    shopping_state=shopping_state,
    candidates_100=candidates_100,
    top_k=10,
)
```

最终仍使用现有共享对象：

```python
RankedCandidate(
    item=candidate.item,
    bm25_score=candidate.bm25_score,
    dense_score=candidate.dense_score,
    retrieval_score=candidate.retrieval_score,
    retrieval_rank=candidate.retrieval_rank,
    rerank_score=final_score,
    rerank_rank=rank,
    matched=["category", "material", "budget"],
    violation=["color:not_matched"],
)
```

更详细的匹配证据和分项得分只作为 Reranking 内部诊断信息，不扩大官方
Agent response。

## 3. 建议的数据流

```text
ShoppingState + Candidates100 + Product Attribute View（1号）
                         |
                         v
              1. Query Serializer
                         |
                         v
              2. Constraint Matcher
                         |
                         v
              3. Feature Extractor
                         |
                         v
       4. Rule / Cross-Encoder / LTR Scorer
                         |
                         v
                   5. Fusion
                         |
                         v
        6. Intent-aware Final Selection
                         |
                         v
               RankedCandidate Top 10
```

每一层都应可以独立替换和关闭，以便进行消融实验。

## 4. Query Serializer

### 4.1 目标

只表达当前仍然有效的购物需求，不直接拼接可能包含旧需求的完整原始对话。

建议输出：

```text
Intent: buying
Category: running shoes
Required:
- budget <= 100 USD
- material = mesh
Preferred:
- color = black
- feature = lightweight
Avoid:
- material = leather
No preference:
- brand
```

建议接口：

```python
def serialize_shopping_state(state: ShoppingState) -> str:
    ...
```

### 4.2 规则

- 只读取 State 已确认的有效约束。
- hard、soft、rejected 和 no preference 必须明确分区。
- 数值范围必须保留比较方向和单位。
- 字段顺序固定，保证实验可复现。
- 不包含 `session_id`、目标商品 ID 等无关信息。
- `history` 只用于 State 上游蒸馏，不默认输入 Reranker 模型。

### 4.3 可实验选择

| 编号 | Query 表达 | 说明 |
| --- | --- | --- |
| Q1 | 紧凑关键词 | 只拼接 category 和约束值，速度快 |
| Q2 | 带字段标签的结构化文本 | 推荐默认方案，语义边界清楚 |
| Q3 | 结构化文本 + 当前用户消息 | 信息更多，但可能引入重复或旧信息 |

第一轮建议比较 Q1 和 Q2，暂不默认使用完整历史。

## 5. Constraint Matcher

### 5.1 统一匹配结果

```python
class MatchStatus(str, Enum):
    SATISFIED = "satisfied"
    UNKNOWN = "unknown"
    VIOLATED = "violated"


@dataclass
class ConstraintMatch:
    attribute: AttributeName
    status: MatchStatus
    score: float
    requested_values: list[str]
    observed_values: list[str]
    evidence: list[str]
```

当前实现位于 `constraint_matcher.py`，并额外提供：

```python
class MultiValuePolicy(str, Enum):
    ANY = "any"
    ALL = "all"

matcher.match_candidate(
    product,
    hard=shopping_state.hard_constraint,
    soft=shopping_state.soft_constraint,
    rejected=shopping_state.rejected_values,
)
```

`CandidateConstraintMatches` 分别保留 hard、soft、rejected 的完整匹配结果，并
提供 hard satisfied/unknown/violation 和 rejected hit 的计数，供下一步 Feature
Extractor 直接使用。

必须区分：

```text
SATISFIED：商品信息明确满足要求
UNKNOWN：商品没有足够信息作出判断
VIOLATED：商品信息明确与要求冲突
```

例如预算不超过 100 美元：

```text
price = 80     -> SATISFIED
price = 150    -> VIOLATED
price = None   -> UNKNOWN
```

缺失值不能直接作为 violation。当前 catalog 中大量商品缺少 price 或部分
`details`，严格删除所有缺失值商品会产生较高误杀风险。

### 5.2 属性专用 matcher

不要让全部属性共用同一个 token overlap 函数。

| 属性 | 主判断方式 | Fallback |
| --- | --- | --- |
| `category` | 类别层级和标准化类别值 | title/token/fuzzy |
| `material` | 标准化材质值 | features/searchable text |
| `color` | 标准化颜色值和别名 | title/searchable text |
| `size` | 尺码类型与单位比较 | 文本匹配 |
| `brand` | brand/manufacturer | store/title |
| `budget` | 数值上下界比较 | 无；缺失即 UNKNOWN |
| `feature` | 关键词、同义词、语义分数 | fuzzy/semantic model |
| `use_case` | 场景词和语义分数 | semantic model |
| `style` | style/fit/pattern 等映射后的结构值 | title/features |
| `other` | 对应的原始子字段 | searchable text |

### 5.3 多值语义

需要在配置中约定一个属性有多个值时的含义：

- `color=[black, navy]` 通常表示 OR，即任意一个可接受。
- `feature=[lightweight, waterproof]` 可能表示 AND，即希望同时满足。
- rejected values 中任意一个命中都应视为风险。

第一版可采用：

```text
category/material/color/size/brand/style -> ANY
feature/use_case/other                   -> ALL
rejected_values                          -> ANY_MATCH_VIOLATES
```

之后通过实验判断 `feature/use_case` 使用 ALL 还是按覆盖率更合理。

## 6. Hard Constraint 策略实验

### 6.1 H1：Soft Penalty

不删除候选，只通过分数调整：

```text
hard SATISFIED  +2.0
hard UNKNOWN     0.0
hard VIOLATED   -4.0
soft SATISFIED  +1.0
soft UNKNOWN     0.0
soft VIOLATED    0.0
rejected match  -6.0
```

优点：对 metadata 缺失更安全。缺点：明确违反约束的商品仍可能进入 Top 10。

### 6.2 H2：Feasibility Tier

先按可行性分层，再在层内按相关度排序：

```text
Tier 0：全部 hard constraint 明确满足
Tier 1：没有违反，但存在 UNKNOWN
Tier 2：至少违反一个 hard constraint
Tier 3：命中 rejected value
```

```python
sort_key = (
    feasibility_tier,
    -semantic_score,
    -retrieval_score,
)
```

优点：Buying 的约束执行更可靠。缺点：State 或属性解析错误时影响较大。

### 6.3 第一版采用策略

```text
Buying：H2 Feasibility Tier
Browsing：H1 Soft Penalty
```

两条路径均不删除候选。Buying 使用 `(tier, -relevance)` 排序；Browsing 完全不
读取 tier，只把 H1 权重计入最终分数。后续消融仍应保留 H1/H2 对调配置，并单独
记录 promotion 和 demotion。

## 7. Candidate Feature Extractor

建议的内部对象：

```python
@dataclass
class CandidateSignals:
    candidate: Candidate
    constraint_matches: CandidateConstraintMatches

    normalized_retrieval_score: float
    dense_score: float | None
    bm25_score: float | None
    retrieval_rank_score: float

    hard_satisfied_count: int
    hard_unknown_count: int
    hard_violation_count: int
    hard_match_score: float
    hard_weighted_score: float

    soft_satisfied_count: int
    soft_unknown_count: int
    soft_violation_count: int
    soft_match_score: float
    soft_weighted_score: float

    rejected_match_count: int
    rejected_unknown_count: int
    rejected_weighted_score: float
    profile_match_score: float

    semantic_score: float | None
    feasibility_tier: int
    soft_penalty_adjustment: float
```

当前实现位于 `feature_extractor.py`。每个属性的 `ConstraintMatch` 保存在内部
signals 中，但不把完整诊断塞入官方 response。

### 7.1 Retrieval 分数标准化选择

| 编号 | 方案 | 说明 |
| --- | --- | --- |
| N1 | 每个 query 内 min-max | 简单，但容易受极端分数影响 |
| N2 | rank score：`1 / (k + rank)` | 稳定且不依赖分数尺度 |
| N3 | z-score/robust scaling | 适合后续线性融合 |

第一轮保留 N1 作为基线，同时增加 N2 对照。

## 8. 语义相关度 Scorer

### 8.1 S1：规则与 fuzzy baseline

包括：

- token/phrase overlap
- 属性同义词表
- fuzzy string match
- 数值距离
- 类别层级匹配

优点是快速、确定、可解释，并且可以作为所有模型方案的 fallback。

当前实现位于 `scorers/rule_scorer.py`，使用统一接口：

```python
relevance = scorer.score(
    product,
    hard_constraints=shopping_state.hard_constraint,
    soft_constraints=shopping_state.soft_constraint,
    query_text=shopping_state.user_message,
)
```

输出 `RelevanceScore`，包括总分、attribute scores、phrase/token/fuzzy、category、
numeric、matched terms 和 evidence。S1 不读取 rejected values；只有 State 没有
结构化正向约束时才使用当前消息 fallback，避免旧 history 污染排序。

### 8.2 S2：MiniLM Cross-Encoder

第一个建议实验的神经 reranker：

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

输入：

```python
(serialized_shopping_state, serialized_product)
```

输出：

```python
semantic_score: float
```

建议比较：

```text
S2-20：只重排规则候选 Top 20
S2-50：只重排规则候选 Top 50
S2-100：重排全部 candidates_100
```

Cross-Encoder 逐个计算 query-product pair，无法像 bi-encoder 一样只预计算
商品 embedding，因此必须记录 P50/P95 延迟。

### 8.3 S3：BGE Reranker

可选模型：

```text
BAAI/bge-reranker-v2-m3
```

它是直接为 query-document relevance scoring 设计的多语言 reranker。由于模型
比 MiniLM 更大，只有在本地资源和延迟允许时再进行 S2 vs S3 对照。

### 8.4 S4：monoT5

monoT5 将排序转化为生成 relevance label 的任务，可作为不同架构路线的实验。
它通常比 MiniLM 的接入和推理更重，建议排在 S2、S3 之后。

### 8.5 商品文本序列化实验

Product Attribute View 由 1 号提供，Reranking 负责决定给语义模型看哪些字段。

| 编号 | 商品文本 | 目的 |
| --- | --- | --- |
| P1 | title + category | 最小、低延迟基线 |
| P2 | P1 + brand + price + 标准属性 | 推荐默认方案 |
| P3 | P2 + top features + description 摘要 | 增加语义信息 |
| P4 | 根据当前约束动态把相关属性放在前面 | 属性感知输入 |

先比较 P1、P2、P3；P4 在输入和 matcher 稳定后再做。

## 9. Score Fusion

### 9.1 F1：人工线性融合

```python
final_score = (
    w_retrieval * retrieval_score
    + w_semantic * semantic_score
    + w_hard * hard_score
    + w_soft * soft_score
    + w_profile * profile_score
    - w_violation * violation_score
)
```

可使用少量离散权重组合，不要无约束地搜索大量浮点参数。

Buying 初始对照：

```text
retrieval  0.20
semantic   0.35
hard       0.30
soft       0.10
profile    0.05
```

Browsing 初始对照：

```text
retrieval  0.60
semantic   0.30
profile    0.10
+ H1 soft_penalty_adjustment
```

Browsing 的 hard/soft/rejected 已通过 H1 adjustment 进入分数，因此不在线性部分
重复计算。Buying 则先按 feasibility tier，再在同层使用上述 Buying 融合权重。
这些数值只是实验起点，不是固定结论。

### 9.2 F2：RRF

当同时存在 BM25 rank、Dense rank 和 Cross-Encoder rank 时，可以比较：

```python
rrf_score = (
    1 / (k + bm25_rank)
    + 1 / (k + dense_rank)
    + 1 / (k + cross_encoder_rank)
)
```

RRF 只能融合排名；如果使用原始模型分数，需要先转换为 rank 或选择线性融合。

### 9.3 F3：Learning-to-Rank

当规则特征和语义分数稳定后，可尝试 LightGBM LambdaMART。候选特征可包括：

```text
bm25/dense/retrieval score 和 rank
semantic score
各属性 match score
hard satisfied/unknown/violation count
rejected match count
price distance
category overlap
attribute coverage
rating/rating_number
intent/turn
```

训练标签：

```text
target product = 1
other candidate = 0
```

一个 session-turn 的候选集合是一个 ranking group。由于公开 session 数量较少，
F3 必须采用按 session 分组的交叉验证，并放在规则和 zero-shot 模型实验之后。

## 10. Intent-aware Final Selection

### 10.1 Buying

优先顺序：

```text
hard feasibility
-> semantic relevance
-> soft preference
-> retrieval signal
```

主要实验为 H1 与 H2，不默认做多样性。

### 10.2 Browsing

Browsing 的 Top 10 应在保持相关的同时覆盖不同可能性。

#### D1：无多样性

直接输出 relevance Top 10，作为对照。

#### D2：Facet Coverage

鼓励 Top 10 覆盖不同：

```text
brand
color
style
material
price bucket
```

这是与 Product Attribute View 最直接配合、也最容易解释的方案。

#### D3：MMR

```text
MMR = lambda * relevance
      - (1 - lambda) * max_similarity_to_selected
```

建议测试：

```text
lambda = 1.00  # 无多样性
lambda = 0.85  # 轻度多样化
lambda = 0.70  # 较强多样化
```

需要同时观察 HitRate@10 和 MRR：多样性可能提高覆盖，却降低目标的具体名次。

## 11. Reranking Replay Evaluator

### 11.1 为什么需要 replay

官方 evaluator 只能说明整个系统变好或变差，不能隔离 Reranking 的真实贡献。
应保存固定的 State 和 candidates_100，然后让多个 Reranker 在完全相同输入上运行。

当前 Pipeline 的 Dialogue 读取 `candidates_100`，不读取 `candidates_10`，因此在
这一版本中可以较干净地进行 reranking-only replay。未来 Dialogue 如果开始使用
排序诊断，则仍需用完整 evaluator 做最终确认。

### 11.2 Case 格式

```python
RerankCase = {
    "session_id": "public_0001",
    "scenario_type": "buying",
    "turn": 3,
    "shopping_state": {...},
    "target_parent_asin": "B001...",
    "candidates_100": [...],
}
```

`target_parent_asin` 只能用于训练标签和离线评测，绝不能输入运行时特征。

### 11.3 必须报告的指标

| 指标 | 说明 |
| --- | --- |
| `coverage@100` | 目标是否出现在 Retrieval 候选中 |
| `conditional_hit@10` | 目标已在候选中时，是否被排进 Top 10 |
| `conditional_mrr@10` | 只评估目标已被召回的 case |
| `promotion_count` | 目标从 11～100 升进 Top 10 的次数 |
| `demotion_count` | 目标从 Top 10 掉出 Top 10 的次数 |
| `mean_rank_change` | 目标平均提升或下降多少名 |
| `hard_violation@10` | Top 10 中明确违反 hard constraint 的比例 |
| `unknown_rate` | 商品属性不足以判断的比例 |
| `p50/p95_latency` | 单轮精排延迟 |

至少满足以下条件才考虑替换基线：

```text
promotion_count > demotion_count
conditional MRR@10 提升
端到端 TechnicalScore 不下降
关键场景没有不可接受的回退
```

### 11.4 数据划分

如果需要训练权重或模型：

- 按 session 分组做 5-fold cross-validation。
- 同一 session 的不同 turn 必须在同一折。
- 尽量同时按 Buying、Browsing、Intent Override、Boundary 分层。
- 同一 target 商品若重复出现，也应尽量放在同一折，避免商品级泄漏。
- candidates_100 中不存在正例的 group 不用于训练排序模型，但保留在 Retrieval
  coverage 报告中。

## 12. 实验矩阵

不要直接穷举全部组合，按阶段只改变一个因素。

| 实验 | 内容 | 主要问题 |
| --- | --- | --- |
| R0 | 当前 `SimpleReranker` | 当前基线 |
| R1 | Query Serializer + 属性专用 Matcher | 结构化匹配是否提升 |
| R2A | R1 + Soft Penalty | 温和约束效果 |
| R2B | R1 + Feasibility Tier | 严格约束效果 |
| R3A | 最优 R2 + MiniLM Top 20 | 小范围语义精排 |
| R3B | 最优 R2 + MiniLM Top 50 | 推荐主实验 |
| R3C | 最优 R2 + MiniLM Top 100 | 效果/延迟上限 |
| R4 | MiniLM vs BGE Reranker | 模型选择 |
| R5 | P1/P2/P3 商品文本 | 输入序列化选择 |
| R6A | 最优相关度 + Facet Coverage | Browsing 多样性 |
| R6B | 最优相关度 + MMR | 多样性方案对照 |
| R7 | 人工融合 vs RRF | 融合选择 |
| R8 | 人工融合 vs LambdaMART | LTR 是否值得 |
| R9 | 完整官方 evaluator | 最终端到端确认 |

每个实验保存独立配置和结果，不覆盖先前结果。

## 13. 建议文件结构

```text
src/reranking/
|-- README.md
|-- RERANKING_OPTIMIZATION_PLAN.md
|-- reranker.py
|-- config.py
|-- query_serializer.py
|-- constraint_matcher.py
|-- feature_extractor.py
|-- fusion.py
|-- diversity.py
|
|-- scorers/
|   |-- __init__.py
|   |-- base.py
|   |-- rule_scorer.py
|   |-- cross_encoder_scorer.py
|   `-- ltr_scorer.py
|
|-- evaluation/
|   |-- __init__.py
|   |-- replay.py
|   |-- metrics.py
|   `-- compare.py
|
`-- tests/
    |-- test_query_serializer.py
    |-- test_constraint_matcher.py
    |-- test_fusion.py
    |-- test_diversity.py
    `-- test_reranker.py
```

Product Attribute View 的源码位置由 1 号决定；Reranking 只通过正式导出接口或
adapter 使用它。

## 14. 推荐实施顺序

### 阶段 A：建立可测基线

1. 在当前最新 `beta` 上重新运行完整 evaluator，记录新的 R0。
2. 已实现版本化 RerankCase 录制，保存 State、Candidates100、Git 和各组件指纹。
3. 已实现 promotion/demotion、conditional hit/MRR、约束质量和 latency 指标。

### 阶段 B：结构化规则精排

1. 接入 1 号的 Product Attribute View。
2. 实现 Query Serializer。
3. 已实现 `ConstraintMatch` 以及属性专用 matcher。
4. 已区分 SATISFIED、UNKNOWN、VIOLATED。
5. 已实现 Candidate Feature Extractor。
6. 第一版采用 Browsing H1、Buying H2；后续 replay 中再做对调消融。

### 阶段 C：语义精排

1. 接入 MiniLM Cross-Encoder。
2. 比较 Top 20、50、100。
3. 比较商品文本 P1、P2、P3。
4. 只有 MiniLM 明显受限且资源允许时，再比较 BGE/monoT5。

### 阶段 D：Browsing 与融合

1. 比较无多样性、Facet Coverage 和 MMR。
2. 比较线性融合与 RRF。
3. 只有 replay 数据和特征稳定后，再尝试 LambdaMART。

### 阶段 E：端到端验收

1. 跑全部单元测试。
2. 跑 200-session evaluator。
3. 按 scenario 和 turn 检查回退。
4. 保存配置、结果、依赖、模型名称、延迟和硬件说明。

## 15. 第一轮推荐范围

第一轮建议只实现：

```text
Replay Evaluator
+ Query Serializer
+ Constraint Matcher
+ UNKNOWN/VIOLATED 分离
+ Soft Penalty vs Feasibility Tier
+ MiniLM Cross-Encoder Top 50
+ Browsing Facet Coverage
```

第一轮暂不实现：

```text
大模型 listwise reranking
Cross-Encoder fine-tuning
LambdaMART
ColBERT
复杂 ensemble
```

先让 R1～R3 得到可信的消融结果，再决定是否值得增加复杂度。

## 16. 参考材料

- [ProductAgent: Benchmarking Conversational Product Search Agent with Asking Clarification Questions](https://aclanthology.org/2025.emnlp-industry.25/)
  - 结构化对话记忆、候选商品统计、symbolic+dense 商品检索闭环。
- [Sentence Transformers: Cross-Encoder Training Overview](https://www.sbert.net/docs/cross_encoder/training_overview.html)
  - Cross-Encoder 作为第二阶段 reranker 的官方用法、训练和评测入口。
- [Sentence Transformers: Pretrained Cross-Encoder Models](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
  - MiniLM 等不同尺寸 reranker 的官方模型列表。
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
  - BGE 多语言 reranker 的官方模型卡和推理说明。
- [Document Ranking with a Pretrained Sequence-to-Sequence Model](https://aclanthology.org/2020.findings-emnlp.63/)
  - monoT5 的 pointwise sequence-to-sequence reranking 方法。
- [Summarization: Using MMR for Diversity-Based Reranking](https://aclanthology.org/X98-1025/)
  - MMR 的相关度与多样性权衡方法。
- [An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934)
  - 线性融合、分数标准化与 RRF 的对照和注意事项。
- [A Semantic Alignment System for Multilingual Query-Product Retrieval](https://arxiv.org/abs/2208.02958)
  - Amazon ESCI 商品 query-product 排序竞赛优胜方案。
