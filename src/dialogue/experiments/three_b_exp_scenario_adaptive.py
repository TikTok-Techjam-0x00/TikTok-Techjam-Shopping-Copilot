# 3B EXPERIMENT VARIANT
# Baseline: current production src/dialogue/three_b.py at e788c3b.
# Differences from current baseline:
#   Adds scenario/policy routing, Ranking Impact, Expected Answer Yield,
#   buying-mode conditional candidate analysis, boundary/override recovery,
#   override asked_attributes epoch reset, and normalized Semantic Prior.
# Variables changed:
#   The complete clarification scoring policy is replaced as one composite experiment.
# Variables intentionally kept unchanged:
#   Official attributes/aliases, State and Candidate compatibility, attribute extraction,
#   Module 1 Top100 input, category gate, question templates, public return schema,
#   turn-10 stop rule, and caller-owned state recording.
# Purpose:
#   Standalone A/B benchmark of a scenario-adaptive 3B policy.
# Runtime inputs only:
#   Uses only Module 2 State and Module 1 Retrieval candidates; no evaluator labels,
#   public-set statistics, session IDs, target products, or fixed benchmark answers.
# Production safety:
#   This file does not modify production three_b.py and can be copied over it for an
#   orchestrator experiment without depending on another experimental module.

"""3B: choose the next clarification attribute and render its question.

Module 2 owns ``shopping_state``; module 1 owns ``candidates_100``. 3B only
reads these inputs and never changes the product ranking produced by module 3A.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypedDict

from ..attribute import product_attribute_values
from ..item import Candidate, Item

if TYPE_CHECKING:
    from ..attribute import AttributeMap, AttributeName


# 官方属性集合与兼容别名保持和生产版一致。
ATTRIBUTES = (
    "category", "use_case", "feature", "size", "material",
    "budget", "style", "color", "brand", "other",
)

# 共享 AttributeName 已与官方 ask_attribute 完全一致；仅兼容旧版 fit 输入。
ATTRIBUTE_NAME_ALIASES = {"fit": "style"}

# 四种运行时策略模式。路由只读取 Module 2 本轮 State，不使用测试集先验。
PolicyMode = Literal["EXPLORE", "CONSTRAIN", "BOUNDARY_RECOVER", "OVERRIDE_RECOVER"]

# 语义先验统一归一化到 0～1，仅作为弱信号参与评分。
SEMANTIC_PRIOR = {
    "category": 1.00,
    "material": 0.95,
    "color": 0.95,
    "size": 0.90,
    "budget": 0.90,
    "use_case": 0.85,
    "style": 0.80,
    "feature": 0.70,
    "brand": 0.55,
    "other": 0.10,
}

# Buying 条件池过小时回退 Top100，避免少量或缺失字段制造虚假的确定性。
MIN_CONDITIONAL_ITEMS = 12
MIN_CONDITIONAL_RATIO = 0.20
CONDITIONAL_ATTRIBUTES = (
    "material", "color", "size", "use_case", "style", "brand",
)

# 选项需要具备最低重复性与集中度，避免向用户展示大量一次性噪声值。
MIN_OPTION_REPEATABILITY = 0.20
MIN_OPTION_CONCENTRATION = 0.50
UTILITY_TIE_TOLERANCE = 1e-9

# Retrieval candidate 不一定有结构化属性；正则用于从标题、详情等文本中兜底提取。
VALUE_PATTERNS = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric|linen|suede|denim)\b",
        re.I,
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|orange|beige)\b", re.I
    ),
    "size": re.compile(r"\b(xxs|xs|small|medium|large|xl|xxl|wide|narrow|petite|plus size)\b", re.I),
    "style": re.compile(r"\b(casual|formal|classic|modern|vintage|sporty|slim|relaxed)\b", re.I),
    "use_case": re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I),
}

# 统一常见商品文本的官方 ask_attribute 边界；未命中规则的文本属于 feature。
CONSTRAINT_ATTRIBUTE_PATTERNS = (
    ("budget", re.compile(r"budget|(?:\$|<=|under)\s*\d", re.I)),
    (
        "material",
        re.compile(
            r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b",
            re.I,
        ),
    ),
    ("color", re.compile(r"\b(color|black|white|blue|red|pink|green)\b", re.I)),
    ("size", re.compile(r"\b(size|sizing|width|wide|narrow)\b", re.I)),
    ("style", re.compile(r"\b(department|style|fit|sleeve|neck)\b", re.I)),
    ("use_case", re.compile(r"\b(hiking|running|gym|winter|outdoor|work)\b", re.I)),
)

# details 的 key 经常不是官方属性名，例如 Fabric Type、Fit Type、Item Width。
DETAIL_KEY_MARKERS = {
    "material": ("material", "fabric"),
    "size": ("size", "sizing", "width"),
    "style": ("style", "department", "fit", "sleeve", "neck"),
    "feature": ("feature",),
}


class AskDecision(TypedDict):
    ask_attribute: str | None
    message: str


class ShoppingStateProtocol(Protocol):
    """Module 2 正式 shopping_state 的结构接口。"""

    session_id: str
    user_profile: Mapping[str, Any]
    user_message: str
    turn: int
    intent: Literal["buying", "browsing"]
    hard_constraint: AttributeMap
    soft_constraint: AttributeMap
    no_prefernce: Sequence[AttributeName]
    asked_attributes: Any
    override_detected: bool
    boundary_detected: bool


ShoppingStateInput: TypeAlias = ShoppingStateProtocol | Mapping[str, Any]


def _state_value(
    shopping_state: ShoppingStateInput,
    field: str,
    default: Any = None,
) -> Any:
    """统一读取对象 State 和字典 State，保持与 3A 相同的访问方式。"""
    if isinstance(shopping_state, Mapping):
        return shopping_state.get(field, default)
    return getattr(shopping_state, field, default)


def _is_value(value: object) -> bool:
    """AttributeValue 等结构化对象只要存在，就代表用户表达过该属性。"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)):
        return bool(value)
    return True


def _attribute_name(value: object) -> str | None:
    """把字符串或 AttributeName 统一成官方 ask_attribute 名称。"""
    raw_value = getattr(value, "value", value)
    name = str(raw_value).strip().lower()
    name = ATTRIBUTE_NAME_ALIASES.get(name, name)
    return name if name in ATTRIBUTES else None


def _as_names(value: object) -> set[str]:
    """读取属性名称列表，也兼容 AttributeName 和 Mapping keys。"""
    names: set[str] = set()
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, Mapping):
        value = list(value.keys())
    if not isinstance(value, Sequence):
        return names
    for item in value:
        raw_name = item.get("ask_attribute") if isinstance(item, Mapping) else item
        name = _attribute_name(raw_name)
        if name:
            names.add(name)
    return names


def _known_attributes(shopping_state: ShoppingStateInput) -> set[str]:
    """读取 Module 2 已确认的 hard_constraint 和 soft_constraint。"""
    known: set[str] = set()
    for field in ("hard_constraint", "soft_constraint"):
        constraints = _state_value(shopping_state, field)
        if isinstance(constraints, Mapping):
            for raw_name, value in constraints.items():
                name = _attribute_name(raw_name)
                if name and _is_value(value):
                    known.add(name)
    return known


def _asked_attributes(shopping_state: ShoppingStateInput) -> set[str]:
    """读取历史提问，防止同一属性在后续轮次被重复询问。"""
    return _as_names(_state_value(shopping_state, "asked_attributes"))


def _unavailable_attributes(shopping_state: ShoppingStateInput) -> set[str]:
    """读取用户不关心的属性；不要与 rejected_values 中的具体值混用。"""
    unavailable: set[str] = set()
    # no_prefernce 是正式字段；低成本保留正确拼写以便迁移。
    for field in ("no_prefernce", "no_preference"):
        unavailable |= _as_names(_state_value(shopping_state, field))
    return unavailable


def _retrieval_items(
    candidates_100: Sequence[Candidate | Mapping[str, Any]],
) -> list[Candidate | Mapping[str, Any]]:
    """读取 Module 1 Retrieval 的前 100 个候选，用于 clarification analysis。

    正式路径显式接受具有 ``Candidate.item`` 的对象；单个 Mapping 候选仅作为
    数据迁移和测试兼容。这里不读取 Module 3A 的 ``candidates_10``。
    """
    accepted: list[Candidate | Mapping[str, Any]] = []
    for value in candidates_100[:100]:
        if isinstance(value, (Candidate, Mapping)):
            accepted.append(value)
    return accepted


def _product(value: Candidate | Mapping[str, Any]) -> Item | dict[str, Any] | None:
    """读取正式 Candidate.item；旧 Mapping 只作为兼容输入。"""
    if isinstance(value, Candidate):
        return value.item

    if not isinstance(value, Mapping):
        return None
    nested_item = value.get("item")
    nested_to_dict = getattr(nested_item, "to_dict", None)
    if callable(nested_to_dict):
        product = nested_to_dict()
        return dict(product) if isinstance(product, Mapping) else None
    if isinstance(nested_item, Mapping):
        return dict(nested_item)
    # 兼容直接使用 catalog 商品字段的旧字典候选。
    return dict(value)


def _text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _detail_value(product: Mapping[str, Any], attribute: str) -> object:
    """按官方 ask_attribute 边界读取异构 details key。"""
    details = product.get("details")
    if not isinstance(details, Mapping):
        return None
    wanted = attribute.casefold().replace("_", " ")
    markers = DETAIL_KEY_MARKERS.get(attribute, (wanted,))
    for key, value in details.items():
        normalized_key = str(key).casefold().replace("_", " ")
        if normalized_key == wanted or any(
            marker in normalized_key for marker in markers
        ):
            return value
    return None


def _first_value(*values: object) -> object:
    """按数据契约规定的优先级返回第一个有效属性值。"""
    for value in values:
        if _is_value(value):
            return value
    return None


def _constraint_attribute(value: str) -> str:
    """将一条商品约束归入官方 ask_attribute。"""
    for attribute, pattern in CONSTRAINT_ATTRIBUTE_PATTERNS:
        if pattern.search(value):
            return attribute
    return "feature"


def _values(product: Mapping[str, Any], attribute: str) -> set[str]:
    """提取候选属性值；这些值只用于判断候选多样性和生成选项。"""
    # Item 在 catalog 加载时已统一提取属性。预算仍读取原始 price，便于
    # 转换成离散价格区间；其他属性优先使用共享的派生结果。
    # 远端3B对 use_case 有更严格的官方六词边界，必须继续走下方专用逻辑。
    if attribute not in {"budget", "use_case"}:
        derived = getattr(product, "attributes", None)
        if isinstance(derived, Mapping):
            values = product_attribute_values(
                derived,
                attribute,
                include_details=False,
            )
            if attribute == "feature":
                values = [
                    value
                    for value in values
                    if _constraint_attribute(value) == "feature"
                ]
            if values:
                return {
                    re.sub(r"\s+", " ", value).strip().lower()[:60]
                    for value in values
                    if value.strip()
                }

    # 先读取统一 item 顶层字段，再读取 details；固定正则只作为最后兜底。
    detail = _detail_value(product, attribute)
    if attribute == "brand":
        # store 只是店铺或品牌相关文本，优先级必须低于明确的 Brand。
        explicit = _first_value(product.get("brand"), detail, product.get("store"))
    elif attribute == "category":
        explicit = _first_value(product.get("category"), product.get("categories"), detail)
    elif attribute == "feature":
        explicit = _first_value(product.get("feature"), product.get("features"), detail)
    elif attribute == "budget":
        explicit = _first_value(product.get("budget"), product.get("price"), detail)
    elif attribute == "use_case":
        # 所有 use_case 值都必须经过统一的六词正则，不能由 details key 绕过词表。
        explicit = None
    else:
        explicit = _first_value(product.get(attribute), detail)

    # 连续价格不适合直接作为问题选项，因此先转换成有限的价格区间。
    if attribute == "budget" and _is_value(explicit):
        match = re.search(r"\d+(?:\.\d+)?", str(explicit).replace(",", ""))
        if match:
            price = float(match.group())
            if price < 25:
                return {"under $25"}
            if price < 50:
                return {"$25–50"}
            if price < 100:
                return {"$50–100"}
            return {"$100+"}

    if _is_value(explicit):
        raw_values = explicit if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)) else [explicit]
        cleaned = {re.sub(r"\s+", " ", str(value)).strip().lower() for value in raw_values if _is_value(value)}
        if attribute == "feature":
            # Material、color 等不能同时贡献给 feature，否则会高估 feature 的排名影响。
            cleaned = {value for value in cleaned if _constraint_attribute(value) == "feature"}
        return {value[:60] for value in cleaned if value}

    # 没有显式字段时，再从可搜索文本中提取有限的常见值。
    pattern = VALUE_PATTERNS.get(attribute)
    if pattern:
        if attribute == "use_case":
            # 官方 hidden constraints 的 use_case 来源只覆盖 features 和 details。
            searchable = _text({
                "features": product.get("features"),
                "details": product.get("details"),
            })
        else:
            searchable = _text({
                key: product.get(key)
                for key in ("title", "details", "features", "description", "categories")
            })
        return {match.lower() for match in pattern.findall(searchable)}
    return set()


class AttributeSignals(TypedDict):
    """一个属性在当前候选池中的全部归一化运行时信号。"""

    coverage: float
    gini_top20: float
    gini_top100: float
    repeatability: float
    option_concentration: float
    answer_yield: float
    semantic_prior: float
    options: list[str]


def _rank_weighted_distribution(
    items: Sequence[Candidate | Mapping[str, Any]],
    attribute: str,
    limit: int,
) -> tuple[Counter[str], Counter[str], float, float]:
    """统计属性值的排名加权分布及候选级出现次数。

    每个候选的权重是 1/sqrt(rank)。一个候选含有多个值时均分该候选权重，
    避免多值商品天然比单值商品贡献更多质量。
    """
    weighted_counts: Counter[str] = Counter()
    candidate_occurrences: Counter[str] = Counter()
    covered_rank_weight = 0.0
    total_rank_weight = 0.0

    for rank, candidate in enumerate(items[:limit], start=1):
        rank_weight = 1.0 / math.sqrt(rank)
        total_rank_weight += rank_weight
        product = _product(candidate)
        if product is None:
            continue
        values = sorted(_values(product, attribute))
        if not values:
            continue

        covered_rank_weight += rank_weight
        value_weight = rank_weight / len(values)
        for value in values:
            weighted_counts[value] += value_weight
            candidate_occurrences[value] += 1

    return (
        weighted_counts,
        candidate_occurrences,
        covered_rank_weight,
        total_rank_weight,
    )


def _gini_impurity(weighted_counts: Mapping[str, float]) -> float:
    """用 Gini impurity 表示一个属性区分当前候选的能力。"""
    total = sum(weighted_counts.values())
    if total <= 0.0 or len(weighted_counts) < 2:
        return 0.0
    return 1.0 - sum((weight / total) ** 2 for weight in weighted_counts.values())


def _attribute_signals(
    items: Sequence[Candidate | Mapping[str, Any]],
    attribute: str,
) -> AttributeSignals:
    """计算 coverage、Ranking Impact 与 Expected Answer Yield。

    所有数值都来自本轮 Retrieval，范围保持在 0～1，便于不同信号直接组合。
    """
    top20_counts, _, _, _ = _rank_weighted_distribution(items, attribute, 20)
    (
        top100_counts,
        candidate_occurrences,
        covered_rank_weight,
        total_rank_weight,
    ) = _rank_weighted_distribution(items, attribute, 100)

    total_value_mass = sum(top100_counts.values())
    coverage = (
        covered_rank_weight / total_rank_weight
        if total_rank_weight > 0.0
        else 0.0
    )
    repeated_mass = sum(
        mass
        for value, mass in top100_counts.items()
        if candidate_occurrences[value] >= 2
    )
    repeatability = (
        repeated_mass / total_value_mass
        if total_value_mass > 0.0
        else 0.0
    )
    ranked_values = sorted(
        top100_counts.items(),
        key=lambda entry: (-entry[1], entry[0]),
    )
    option_concentration = (
        sum(mass for _, mass in ranked_values[:3]) / total_value_mass
        if total_value_mass > 0.0
        else 0.0
    )
    answer_yield = coverage * repeatability * option_concentration

    # 只在候选值可重复、且前三个选项覆盖足够质量时向用户展示选项。
    options = (
        [value for value, _ in ranked_values[:3]]
        if (
            repeatability >= MIN_OPTION_REPEATABILITY
            and option_concentration >= MIN_OPTION_CONCENTRATION
        )
        else []
    )
    return {
        "coverage": coverage,
        "gini_top20": _gini_impurity(top20_counts),
        "gini_top100": _gini_impurity(top100_counts),
        "repeatability": repeatability,
        "option_concentration": option_concentration,
        "answer_yield": answer_yield,
        "semantic_prior": SEMANTIC_PRIOR[attribute],
        "options": options,
    }


def _policy_mode(shopping_state: ShoppingStateInput) -> PolicyMode:
    """按 override、boundary、buying、默认的顺序选择本轮策略。"""
    if bool(_state_value(shopping_state, "override_detected", False)):
        return "OVERRIDE_RECOVER"
    if bool(_state_value(shopping_state, "boundary_detected", False)):
        return "BOUNDARY_RECOVER"
    if str(_state_value(shopping_state, "intent", "browsing")).lower() == "buying":
        return "CONSTRAIN"
    return "EXPLORE"


def _constraint_text_values(value: object) -> set[str]:
    """保守读取 hard_constraint 的离散文本值；数值区间等复杂结构不参与过滤。"""
    raw_values = getattr(value, "values", None)
    if raw_values is None and isinstance(value, Mapping):
        raw_values = value.get("values", value.get("value"))
    if raw_values is None:
        raw_values = value

    if isinstance(raw_values, str):
        candidates: Sequence[object] = [raw_values]
    elif isinstance(raw_values, Sequence):
        candidates = raw_values
    else:
        return set()

    return {
        re.sub(r"\s+", " ", str(entry)).strip().lower()[:60]
        for entry in candidates
        if isinstance(entry, (str, int, float)) and str(entry).strip()
    }


def _hard_constraint_values(
    shopping_state: ShoppingStateInput,
    attribute: str,
) -> set[str]:
    """读取指定 hard constraint 的低歧义文本值。"""
    constraints = _state_value(shopping_state, "hard_constraint")
    if not isinstance(constraints, Mapping):
        return set()
    for raw_name, value in constraints.items():
        if _attribute_name(raw_name) == attribute:
            return _constraint_text_values(value)
    return set()


def _values_match(candidate_values: set[str], wanted_values: set[str]) -> bool:
    """宽松判断两个离散值集合是否相容，避免因轻微文本差异误删候选。"""
    for candidate_value in candidate_values:
        for wanted_value in wanted_values:
            if (
                candidate_value == wanted_value
                or candidate_value in wanted_value
                or wanted_value in candidate_value
            ):
                return True
    return False


def _conditional_items_for_buying(
    shopping_state: ShoppingStateInput,
    items: list[Candidate | Mapping[str, Any]],
) -> list[Candidate | Mapping[str, Any]]:
    """用低歧义 hard constraints 构建安全的 buying 条件候选池。

    候选缺少某个属性时保留；只有双方都有明确值且显式冲突时才排除。
    条件池过小则回退原始 Top100，避免缺字段或稀疏数据导致不稳定选择。
    """
    active_constraints = {
        attribute: values
        for attribute in CONDITIONAL_ATTRIBUTES
        if (values := _hard_constraint_values(shopping_state, attribute))
    }
    if not active_constraints or not items:
        return items

    filtered: list[Candidate | Mapping[str, Any]] = []
    for candidate in items:
        product = _product(candidate)
        if product is None:
            filtered.append(candidate)
            continue

        explicit_mismatch = False
        for attribute, wanted_values in active_constraints.items():
            candidate_values = _values(product, attribute)
            if candidate_values and not _values_match(candidate_values, wanted_values):
                explicit_mismatch = True
                break
        if not explicit_mismatch:
            filtered.append(candidate)

    required_by_ratio = math.ceil(len(items) * MIN_CONDITIONAL_RATIO)
    if (
        len(filtered) < MIN_CONDITIONAL_ITEMS
        or len(filtered) < required_by_ratio
    ):
        return items
    return filtered


def _ranking_impact(
    mode: PolicyMode,
    turn: int,
    signals: AttributeSignals,
) -> float:
    """让不同场景关注不同深度的候选分布。"""
    if mode == "EXPLORE":
        if turn <= 2:
            top20_weight, top100_weight = 0.45, 0.55
        elif turn <= 5:
            top20_weight, top100_weight = 0.65, 0.35
        else:
            top20_weight, top100_weight = 0.85, 0.15
    elif mode == "CONSTRAIN":
        top20_weight, top100_weight = 0.75, 0.25
    elif mode == "BOUNDARY_RECOVER":
        top20_weight, top100_weight = 0.65, 0.35
    else:
        top20_weight, top100_weight = 0.70, 0.30
    return (
        top20_weight * signals["gini_top20"]
        + top100_weight * signals["gini_top100"]
    )


def _attribute_utility(
    mode: PolicyMode,
    turn: int,
    signals: AttributeSignals,
    reask_bonus: float,
) -> float:
    """按场景组合归一化信号，返回可直接比较的属性效用。"""
    impact = _ranking_impact(mode, turn, signals)
    if mode == "EXPLORE":
        return (
            0.45 * impact
            + 0.35 * signals["answer_yield"]
            + 0.20 * signals["semantic_prior"]
        )
    if mode == "CONSTRAIN":
        return (
            0.50 * impact
            + 0.35 * signals["answer_yield"]
            + 0.15 * signals["semantic_prior"]
        )
    if mode == "BOUNDARY_RECOVER":
        return (
            0.35 * impact
            + 0.50 * signals["answer_yield"]
            + 0.15 * signals["semantic_prior"]
        )
    return (
        0.40 * impact
        + 0.35 * signals["answer_yield"]
        + 0.15 * signals["semantic_prior"]
        + 0.10 * reask_bonus
    )


def _turn_number(shopping_state: ShoppingStateInput) -> int:
    """安全读取轮次；非法值回退到第一轮，不让 3B 中断整个会话。"""
    raw_turn = _state_value(shopping_state, "turn", 1)
    try:
        turn = int(raw_turn)
    except (TypeError, ValueError):
        return 1
    return max(1, turn)


def choose_ask_attribute(
    shopping_state: ShoppingStateInput,
    candidates_100: Sequence[Candidate | Mapping[str, Any]],
) -> tuple[str | None, list[str]]:
    """按场景路由，在未耗尽的官方属性中选择下一项 clarification。"""
    turn = _turn_number(shopping_state)
    if turn >= 10:
        return None, []

    mode = _policy_mode(shopping_state)
    known = _known_attributes(shopping_state)
    previously_asked = _asked_attributes(shopping_state)
    unavailable = _unavailable_attributes(shopping_state)

    # Override 代表新需求阶段：旧 asked 属性可重新竞争，但已知和明确不关心的
    # 属性仍然不能询问。其他模式继续保持生产版的防重复规则。
    excluded = known | unavailable
    if mode != "OVERRIDE_RECOVER":
        excluded |= previously_asked

    items = _retrieval_items(candidates_100)

    # Category 仍是硬规则，不与任何评分信号竞争。
    if "category" not in excluded:
        return "category", _attribute_signals(items, "category")["options"]

    # other 只在所有常规属性都耗尽时兜底，不参加常规分数竞争。
    eligible = [
        attribute
        for attribute in ATTRIBUTES
        if attribute not in {"category", "other"} and attribute not in excluded
    ]
    if not eligible:
        if "other" not in excluded:
            return "other", []
        return None, []

    scoring_items = (
        _conditional_items_for_buying(shopping_state, items)
        if mode == "CONSTRAIN"
        else items
    )

    best_score = -1.0
    best_attribute: str | None = None
    best_options: list[str] = []
    for attribute in ATTRIBUTES:
        if attribute not in eligible:
            continue
        signals = _attribute_signals(scoring_items, attribute)
        reask_bonus = (
            1.0
            if (
                mode == "OVERRIDE_RECOVER"
                and attribute in previously_asked
                and attribute not in known
                and attribute not in unavailable
            )
            else 0.0
        )
        utility = _attribute_utility(mode, turn, signals, reask_bonus)
        # 只在明确高于当前最佳值时替换；同分或极近分保留 ATTRIBUTES 中较早者。
        if utility > best_score + UTILITY_TIE_TOLERANCE:
            best_score = utility
            best_attribute = attribute
            best_options = signals["options"]

    return best_attribute, best_options


def build_question(attribute: str | None, options: Sequence[str] = ()) -> str:
    """使用固定模板生成问题，并尽量展示 Retrieval 中的主要候选值。"""
    useful_options = [str(value) for value in options[:3] if value]
    if len(useful_options) == 1:
        choices = useful_options[0]
    elif len(useful_options) == 2:
        choices = " or ".join(useful_options)
    else:
        choices = ", ".join(useful_options[:-1]) + (f", or {useful_options[-1]}" if useful_options else "")

    if attribute == "category":
        return f"Which product category do you mean{f': {choices}' if choices else ''}?"
    if attribute == "use_case":
        return f"Where or when do you plan to use it{f': {choices}' if choices else ''}?"
    if attribute == "feature":
        return f"Which feature matters most to you{f': {choices}' if choices else ''}?"
    if attribute == "size":
        return f"What size or fit do you need{f': {choices}' if choices else ''}?"
    if attribute == "material":
        return f"Which material do you prefer{f': {choices}' if choices else ''}?"
    if attribute == "budget":
        return f"What budget range should I use{f': {choices}' if choices else ''}?"
    if attribute == "style":
        return f"Which style do you prefer{f': {choices}' if choices else ''}?"
    if attribute == "color":
        return f"Which color do you prefer{f': {choices}' if choices else ''}?"
    if attribute == "brand":
        return f"Do you have a preferred brand{f', such as {choices}' if choices else ''}?"
    if attribute == "other":
        return "Is there any other requirement I should consider?"
    return ""


def decide_ask(
    shopping_state: ShoppingStateInput,
    candidates_100: Sequence[Candidate | Mapping[str, Any]],
) -> AskDecision:
    """3B 主入口；读取 Module 2 State 和 Module 1 Retrieval candidates。"""
    attribute, options = choose_ask_attribute(shopping_state, candidates_100)
    return {"ask_attribute": attribute, "message": build_question(attribute, options)}


def record_asked_attribute(
    shopping_state: ShoppingStateInput, attribute: str | None
) -> None:
    """把本轮提问写回 Module 2 State，并在 override 时开启新的提问 epoch。

    decide_ask 保持无状态和只读。override_detected 为真时，旧需求阶段的
    asked_attributes 会先清空，再只记录新阶段的本轮属性。
    """
    if attribute is None or attribute not in ATTRIBUTES:
        return

    if bool(_state_value(shopping_state, "override_detected", False)):
        asked: set[str] = set()
    else:
        asked = _asked_attributes(shopping_state)
    asked.add(attribute)

    # 统一写回列表，便于 JSON 序列化和模块之间传递。
    updated = sorted(asked, key=ATTRIBUTES.index)
    if isinstance(shopping_state, MutableMapping):
        shopping_state["asked_attributes"] = updated
        return
    try:
        setattr(shopping_state, "asked_attributes", updated)
    except (AttributeError, TypeError) as error:
        raise TypeError("shopping_state must allow asked_attributes to be updated") from error


class AskAttributeSelector:
    """为使用组件类的主流程提供一个很薄的对象封装。"""

    def decide(
        self,
        shopping_state: ShoppingStateInput,
        candidates_100: Sequence[Candidate | Mapping[str, Any]],
    ) -> AskDecision:
        return decide_ask(shopping_state, candidates_100)
