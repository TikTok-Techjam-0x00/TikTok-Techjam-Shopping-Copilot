"""3B: choose the next clarification attribute and render its question.

3B is intentionally stateless. State is owned by module 2 and ranking results
are owned by module 3; this module only reads both inputs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, MutableMapping, Sequence
from typing import Any, TypedDict


# 一、属性定义与基础优先级。
# 当 Ranking Result 缺少商品元数据时，系统会退回到这组稳定的默认顺序。
ATTRIBUTES = (
    "category", "use_case", "feature", "size", "material",
    "budget", "style", "color", "brand", "other",
)

BASE_PRIORITY = {
    "category": 90.0,
    "use_case": 70.0,
    "feature": 68.0,
    "size": 66.0,
    "material": 64.0,
    "budget": 60.0,
    "style": 58.0,
    "color": 52.0,
    "brand": 45.0,
    "other": 5.0,
}

# 用户画像只负责小幅调整优先级，不会被当成用户已经明确表达的约束。
PROFILE_TAG_TO_ATTRIBUTE = {
    "material": "material", "fabric": "material",
    "fit": "size", "size": "size",
    "style": "style", "fashion": "style",
    "comfort": "feature", "durability": "feature", "warmth": "feature",
    "weather": "use_case", "occasion": "use_case",
    "price": "budget", "value": "budget",
    "brand": "brand", "color": "color", "colour": "color",
}

# Ranking Result 不一定有结构化属性；正则用于从标题、详情等文本中做兜底提取。
VALUE_PATTERNS = {
    "material": re.compile(
        r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|linen|suede|denim)\b", re.I
    ),
    "color": re.compile(
        r"\b(black|white|blue|red|pink|green|brown|gr[ae]y|purple|yellow|orange|beige)\b", re.I
    ),
    "size": re.compile(r"\b(xxs|xs|small|medium|large|xl|xxl|wide|narrow|petite|plus size)\b", re.I),
    "style": re.compile(r"\b(casual|formal|classic|modern|vintage|sporty|slim|relaxed)\b", re.I),
    "use_case": re.compile(r"\b(hiking|running|gym|winter|outdoor|work|wedding|travel|daily)\b", re.I),
}


class AskDecision(TypedDict):
    ask_attribute: str | None
    message: str


def _is_value(value: object) -> bool:
    return value not in (None, "", [], {}, ())


def _as_names(value: object) -> set[str]:
    """兼容 ["size"] 和 [{"ask_attribute": "size"}] 两种列表格式。"""
    names: set[str] = set()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence):
        return names
    for item in value:
        name = item.get("ask_attribute") if isinstance(item, Mapping) else item
        if isinstance(name, str) and name in ATTRIBUTES:
            names.add(name)
    return names


def _known_attributes(state: Mapping[str, Any]) -> set[str]:
    """读取模块 2 已经确认的属性，兼容字典和列表两种 State 格式。"""
    known: set[str] = set()
    for field in ("known_attributes", "constraints", "preferences", "slots", "attributes"):
        values = state.get(field)
        # 例如 {"known_attributes": ["category", "color"]}。
        known |= _as_names(values)
        # 例如 {"known_attributes": {"category": "shoes"}}。
        if isinstance(values, Mapping):
            for name, value in values.items():
                if name in ATTRIBUTES and _is_value(value):
                    known.add(name)
    for name in ATTRIBUTES:
        if _is_value(state.get(name)):
            known.add(name)
    return known


def _asked_attributes(state: Mapping[str, Any]) -> set[str]:
    """读取历史提问，防止同一属性在后续轮次被重复询问。"""
    asked: set[str] = set()
    for field in ("asked_attributes", "question_history", "asked"):
        asked |= _as_names(state.get(field))
    return asked


def _unavailable_attributes(state: Mapping[str, Any]) -> set[str]:
    """读取用户明确表示无偏好或拒绝回答的属性。"""
    unavailable: set[str] = set()
    for field in ("unavailable_attributes", "no_preference_attributes", "rejected_attributes"):
        unavailable |= _as_names(state.get(field))
    return unavailable


def _ranking_items(ranking_result: object) -> list[Mapping[str, Any]]:
    """二、读取模块 3 的 Top 10，兼容常见的外层字段名称。"""
    if isinstance(ranking_result, Mapping):
        for field in ("results", "candidates", "items", "ranked_products", "recommendations"):
            value = ranking_result.get(field)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [item for item in value if isinstance(item, Mapping)][:10]
        return []
    if isinstance(ranking_result, Sequence) and not isinstance(ranking_result, (str, bytes)):
        return [item for item in ranking_result if isinstance(item, Mapping)][:10]
    return []


def _product(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """商品属性可以直接放在候选项里，也可以嵌套在常见的元数据字段中。"""
    result = dict(candidate)
    for field in ("product", "item", "metadata"):
        nested = candidate.get(field)
        if isinstance(nested, Mapping):
            result.update(nested)
    return result


def _text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _values(product: Mapping[str, Any], attribute: str) -> set[str]:
    """三、提取候选属性值；这些值只用于判断候选多样性和生成选项。"""
    explicit = product.get(attribute)
    if attribute == "brand" and not _is_value(explicit):
        explicit = product.get("store")
    if attribute == "category" and not _is_value(explicit):
        explicit = product.get("categories")
    if attribute == "feature" and not _is_value(explicit):
        explicit = product.get("features")
    if attribute == "budget" and not _is_value(explicit):
        explicit = product.get("price")

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
        return {value[:60] for value in cleaned if value}

    # 没有显式字段时，再从可搜索文本中提取有限的常见值。
    pattern = VALUE_PATTERNS.get(attribute)
    if pattern:
        searchable = _text({
            key: product.get(key)
            for key in ("title", "details", "features", "description", "categories")
        })
        return {match.lower() for match in pattern.findall(searchable)}
    return set()


def _candidate_diversity_signal(
    items: list[Mapping[str, Any]], attribute: str
) -> tuple[float, list[str]]:
    """计算候选多样性启发分，而非严格的信息增益。"""
    if len(items) < 2:
        return 0.0, []

    # 排名靠前的商品权重更高，避免第十名候选过度影响提问方向。
    weighted_counts: Counter[str] = Counter()
    covered = 0
    for rank, candidate in enumerate(items, start=1):
        values = _values(_product(candidate), attribute)
        if not values:
            continue
        covered += 1
        weight = 1.0 / math.sqrt(rank)
        for value in values:
            weighted_counts[value] += weight / len(values)

    if covered < 2 or len(weighted_counts) < 2:
        return 0.0, [value for value, _ in weighted_counts.most_common(3)]

    # 熵表示候选值是否分散；覆盖率表示该属性在多少候选商品上可用。
    # 这只是对“这个问题能否有效区分候选”的近似，并不模拟用户回答后的重排。
    total = sum(weighted_counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in weighted_counts.values())
    normalized_entropy = entropy / math.log(len(weighted_counts))
    coverage = covered / len(items)
    # 2～5 个主要取值比较适合追问；取值过多通常会让问题过于宽泛。
    cardinality_factor = min(1.0, 5.0 / len(weighted_counts))
    boost = 38.0 * coverage * normalized_entropy * cardinality_factor
    options = [value for value, _ in weighted_counts.most_common(3)]
    return boost, options


def _profile_boosts(state: Mapping[str, Any]) -> Counter[str]:
    """四、将用户画像标签转换成较小的属性优先级增量。"""
    profile = state.get("user_profile")
    tags = profile.get("preference_tags", []) if isinstance(profile, Mapping) else []
    boosts: Counter[str] = Counter()
    if not isinstance(tags, Sequence) or isinstance(tags, (str, bytes)):
        return boosts
    for tag in tags:
        attribute = PROFILE_TAG_TO_ATTRIBUTE.get(str(tag).lower())
        if attribute:
            boosts[attribute] += 12.0
    return boosts


def _turn_number(state: Mapping[str, Any]) -> int:
    """安全读取轮次；非法值回退到第一轮，不让 3B 中断整个会话。"""
    raw_turn = state.get("turn") or state.get("current_turn") or 1
    try:
        turn = int(raw_turn)
    except (TypeError, ValueError):
        return 1
    return max(1, turn)


def choose_ask_attribute(
    state: Mapping[str, Any],
    ranking_result: object,
) -> tuple[str | None, list[str]]:
    """五、综合 State 覆盖情况与候选多样性，选择一个尚未问过的属性。"""
    turn = _turn_number(state)
    if turn >= 10:
        return None, []

    excluded = _known_attributes(state) | _asked_attributes(state) | _unavailable_attributes(state)
    items = _ranking_items(ranking_result)
    profile_boosts = _profile_boosts(state)

    # 类别是基础约束：模块 2 尚未确认类别时，先不比较更细的商品属性。
    if "category" not in excluded:
        _, options = _candidate_diversity_signal(items, "category")
        return "category", options

    scored: list[tuple[float, int, str, list[str]]] = []
    for order, attribute in enumerate(ATTRIBUTES):
        if attribute in excluded:
            continue
        diversity_boost, options = _candidate_diversity_signal(items, attribute)
        score = BASE_PRIORITY[attribute] + profile_boosts[attribute] + diversity_boost
        # 前两轮偏向确认大方向，后续轮次偏向能直接缩小候选集的具体属性。
        if turn <= 2 and attribute in ("category", "use_case"):
            score += 8.0
        if turn >= 4 and attribute in ("feature", "size", "material", "budget"):
            score += 5.0
        scored.append((score, -order, attribute, options))

    if not scored:
        return None, []
    _, _, attribute, options = max(scored)
    return attribute, options


def build_question(attribute: str | None, options: Sequence[str] = ()) -> str:
    """六、使用固定模板生成问题，并尽量展示 Ranking Result 中的主要候选值。"""
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


def decide_ask(state: Mapping[str, Any], ranking_result: object) -> AskDecision:
    """3B 主入口；只读取 State，不会自动修改模块 2 的状态。"""
    attribute, options = choose_ask_attribute(state, ranking_result)
    return {"ask_attribute": attribute, "message": build_question(attribute, options)}


def record_asked_attribute(
    state: MutableMapping[str, Any], attribute: str | None
) -> None:
    """七、供调用方显式把本轮提问写回模块 2 的可变 State。

    ``decide_ask`` 保持无状态和只读；主流程得到决策后应调用本函数，
    否则下一轮 3B 无法知道该属性已经问过。
    """
    if attribute is None or attribute not in ATTRIBUTES:
        return
    asked = _asked_attributes(state)
    asked.add(attribute)
    # 统一写回列表，便于 JSON 序列化和模块之间传递。
    state["asked_attributes"] = sorted(asked, key=ATTRIBUTES.index)


class AskAttributeSelector:
    """为使用组件类的主流程提供一个很薄的对象封装。"""

    def decide(self, state: Mapping[str, Any], ranking_result: object) -> AskDecision:
        return decide_ask(state, ranking_result)
