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

from ..item import Candidate

if TYPE_CHECKING:
    from ..attribute import AttributeMap, AttributeName


# 属性定义与基础优先级。
# 当 Retrieval candidates 缺少商品元数据时，系统会退回到这组稳定的默认顺序。
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

# Retrieval candidate 不一定有结构化属性；正则用于从标题、详情等文本中兜底提取。
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
    if name == "others":
        name = "other"
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


def _product(value: Candidate | Mapping[str, Any]) -> dict[str, Any] | None:
    """把正式 Candidate.item 或兼容 Mapping 统一成商品字典。"""
    if isinstance(value, Candidate):
        return value.item.to_dict()

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
    """大小写不敏感地读取 details，例如 Material、material、use_case。"""
    details = product.get("details")
    if not isinstance(details, Mapping):
        return None
    wanted = attribute.casefold().replace("_", " ")
    for key, value in details.items():
        normalized_key = str(key).casefold().replace("_", " ")
        if normalized_key == wanted:
            return value
    return None


def _first_value(*values: object) -> object:
    """按数据契约规定的优先级返回第一个有效属性值。"""
    for value in values:
        if _is_value(value):
            return value
    return None


def _values(product: Mapping[str, Any], attribute: str) -> set[str]:
    """提取候选属性值；这些值只用于判断候选多样性和生成选项。"""
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
    items: list[Candidate | Mapping[str, Any]], attribute: str
) -> tuple[float, list[str]]:
    """计算候选多样性启发分，而非严格的信息增益。"""
    if len(items) < 2:
        return 0.0, []

    # 排名靠前的商品权重更高，避免排名靠后的候选过度影响提问方向。
    weighted_counts: Counter[str] = Counter()
    covered = 0
    for rank, candidate in enumerate(items, start=1):
        product = _product(candidate)
        if product is None:
            continue
        values = _values(product, attribute)
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
    # normalized entropy 已将不同候选值数量统一到 0～1，无需额外惩罚高 cardinality。
    boost = 24.0 * coverage * normalized_entropy
    options = [value for value, _ in weighted_counts.most_common(3)]
    return boost, options


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
    """综合 State 覆盖情况与候选多样性，选择一个尚未问过的属性。"""
    turn = _turn_number(shopping_state)
    if turn >= 10:
        return None, []

    excluded = (
        _known_attributes(shopping_state)
        | _asked_attributes(shopping_state)
        | _unavailable_attributes(shopping_state)
    )
    items = _retrieval_items(candidates_100)

    # 类别是基础约束：模块 2 尚未确认类别时，先不比较更细的商品属性。
    if "category" not in excluded:
        _, options = _candidate_diversity_signal(items, "category")
        return "category", options

    scored: list[tuple[float, int, str, list[str]]] = []
    for order, attribute in enumerate(ATTRIBUTES):
        if attribute in excluded:
            continue
        diversity_boost, options = _candidate_diversity_signal(items, attribute)
        score = BASE_PRIORITY[attribute] + diversity_boost
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
    """供调用方显式把本轮提问写回模块 2 的可变 State。

    ``decide_ask`` 保持无状态和只读；主流程得到决策后应调用本函数，
    否则下一轮 3B 无法知道该属性已经问过。
    """
    if attribute is None or attribute not in ATTRIBUTES:
        return
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
