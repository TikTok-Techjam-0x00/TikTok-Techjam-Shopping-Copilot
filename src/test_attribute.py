from __future__ import annotations

import json
import unittest

from src.attribute import (
    ATTRIBUTE_NAMES,
    AttributeName,
    AttributeValue,
    attribute_map_to_dict,
    normalize_attribute_map,
    normalize_attribute_name,
    to_official_ask_attribute,
)


class AttributeContractTest(unittest.TestCase):
    def test_final_attribute_names_are_unique_and_include_others(self) -> None:
        self.assertEqual(len(ATTRIBUTE_NAMES), len({name.value for name in ATTRIBUTE_NAMES}))
        self.assertIn(AttributeName.OTHERS, ATTRIBUTE_NAMES)
        self.assertIn(AttributeName.TARGET_USER, ATTRIBUTE_NAMES)
        self.assertIn(AttributeName.RATING, ATTRIBUTE_NAMES)

    def test_dataset_and_llm_aliases_use_canonical_names(self) -> None:
        self.assertIs(normalize_attribute_name("categories"), AttributeName.CATEGORY)
        self.assertIs(normalize_attribute_name("Fabric Type"), AttributeName.MATERIAL)
        self.assertIs(normalize_attribute_name("Department"), AttributeName.TARGET_USER)
        self.assertIs(normalize_attribute_name("Occasion"), AttributeName.USE_CASE)
        self.assertIs(normalize_attribute_name("Closure Type"), AttributeName.FEATURE)
        self.assertIs(normalize_attribute_name("unrecognized field"), AttributeName.OTHERS)

    def test_image_style_attributes_normalize_to_one_value_type(self) -> None:
        attributes = normalize_attribute_map(
            {
                "category": ["earrings", "hoop earrings", "women's jewelry"],
                "material": ["fabric", "stainless steel"],
                "color": [],
                "size": {
                    "diameter": "approximately 2 inches",
                    "dimensions": "1.97 x 1.97 x 0.08 inches",
                    "weight": "0.5 ounces",
                },
                "style": ["statement", "hoop", "artsy"],
                "brand": ["Spirit Hoops"],
                "budget": None,
                "feature": ["lightweight", "comfortable", "hypoallergenic"],
                "use_case": ["gift giving", "events", "party"],
                "target_user": ["women"],
            }
        )

        self.assertNotIn(AttributeName.COLOR, attributes)
        self.assertNotIn(AttributeName.BUDGET, attributes)
        self.assertEqual(attributes[AttributeName.MATERIAL].values, ["fabric", "stainless steel"])
        self.assertEqual(
            attributes[AttributeName.SIZE].details["diameter"],
            ["approximately 2 inches"],
        )
        self.assertTrue(all(isinstance(value, AttributeValue) for value in attributes.values()))

    def test_range_representation_supports_budget_and_rating(self) -> None:
        attributes = normalize_attribute_map(
            {
                "price": {"min": 50, "max": 100, "currency": "USD"},
                "average_rating": {"minimum": 4.2, "unit": "stars"},
            }
        )
        budget = attributes[AttributeName.BUDGET]
        self.assertEqual((budget.minimum, budget.maximum, budget.unit), (50.0, 100.0, "USD"))
        self.assertEqual(attributes[AttributeName.RATING].minimum, 4.2)

    def test_unknown_fields_are_preserved_under_others(self) -> None:
        attributes = normalize_attribute_map(
            {
                "Care Instructions": "hand wash only",
                "Country of Origin": "USA",
                "others": {"custom engraving": "available"},
            }
        )
        others = attributes[AttributeName.OTHERS]
        self.assertEqual(others.details["care_instructions"], ["hand wash only"])
        self.assertEqual(others.details["country_of_origin"], ["USA"])
        self.assertEqual(others.details["custom_engraving"], ["available"])

    def test_attribute_map_is_json_serializable(self) -> None:
        attributes = normalize_attribute_map(
            {
                "color": ["black", "Black", "white"],
                "budget": {"max": 80, "currency": "USD"},
            }
        )
        serialized = attribute_map_to_dict(attributes)
        json.dumps(serialized)
        self.assertEqual(serialized["color"]["values"], ["black", "white"])
        self.assertEqual(serialized["budget"]["max"], 80.0)

    def test_internal_names_map_to_official_ask_attribute(self) -> None:
        self.assertEqual(to_official_ask_attribute(AttributeName.MATERIAL), "material")
        self.assertEqual(to_official_ask_attribute(AttributeName.FIT), "size")
        self.assertEqual(to_official_ask_attribute(AttributeName.PATTERN), "style")
        self.assertEqual(to_official_ask_attribute(AttributeName.TARGET_USER), "other")
        self.assertEqual(to_official_ask_attribute(AttributeName.OTHERS), "other")


if __name__ == "__main__":
    unittest.main()
