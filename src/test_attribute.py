from __future__ import annotations

import json
import unittest

from src.attribute import (
    ATTRIBUTE_NAMES,
    AttributeName,
    AttributeValue,
    attribute_map_to_dict,
    extract_product_attributes,
    normalize_attribute_map,
    normalize_attribute_name,
    product_attribute_text,
    product_attribute_values,
    to_official_ask_attribute,
)


class AttributeContractTest(unittest.TestCase):
    def test_attribute_names_exactly_match_official_ask_attributes(self) -> None:
        self.assertEqual(len(ATTRIBUTE_NAMES), len({name.value for name in ATTRIBUTE_NAMES}))
        self.assertEqual(
            {name.value for name in ATTRIBUTE_NAMES},
            {
                "category",
                "material",
                "color",
                "size",
                "style",
                "brand",
                "budget",
                "feature",
                "use_case",
                "other",
            },
        )

    def test_dataset_and_llm_aliases_use_canonical_names(self) -> None:
        self.assertIs(normalize_attribute_name("categories"), AttributeName.CATEGORY)
        self.assertIs(normalize_attribute_name("Fabric Type"), AttributeName.MATERIAL)
        self.assertIs(normalize_attribute_name("Department"), AttributeName.STYLE)
        self.assertIs(normalize_attribute_name("Pattern Type"), AttributeName.STYLE)
        self.assertIs(normalize_attribute_name("Average Rating"), AttributeName.FEATURE)
        self.assertIs(normalize_attribute_name("Occasion"), AttributeName.USE_CASE)
        self.assertIs(normalize_attribute_name("Closure Type"), AttributeName.FEATURE)
        self.assertIs(normalize_attribute_name("unrecognized field"), AttributeName.OTHER)

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

    def test_range_representation_supports_budget_and_mapped_fields(self) -> None:
        attributes = normalize_attribute_map(
            {
                "price": {"min": 50, "max": 100, "currency": "USD"},
                "average_rating": {"minimum": 4.2, "unit": "stars"},
            }
        )
        budget = attributes[AttributeName.BUDGET]
        self.assertEqual((budget.minimum, budget.maximum, budget.unit), (50.0, 100.0, "USD"))
        self.assertEqual(attributes[AttributeName.FEATURE].minimum, 4.2)

    def test_unknown_fields_are_preserved_under_singular_other(self) -> None:
        attributes = normalize_attribute_map(
            {
                "Care Instructions": "hand wash only",
                "Country of Origin": "USA",
                "other": {"custom engraving": "available"},
            }
        )
        other = attributes[AttributeName.OTHER]
        self.assertEqual(other.details["care_instructions"], ["hand wash only"])
        self.assertEqual(other.details["country_of_origin"], ["USA"])
        self.assertEqual(other.details["custom_engraving"], ["available"])
        serialized = attribute_map_to_dict(attributes)
        self.assertEqual(list(serialized), ["other"])

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
        self.assertEqual(to_official_ask_attribute("fit"), "style")
        self.assertEqual(to_official_ask_attribute("pattern"), "style")
        self.assertEqual(to_official_ask_attribute("target_user"), "feature")
        self.assertEqual(to_official_ask_attribute("unrecognized field"), "other")

    def test_product_attributes_use_catalog_fields_and_detail_aliases(self) -> None:
        attributes = extract_product_attributes(
            {
                "title": "Black trail shoes",
                "categories": ["Shoes", "Hiking"],
                "features": ["Waterproof"],
                "price": 89.5,
                "store": "Store Fallback",
                "details": {
                    "Brand": "Trail Works",
                    "Fabric Type": "Bamboo Viscose",
                    "Colour": "Forest Green",
                    "Department": "Women",
                    "Care Instructions": "Hand Wash",
                },
            }
        )

        self.assertEqual(attributes[AttributeName.CATEGORY].values, ["Shoes", "Hiking"])
        self.assertEqual(attributes[AttributeName.MATERIAL].values, ["Bamboo Viscose"])
        self.assertEqual(attributes[AttributeName.COLOR].values, ["Forest Green"])
        self.assertEqual(attributes[AttributeName.STYLE].values, ["Women"])
        self.assertEqual(attributes[AttributeName.BRAND].values, ["Trail Works"])
        self.assertEqual(attributes[AttributeName.BUDGET].minimum, 89.5)
        self.assertEqual(
            attributes[AttributeName.OTHER].details["care_instructions"],
            ["Hand Wash"],
        )
        self.assertEqual(
            product_attribute_text(attributes, AttributeName.OTHER),
            "Hand Wash",
        )

    def test_product_text_fallback_only_fills_missing_attributes(self) -> None:
        attributes = extract_product_attributes(
            {
                "title": "Black cotton running shirt",
                "details": {"Color": "Navy"},
            }
        )
        self.assertEqual(attributes[AttributeName.COLOR].values, ["Navy"])
        self.assertEqual(attributes[AttributeName.MATERIAL].values, ["cotton"])
        self.assertEqual(attributes[AttributeName.USE_CASE].values, ["running"])

    def test_generic_category_and_package_dimensions_do_not_pollute_options(self) -> None:
        attributes = extract_product_attributes(
            {
                "categories": ["Clothing, Shoes & Jewelry", "Women", "Dresses"],
                "details": {"Package Dimensions": "10 x 8 x 1 inches"},
            }
        )
        self.assertEqual(
            attributes[AttributeName.CATEGORY].values,
            ["Women", "Dresses"],
        )
        self.assertEqual(
            product_attribute_values(
                attributes,
                AttributeName.SIZE,
                include_details=False,
            ),
            [],
        )
        self.assertEqual(
            product_attribute_values(attributes, AttributeName.SIZE),
            ["10 x 8 x 1 inches"],
        )


if __name__ == "__main__":
    unittest.main()
