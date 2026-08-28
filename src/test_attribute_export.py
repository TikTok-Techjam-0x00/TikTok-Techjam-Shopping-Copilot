from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path

from extract_attributes import export_catalog_attributes


class AttributeExportTest(unittest.TestCase):
    def test_export_streams_deduplicated_attributes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            output = root / "attributes.jsonl"
            product = {
                "parent_asin": "A1",
                "title": "Black cotton shirt",
                "categories": ["Clothing, Shoes & Jewelry", "Shirts"],
                "details": {"Brand": "Example"},
                "price": 25,
            }
            catalog.write_text(
                "\n".join([json.dumps(product), json.dumps(product), "not-json"]),
                encoding="utf-8",
            )

            stats = export_catalog_attributes(catalog, output)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(stats["items_written"], 1)
        self.assertEqual(stats["duplicate_asins"], 1)
        self.assertEqual(stats["malformed_rows"], 1)
        self.assertEqual(rows[0]["parent_asin"], "A1")
        self.assertEqual(rows[0]["attributes"]["category"]["values"], ["Shirts"])
        self.assertEqual(rows[0]["attributes"]["brand"]["values"], ["Example"])

    def test_gzip_output_is_actually_compressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.jsonl"
            output = root / "attributes.jsonl.gz"
            catalog.write_text(
                json.dumps({"parent_asin": "A1", "details": {"Color": "Blue"}}),
                encoding="utf-8",
            )

            export_catalog_attributes(catalog, output)
            with gzip.open(output, "rt", encoding="utf-8") as handle:
                row = json.loads(handle.readline())

        self.assertEqual(row["attributes"]["color"]["values"], ["Blue"])


if __name__ == "__main__":
    unittest.main()
