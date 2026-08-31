from __future__ import annotations

import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_catalog as preparation


class PrepareCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.content = b'{"parent_asin":"test-product"}\n'
        self.archive = self.root / "catalog.jsonl.gz"
        self.archive.write_bytes(gzip.compress(self.content))
        self.destination = self.root / "data/catalog.jsonl"
        for name, digest in (
            ("ARCHIVE_SHA256", preparation.sha256(self.archive)),
            ("CATALOG_SHA256", hashlib.sha256(self.content).hexdigest()),
        ):
            patcher = patch.object(preparation, name, digest)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_prepare_and_repeat_are_safe(self) -> None:
        self.assertTrue(preparation.prepare_catalog(self.archive, self.destination))
        self.assertEqual(self.destination.read_bytes(), self.content)
        self.assertFalse(preparation.prepare_catalog(self.archive, self.destination))

    def test_different_existing_file_is_not_overwritten(self) -> None:
        self.destination.parent.mkdir()
        self.destination.write_bytes(b"user data")
        with self.assertRaises(FileExistsError):
            preparation.prepare_catalog(self.archive, self.destination)
        self.assertEqual(self.destination.read_bytes(), b"user data")

    def test_wrong_archive_is_rejected(self) -> None:
        self.archive.write_bytes(b"invalid archive")
        with self.assertRaisesRegex(ValueError, "Archive SHA256"):
            preparation.prepare_catalog(self.archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_wrong_catalog_is_not_published(self) -> None:
        with patch.object(preparation, "CATALOG_SHA256", "0" * 64):
            with self.assertRaisesRegex(ValueError, "Decompressed catalog"):
                preparation.prepare_catalog(self.archive, self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_archive_is_reported(self) -> None:
        with self.assertRaises(FileNotFoundError):
            preparation.prepare_catalog(self.root / "missing.gz", self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
