"""Validate and unpack the organizer's frozen catalog without network access."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
# Archive digest from the organizer's participant-kit SHA256SUMS.
ARCHIVE_SHA256 = "07fd142631fd6b03e2b4d09988c3eb7d53720e9d57010c79db48eeaada50a8f8"
CATALOG_SHA256 = "da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_catalog(archive: Path, destination: Path) -> bool:
    """Return True after preparing data, False if already valid; never overwrite."""
    if destination.exists():
        if sha256(destination) != CATALOG_SHA256:
            raise FileExistsError(f"Refusing to overwrite a different catalog: {destination}")
        return False
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive SHA256 mismatch; use the frozen participant-kit catalog.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Validate the decompressed bytes before creating the final file. Opening
    # the destination exclusively also protects a file created concurrently.
    with tempfile.TemporaryDirectory(prefix="catalog-", dir=destination.parent) as directory:
        unpacked = Path(directory) / "catalog.jsonl"
        with gzip.open(archive, "rb") as source, unpacked.open("wb") as output:
            shutil.copyfileobj(source, output)
        if sha256(unpacked) != CATALOG_SHA256:
            raise ValueError("Decompressed catalog SHA256 mismatch.")
        with unpacked.open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPOSITORY_ROOT / "catalog.jsonl.gz")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "data/catalog.jsonl")
    args = parser.parse_args()
    try:
        created = prepare_catalog(args.archive, args.output)
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"Catalog preparation failed: {error}\n")
    print(f"{'Prepared' if created else 'Already verified'}: {args.output}")
    print(f"SHA256: {CATALOG_SHA256}")


if __name__ == "__main__":
    main()
