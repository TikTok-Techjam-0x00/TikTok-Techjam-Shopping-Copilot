"""Git, component, and input provenance for reproducible replay runs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


COMPONENT_PATHS: dict[str, tuple[str, ...]] = {
    "shared_contracts": ("src/item.py", "src/attribute.py"),
    "state": ("src/state",),
    "retrieval": ("src/retrieval",),
    "reranking": ("src/reranking",),
    "dialogue": ("src/dialogue",),
    "pipeline": ("src/pipeline", "starter/agent.py"),
    "official_local_evaluator": (
        "evaluator/local_evaluator.py",
        "docs/evaluation_config.json",
    ),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(project_root: Path, *args: str) -> str | None:
    try:
        process = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    return process.stdout.strip() if process.returncode == 0 else None


def _source_files(project_root: Path, entries: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for entry in entries:
        path = project_root / entry
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("*.py")))
    return sorted(set(files))


def _component_version(project_root: Path, entries: Iterable[str], commit: str) -> dict[str, Any]:
    entry_list = list(entries)
    rows: list[dict[str, str]] = []
    aggregate = hashlib.sha256()
    for path in _source_files(project_root, entry_list):
        relative = path.relative_to(project_root).as_posix()
        digest = sha256_file(path)
        rows.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    last_change_text = _git(
        project_root,
        "log",
        "-1",
        "--format=%H%x00%cI%x00%s",
        "--",
        *entry_list,
    )
    last_change: dict[str, str] | None = None
    if last_change_text:
        parts = last_change_text.split("\0", 2)
        if len(parts) == 3:
            last_change = {
                "commit": parts[0],
                "committed_at": parts[1],
                "subject": parts[2],
            }
    return {
        "git_commit": commit,
        "last_committed_change": last_change,
        "content_sha256": aggregate.hexdigest(),
        "files": rows,
    }


def _display_path(project_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def collect_provenance(
    project_root: str | Path,
    *,
    catalog_path: str | Path,
    dataset_path: str | Path,
    dataset_role: str = "public_evaluator_set",
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Capture both the Git revision and dirty-file content fingerprints."""

    root = Path(project_root).resolve()
    catalog = Path(catalog_path).resolve()
    dataset = Path(dataset_path).resolve()
    commit = _git(root, "rev-parse", "HEAD") or "unknown"
    status_text = _git(root, "status", "--porcelain=v1")
    dirty_files = status_text.splitlines() if status_text else []
    diff = _git(root, "diff", "--binary", "HEAD") or ""
    remote = _git(root, "remote", "get-url", "origin")

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "short_commit": commit[:12] if commit != "unknown" else "unknown",
            "branch": _git(root, "branch", "--show-current") or "detached-or-unknown",
            "remote": remote,
            "dirty": bool(dirty_files),
            "dirty_files": dirty_files,
            "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        },
        "component_versions": {
            name: _component_version(root, entries, commit)
            for name, entries in COMPONENT_PATHS.items()
        },
        "inputs": {
            "catalog": {
                "path": _display_path(root, catalog),
                "sha256": sha256_file(catalog),
                "bytes": catalog.stat().st_size,
            },
            "dataset": {
                "role": dataset_role,
                "path": _display_path(root, dataset),
                "sha256": sha256_file(dataset),
                "bytes": dataset.stat().st_size,
            },
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "command": list(command or sys.argv),
        },
    }
