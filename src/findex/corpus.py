"""Lazy corpus reader.

``iter_documents(root)`` is a *generator*: it opens one file at a time, yields
one ``Document`` and moves on. The corpus is never loaded into memory as a
whole -- only the current document is alive.

Supported inputs:

* a directory tree of text files (``.txt``, ``.md``, ``.rst``), walked with
  ``Path.rglob``;
* a JSON-lines file (``.jsonl`` or ``.jsonl.gz``), one document per line, with a
  ``"text"`` field and an optional ``"title"`` field.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger(__name__)

TEXT_SUFFIXES: frozenset[str] = frozenset({".txt", ".md", ".rst"})


class Document(NamedTuple):
    """One document of the corpus."""

    doc_id: int
    path: str
    text: str
    title: str = ""


def _decode(raw: bytes, path: Path) -> str:
    """Decode UTF-8; on bad bytes log a warning and replace them with U+FFFD."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        log.warning("bad UTF-8 in %s (%s); replacing undecodable bytes", path, exc.reason)
        return raw.decode("utf-8", errors="replace")


def _title_from_text(text: str, fallback: str) -> str:
    """First non-empty line that is not an RST/Markdown decoration."""
    for line in text.splitlines():
        stripped = line.strip().strip("#").strip()
        if stripped and not set(stripped) <= set("=-~*^`:._+"):
            if stripped.startswith(".."):  # RST directive / comment
                continue
            return stripped[:120]
    return fallback


def iter_text_files(root: Path) -> Iterator[Path]:
    """Yield text files under ``root`` in a stable (sorted) order.

    Sorting needs the list of *paths* (small), never the file contents.
    Stable order gives stable ``doc_id`` values between runs.
    """
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in TEXT_SUFFIXES and p.is_file())
    yield from paths


def read_document(path: Path, doc_id: int, root: Path) -> Document | None:
    """Read ONE file as a Document; ``None`` (with a warning) if it cannot be read.

    Used both by the streaming reader and by parallel workers (Lab 5), which receive
    *paths* and read their own documents instead of being sent the text.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        log.warning("cannot read %s: %s", path, exc)
        return None
    text = _decode(raw, path)
    rel = path.relative_to(root).as_posix()
    return Document(doc_id, str(path), text, _title_from_text(text, rel))


def _iter_dir(root: Path) -> Iterator[Document]:
    for doc_id, path in enumerate(iter_text_files(root)):
        doc = read_document(path, doc_id, root)
        if doc is not None:
            yield doc


def _iter_jsonl(path: Path) -> Iterator[Document]:
    opener = gzip.open if path.suffix == ".gz" else open
    doc_id = 0
    with opener(path, "rb") as f:
        for line_no, raw in enumerate(f, start=1):
            line = _decode(raw, path).strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d: invalid JSON (%s); skipped", path, line_no, exc.msg)
                continue
            text = str(obj.get("text", ""))
            title = str(obj.get("title") or _title_from_text(text, f"line {line_no}"))
            yield Document(doc_id, f"{path}#L{line_no}", text, title)
            doc_id += 1


def iter_documents(root: Path) -> Iterator[Document]:
    """Stream documents from a directory tree or a JSON-lines file, one at a time."""
    if root.is_dir():
        yield from _iter_dir(root)
    elif root.name.endswith((".jsonl", ".jsonl.gz")):
        yield from _iter_jsonl(root)
    elif root.is_file():
        text = _decode(root.read_bytes(), root)
        yield Document(0, str(root), text, _title_from_text(text, root.name))
    else:
        raise FileNotFoundError(f"corpus not found: {root}")
