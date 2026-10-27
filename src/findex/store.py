"""Saving and loading the index. Format is chosen by file extension.

* ``.pkl`` -- pickle. Fast and simple, BUT: ``pickle.load`` can execute arbitrary
  code embedded in the file (a pickle may instruct the loader to call any importable
  callable, e.g. ``os.system``, via ``__reduce__``). Loading a pickle from an untrusted
  source is a remote-code-execution hole. We only ever load index files we built.
* ``.json`` -- safe, human-readable, portable. Tuples become lists and int keys become
  strings, so they are converted back on load.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TypedDict, cast

from findex.decorators import timed
from findex.index import Index
from findex.models import DocMeta, Posting

FORMAT_VERSION = 1


class IndexJson(TypedDict):
    """Shape of the JSON index file (JSON object keys are always strings)."""

    version: int
    has_positions: bool
    doc_lengths: dict[str, int]
    doc_meta: dict[str, tuple[str, str]]
    postings: dict[str, list[tuple[int, int, list[int]]]]


class IndexFormatError(ValueError):
    """The file exists but is not a findex index we can read."""


@timed
def save(index: Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        _save_json(index, path)
    else:
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


@timed
def load(path: Path) -> Index:
    if not path.is_file():
        raise FileNotFoundError(f"index file not found: {path}")
    if path.suffix == ".json":
        return _load_json(path)
    try:
        with path.open("rb") as f:
            # SECURITY: never point this at a file you did not create yourself.
            obj = pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise IndexFormatError(f"{path} is not a valid findex index: {exc}") from exc
    if not isinstance(obj, Index):
        raise IndexFormatError(f"{path} does not contain a findex index")
    return obj


@contextmanager
def open_index(path: Path) -> Generator[Index]:
    """``with open_index(p) as ix: ...`` -- the index is released even if the body raises.

    Everything before ``yield`` is ``__enter__``, everything in ``finally`` is ``__exit__``.
    """
    index = load(path)
    try:
        yield index
    finally:
        index.close()


def _save_json(index: Index, path: Path) -> None:
    data: IndexJson = {
        "version": FORMAT_VERSION,
        "has_positions": index.has_positions,
        "doc_lengths": {str(d): index.doc_length(d) for d in index.all_doc_ids},
        "doc_meta": {
            str(d): (index.doc_meta(d).path, index.doc_meta(d).title) for d in index.all_doc_ids
        },
        "postings": {
            term: [(p.doc_id, p.tf, list(p.positions)) for p in plist]
            for term, plist in index.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> Index:
    try:
        with path.open(encoding="utf-8") as f:
            raw: object = json.load(f)
    except json.JSONDecodeError as exc:
        raise IndexFormatError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or cast(dict[str, object], raw).get("version") != FORMAT_VERSION:
        raise IndexFormatError(f"unsupported index format in {path}")
    # json.load returns untyped data; after the version check we trust our own file format.
    data = cast(IndexJson, raw)
    return Index(
        postings={
            term: [Posting(d, tf, tuple(pos)) for d, tf, pos in plist]
            for term, plist in data["postings"].items()
        },
        doc_lengths={int(k): v for k, v in data["doc_lengths"].items()},
        doc_meta={int(k): DocMeta(int(k), p, t) for k, (p, t) in data["doc_meta"].items()},
        has_positions=data["has_positions"],
    )
