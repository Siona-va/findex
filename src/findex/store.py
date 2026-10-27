"""Saving and loading the index. Format is chosen by file extension.

* ``.pkl`` -- pickle. Fast and compact, BUT: ``pickle.load`` can execute arbitrary
  code embedded in the file (a pickle may tell the loader to call any importable
  callable, e.g. ``os.system``). Loading a pickle from an untrusted source is a
  remote-code-execution hole. We only ever load index files *we built ourselves*.
* ``.json`` -- safe, human-readable, portable; bigger and slower. Tuples become
  lists and int keys become strings, so we convert them back on load.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

from findex.index import IndexData
from findex.models import DocMeta, Posting

FORMAT_VERSION = 1


def save(index: IndexData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        _save_json(index, path)
    else:
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: Path) -> IndexData:
    if not path.exists():
        raise FileNotFoundError(f"index file not found: {path}")
    if path.suffix == ".json":
        return _load_json(path)
    with path.open("rb") as f:
        # SECURITY: never point this at a file you did not create yourself.
        obj = pickle.load(f)
    if not isinstance(obj, IndexData):
        raise ValueError(f"{path} does not contain a findex index")
    return obj


def _save_json(index: IndexData, path: Path) -> None:
    data: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "has_positions": index.has_positions,
        "doc_lengths": index.doc_lengths,
        "doc_meta": {d: [m.path, m.title] for d, m in index.doc_meta.items()},
        "postings": {
            term: [[p.doc_id, p.tf, list(p.positions)] for p in plist]
            for term, plist in index.postings.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))


def _load_json(path: Path) -> IndexData:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"unsupported index format version in {path}")
    return IndexData(
        postings={
            term: [Posting(d, tf, tuple(pos)) for d, tf, pos in plist]
            for term, plist in data["postings"].items()
        },
        doc_lengths={int(k): v for k, v in data["doc_lengths"].items()},
        doc_meta={int(k): DocMeta(int(k), p, t) for k, (p, t) in data["doc_meta"].items()},
        has_positions=data["has_positions"],
    )
