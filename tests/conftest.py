"""Shared fixtures: a tiny corpus on disk and an index built from it (once per session)."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.store import save

# Coverage tracing makes Hypothesis' data generation slower; don't fail on that.
settings.register_profile("default", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("default")

TINY_CORPUS = {
    "asyncio.txt": "Asyncio: the event loop runs coroutines. The event loop is the core.",
    "threads.txt": "Threads share memory. The GIL lets one thread run Python bytecode at a time.",
    "loops.md": "# Loops\nA for loop iterates. The loop event happens later.",
    "ukr/вірш.txt": "Пʼять котів сиділи на вікні. П'ять — гарне число!",
    "long.rst": "Python " + "filler " * 300 + "zebra",
    "short.rst": "zebra crossing",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in TINY_CORPUS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    (root / "broken.txt").write_bytes(b"caf\xe9 latin-1 bytes")  # not valid UTF-8
    return root


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    """Session-scoped: built once. Tests must NOT mutate it (it is shared)."""
    return build_index(iter_documents(corpus_dir), positions=True)


@pytest.fixture(scope="session")
def index_file(index: Index, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "index.pkl"
    save(index, path)
    return path


def doc_id_of(index: Index, suffix: str) -> int:
    return next(d for d in index.all_doc_ids if index.doc_meta(d).path.endswith(suffix))
