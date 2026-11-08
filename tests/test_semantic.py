"""Lab 8: chunking, RRF, semantic + hybrid modes (LSA embedder: offline, no download)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from findex.fast import NumpySearchEngine
from findex.index import Index
from findex.semantic import (
    SemanticIndex,
    build_embeddings,
    chunks,
    emb_dir,
    rrf,
    search_mode,
)
from findex.web import create_app
from findex.web.settings import Settings

pytest.importorskip("sklearn")


@pytest.fixture(scope="module")
def sem_dir(index: Index, index_file: Path) -> Path:
    out = emb_dir(index_file)
    build_embeddings(index, out, model="lsa", words=4)
    return out


def test_chunks() -> None:
    assert list(chunks("a b c d e", words=2)) == [(0, "a b"), (2, "c d"), (4, "e")]
    assert list(chunks("", words=2)) == []


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_embeddings_are_unit_vectors(sem_dir: Path) -> None:
    sem = SemanticIndex(sem_dir)
    norms = np.linalg.norm(sem.vectors, axis=1)
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)
    assert len(sem.chunk_doc) == len(sem.vectors) == len(sem.chunk_start)


def test_semantic_and_hybrid_modes(index: Index, sem_dir: Path) -> None:
    sem = SemanticIndex(sem_dir)
    engine = NumpySearchEngine(index)
    semantic = search_mode(engine, sem, "event loop coroutines", k=3, mode="semantic")
    assert semantic and semantic[0].chunk_row is not None
    assert [r.score for r in semantic] == sorted((r.score for r in semantic), reverse=True)
    hybrid = search_mode(engine, sem, "event loop", k=3, mode="hybrid")
    assert hybrid[0].title.startswith("Asyncio")  # top in both lists -> top after RRF
    keyword = search_mode(engine, None, "zebra", k=5, mode="keyword")
    assert len(keyword) == 2
    with pytest.raises(LookupError):
        search_mode(engine, None, "zebra", mode="semantic")


def test_hybrid_survives_a_free_text_query_that_is_not_valid_syntax(
    index: Index, sem_dir: Path
) -> None:
    sem = SemanticIndex(sem_dir)
    res = search_mode(NumpySearchEngine(index), sem, "loop (unbalanced", k=3, mode="hybrid")
    assert res  # keyword side failed to parse -> semantic side alone


def test_api_modes(index_file: Path, sem_dir: Path, tmp_path: Path) -> None:
    with TestClient(create_app(Settings(index_path=index_file))) as c:
        r = c.get("/search", params={"q": "event loop", "mode": "hybrid", "k": 3})
        assert r.status_code == 200 and r.json()["mode"] == "hybrid"
        assert c.get("/stats").json()["semantic"] == "lsa"
        assert "Semantic" in c.get("/").text
        assert c.get("/search", params={"q": "x", "mode": "magic"}).status_code == 422


@pytest.fixture
def no_embeddings(index: Index, tmp_path: Path) -> Iterator[TestClient]:
    from findex.store import save

    path = tmp_path / "plain.pkl"
    save(index, path)
    with TestClient(create_app(Settings(index_path=path))) as c:
        yield c


def test_semantic_without_embeddings_is_400(no_embeddings: TestClient) -> None:
    r = no_embeddings.get("/search", params={"q": "loop", "mode": "semantic"})
    assert r.status_code == 400 and "findex embed" in r.json()["detail"]
