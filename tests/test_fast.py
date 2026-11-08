"""Lab 8: the NumPy scorer ranks EXACTLY like the Lab 3 scorer."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from findex.corpus import Document
from findex.fast import NumpyPostings, NumpySearchEngine, top_k
from findex.index import Index, build_index
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchEngine

QUERIES = ["event loop", "the OR loop OR zebra", "zebra", "NOT zebra", '"event loop"', "python"]


@pytest.mark.parametrize("scorer", [BM25(), TfIdf(), BM25(k1=1.2, b=0.0)], ids=repr)
@pytest.mark.parametrize("q", QUERIES)
def test_identical_ranking_on_fixture(index: Index, scorer: Scorer, q: str) -> None:
    slow, fast = SearchEngine(index, scorer), NumpySearchEngine(index, scorer)
    a, b = slow.search(q, k=20), fast.search(q, k=20)
    assert [r.doc_id for r in a] == [r.doc_id for r in b]
    assert [r.score for r in a] == pytest.approx([r.score for r in b], rel=1e-12)


words = st.sampled_from(["cat", "dog", "кіт", "the", "a", "zebra"])


@given(
    st.lists(st.lists(words, max_size=15).map(" ".join), min_size=1, max_size=30),
    st.sampled_from(["cat", "the OR dog", "cat dog", "zebra OR кіт OR the"]),
    st.integers(1, 8),
)
def test_identical_ranking_property(texts: list[str], q: str, k: int) -> None:
    ix = build_index(Document(i, str(i), t) for i, t in enumerate(texts))
    for scorer in (BM25(), TfIdf()):
        a = SearchEngine(ix, scorer).search(q, k=k)
        b = NumpySearchEngine(ix, scorer).search(q, k=k)
        assert [r.doc_id for r in a] == [r.doc_id for r in b]


def test_top_k_breaks_ties_by_doc_id() -> None:
    cand = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    scores = np.array([1.0, 5.0, 3.0, 5.0, 3.0])
    assert [d for d, _ in top_k(cand, scores, 3)] == [1, 3, 2]
    assert [d for d, _ in top_k(cand, scores, 10)] == [1, 3, 2, 4, 0]


def test_postings_are_views_not_copies(index: Index) -> None:
    p = NumpyPostings(index)
    ids, _ = p.postings("event")
    assert ids.base is p.doc_ids  # a slice of the one big array
    assert p.doc_ids.dtype == np.int32 and len(p.postings("nope")[0]) == 0
