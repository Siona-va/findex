"""The three ranking sanity checks from Lab 3, as tests; plus search/snippets."""

from __future__ import annotations

import logging

import pytest
from conftest import doc_id_of

from findex.corpus import Document
from findex.index import Index, build_index
from findex.models import Posting
from findex.scoring import BM25, Scorer, TfIdf, make_scorer
from findex.search import SearchEngine, SearchResult
from findex.snippets import make_snippet


def test_both_scorers_satisfy_the_protocol() -> None:
    scorers: list[Scorer] = [BM25(), TfIdf(), make_scorer("bm25"), make_scorer("tfidf")]
    ix = build_index([Document(0, "0", "a b"), Document(1, "1", "b")])
    for s in scorers:
        assert s.score("a", ix["a"][0], ix) > 0


def test_rare_term_outranks_common_term() -> None:
    docs = [Document(i, str(i), "the the the common text") for i in range(9)]
    docs.append(Document(9, "9", "the rare zebra"))
    ix = build_index(docs)
    for scorer in (BM25(), TfIdf()):
        results = SearchEngine(ix, scorer).search("the OR zebra", k=3)
        assert results[0].doc_id == 9


def test_bm25_saturates_term_frequency() -> None:
    bm25 = BM25()
    w = [bm25.tf_weight(tf, 100, 100.0) for tf in range(1, 21)]
    assert w[1] - w[0] > 10 * (w[19] - w[18])  # 2nd occurrence >> 20th occurrence
    assert w[-1] < bm25.k1 + 1


def test_short_document_beats_long_document(index: Index) -> None:
    results = SearchEngine(index, BM25()).search("zebra", k=2)
    assert results[0].doc_id == doc_id_of(index, "short.rst")
    tfidf = TfIdf()
    long_p, short_p = sorted(index["zebra"], key=lambda p: -index.doc_length(p.doc_id))
    assert tfidf.score("zebra", long_p, index) == tfidf.score("zebra", short_p, index)


def test_b_zero_disables_length_normalization() -> None:
    bm = BM25(b=0.0)
    assert bm.tf_weight(1, 10, 100.0) == bm.tf_weight(1, 1000, 100.0)


def test_results_are_ordered_and_limited(index: Index) -> None:
    results = SearchEngine(index).search("the OR loop OR event", k=3)
    assert len(results) == 3
    assert results == sorted(results, reverse=True)
    assert isinstance(results[0], SearchResult)


def test_phrase_search_and_cache_hit(index: Index, caplog: pytest.LogCaptureFixture) -> None:
    engine = SearchEngine(index)
    with caplog.at_level(logging.DEBUG, logger="findex.search"):
        first = engine.search('"event loop"')
        second = engine.search('"event loop"')
    assert first == second
    assert {r.doc_id for r in first} == {doc_id_of(index, "asyncio.txt")}  # not loops.md
    assert "cache hit" in caplog.text


def test_snippet_highlights_and_trims() -> None:
    text = "x " * 200 + "the Event Loop runs here " + "y " * 200
    snip = make_snippet(text, {"event", "loop"}, width=30)
    assert "[Event] [Loop]" in snip
    assert snip.startswith("…") and snip.endswith("…")
    assert len(snip) < 120


def test_snippet_without_match_shows_beginning() -> None:
    assert make_snippet("hello world", {"zzz"}) == "hello world"


def test_search_snippet_reads_document(index: Index) -> None:
    engine = SearchEngine(index)
    r = engine.search("гарне", k=1)[0]
    assert "[гарне]" in engine.snippet(r, "гарне")


def test_tf_weight_matches_formula() -> None:
    bm = BM25(k1=1.2, b=0.75)
    assert bm.tf_weight(3, 50, 100.0) == pytest.approx(3 * 2.2 / (3 + 1.2 * (0.25 + 0.375)))
    assert TfIdf().score("x", Posting(0, 0), build_index([Document(0, "0", "x")])) == 0.0
