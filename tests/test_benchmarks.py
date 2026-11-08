"""Lab 8: performance tests (pytest-benchmark).

    uv run pytest tests/test_benchmarks.py --benchmark-only     # timings table
    uv run pytest                                               # also run as normal tests

The guard test fails CI if the NumPy scorer stops being faster than the Python one --
a regression check that does not depend on how fast the CI machine is.
"""

from __future__ import annotations

import timeit
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.corpus import Document, iter_documents
from findex.fast import NumpySearchEngine
from findex.index import Index, build_index
from findex.models import DocMeta, Posting
from findex.scoring import BM25
from findex.search import SearchEngine


@pytest.fixture(scope="module")
def big_index() -> Index:
    n = 20_000
    postings = {"term": [Posting(d, 1 + d % 7) for d in range(n)]}
    lengths = {d: 100 + d % 900 for d in range(n)}
    return Index(postings, lengths, {d: DocMeta(d, str(d), str(d)) for d in range(n)})


def test_bench_build_index(benchmark: BenchmarkFixture, corpus_dir: Path) -> None:
    ix = benchmark(lambda: build_index(iter_documents(corpus_dir)))
    assert ix.num_docs == 7


def test_bench_search_python(benchmark: BenchmarkFixture, big_index: Index) -> None:
    engine = SearchEngine(big_index, BM25())
    assert len(benchmark(engine.search, "term", 10)) == 10


def test_bench_search_numpy(benchmark: BenchmarkFixture, big_index: Index) -> None:
    engine = NumpySearchEngine(big_index, BM25())
    assert len(benchmark(engine.search, "term", 10)) == 10


def test_numpy_scorer_stays_faster(big_index: Index) -> None:
    """Relative guard: machine-independent, so it can run in CI."""
    slow, fast = SearchEngine(big_index, BM25()), NumpySearchEngine(big_index, BM25())
    t_slow = min(timeit.repeat(lambda: slow.search("term"), number=5, repeat=3))
    t_fast = min(timeit.repeat(lambda: fast.search("term"), number=5, repeat=3))
    assert t_fast * 3 < t_slow, f"NumPy {t_fast:.4f}s vs Python {t_slow:.4f}s"


def test_tiny_docs_still_work() -> None:
    ix = build_index([Document(0, "0", "a b"), Document(1, "1", "b")])
    assert [r.doc_id for r in NumpySearchEngine(ix).search("b")] == [1, 0]
