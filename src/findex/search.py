"""Search: Boolean matching (Lab 2) and ranked retrieval (Lab 3).

    python -m findex.search index.pkl "query" [--scorer bm25|tfidf] [--k 10] [-v]
    python -m findex.search index.pkl "a b OR c" --boolean [--engine merge|set]

Ranked search = the query tree decides WHICH documents match; the scorer decides
their ORDER; ``heapq.nlargest`` takes the top-k in O(n log k).
"""

from __future__ import annotations

import argparse
import functools
import heapq
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from operator import itemgetter
from pathlib import Path

from findex.decorators import timed
from findex.index import Index
from findex.merge import ENGINES, DocIds, EngineName
from findex.query import parse
from findex.scoring import BM25, Scorer, ScorerName, make_scorer
from findex.snippets import Highlighter, brackets, load_text, make_snippet
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


# --- Lab 2: simple left-to-right Boolean search -----------------------------------------


def doc_ids(index: Index, term: str) -> DocIds:
    return [p.doc_id for p in index.get(term, [])]


def boolean_search(index: Index, query: str, engine: EngineName = "merge") -> DocIds:
    """``a b OR c NOT d`` evaluated left to right: ``((a AND b) OR c) NOT d``."""
    ops = ENGINES[engine]
    result: DocIds | None = None
    pending = "AND"
    for word in query.split():
        if word in ("AND", "OR", "NOT"):
            pending = word
            continue
        for term in tokenize(word):
            ids = doc_ids(index, term)
            result = ids if result is None else ops[pending](result, ids)
            pending = "AND"
    return result or []


# --- Lab 3: ranked search ---------------------------------------------------------------


@dataclass(order=True, frozen=True)
class SearchResult:
    """Ordered by score first (then doc_id), so ``sorted(results, reverse=True)`` ranks."""

    score: float
    doc_id: int
    title: str = field(compare=False)
    path: str = field(compare=False)


class SearchEngine:
    """An index + a scorer + an LRU cache of ``query string -> matching doc_ids``."""

    def __init__(self, index: Index, scorer: Scorer | None = None, cache_size: int = 256) -> None:
        self.index = index
        self.scorer: Scorer = scorer if scorer is not None else BM25()
        # A per-instance cache: lru_cache needs hashable arguments -> the query *string*.
        self._match = functools.lru_cache(maxsize=cache_size)(self._match_uncached)

    def _match_uncached(self, query: str) -> tuple[int, ...]:
        return tuple(parse(query).evaluate(self.index))

    def match(self, query: str) -> tuple[int, ...]:
        hits_before = self._match.cache_info().hits
        result = self._match(query)
        if self._match.cache_info().hits > hits_before:
            log.debug("cache hit for %r", query)
        return result

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        matched = self.match(query)
        if not matched:
            return []
        matched_set = set(matched)
        scores: dict[int, float] = dict.fromkeys(matched, 0.0)
        acc: defaultdict[int, float] = defaultdict(float)
        for term in sorted(set(parse(query).positive_terms())):  # fixed order: reproducible sums
            for posting in self.index.get(term, []):
                if posting.doc_id in matched_set:
                    acc[posting.doc_id] += self.scorer.score(term, posting, self.index)
        scores.update(acc)
        top = heapq.nlargest(k, scores.items(), key=itemgetter(1))
        return [
            SearchResult(score, d, self.index.doc_meta(d).title, self.index.doc_meta(d).path)
            for d, score in top
        ]

    def snippet(self, result: SearchResult, query: str, highlight: Highlighter = brackets) -> str:
        terms = set(parse(query).positive_terms())
        try:
            text = load_text(self.index.doc_meta(result.doc_id))
        except OSError:
            return "(document text not available)"
        return make_snippet(text, terms, highlight=highlight)


def main(argv: list[str] | None = None) -> None:
    from findex.store import open_index

    parser = argparse.ArgumentParser(prog="python -m findex.search")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--scorer", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--boolean", action="store_true", help="Lab 2 unranked search")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    parser.add_argument("--repeat", type=int, default=1, help="run the query N times")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2), format="%(name)s: %(message)s"
    )
    scorer_name: ScorerName = args.scorer

    with open_index(args.index) as index:
        if args.boolean:
            t0 = time.perf_counter()
            hits = boolean_search(index, args.query, args.engine)
            print(f"{len(hits)} documents match ({(time.perf_counter() - t0) * 1000:.2f} ms)")
            for d in hits[: args.k]:
                print(f"  [{d:>4}] {index.doc_meta(d).title}")
            return
        engine = SearchEngine(index, make_scorer(scorer_name))
        results: list[SearchResult] = []
        for _ in range(args.repeat):
            results = engine.search(args.query, k=args.k)
        print(f"{index!r} · scorer={scorer_name} · query={parse(args.query)}")
        for rank, r in enumerate(results, start=1):
            print(f"{rank:>2}. {r.score:7.3f}  {r.title}  ({r.path})")
            print(f"     {engine.snippet(r, args.query)}")


if __name__ == "__main__":
    main()
