"""Vectorized scoring with NumPy (Lab 8).

Postings live in two flat ``int32`` arrays (``doc_ids``, ``tfs``) -- built with
``np.frombuffer`` straight from ``Index.flat_postings()``'s ``array('I')`` buffers, no
per-element copy -- and every term owns a *slice* (a view, not a copy) of them.
Document lengths are one ``int32[max_doc_id + 1]`` array.

Scoring a term for ALL its documents is then one array expression (ufuncs + fancy
indexing + scatter-add) instead of a Python loop over postings; top-k is
``np.argpartition`` (O(n)) instead of a heap over Python tuples.

``NumpySearchEngine`` is a drop-in ``SearchEngine``: same query language, same matching,
same ranking (tested against the Lab 3 scorer, including tie-breaking by doc_id).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt

from findex.decorators import timed
from findex.index import Index
from findex.query import parse
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchEngine, SearchResult

log = logging.getLogger(__name__)

I32 = npt.NDArray[np.int32]
F64 = npt.NDArray[np.float64]


class NumpyPostings:
    """Read-only postings as NumPy arrays."""

    def __init__(self, index: Index) -> None:
        terms, offsets, ids, tfs = index.flat_postings()
        # frombuffer: a view on the array('I') buffer, no copy; then ONE cast to int32
        self.doc_ids: I32 = np.frombuffer(ids, dtype=np.uint32).astype(np.int32)
        self.tfs: I32 = np.frombuffer(tfs, dtype=np.uint32).astype(np.int32)
        off = np.frombuffer(offsets, dtype=np.uint32)
        self.span: dict[str, tuple[int, int]] = {
            t: (int(off[i]), int(off[i + 1])) for i, t in enumerate(terms)
        }
        ids_all = index.all_doc_ids
        self.doc_len: I32 = np.zeros((ids_all[-1] + 1) if ids_all else 0, dtype=np.int32)
        for d in ids_all:
            self.doc_len[d] = index.doc_length(d)
        self.num_docs = index.num_docs
        self.avg_len = index.avg_doc_length

    def postings(self, term: str) -> tuple[I32, I32]:
        lo, hi = self.span.get(term, (0, 0))
        return self.doc_ids[lo:hi], self.tfs[lo:hi]  # views

    @property
    def nbytes(self) -> int:
        return self.doc_ids.nbytes + self.tfs.nbytes + self.doc_len.nbytes

    # --- the vectorized scorers ----------------------------------------------------------

    def bm25(self, terms: list[str], k1: float = 1.5, b: float = 0.75) -> F64:
        scores = np.zeros(self.doc_len.shape[0], dtype=np.float64)
        for term in terms:
            ids, tfs = self.postings(term)
            if not len(ids):
                continue
            df = len(ids)
            idf = math.log(1.0 + (self.num_docs - df + 0.5) / (df + 0.5))
            norm = 1.0 - b + b * (self.doc_len[ids] / self.avg_len)  # fancy indexing
            tf = tfs.astype(np.float64)
            scores[ids] += idf * (tf * (k1 + 1.0) / (tf + k1 * norm))  # ufuncs + scatter-add
        return scores

    def tfidf(self, terms: list[str]) -> F64:
        scores = np.zeros(self.doc_len.shape[0], dtype=np.float64)
        for term in terms:
            ids, tfs = self.postings(term)
            if not len(ids):
                continue
            idf = math.log(self.num_docs / len(ids))
            scores[ids] += (1.0 + np.log(tfs.astype(np.float64))) * idf
        return scores


def top_k(candidates: I32, scores: F64, k: int) -> list[tuple[int, float]]:
    """Best ``k`` (doc_id, score), score desc, ties by doc_id asc -- like Lab 3's heapq.

    ``argpartition`` finds the k-th best score in O(n); every doc above it is in, and
    the ties AT it are taken lowest doc_id first (candidates are sorted ascending).
    """
    s = scores[candidates]
    if len(candidates) > k:
        kth = s[np.argpartition(-s, k - 1)[k - 1]]
        above = s > kth
        ties = np.flatnonzero(s == kth)[: k - int(above.sum())]
        keep = np.concatenate([np.flatnonzero(above), ties])
        candidates, s = candidates[keep], s[keep]
    order = np.lexsort((candidates, -s))  # last key is primary: -score, then doc_id
    return [(int(candidates[i]), float(s[i])) for i in order]


class NumpySearchEngine(SearchEngine):
    """Same API as ``SearchEngine``; scoring and top-k in NumPy."""

    def __init__(
        self,
        index: Index,
        scorer: Scorer | None = None,
        cache_size: int = 256,
        postings: NumpyPostings | None = None,
    ) -> None:
        super().__init__(index, scorer, cache_size)
        self.np = postings if postings is not None else NumpyPostings(index)

    @timed
    def search(self, query: str, k: int = 10) -> list[SearchResult]:
        matched = self.match(query)
        if not matched:
            return []
        terms = sorted(set(parse(query).positive_terms()))
        if isinstance(self.scorer, BM25):
            scores = self.np.bm25(terms, self.scorer.k1, self.scorer.b)
        elif isinstance(self.scorer, TfIdf):
            scores = self.np.tfidf(terms)
        else:  # an unknown scorer: fall back to the per-posting Python loop
            return super().search(query, k)
        cand = np.fromiter(matched, dtype=np.int32, count=len(matched))
        return [
            SearchResult(score, d, self.index.doc_meta(d).title, self.index.doc_meta(d).path)
            for d, score in top_k(cand, scores, k)
        ]
