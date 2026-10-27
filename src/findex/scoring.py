"""Ranking functions behind one structural interface.

``Scorer`` is a ``typing.Protocol``: any object with a matching ``score`` method *is*
a scorer -- no inheritance needed. ``TfIdf`` and ``BM25`` are independent classes.
Both are also callable (``scorer(term, posting, index)``) via ``__call__``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

from findex.index import Index
from findex.models import Posting

ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def score(self, term: str, posting: Posting, index: Index) -> float: ...


@dataclass(frozen=True)
class TfIdf:
    """``(1 + log tf) * log(N / df)`` -- sublinear tf, rare terms weigh more."""

    def score(self, term: str, posting: Posting, index: Index) -> float:
        if posting.tf <= 0:
            return 0.0
        return (1.0 + math.log(posting.tf)) * index.idf(term)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


@dataclass(frozen=True)
class BM25:
    """Okapi BM25.

    * ``k1`` -- term-frequency saturation: the 20th occurrence adds almost nothing.
    * ``b``  -- document-length normalization: 0 = off, 1 = full.
    """

    k1: float = 1.5
    b: float = 0.75

    def idf(self, term: str, index: Index) -> float:
        n, df = index.num_docs, index.df(term)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def tf_weight(self, tf: int, doc_len: int, avg_len: float) -> float:
        norm = 1.0 - self.b + self.b * (doc_len / avg_len if avg_len else 1.0)
        return tf * (self.k1 + 1.0) / (tf + self.k1 * norm)

    def score(self, term: str, posting: Posting, index: Index) -> float:
        doc_len = index.doc_length(posting.doc_id)
        return self.idf(term, index) * self.tf_weight(posting.tf, doc_len, index.avg_doc_length)

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        return self.score(term, posting, index)


def make_scorer(name: ScorerName) -> Scorer:
    return BM25() if name == "bm25" else TfIdf()
