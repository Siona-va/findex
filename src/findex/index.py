"""The inverted index as a Python object, and building it in one streaming pass.

    python -m findex.index data/pydocs --out index.pkl [--positions] [--limit N]

``Index`` is a read-only ``Mapping[str, list[Posting]]`` (term -> postings), so it
behaves like a built-in: ``len(ix)``, ``"python" in ix``, ``ix["python"]``,
``for term in ix``, ``ix.get(term, [])``, ``ix.keys()`` ... all work.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from functools import cached_property
from itertools import islice
from pathlib import Path
from typing import Any

from findex.corpus import Document, iter_documents
from findex.decorators import timed
from findex.models import DocMeta, Posting
from findex.tokenize import tokenize

log = logging.getLogger(__name__)


class Index(Mapping[str, list[Posting]]):
    """Term -> postings sorted by doc_id, plus per-document lengths and metadata."""

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
        has_positions: bool = False,
    ) -> None:
        self._postings = postings
        self._doc_lengths = doc_lengths
        self._doc_meta = doc_meta
        self.has_positions = has_positions

    # --- the Mapping protocol: three abstract methods, the rest comes for free ---------

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]  # KeyError if absent, as a Mapping should

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:  # faster than Mapping's try/except
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_}, positions={self.has_positions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return (
            self._postings == other._postings
            and self._doc_lengths == other._doc_lengths
            and self._doc_meta == other._doc_meta
            and self.has_positions == other.has_positions
        )

    __hash__ = None  # type: ignore[assignment]  # mutable container: explicitly unhashable

    def flat_postings(self) -> tuple[tuple[str, ...], array[int], array[int], array[int]]:
        """All postings as flat buffers: term ``terms[i]`` owns ``offsets[i]:offsets[i+1]``.

        Used for compact pickling and by the NumPy scorer (``np.frombuffer`` views).
        """
        terms = tuple(self._postings)
        offsets, ids, tfs = array("I", [0]), array("I"), array("I")
        for term in terms:
            for p in self._postings[term]:
                ids.append(p.doc_id)
                tfs.append(p.tf)
            offsets.append(len(ids))
        return terms, offsets, ids, tfs

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle as flat ``array('I')`` buffers instead of 332k ``Posting`` objects.

        Profiling (Lab 8) showed loading spent 1.3 of 1.5 s in ``dataclasses``'
        per-object ``__setstate__``. Arrays unpickle as one memcpy; ``_rebuild_index``
        then creates the postings in one tight loop. Same object back, ~4x faster.
        """
        terms, offsets, ids, tfs = self.flat_postings()
        pos_off, pos = array("I", [0]), array("I")
        if self.has_positions:
            for term in terms:
                for p in self._postings[term]:
                    pos.extend(p.positions)
                    pos_off.append(len(pos))
        meta = tuple((m.doc_id, m.path, m.title) for m in self._doc_meta.values())
        state = (
            terms,
            offsets,
            ids,
            tfs,
            pos_off,
            pos,
            self._doc_lengths,
            meta,
            self.has_positions,
        )
        return (_rebuild_index, state)

    # --- statistics -----------------------------------------------------------------

    @property
    def num_docs(self) -> int:
        return len(self._doc_lengths)

    @cached_property
    def avg_doc_length(self) -> float:
        """Computed once per index (BM25 needs it for every scored document)."""
        return sum(self._doc_lengths.values()) / self.num_docs if self.num_docs else 0.0

    @cached_property
    def all_doc_ids(self) -> list[int]:
        return sorted(self._doc_lengths)

    @cached_property
    def total_tokens(self) -> int:
        return sum(self._doc_lengths.values())

    def doc_length(self, doc_id: int) -> int:
        return self._doc_lengths[doc_id]

    def doc_meta(self, doc_id: int) -> DocMeta:
        return self._doc_meta[doc_id]

    def df(self, term: str) -> int:
        """Document frequency: in how many documents the term occurs."""
        postings = self._postings.get(term)
        return len(postings) if postings else 0

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log(self.num_docs / df) if df else 0.0

    def collection_frequencies(self) -> Counter[str]:
        return Counter({t: sum(p.tf for p in plist) for t, plist in self._postings.items()})

    def close(self) -> None:
        """Release the big structures (used by ``open_index``)."""
        self._postings = {}
        self._doc_lengths = {}
        self._doc_meta = {}
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("all_doc_ids", None)
        self.__dict__.pop("total_tokens", None)


def _rebuild_index(
    terms: tuple[str, ...],
    offsets: array[int],
    ids: array[int],
    tfs: array[int],
    pos_off: array[int],
    pos: array[int],
    doc_lengths: dict[int, int],
    meta: tuple[tuple[int, str, str], ...],
    has_positions: bool,
) -> Index:
    """Inverse of ``Index.__reduce__`` (module level, so pickle can find it)."""
    postings: dict[str, list[Posting]] = {}
    for i, term in enumerate(terms):
        lo, hi = offsets[i], offsets[i + 1]
        if has_positions:
            postings[term] = [
                Posting(ids[j], tfs[j], tuple(pos[pos_off[j] : pos_off[j + 1]]))
                for j in range(lo, hi)
            ]
        else:
            postings[term] = [Posting(d, f) for d, f in zip(ids[lo:hi], tfs[lo:hi], strict=True)]
    doc_meta = {d: DocMeta(d, p, t) for d, p, t in meta}
    return Index(postings, doc_lengths, doc_meta, has_positions)


@timed
def build_index(docs: Iterable[Document], positions: bool = False) -> Index:
    """One pass over a lazily streamed corpus; only the index grows."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc in docs:
        if positions:
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for i, tok in enumerate(tokenize(doc.text)):
                pos_map[tok].append(i)
                length = i + 1
            for term, plist in pos_map.items():
                postings[term].append(Posting(doc.doc_id, len(plist), tuple(plist)))
        else:
            counts = Counter(tokenize(doc.text))
            length = counts.total()
            for term, tf in counts.items():
                postings[term].append(Posting(doc.doc_id, tf))
        doc_lengths[doc.doc_id] = length
        doc_meta[doc.doc_id] = DocMeta(doc.doc_id, doc.path, doc.title)

    log.info("indexed %d documents, %d terms", len(doc_lengths), len(postings))
    return Index(dict(postings), doc_lengths, doc_meta, positions)


def main(argv: list[str] | None = None) -> None:
    from findex.store import save

    parser = argparse.ArgumentParser(prog="python -m findex.index")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True, help=".pkl (pickle) or .json")
    parser.add_argument("--positions", action="store_true", help="store token positions")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING - 10 * min(args.verbose, 2))

    docs: Iterable[Document] = iter_documents(args.corpus)
    if args.limit is not None:
        docs = islice(docs, args.limit)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(docs, positions=args.positions)
    t_build = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t0 = time.perf_counter()
    save(index, args.out)
    t_save = time.perf_counter() - t0

    print(repr(index))
    print(f"postings  : {sum(len(p) for p in index.values()):,}")
    print(f"build     : {t_build:.2f} s, peak memory {peak / 2**20:.1f} MB")
    print(f"save      : {t_save:.2f} s -> {args.out} ({args.out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    # Pitfall: under `python -m findex.index` this file runs as module `__main__`, so the
    # classes defined here would be pickled as `__main__.Index` and could not be loaded
    # from anywhere else. Re-import the real module and run *its* main().
    from findex.index import main as _main

    _main()
