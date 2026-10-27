"""Boolean search over the inverted index.

    python -m findex.search index.pkl "query" [--engine merge|set]

Query syntax (Lab 2, evaluated left to right):  ``a b`` (implicit AND),
``a OR b``, ``a NOT b``.

Two interchangeable engines:

* ``merge`` -- the classic two-pointer merge over doc_id lists sorted ascending.
  O(len(a) + len(b)), no hashing, streams, works on compressed postings.
* ``set``   -- ``set(a) & set(b)`` etc. Simple; pays for building hash tables.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections.abc import Callable, Sequence
from pathlib import Path

from findex.index import IndexData
from findex.tokenize import tokenize

DocIds = list[int]


# --- two-pointer merges on sorted lists ----------------------------------------------


def merge_and(a: Sequence[int], b: Sequence[int]) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: Sequence[int], b: Sequence[int]) -> DocIds:
    out: DocIds = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: Sequence[int], b: Sequence[int]) -> DocIds:
    """Documents in ``a`` but not in ``b`` (``a NOT b``)."""
    out: DocIds = []
    i = j = 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# --- set-based alternatives -----------------------------------------------------------


def set_and(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) & set(b))


def set_or(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) | set(b))


def set_not(a: Sequence[int], b: Sequence[int]) -> DocIds:
    return sorted(set(a) - set(b))


Op = Callable[[Sequence[int], Sequence[int]], DocIds]
ENGINES: dict[str, dict[str, Op]] = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def doc_ids(index: IndexData, term: str) -> DocIds:
    return [p.doc_id for p in index.postings.get(term, [])]


def boolean_search(index: IndexData, query: str, engine: str = "merge") -> DocIds:
    """Left-to-right evaluation: ``a b OR c NOT d`` == ``((a AND b) OR c) NOT d``."""
    ops = ENGINES[engine]
    result: DocIds | None = None
    pending = "AND"
    for word in query.split():
        if word in ("AND", "OR", "NOT"):
            pending = word
            continue
        for term in tokenize(word):  # normalize exactly like the indexer did
            ids = doc_ids(index, term)
            result = ids if result is None else ops[pending](result, ids)
            pending = "AND"
    return result or []


def main(argv: list[str] | None = None) -> None:
    from findex.store import load

    parser = argparse.ArgumentParser(prog="python -m findex.search")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--engine", choices=sorted(ENGINES), default="merge")
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = load(args.index)
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    hits = boolean_search(index, args.query, args.engine)
    t_search = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"{len(hits)} documents match {args.query!r} (engine={args.engine})")
    for doc_id in hits[: args.show]:
        meta = index.doc_meta[doc_id]
        print(f"  [{doc_id:>4}] {meta.title}  ({meta.path})")
    print(f"load {t_load:.2f} s · search {t_search * 1000:.2f} ms · peak {peak / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
