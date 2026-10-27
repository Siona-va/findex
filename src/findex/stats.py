"""Corpus statistics in a single streaming pass.

    python -m findex.stats data/pydocs              # lazy pipeline (default)
    python -m findex.stats data/pydocs --limit 50   # first 50 documents only
    python -m findex.stats data/pydocs --eager      # deliberately eager version

Prints document count, token count, vocabulary size, top-N terms,
elapsed wall time and peak memory (tracemalloc).
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain, islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    top: list[tuple[str, int]]


def limited(docs: Iterator[Document], limit: int | None) -> Iterator[Document]:
    """``--limit N`` via ``itertools.islice`` -- still lazy."""
    return docs if limit is None else islice(docs, limit)


def compute_stats(docs: Iterable[Document], top: int = 50) -> CorpusStats:
    """Lazy version: one pass, only the Counter grows."""
    counts: Counter[str] = Counter()
    n_docs = 0
    for doc in docs:
        n_docs += 1
        counts.update(tokenize(doc.text))  # consumes the token generator
    return CorpusStats(n_docs, counts.total(), len(counts), counts.most_common(top))


def compute_stats_eager(docs: Iterable[Document], top: int = 50) -> CorpusStats:
    """Eager version -- written badly ON PURPOSE for the measurement (Lab 1 M4).

    1. every document's text is read into one list;
    2. every document is tokenized into a list, giving a list of lists;
    3. only then do we count.
    """
    all_docs = list(docs)  # whole corpus text in RAM
    token_lists = [list(tokenize(d.text)) for d in all_docs]  # every token object in RAM
    counts = Counter(chain.from_iterable(token_lists))
    return CorpusStats(len(all_docs), counts.total(), len(counts), counts.most_common(top))


def _fmt_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m findex.stats", description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--limit", type=int, default=None, help="only the first N documents")
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--eager", action="store_true", help="use the list-based version")
    args = parser.parse_args(argv)

    docs = limited(iter_documents(args.corpus), args.limit)
    fn = compute_stats_eager if args.eager else compute_stats

    tracemalloc.start()
    t0 = time.perf_counter()
    stats = fn(docs, top=args.top)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"version     : {'eager (lists)' if args.eager else 'lazy (generators)'}")
    print(f"documents   : {stats.documents:,}")
    print(f"tokens      : {stats.tokens:,}")
    print(f"vocabulary  : {stats.vocabulary:,}")
    print(f"elapsed     : {elapsed:.2f} s")
    print(f"peak memory : {_fmt_bytes(peak)}")
    print(f"top-{args.top} terms:")
    for rank, (term, count) in enumerate(stats.top, start=1):
        print(f"  {rank:>3}. {term:<20} {count:,}")


if __name__ == "__main__":
    main()
