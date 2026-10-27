"""Building the inverted index in one streaming pass.

    python -m findex.index data/pydocs --out index.pkl [--positions] [--limit N]

The corpus still streams (Lab 1 pipeline); only the index grows.
Postings are sorted by ``doc_id`` *for free*: documents arrive in increasing
``doc_id`` order and we append each document's postings when it ends.
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.models import DocMeta, Posting
from findex.tokenize import tokenize


@dataclass
class IndexData:
    """Raw index: three dictionaries (Lab 3 wraps this into a proper class)."""

    postings: dict[str, list[Posting]] = field(default_factory=dict)
    doc_lengths: dict[int, int] = field(default_factory=dict)
    doc_meta: dict[int, DocMeta] = field(default_factory=dict)
    has_positions: bool = False


def build_index(docs: Iterable[Document], positions: bool = False) -> IndexData:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}

    for doc in docs:
        if positions:
            # term -> list of token offsets in *this* document
            pos_map: defaultdict[str, list[int]] = defaultdict(list)
            length = 0
            for i, tok in enumerate(tokenize(doc.text)):
                pos_map[tok].append(i)
                length = i + 1
            for term, plist in pos_map.items():
                postings[term].append(Posting(doc.doc_id, len(plist), tuple(plist)))
        else:
            counts = Counter(tokenize(doc.text))  # the current document's Counter
            length = counts.total()
            for term, tf in counts.items():  # "flush" it when the document ends
                postings[term].append(Posting(doc.doc_id, tf))
        doc_lengths[doc.doc_id] = length
        doc_meta[doc.doc_id] = DocMeta(doc.doc_id, doc.path, doc.title)

    return IndexData(dict(postings), doc_lengths, doc_meta, positions)


def main(argv: list[str] | None = None) -> None:
    from findex.store import save  # local import: store depends on this module

    parser = argparse.ArgumentParser(prog="python -m findex.index")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, required=True, help=".pkl (pickle) or .json")
    parser.add_argument("--positions", action="store_true", help="store token positions")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

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

    print(f"documents : {len(index.doc_meta):,}")
    print(f"terms     : {len(index.postings):,}")
    print(f"postings  : {sum(len(p) for p in index.postings.values()):,}")
    print(f"build     : {t_build:.2f} s, peak memory {peak / 2**20:.1f} MB")
    print(f"save      : {t_save:.2f} s -> {args.out} ({args.out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
