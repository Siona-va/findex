"""Lab 2 measurements: memory study, storage formats, merge vs set.

    uv run python scripts/lab02_measure.py data/pydocs

Prints three Markdown tables that go straight into the README.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
import time
import timeit
import tracemalloc
from array import array
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from findex.corpus import iter_documents
from findex.index import build_index
from findex.models import PlainPosting, Posting
from findex.search import ENGINES, doc_ids
from findex.store import load, save
from findex.tokenize import tokenize


def build_objects(root: Path, cls: type[Any]) -> dict[str, list[Any]]:
    postings: defaultdict[str, list[Any]] = defaultdict(list)
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            postings[term].append(cls(doc.doc_id, tf))
    return dict(postings)


def build_arrays(root: Path) -> dict[str, tuple[array[int], array[int]]]:
    """No Posting objects at all: two parallel arrays of raw 4-byte unsigned ints."""
    postings: dict[str, tuple[array[int], array[int]]] = {}
    for doc in iter_documents(root):
        for term, tf in Counter(tokenize(doc.text)).items():
            pair = postings.get(term)
            if pair is None:
                pair = postings[term] = (array("I"), array("I"))
            pair[0].append(doc.doc_id)
            pair[1].append(tf)
    return postings


def measure(build: Callable[[], Any], tmp: Path, name: str) -> tuple[str, float, float, float]:
    # Build with tracemalloc, but keep the result only long enough to pickle it.
    tracemalloc.start()
    obj = build()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    f = tmp / f"{name}.pkl"
    with f.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    del obj
    t0 = time.perf_counter()
    with f.open("rb") as fh:
        pickle.load(fh)
    t_load = time.perf_counter() - t0
    return name, peak / 2**20, f.stat().st_size / 2**20, t_load


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pydocs")
    tmp = Path(tempfile.mkdtemp())

    print("\n### Memory study (postings only)\n")
    print("| Postings representation | Peak memory (build) | Index file size | Load time |")
    print("|---|---|---|---|")
    rows = [
        measure(lambda: build_objects(root, PlainPosting), tmp, "plain"),
        measure(lambda: build_objects(root, Posting), tmp, "slots"),
        measure(lambda: build_arrays(root), tmp, "arrays"),
    ]
    labels = {
        "plain": "`list[Posting]` with plain `@dataclass`",
        "slots": "`list[Posting]` with `slots=True`",
        "arrays": "`array('I')` pairs (doc_ids, tfs)",
    }
    for name, peak, size, t_load in rows:
        print(f"| {labels[name]} | {peak:.1f} MB | {size:.1f} MB | {t_load:.2f} s |")

    print("\n### Per-object size\n")
    print(
        f"- PlainPosting: {sys.getsizeof(PlainPosting(1, 1))} B + __dict__ "
        f"{sys.getsizeof(PlainPosting(1, 1).__dict__)} B"
    )
    print(f"- Posting (slots): {sys.getsizeof(Posting(1, 1))} B")
    print(f"- one array('I') element: {array('I').itemsize} B")

    print("\n### Storage formats (full index, no positions)\n")
    index = build_index(iter_documents(root))
    print("| Format | File size | Save | Load |")
    print("|---|---|---|---|")
    for ext in ("pkl", "json"):
        f = tmp / f"index.{ext}"
        t0 = time.perf_counter()
        save(index, f)
        t_save = time.perf_counter() - t0
        t0 = time.perf_counter()
        load(f)
        t_load = time.perf_counter() - t0
        print(f"| {ext} | {f.stat().st_size / 2**20:.1f} MB | {t_save:.2f} s | {t_load:.2f} s |")

    print("\n### merge vs set\n")
    by_df = sorted(index, key=index.df)
    common = by_df[-2:]
    rare = [t for t in by_df if index.df(t) >= 3][:2]
    print("| Terms (df) | Op | merge | set |")
    print("|---|---|---|---|")
    for pair in (common, rare, [common[-1], rare[0]]):
        a, b = doc_ids(index, pair[0]), doc_ids(index, pair[1])
        label = f"`{pair[0]}` ({len(a)}) · `{pair[1]}` ({len(b)})"
        for op in ("AND", "OR"):
            res = []
            for engine in ("merge", "set"):
                fn = ENGINES[engine][op]
                n = 2000
                t = timeit.timeit(lambda fn=fn, a=a, b=b: fn(a, b), number=n) / n
                res.append(f"{t * 1e6:.1f} µs")
            print(f"| {label} | {op} | {res[0]} | {res[1]} |")


if __name__ == "__main__":
    main()
