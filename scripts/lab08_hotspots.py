"""Lab 8 M1: time the three hotspots found by the profiler (run before AND after a fix).

uv run python scripts/lab08_hotspots.py data/pydocs data/index.pkl
"""

from __future__ import annotations

import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import build_index
from findex.search import SearchEngine
from findex.store import load, save
from findex.tokenize import tokenize


def bench(fn: Callable[[], object], runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    corpus, index_path = Path(sys.argv[1]), Path(sys.argv[2])
    texts = [d.text for d in iter_documents(corpus)]
    ix = build_index(iter_documents(corpus), positions=True)
    save(ix, index_path)
    engine = SearchEngine(ix)

    def snippets(q: str) -> None:
        for r in engine.search(q, k=10):
            engine.snippet(r, q)

    rows = [
        ("1. load index (pickle, 332k postings)", lambda: load(index_path)),
        ("2. search + 10 snippets: `the OR a OR of`", lambda: snippets("the OR a OR of")),
        ("2. search + 10 snippets: `python async await`", lambda: snippets("python async await")),
        (
            "3. tokenize whole corpus (1.8M tokens)",
            lambda: sum(1 for t in texts for _ in tokenize(t)),
        ),
        ("   build_index, serial", lambda: build_index(iter_documents(corpus))),
    ]
    for label, fn in rows:
        print(f"| {label} | {bench(fn):.3f} s |")


if __name__ == "__main__":
    main()
