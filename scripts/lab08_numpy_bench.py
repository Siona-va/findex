"""Lab 8 M2: Lab 3's Python scorer vs the NumPy scorer, same queries, same results.

uv run python scripts/lab08_numpy_bench.py data/index.pkl
"""

from __future__ import annotations

import sys
import timeit
from pathlib import Path

from findex.fast import NumpySearchEngine
from findex.index import Index
from findex.scoring import BM25
from findex.search import SearchEngine
from findex.store import load

QUERIES = [
    ("common term", "the"),
    ("rare term", "zipapp"),
    ("3 terms, AND", "python async await"),
    ("3 terms, OR", "python OR async OR await"),
    ("3 common terms, OR", "the OR a OR of"),
]


def per_call_ms(fn: object) -> float:
    timer = timeit.Timer(fn)  # type: ignore[arg-type]
    n, _ = timer.autorange()
    return min(timer.repeat(repeat=5, number=n)) / n * 1000


def main() -> None:
    ix = load(Path(sys.argv[1] if len(sys.argv) > 1 else "data/index.pkl"))
    py, fast = SearchEngine(ix, BM25()), NumpySearchEngine(ix, BM25())
    print(
        "| Query | Docs matched | Postings scored | Lab 3 scorer (ms) | NumPy scorer (ms) "
        "| Speedup |"
    )
    print("|---|---|---|---|---|---|")
    for label, q in QUERIES:
        assert [r.doc_id for r in py.search(q)] == [r.doc_id for r in fast.search(q)]
        matched = len(py.match(q))  # both engines now have q in their match cache
        scored = sum(ix.df(t) for t in set(q.lower().split()) - {"or"})
        t_py = per_call_ms(lambda q=q: py.search(q, k=10))
        t_np = per_call_ms(lambda q=q: fast.search(q, k=10))
        print(
            f"| {label}: `{q}` | {matched} | {scored} | {t_py:.3f} | {t_np:.3f} | "
            f"{t_py / t_np:.1f}× |"
        )
    scale()


def scale() -> None:
    """Synthetic: one term present in ALL N documents -- how the gap grows with N."""
    import random

    from findex.models import DocMeta, Posting

    print("\n| Postings for the term (N docs) | Lab 3 scorer (ms) | NumPy scorer (ms) | Speedup |")
    print("|---|---|---|---|")
    rng = random.Random(42)
    for n in (1_000, 10_000, 100_000):
        lengths = {d: rng.randint(50, 5000) for d in range(n)}
        postings = {"term": [Posting(d, rng.randint(1, 20)) for d in range(n)]}
        ix = Index(postings, lengths, {d: DocMeta(d, str(d), str(d)) for d in range(n)})
        py, fast = SearchEngine(ix, BM25()), NumpySearchEngine(ix, BM25())
        assert [r.doc_id for r in py.search("term")] == [r.doc_id for r in fast.search("term")]
        t_py = per_call_ms(lambda py=py: py.search("term", k=10))
        t_np = per_call_ms(lambda fast=fast: fast.search("term", k=10))
        print(f"| {n:,} | {t_py:.2f} | {t_np:.2f} | {t_py / t_np:.0f}× |")


if __name__ == "__main__":
    main()
