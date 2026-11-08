"""Lab 8 M3: precision@5 for keyword / semantic / hybrid on 10 + 5 paraphrase queries.

    uv run findex embed data/index.pkl --model all-MiniLM-L6-v2      # or --model lsa
    uv run python scripts/lab08_eval.py data/index.pkl data/pydocs

Paraphrase queries deliberately avoid the documents' own vocabulary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab03_evaluate import QRELS  # noqa: E402  (the 10 hand-labelled queries of Lab 3)

from findex.fast import NumpySearchEngine  # noqa: E402
from findex.semantic import SemanticIndex, emb_dir, search_mode  # noqa: E402
from findex.store import load  # noqa: E402

PARAPHRASES: dict[str, set[str]] = {
    "turn objects into bytes and back": {
        "library/pickle.rst",
        "library/marshal.rst",
        "library/struct.rst",
        "library/json.rst",
        "library/copyreg.rst",
        "library/shelve.rst",
    },
    "make my program faster, find the slow parts": {
        "library/profile.rst",
        "library/timeit.rst",
        "howto/perf_profiling.rst",
        "library/tracemalloc.rst",
        "library/profiling.tracing.rst",
        "library/pstats.rst",
    },
    "send mail to someone from a script": {
        "library/smtplib.rst",
        "library/email.rst",
        "library/email.message.rst",
        "library/email.examples.rst",
    },
    "remember the results of an expensive function": {"library/functools.rst"},
    "run several jobs at the same time": {
        "library/threading.rst",
        "library/concurrent.futures.rst",
        "library/multiprocessing.rst",
        "library/concurrency.rst",
        "library/asyncio.rst",
        "library/asyncio-task.rst",
        "howto/free-threading-python.rst",
    },
}


def main() -> None:
    index_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index.pkl")
    root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/pydocs")
    ix = load(index_path)
    sem = SemanticIndex(emb_dir(index_path))
    engine = NumpySearchEngine(ix)
    print(f"embeddings: {sem.meta['model']} ({sem.meta['dim']}-d, {sem.meta['chunks']} chunks)\n")
    print("| Query | Kind | Keyword P@5 | Semantic P@5 | Hybrid P@5 |")
    print("|---|---|---|---|---|")
    sums = {"keyword": 0.0, "semantic": 0.0, "hybrid": 0.0}
    sub = {"original": dict(sums), "paraphrase": dict(sums)}
    for kind, qrels in (("original", QRELS), ("paraphrase", PARAPHRASES)):
        for query, relevant in qrels.items():
            row = []
            for mode in ("keyword", "semantic", "hybrid"):
                # free text: words ORed for the keyword side (as in Lab 3)
                q = " OR ".join(query.replace(",", "").split()) if mode != "semantic" else query
                if mode == "hybrid":
                    hits = search_mode(engine, sem, q, 5, "hybrid")  # keyword side uses q
                else:
                    hits = search_mode(engine, sem, q, 5, mode)  # type: ignore[arg-type]
                top = [Path(h.path).relative_to(root).as_posix() for h in hits]
                p5 = sum(t in relevant for t in top) / 5
                sums[mode] += p5
                sub[kind][mode] += p5
                row.append(p5)
            print(f"| {query} | {kind} | " + " | ".join(f"{x:.1f}" for x in row) + " |")
    for kind, n in (("original", len(QRELS)), ("paraphrase", len(PARAPHRASES))):
        m = sub[kind]
        print(
            f"| **mean, {kind} ({n})** | | **{m['keyword'] / n:.2f}** | "
            f"**{m['semantic'] / n:.2f}** | **{m['hybrid'] / n:.2f}** |"
        )
    n = len(QRELS) + len(PARAPHRASES)
    print(
        f"| **mean, all {n}** | | **{sums['keyword'] / n:.2f}** | "
        f"**{sums['semantic'] / n:.2f}** | **{sums['hybrid'] / n:.2f}** |"
    )


if __name__ == "__main__":
    main()
