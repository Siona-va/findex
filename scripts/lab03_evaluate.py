"""Lab 3: ranking sanity checks and precision@5 for TF-IDF vs BM25.

    uv run python scripts/lab03_evaluate.py data/pydocs

The relevance labels below were assigned by hand by reading the Python docs table of
contents: for each query, the files a person would *want* in the top 5.
"""

from __future__ import annotations

import sys
from pathlib import Path

from findex.corpus import iter_documents
from findex.index import Index, build_index
from findex.models import Posting
from findex.scoring import BM25, Scorer, TfIdf
from findex.search import SearchEngine

QRELS: dict[str, set[str]] = {
    "asyncio event loop": {
        "library/asyncio-eventloop.rst",
        "library/asyncio.rst",
        "howto/a-conceptual-overview-of-asyncio.rst",
        "library/asyncio-runner.rst",
        "library/asyncio-dev.rst",
        "library/asyncio-llapi-index.rst",
        "library/asyncio-extending.rst",
        "library/asyncio-platforms.rst",
    },
    "regular expression": {"library/re.rst", "howto/regex.rst"},
    "unicode normalization": {"howto/unicode.rst", "library/unicodedata.rst"},
    "logging handler": {
        "library/logging.rst",
        "library/logging.handlers.rst",
        "howto/logging.rst",
        "howto/logging-cookbook.rst",
        "library/logging.config.rst",
    },
    "descriptor": {"howto/descriptor.rst", "reference/datamodel.rst", "builtins/functions.rst"},
    "dataclass frozen": {"library/dataclasses.rst"},
    "garbage collection reference cycles": {
        "library/gc.rst",
        "c-api/gcsupport.rst",
        "library/weakref.rst",
        "reference/datamodel.rst",
    },
    "serialize objects": {
        "library/pickle.rst",
        "library/json.rst",
        "library/marshal.rst",
        "library/shelve.rst",
        "library/copyreg.rst",
        "library/pickletools.rst",
    },
    "thread pool executor": {
        "library/concurrent.futures.rst",
        "library/asyncio-eventloop.rst",
        "library/threading.rst",
        "library/concurrent.rst",
    },
    "command line arguments parser": {
        "library/argparse.rst",
        "howto/argparse.rst",
        "library/optparse.rst",
        "library/getopt.rst",
        "howto/argparse-optparse.rst",
        "library/sys.rst",
    },
}


def rel(path: str, root: Path) -> str:
    return Path(path).relative_to(root).as_posix()


def precision_at_5(index: Index, scorer: Scorer, root: Path) -> list[float]:
    engine = SearchEngine(index, scorer)
    out: list[float] = []
    for query, relevant in QRELS.items():
        # Free-text query: any term may match (OR), ranking does the rest.
        q = " OR ".join(query.split())
        top = [rel(r.path, root) for r in engine.search(q, k=5)]
        out.append(sum(p in relevant for p in top) / 5)
    return out


def sanity_checks(index: Index, root: Path) -> None:
    bm25, tfidf = BM25(), TfIdf()
    print("### 1. A rare query term outranks a common one\n")
    engine = SearchEngine(index, bm25)
    res = engine.search("the OR zipapp", k=len(index.all_doc_ids))
    has_rare = {p.doc_id for p in index["zipapp"]}
    best_rare = next(r for r in res if r.doc_id in has_rare)
    best_common = next(r for r in res if r.doc_id not in has_rare)
    print(f"query `the OR zipapp`: df(the)={index.df('the')}, df(zipapp)={index.df('zipapp')}")
    print(f"- best doc WITH zipapp:    {best_rare.score:.3f}  {rel(best_rare.path, root)}")
    print(f"- best doc with only `the`: {best_common.score:.3f}  {rel(best_common.path, root)}")
    print(
        f"- idf(the) = {bm25.idf('the', index):.3f}, idf(zipapp) = "
        f"{bm25.idf('zipapp', index):.3f}\n"
    )

    print("### 2. BM25 saturates term frequency\n")
    avg = index.avg_doc_length
    print("| tf | BM25 tf-weight | gain of this occurrence | TF-IDF tf-weight |")
    print("|---|---|---|---|")
    for tf in (1, 2, 3, 5, 10, 19, 20):
        w = bm25.tf_weight(tf, round(avg), avg)
        gain = w - bm25.tf_weight(tf - 1, round(avg), avg)
        t = tfidf.score("asyncio", Posting(0, tf), index) / index.idf("asyncio")
        print(f"| {tf} | {w:.3f} | +{gain:.3f} | {t:.3f} |")
    print(f"\n(BM25 weight can never exceed k1 + 1 = {bm25.k1 + 1})\n")

    print("### 3. A short document with one match beats a long one with one match\n")
    for term in sorted(index, key=index.df):
        ones = [p for p in index[term] if p.tf == 1]
        if index.df(term) < 4 or len(ones) < 2:
            continue
        short = min(ones, key=lambda p: index.doc_length(p.doc_id))
        long = max(ones, key=lambda p: index.doc_length(p.doc_id))
        if index.doc_length(long.doc_id) > 20 * index.doc_length(short.doc_id):
            break
    else:  # pragma: no cover
        return
    for label, p in (("short", short), ("long", long)):
        meta = index.doc_meta(p.doc_id)
        print(
            f"- {label}: `{rel(meta.path, root)}` length={index.doc_length(p.doc_id)} "
            f"tf=1 → BM25 {bm25.score(term, p, index):.3f}, "
            f"TF-IDF {tfidf.score(term, p, index):.3f}"
        )
    print(f"(term `{term}`; TF-IDF ignores length, so both get the same score)\n")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/pydocs")
    index = build_index(iter_documents(root))
    sanity_checks(index, root)

    print("### precision@5\n")
    p_tfidf = precision_at_5(index, TfIdf(), root)
    p_bm25 = precision_at_5(index, BM25(), root)
    print("| Query | TF-IDF P@5 | BM25 P@5 |")
    print("|---|---|---|")
    for q, a, b in zip(QRELS, p_tfidf, p_bm25, strict=True):
        print(f"| {q} | {a:.1f} | {b:.1f} |")
    print(f"| **mean** | **{sum(p_tfidf) / 10:.2f}** | **{sum(p_bm25) / 10:.2f}** |")


if __name__ == "__main__":
    main()
