"""Lab 2 checks: hashable records, sorted postings, merges agree with sets, round trip."""

from pathlib import Path

from findex.corpus import Document
from findex.index import build_index
from findex.models import DocMeta, Posting
from findex.search import boolean_search, merge_and, merge_not, merge_or
from findex.store import load, save

DOCS = [
    Document(0, "a.txt", "the cat sat on the mat", "a"),
    Document(1, "b.txt", "the dog sat", "b"),
    Document(2, "c.txt", "a cat and a dog", "c"),
]


def test_records_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert len({DocMeta(1, "p", "t"), DocMeta(1, "p", "t")}) == 1


def test_postings_sorted_and_tf() -> None:
    ix = build_index(DOCS)
    assert [p.doc_id for p in ix.postings["the"]] == [0, 1]
    assert ix.postings["the"][0].tf == 2
    assert ix.doc_lengths == {0: 6, 1: 3, 2: 5}


def test_merges() -> None:
    a, b = [1, 3, 5, 7], [3, 4, 5, 8]
    assert merge_and(a, b) == [3, 5]
    assert merge_or(a, b) == [1, 3, 4, 5, 7, 8]
    assert merge_not(a, b) == [1, 7]


def test_boolean_engines_agree() -> None:
    ix = build_index(DOCS)
    for q in ["cat", "cat dog", "cat OR dog", "the NOT cat", "sat dog OR a"]:
        assert boolean_search(ix, q, "merge") == boolean_search(ix, q, "set"), q
    assert boolean_search(ix, "cat dog") == [2]


def test_round_trip(tmp_path: Path) -> None:
    ix = build_index(DOCS, positions=True)
    for name in ("i.pkl", "i.json"):
        assert load_eq(ix, tmp_path / name)


def load_eq(ix: object, path: Path) -> bool:
    save(ix, path)  # type: ignore[arg-type]
    return load(path) == ix
