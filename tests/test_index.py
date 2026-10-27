"""Index build, the Index data model, persistence."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from findex.corpus import Document
from findex.index import Index, build_index
from findex.models import DocMeta, Posting
from findex.search import boolean_search
from findex.store import IndexFormatError, load, open_index, save

DOCS = [
    Document(0, "a.txt", "the cat sat on the mat", "a"),
    Document(1, "b.txt", "the dog sat", "b"),
    Document(2, "c.txt", "a cat and a dog", "c"),
]


def test_records_are_hashable() -> None:
    assert len({Posting(1, 2), Posting(1, 2), Posting(2, 1)}) == 2
    assert len({DocMeta(1, "p", "t"), DocMeta(1, "p", "t")}) == 1


def test_postings_sorted_with_tf_and_lengths() -> None:
    ix = build_index(DOCS)
    assert [p.doc_id for p in ix["the"]] == [0, 1]
    assert ix["the"][0].tf == 2
    assert [ix.doc_length(d) for d in range(3)] == [6, 3, 5]


def test_positions_flag() -> None:
    ix = build_index(DOCS, positions=True)
    assert ix["the"][0].positions == (0, 4)
    assert build_index(DOCS)["the"][0].positions == ()


def test_index_behaves_like_a_mapping() -> None:
    ix = build_index(DOCS)
    assert len(ix) == len(set(ix))
    assert "cat" in ix and "zebra" not in ix
    assert ix.get("zebra") is None
    with pytest.raises(KeyError):
        ix["zebra"]
    assert repr(ix) == "Index(terms=8, docs=3, positions=False)"
    assert ix.df("cat") == 2 and ix.df("zebra") == 0
    assert ix.avg_doc_length == pytest.approx(14 / 3)
    assert ix.num_docs == 3


def test_boolean_engines_agree() -> None:
    ix = build_index(DOCS)
    for q in ["cat", "cat dog", "cat OR dog", "the NOT cat", "sat dog OR a"]:
        assert boolean_search(ix, q, "merge") == boolean_search(ix, q, "set"), q
    assert boolean_search(ix, "cat dog") == [2]


@pytest.mark.parametrize("name", ["i.pkl", "i.json"])
def test_save_load_round_trip(tmp_path: Path, index: Index, name: str) -> None:
    save(index, tmp_path / name)
    assert load(tmp_path / name) == index


def test_open_index_releases_even_on_error(index_file: Path) -> None:
    captured: list[Index] = []
    with pytest.raises(RuntimeError), open_index(index_file) as ix:
        captured.append(ix)
        assert len(ix) > 0
        raise RuntimeError("boom")
    assert len(captured[0]) == 0  # closed by the context manager


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.pkl")
    junk = tmp_path / "junk.pkl"
    junk.write_bytes(b"not a pickle")
    with pytest.raises(IndexFormatError):
        load(junk)
    other = tmp_path / "other.pkl"
    other.write_bytes(pickle.dumps({"just": "a dict"}))
    with pytest.raises(IndexFormatError):
        load(other)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text('{"version": 99}')
    with pytest.raises(IndexFormatError):
        load(bad_json)
