from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path

import pytest

from findex.corpus import iter_documents


def test_reads_directory_tree(corpus_dir: Path) -> None:
    docs = list(iter_documents(corpus_dir))
    assert len(docs) == 7
    assert [d.doc_id for d in docs] == list(range(7))  # stable, sorted order
    assert any(d.path.endswith("вірш.txt") for d in docs)


def test_first_document_is_lazy(corpus_dir: Path) -> None:
    gen = iter_documents(corpus_dir)
    first = next(gen)
    assert first.doc_id == 0
    gen.close()


def test_bad_encoding_is_logged_not_fatal(
    corpus_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="findex.corpus"):
        docs = {Path(d.path).name: d for d in iter_documents(corpus_dir)}
    assert "�" in docs["broken.txt"].text
    assert "bad UTF-8" in caplog.text


def test_jsonl_and_gzip(tmp_path: Path) -> None:
    lines = [{"title": "One", "text": "first doc"}, {"text": "second doc"}]
    payload = "\n".join(json.dumps(x) for x in lines) + "\n\nnot json\n"
    plain = tmp_path / "c.jsonl"
    plain.write_text(payload, encoding="utf-8")
    packed = tmp_path / "c.jsonl.gz"
    packed.write_bytes(gzip.compress(payload.encode()))
    for path in (plain, packed):
        docs = list(iter_documents(path))
        assert [d.text for d in docs] == ["first doc", "second doc"]
        assert docs[0].title == "One"


def test_missing_corpus_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        next(iter_documents(tmp_path / "nope"))
