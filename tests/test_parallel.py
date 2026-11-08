"""Lab 5: every executor builds the byte-identical index; worker errors surface."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from findex.corpus import iter_documents, iter_text_files
from findex.index import build_index
from findex.parallel import Chunk, build_parallel, build_partial, make_chunks, make_pool, merge


def test_serial_map_reduce_equals_lab4_build(corpus_dir: Path) -> None:
    whole = Chunk(str(corpus_dir), 0, tuple(str(p) for p in iter_text_files(corpus_dir)))
    reference = build_index(iter_documents(corpus_dir))
    rebuilt = merge([build_partial(whole)])
    assert rebuilt == reference
    assert pickle.dumps(rebuilt) == pickle.dumps(reference)


def test_chunks_cover_all_paths_contiguously(corpus_dir: Path) -> None:
    chunks = make_chunks(corpus_dir, 3)
    assert [c.start for c in chunks] == [0, 3, 5]
    assert sum(len(c.paths) for c in chunks) == 7
    assert make_chunks(corpus_dir, 100)[-1].start == 6  # never more chunks than files


@pytest.mark.slow
@pytest.mark.parametrize("positions", [False, True])
def test_all_executors_byte_identical(corpus_dir: Path, positions: bool) -> None:
    reference = pickle.dumps(build_index(iter_documents(corpus_dir), positions=positions))
    for executor in ("serial", "threads", "processes"):
        ix, stats = build_parallel(
            corpus_dir,
            executor,
            workers=2,
            positions=positions,
            chunks_per_worker=2,
            start_method="spawn",
        )
        assert pickle.dumps(ix) == reference, executor
        assert stats.merge_seconds >= 0


@pytest.mark.parametrize("positions", [False, True])
def test_merge_order_does_not_matter(corpus_dir: Path, positions: bool) -> None:
    parts = [build_partial(c) for c in make_chunks(corpus_dir, 4, positions)]
    reference = build_index(iter_documents(corpus_dir), positions=positions)
    assert merge(reversed(parts)) == merge(parts) == reference
    assert len(merge([])) == 0


def test_partial_is_compact(corpus_dir: Path) -> None:
    part = build_partial(make_chunks(corpus_dir, 1)[0])
    assert part.offsets[-1] == len(part.doc_ids) == len(part.tfs)
    assert len(part.terms) + 1 == len(part.offsets)
    assert pickle.loads(pickle.dumps(part)) == part


@pytest.mark.slow
@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_worker_exception_surfaces(tmp_path: Path, corpus_dir: Path, executor: str) -> None:
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not under the corpus root")
    bad = Chunk(str(corpus_dir), 0, (str(outside),))
    with (
        make_pool(executor, 2, "spawn") as pool,  # type: ignore[arg-type]
        pytest.raises(ValueError, match="is not in the subpath") as info,
    ):
        pool.submit(build_partial, bad).result()
    if executor == "processes":  # the worker's own traceback travels back as __cause__
        assert "build_partial" in str(info.value.__cause__)


def test_non_directory_corpus_falls_back_to_serial(tmp_path: Path) -> None:
    f = tmp_path / "c.jsonl"
    f.write_text('{"text": "hello world"}\n{"text": "hello again"}\n')
    ix, stats = build_parallel(f, "processes", workers=2)
    assert ix.num_docs == 2 and stats.chunks == 1
